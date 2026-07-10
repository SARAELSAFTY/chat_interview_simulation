from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

# ── Primary session storage ────────────────────────────────────────────────────
# { session_id (str) -> session_record (dict) }
_sessions: dict[str, dict[str, Any]] = {}

# ── Per-user tag index ─────────────────────────────────────────────────────────
# { user_id (str) -> list[str] } — deduplicated, preserves insertion order
# This is the core of the history / duplicate-avoidance strategy.
_tag_index: dict[str, list[str]] = defaultdict(list)

# ── Per-user session list (ordered, for the "last N sessions" window) ──────────
# { user_id (str) -> list[session_id (str)] } — in chronological order
_user_sessions: dict[str, list[str]] = defaultdict(list)

# How many completed sessions to include in the Interviewer's context window.
RECENT_SESSION_WINDOW = 2

# Starting point on the 1-5 difficulty scale for each experience level.
# Without this, every session started at the same "medium" difficulty
# regardless of the level the user picked.
_LEVEL_START_DIFFICULTY: dict[str, int] = {
    "entry": 1,
    "junior": 2,
    "mid": 3,
    "senior": 4,
    "manager_lead": 4,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Session CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def create_session(
    user_id: str,
    field: str,
    question_mix: dict[str, int],   # {"technical": N, "soft": M}
    level: str = "mid",              # entry | junior | mid | senior | manager_lead
) -> dict[str, Any]:
    """
    Create a new session record, register it under the user's session list,
    and return the full record dict.
    """
    session_id = str(uuid.uuid4())
    now = _utcnow()

    # Build interleaved question sequence (technical-first alternating, remainder appended)
    sequence = _build_question_sequence(question_mix)

    session: dict[str, Any] = {
        "session_id": session_id,
        "user_id": user_id,
        "field": field,
        "level": level,
        "question_mix": question_mix,
        "status": "in_progress",
        # 1–5 scale, scaled to the chosen level instead of always starting at "medium"
        "starting_difficulty": _LEVEL_START_DIFFICULTY.get(level, 3),
        "current_difficulty": _LEVEL_START_DIFFICULTY.get(level, 3),
        "created_at": now,
        "completed_at": None,
        "turns": [],
        "review": None,
        # Internal sequencing — not part of the public schema, used server-side only
        "_question_sequence": sequence,  # list[str], each entry "technical" or "soft_skill"
        "_current_turn_index": 0,        # next question is sequence[this index]
    }

    _sessions[session_id] = session
    _user_sessions[user_id].append(session_id)
    return session


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return the session record or None if not found."""
    return _sessions.get(session_id)


def require_session(session_id: str) -> dict[str, Any]:
    """Return the session record, raising ValueError if not found."""
    s = _sessions.get(session_id)
    if s is None:
        raise ValueError(f"Session '{session_id}' not found.")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Turn management
# ═══════════════════════════════════════════════════════════════════════════════

def append_turn(
    session_id: str,
    *,
    question: str,
    category: str,              # "technical" | "soft_skill"
    tags: list[str],
    difficulty: str,            # "easy" | "medium" | "hard"
    difficulty_delta: int,
    answer: str,
) -> dict[str, Any]:
    """
    Append a completed turn to the session, update adaptive difficulty, and
    merge the turn's tags into the user's cross-session tag index.

    Returns the appended turn dict.
    """
    session = require_session(session_id)

    turn_id = len(session["turns"]) + 1
    turn: dict[str, Any] = {
        "turn_id": turn_id,
        "category": category,
        "question": question,
        "tags": tags,
        "difficulty": difficulty,
        "difficulty_delta": difficulty_delta,
        "answer": answer,
        "answered_at": _utcnow(),
    }

    session["turns"].append(turn)
    session["_current_turn_index"] += 1

    # Adaptive difficulty — clamp to [1, 5]
    new_diff = session["current_difficulty"] + difficulty_delta
    session["current_difficulty"] = max(1, min(5, new_diff))

    # ── TAG INDEX UPDATE (the critical history mechanism) ──────────────────────
    # Merge new tags into the user's deduplicated tag list. This is what ensures
    # the Interviewer can check all-time history without loading old transcripts.
    _merge_tags(session["user_id"], tags)

    return turn


def mark_session_complete(session_id: str, review: dict[str, Any]) -> None:
    """Mark the session complete and attach the Evaluator's review."""
    session = require_session(session_id)
    session["status"] = "complete"
    session["completed_at"] = _utcnow()
    session["review"] = review


# ═══════════════════════════════════════════════════════════════════════════════
# History retrieval (used by Interviewer context builder)
# ═══════════════════════════════════════════════════════════════════════════════

def get_recent_completed_sessions(
    user_id: str,
    n: int = RECENT_SESSION_WINDOW,
) -> list[dict[str, Any]]:
    """
    Return the last `n` *completed* sessions for this user in chronological
    order, each as a lightweight summary dict safe to inject into a prompt.

    Only completed sessions are included — in-progress sessions never appear
    in the Interviewer's historical context.
    """
    all_ids = _user_sessions.get(user_id, [])
    completed = [
        _sessions[sid]
        for sid in reversed(all_ids)           # newest first …
        if sid in _sessions
        and _sessions[sid]["status"] == "complete"
    ][:n]                                       # … take only n
    completed.reverse()                         # back to chronological order

    return [_session_to_context_summary(s) for s in completed]


def get_tag_index(user_id: str, max_tags: int | None = None) -> list[str]:
    """
    Return the user's deduplicated tag list (oldest-first insertion order).

    If `max_tags` is provided, only the most-recent `max_tags` entries are
    returned — this caps prompt size for long-tenured users (see §12.4).
    """
    tags = _tag_index.get(user_id, [])
    if max_tags is not None and len(tags) > max_tags:
        # Keep the most recent tags — they're more likely to be relevant
        tags = tags[-max_tags:]
    return list(tags)


def get_used_tags_current_session(session_id: str) -> list[str]:
    """
    Return tags used so far in the *current* session only.
    Used to help the Interviewer avoid repeating topics mid-session.
    """
    session = require_session(session_id)
    seen: list[str] = []
    for turn in session["turns"]:
        for tag in turn["tags"]:
            if tag not in seen:
                seen.append(tag)
    return seen


def get_all_sessions_for_user(user_id: str) -> list[dict[str, Any]]:
    """
    Return all sessions for the user (completed + in-progress), ordered
    chronologically. Used for debug / admin endpoints.
    """
    return [
        _sessions[sid]
        for sid in _user_sessions.get(user_id, [])
        if sid in _sessions
    ]


def get_full_tag_index_stats(user_id: str) -> dict[str, Any]:
    """
    Return tag statistics for a user — useful for debugging and monitoring
    prompt-size growth without leaking session content.
    """
    tags = _tag_index.get(user_id, [])
    session_count = len(_user_sessions.get(user_id, []))
    return {
        "user_id": user_id,
        "total_unique_tags": len(tags),
        "total_sessions": session_count,
        "completed_sessions": sum(
            1 for sid in _user_sessions.get(user_id, [])
            if _sessions.get(sid, {}).get("status") == "complete"
        ),
        "tag_sample_recent_10": tags[-10:],   # last 10 for quick inspection
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sequencing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def next_question_category(session_id: str) -> str | None:
    """
    Return the category for the next question, or None if the session is done.
    """
    session = require_session(session_id)
    idx = session["_current_turn_index"]
    seq = session["_question_sequence"]
    if idx >= len(seq):
        return None
    return seq[idx]


def is_session_complete(session_id: str) -> bool:
    """
    True when all questions in the sequence have been answered.
    """
    session = require_session(session_id)
    return session["_current_turn_index"] >= len(session["_question_sequence"])


def total_questions(session_id: str) -> int:
    """Total number of questions in this session."""
    session = require_session(session_id)
    return len(session["_question_sequence"])


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_question_sequence(question_mix: dict[str, int]) -> list[str]:
    """
    Build an interleaved list of categories (technical-first alternating),
    with any remainder appended at the end.

    Example: {technical:5, soft:3} → [T,S,T,S,T,S,T,T,T]  (3 interleaved pairs + 2 T remainder)
    """
    t_count = question_mix.get("technical", 0)
    s_count = question_mix.get("soft", 0)

    interleaved: list[str] = []
    t_rem, s_rem = t_count, s_count

    while t_rem > 0 and s_rem > 0:
        interleaved.append("technical")
        t_rem -= 1
        interleaved.append("soft_skill")
        s_rem -= 1

    # Append whatever is left
    interleaved.extend(["technical"] * t_rem)
    interleaved.extend(["soft_skill"] * s_rem)
    return interleaved


def _merge_tags(user_id: str, new_tags: list[str]) -> None:
    """
    Deduplicated append: add each tag from `new_tags` to the user's tag index
    only if it isn't already present.  Insertion order is preserved.
    """
    existing_set = set(_tag_index[user_id])
    for tag in new_tags:
        tag = tag.strip().lower()
        if tag and tag not in existing_set:
            _tag_index[user_id].append(tag)
            existing_set.add(tag)


def _session_to_context_summary(session: dict[str, Any]) -> dict[str, Any]:
    """
    Reduce a full session record to the compact summary dict that gets
    injected into the Interviewer's context window.
    Only includes what the Interviewer is allowed to see (§6.3).
    """
    return {
        "session_id": session["session_id"],
        "field": session["field"],
        "question_mix": session["question_mix"],
        "completed_at": session.get("completed_at"),
        "turns": [
            {
                "turn_id": t["turn_id"],
                "category": t["category"],
                "question": t["question"],
                "tags": t["tags"],
                "difficulty": t["difficulty"],
                # answer is included so the Interviewer knows what was already discussed
                "answer_snippet": t["answer"][:200] if t.get("answer") else "",
            }
            for t in session.get("turns", [])
        ],
    }


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()