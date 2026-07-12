from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

import config

logger = logging.getLogger(__name__)

# ── Prompt template (loaded once) ─────────────────────────────────────────────
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "interviewer.txt"
_PROMPT_TEMPLATE: str = _TEMPLATE_PATH.read_text(encoding="utf-8")

_DIFFICULTY_LABELS = {1: "very easy", 2: "easy", 3: "medium", 4: "hard", 5: "very hard"}


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════

class InterviewerOutput(BaseModel):
    """Structured output from the Interviewer agent."""

    question: str = Field(..., min_length=10)
    category: str = Field(..., pattern=r"^(technical|soft_skill)$")
    tags: list[str] = Field(..., min_length=1, max_length=5)
    difficulty: str = Field(..., pattern=r"^(easy|medium|hard)$")
    difficulty_delta: int = Field(..., ge=-1, le=1)

    @field_validator("tags", mode="before")
    @classmethod
    def normalise_tags(cls, v: list[str]) -> list[str]:
        """Lowercase and strip whitespace from all tags."""
        return [t.strip().lower() for t in v if t.strip()]


class InterviewerContext(BaseModel):
   
    field: str
    level: str = "mid"                  # entry | junior | mid | senior | manager_lead
    category: str                       # "technical" | "soft_skill"
    current_difficulty: int             # 1-5
    turns_done: int
    total_questions: int
    remaining_technical: int
    remaining_soft: int
    current_session_tags: list[str]     # tags used so far this session
    all_time_tags: list[str]            # user's full deduplicated tag index


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def ask_next_question(ctx: InterviewerContext) -> InterviewerOutput:
    """
    Build the Interviewer prompt from ctx, call Groq, validate the response.
    Retries once on malformed JSON / schema mismatch before raising.
    """
    prompt = _build_prompt(ctx)
    logger.debug("Interviewer prompt built (%d chars)", len(prompt))

    for attempt in range(2):            # try up to 2 times
        raw = _call_groq(prompt)
        try:
            data = _extract_json(raw)
            result = InterviewerOutput(**data)
            logger.info(
                "Interviewer → q=%r tags=%s diff=%s delta=%d",
                result.question[:60],
                result.tags,
                result.difficulty,
                result.difficulty_delta,
            )
            return result
        except Exception as exc:
            if attempt == 0:
                logger.warning("Interviewer output invalid on attempt 1: %s — retrying", exc)
            else:
                logger.error("Interviewer output invalid on attempt 2: %s", exc)
                raise ValueError(f"Interviewer returned malformed output after 2 attempts: {exc}") from exc

    # mypy / type-checker unreachable guard
    raise RuntimeError("Unreachable")


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(ctx: InterviewerContext) -> str:
    """Fill the prompt template with values from the context."""

    difficulty_label = _DIFFICULTY_LABELS.get(ctx.current_difficulty, "medium")

    # Format the tag lists for readability inside the prompt
    session_tags_str = (
        ", ".join(ctx.current_session_tags) if ctx.current_session_tags else "(none yet)"
    )
    all_time_tags_str = (
        ", ".join(ctx.all_time_tags) if ctx.all_time_tags else "(no prior history)"
    )

    return _PROMPT_TEMPLATE.format(
        field=ctx.field,
        level=ctx.level,
        category=ctx.category,
        current_difficulty=ctx.current_difficulty,
        turns_done=ctx.turns_done,
        total_questions=ctx.total_questions,
        remaining_technical=ctx.remaining_technical,
        remaining_soft=ctx.remaining_soft,
        current_session_tags=session_tags_str,
        all_time_tags=all_time_tags_str,
        difficulty_label=difficulty_label,
    )


def _call_groq(prompt: str) -> str:
    """Call the Groq chat completions endpoint and return the raw assistant text."""
    response = config.groq_client.chat.completions.create(
        model=config.INTERVIEWER_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        # 512 was too tight for longer/hard questions (e.g. multi-part system
        # design prompts) — the response got truncated mid-JSON, failing
        # _extract_json and burning a retry. getattr keeps this overridable
        # via config without requiring the attribute to exist yet.
        max_tokens=getattr(config, "INTERVIEWER_MAX_TOKENS", 1024),
    )
    return response.choices[0].message.content or ""


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Extract a JSON object from the model's response.
    Handles the common case where the model wraps JSON in markdown fences.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
        text = inner.strip()

    # Find the first { … } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {raw[:200]!r}")

    return json.loads(text[start:end])