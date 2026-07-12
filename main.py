from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import state.session_store as store
from services.interviewer import InterviewerContext, InterviewerOutput, ask_next_question
from services.evaluator import EvaluatorContext, EvaluatorOutput, produce_review

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Retry configuration ────────────────────────────────────────────────────────
# Broad net of "this is probably a transient network/DNS issue, not a real
# application error" exception types. requests/urllib3 are optional imports —
# whichever HTTP client the agent services use under the hood, we try to catch
# its connection-layer exceptions specifically rather than retrying everything
# (retrying a genuine bad-request or auth error just wastes time and delays
# the real error reaching the client).
_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, OSError)

try:
    import requests.exceptions as _requests_exc

    _NETWORK_EXCEPTIONS += (
        _requests_exc.ConnectionError,
        _requests_exc.Timeout,
        _requests_exc.ConnectTimeout,
    )
except ImportError:
    pass

try:
    import urllib3.exceptions as _urllib3_exc

    _NETWORK_EXCEPTIONS += (
        _urllib3_exc.MaxRetryError,
        _urllib3_exc.NameResolutionError,
        _urllib3_exc.NewConnectionError,
    )
except ImportError:
    pass

AGENT_MAX_ATTEMPTS = getattr(config, "AGENT_MAX_ATTEMPTS", 3)
AGENT_RETRY_MIN_WAIT = getattr(config, "AGENT_RETRY_MIN_WAIT_SECONDS", 2)
AGENT_RETRY_MAX_WAIT = getattr(config, "AGENT_RETRY_MAX_WAIT_SECONDS", 15)

T = TypeVar("T")


def _call_with_retry(fn: Callable[[], T], *, label: str) -> T:
    """
    Manual retry with exponential backoff, stdlib only (no external
    dependencies). Retries only on exceptions in _NETWORK_EXCEPTIONS
    (transient connection/DNS issues). Anything else — a real application
    error, bad input, auth failure, etc. — is raised immediately on the
    first attempt so we don't waste time retrying an error that will
    never succeed.
    """
    attempt = 1
    delay = AGENT_RETRY_MIN_WAIT
    while True:
        try:
            return fn()
        except _NETWORK_EXCEPTIONS as e:
            if attempt >= AGENT_MAX_ATTEMPTS:
                logger.error("%s failed after %d attempts: %s", label, attempt, e)
                raise
            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                label, attempt, AGENT_MAX_ATTEMPTS, delay, e,
            )
            time.sleep(delay)
            delay = min(delay * 2, AGENT_RETRY_MAX_WAIT)
            attempt += 1


def _call_interviewer(ctx: InterviewerContext) -> InterviewerOutput:
    return _call_with_retry(lambda: ask_next_question(ctx), label="Interviewer call")


def _call_evaluator(ctx: EvaluatorContext) -> EvaluatorOutput:
    return _call_with_retry(lambda: produce_review(ctx), label="Evaluator call")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Interview Simulator",
    version="1.2.0",
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all so an unexpected error never leaks a raw traceback or internal
    exception string to the client. Everything relevant is still logged.
    """
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Serve index.html at root ───────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent

@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the HTML frontend at the root URL."""
    return FileResponse(_BASE_DIR / "index.html")

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness check. Does not probe external agent APIs — those can fail
    independently of the app being up, which is exactly the failure mode
    this endpoint should stay silent about."""
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class StartRequest(BaseModel):
    user_id: str = Field(..., description="Client-generated UUID identifying the user.")
    field: str = Field(..., min_length=2, max_length=200, description="Job title or field, e.g. 'Backend Developer'.")
    question_mix: dict[str, int] = Field(
        ...,
        description='Question counts, e.g. {"technical": 5, "soft": 3}.',
        examples=[{"technical": 5, "soft": 3}],
    )
    level: str = Field(
        "mid",
        pattern=r"^(entry|junior|mid|senior|manager_lead)$",
        description="Target experience level for this interview.",
    )

    model_config = {"json_schema_extra": {"example": {
        "user_id": "550e8400-e29b-41d4-a716-446655440000",
        "field": "Backend Developer",
        "question_mix": {"technical": 5, "soft": 3},
        "level": "mid",
    }}}


class QuestionResponse(BaseModel):
    session_id: str
    turn_id: int
    question: str
    category: str
    tags: list[str]
    difficulty: str
    turns_done: int
    total_questions: int
    status: Literal["in_progress"]


class AnswerRequest(BaseModel):
    session_id: str = Field(..., description="Session ID returned by /start.")
    answer: str = Field(..., min_length=1, description="The user's answer text.")

    model_config = {"json_schema_extra": {"example": {
        "session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
        "answer": "I would use a message queue to decouple the services...",
    }}}


class AnswerResponse(BaseModel):
    
    session_id: str
    status: Literal["in_progress", "complete", "awaiting_review"]
    turns_done: int
    total_questions: int
    next_question: QuestionResponse | None = None
    review: dict[str, Any] | None = None
    detail: str | None = None


class RetryReviewResponse(BaseModel):
    session_id: str
    status: Literal["complete", "awaiting_review"]
    review: dict[str, Any] | None = None
    detail: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tag/history debug endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/debug/tags/{user_id}", tags=["debug"])
def debug_tag_stats(user_id: str) -> dict[str, Any]:
    """
    Return tag index statistics for a user.
    Useful for monitoring prompt-size growth over time (§12.4).
    """
    return store.get_full_tag_index_stats(user_id)


@app.get("/debug/tags/{user_id}/full", tags=["debug"])
def debug_full_tag_index(user_id: str) -> dict[str, Any]:
    """Return the complete tag list for a user."""
    tags = store.get_tag_index(user_id)
    return {"user_id": user_id, "total": len(tags), "tags": tags}


@app.get("/debug/sessions/{user_id}", tags=["debug"])
def debug_user_sessions(user_id: str) -> list[dict[str, Any]]:
    """Return all sessions for a user (completed + in-progress)."""
    sessions = store.get_all_sessions_for_user(user_id)
    # Strip internal fields before returning
    return [_public_session(s) for s in sessions]


@app.get("/debug/sessions/{user_id}/recent", tags=["debug"])
def debug_recent_sessions(user_id: str) -> list[dict[str, Any]]:
    """Return the last N completed sessions as the Interviewer would see them."""
    return store.get_recent_completed_sessions(user_id)


@app.get("/session/{session_id}", tags=["session"])
def get_session(session_id: str) -> dict[str, Any]:
    """Return the full public view of a session record."""
    try:
        s = store.require_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _public_session(s)


# ═══════════════════════════════════════════════════════════════════════════════
# Core endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/start", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED, tags=["interview"])
def start_session(req: StartRequest) -> QuestionResponse:
    """
    Create a new interview session and return the first question.

    Steps:
      1. Create session record.
      2. Load the user's full tag index + last 2 completed sessions.
      3. Call the Interviewer agent for question #1.
      4. Append the question as the current pending turn (no answer yet).
      5. Return the question to the client.
    """
    # 1. Create session
    session = store.create_session(
        user_id=req.user_id,
        field=req.field,
        question_mix=req.question_mix,
        level=req.level,
    )
    session_id = session["session_id"]
    logger.info("Session created: %s for user %s (field=%s, level=%s)", session_id, req.user_id, req.field, req.level)

    # 2. Get first question category from the sequence
    category = store.next_question_category(session_id)
    if category is None:
        raise HTTPException(status_code=400, detail="question_mix totals zero questions.")

    # 3. Build context and call Interviewer
    ctx = _build_interviewer_context(session, category)
    try:
        q_out: InterviewerOutput = _call_interviewer(ctx)
    except Exception as e:
        logger.error("Interviewer call failed on /start for session %s: %s", session_id, e)
        raise HTTPException(
            status_code=502,
            detail="The interviewer service is temporarily unavailable. Please try again.",
        )

    # 4. Append as a pending turn (answer="" until /answer is called)
    store.append_turn(
        session_id,
        question=q_out.question,
        category=q_out.category,
        tags=q_out.tags,
        difficulty=q_out.difficulty,
        difficulty_delta=q_out.difficulty_delta,
        answer="",          # will be filled in by /answer
    )

    logger.info("Question #1 generated: tags=%s diff=%s", q_out.tags, q_out.difficulty)

    return QuestionResponse(
        session_id=session_id,
        turn_id=1,
        question=q_out.question,
        category=q_out.category,
        tags=q_out.tags,
        difficulty=q_out.difficulty,
        turns_done=0,           # 0 answers recorded yet
        total_questions=store.total_questions(session_id),
        status="in_progress",
    )


@app.post("/answer", response_model=AnswerResponse, tags=["interview"])
def submit_answer(req: AnswerRequest) -> AnswerResponse:
    
    try:
        session = store.require_session(req.session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if session["status"] == "complete":
        raise HTTPException(status_code=400, detail="Session is already complete.")

    turns = session["turns"]
    if not turns:
        raise HTTPException(status_code=400, detail="No pending question found. Call /start first.")

    pending_turn = turns[-1]
    total_q = store.total_questions(req.session_id)
    answered_count = len(turns)
    is_final_turn = answered_count >= total_q

    if pending_turn.get("answer"):
        # This turn already has an answer recorded. The only legitimate reason
        # to hit /answer again in that state is retrying a failed final-turn
        # review — never allow it to silently overwrite an answer.
        if is_final_turn and session.get("status") == "awaiting_review":
            logger.info(
                "Session %s: retrying Evaluator via /answer (answer already recorded).",
                req.session_id,
            )
            return _run_evaluator_and_respond(req.session_id, session, turns, answered_count, total_q)
        raise HTTPException(
            status_code=400,
            detail="The current turn already has an answer. Something is out of sync.",
        )

    # ── Fill in the answer on the pending turn ─────────────────────────────────
    pending_turn["answer"] = req.answer.strip()
    pending_turn["answered_at"] = store._utcnow()

    logger.info(
        "Answer recorded for session %s turn %d/%d",
        req.session_id, answered_count, total_q,
    )

    # ── Check if the interview is done ────────────────────────────────────────
    if is_final_turn:
        return _run_evaluator_and_respond(req.session_id, session, turns, answered_count, total_q)

    # ── Not done: ask the next question ──────────────────────────────────────
    next_category = store.next_question_category(req.session_id)
    if next_category is None:
        raise HTTPException(status_code=500, detail="Sequence exhausted prematurely.")

    ctx = _build_interviewer_context(session, next_category)
    try:
        q_out = _call_interviewer(ctx)
    except Exception as e:
        logger.error("Interviewer failed for session %s: %s", req.session_id, e)
        raise HTTPException(
            status_code=502,
            detail="The interviewer service is temporarily unavailable. Please try again.",
        )

    store.append_turn(
        req.session_id,
        question=q_out.question,
        category=q_out.category,
        tags=q_out.tags,
        difficulty=q_out.difficulty,
        difficulty_delta=q_out.difficulty_delta,
        answer="",
    )

    next_turn_id = len(session["turns"])
    logger.info(
        "Question #%d generated for session %s: tags=%s",
        next_turn_id, req.session_id, q_out.tags,
    )

    return AnswerResponse(
        session_id=req.session_id,
        status="in_progress",
        turns_done=answered_count,
        total_questions=total_q,
        next_question=QuestionResponse(
            session_id=req.session_id,
            turn_id=next_turn_id,
            question=q_out.question,
            category=q_out.category,
            tags=q_out.tags,
            difficulty=q_out.difficulty,
            turns_done=answered_count,
            total_questions=total_q,
            status="in_progress",
        ),
        review=None,
    )


@app.post("/session/{session_id}/retry-review", response_model=RetryReviewResponse, tags=["interview"])
def retry_review(session_id: str) -> RetryReviewResponse:
    """
    Re-run the Evaluator for a session stuck in 'awaiting_review' after the
    final answer was already recorded. Safe to call repeatedly — it never
    touches answer data, only re-attempts the review generation.
    """
    try:
        session = store.require_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if session["status"] == "complete":
        raise HTTPException(status_code=400, detail="Session is already complete.")
    if session.get("status") != "awaiting_review":
        raise HTTPException(
            status_code=400,
            detail="Session is not awaiting review. Nothing to retry.",
        )

    turns = session["turns"]
    total_q = store.total_questions(session_id)
    answered_count = len(turns)

    result = _run_evaluator_and_respond(session_id, session, turns, answered_count, total_q)
    return RetryReviewResponse(
        session_id=session_id,
        status=result.status,  # "complete" or "awaiting_review"
        review=result.review,
        detail=result.detail,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator resilience helper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_evaluator_and_respond(
    session_id: str,
    session: dict[str, Any],
    turns: list[dict[str, Any]],
    answered_count: int,
    total_q: int,
) -> AnswerResponse:
    """
    Attempt the (already-retried-internally) Evaluator call. On success, marks
    the session complete and returns the review. On failure, marks the session
    'awaiting_review' — NOT an HTTP error — so the client can retry later
    without losing the recorded answers or getting stuck.
    """
    logger.info("Session %s: running Evaluator.", session_id)
    eval_ctx = EvaluatorContext(
        field=session["field"],
        level=session.get("level", "mid"),
        question_mix=session["question_mix"],
        turns=turns,
        start_difficulty=session.get("starting_difficulty", 3),
        end_difficulty=session["current_difficulty"],
    )

    try:
        review_out: EvaluatorOutput = _call_evaluator(eval_ctx)
    except Exception as e:
        logger.error("Evaluator failed for session %s after retries: %s", session_id, e)
        session["status"] = "awaiting_review"
        return AnswerResponse(
            session_id=session_id,
            status="awaiting_review",
            turns_done=answered_count,
            total_questions=total_q,
            next_question=None,
            review=None,
            detail=(
                "Your final answer was saved, but generating the review failed. "
                "Call POST /session/{id}/retry-review to try again."
            ),
        )

    store.mark_session_complete(session_id, review_out.model_dump())

    return AnswerResponse(
        session_id=session_id,
        status="complete",
        turns_done=answered_count,
        total_questions=total_q,
        next_question=None,
        review=review_out.model_dump(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Context builders (glue between store and agent services)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_interviewer_context(session: dict[str, Any], category: str) -> InterviewerContext:
    """
    Assemble all context the Interviewer needs for one call.

    This is where the history/tag strategy from §6.1 is implemented:
      • current_session_tags  — tags used so far THIS session (mid-session dedup)
      • all_time_tags          — full deduplicated tag index (all-time dedup)
    """
    session_id = session["session_id"]
    user_id = session["user_id"]

    # ── Tag context ───────────────────────────────────────────────────────────
    current_tags = store.get_used_tags_current_session(session_id)
    all_tags = store.get_tag_index(user_id, max_tags=config.MAX_TAGS_IN_PROMPT)

    # ── Remaining counts ──────────────────────────────────────────────────────
    seq = session["_question_sequence"]
    idx = session["_current_turn_index"]
    upcoming = seq[idx:]            # categories not yet asked

    remaining_technical = upcoming.count("technical")
    remaining_soft = upcoming.count("soft_skill")

    return InterviewerContext(
        field=session["field"],
        level=session["level"],
        category=category,
        current_difficulty=session["current_difficulty"],
        turns_done=len([t for t in session["turns"] if t.get("answer")]),
        total_questions=len(seq),
        remaining_technical=remaining_technical,
        remaining_soft=remaining_soft,
        current_session_tags=current_tags,
        all_time_tags=all_tags,
    )


def _public_session(s: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields before returning a session to the client."""
    return {k: v for k, v in s.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════════════════
# Dev runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)