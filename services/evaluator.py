from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

import config

logger = logging.getLogger(__name__)

# ── Prompt template (loaded once) ─────────────────────────────────────────────
_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "evaluator.txt"
_PROMPT_TEMPLATE: str = _TEMPLATE_PATH.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluatorOutput(BaseModel):
    """Structured output from the Evaluator agent (§7.3 of the architecture)."""

    summary: str = Field(..., min_length=20)
    strengths: list[str] = Field(..., min_length=1, max_length=10)
    weaknesses: list[str] = Field(..., min_length=1, max_length=10)
    skill_level: str = Field(..., pattern=r"^(entry|junior|mid|senior|manager_lead)$")
    level_up_gaps: list[str] = Field(..., min_length=1, max_length=10)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("strengths", "weaknesses", "level_up_gaps", mode="before")
    @classmethod
    def must_be_non_empty_strings(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s.strip()]


class EvaluatorContext(BaseModel):
    """All context the Evaluator needs — the complete current session."""

    field: str
    level: str = "mid"                  # entry | junior | mid | senior | manager_lead
    question_mix: dict[str, int]        # {"technical": N, "soft": M}
    turns: list[dict[str, Any]]         # full turn list from the session record
    start_difficulty: int = 3
    end_difficulty: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def produce_review(ctx: EvaluatorContext) -> EvaluatorOutput:
    
    prompt = _build_prompt(ctx)
    logger.debug("Evaluator prompt built (%d chars)", len(prompt))

    for attempt in range(2):
        raw = _call_groq(prompt)
        try:
            data = _extract_json(raw)
            # Inject timestamp if missing
            data.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
            result = EvaluatorOutput(**data)
            logger.info(
                "Evaluator → skill_level=%s, strengths=%d, weaknesses=%d, gaps=%d",
                result.skill_level,
                len(result.strengths),
                len(result.weaknesses),
                len(result.level_up_gaps),
            )
            return result
        except Exception as exc:
            if attempt == 0:
                logger.warning("Evaluator output invalid on attempt 1: %s — retrying", exc)
            else:
                logger.error("Evaluator output invalid on attempt 2: %s\nRaw: %s", exc, raw[:500])
                raise ValueError(f"Evaluator returned malformed output after 2 attempts: {exc}") from exc

    raise RuntimeError("Unreachable")


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(ctx: EvaluatorContext) -> str:
    
    transcript = _format_transcript(ctx.turns)
    return _PROMPT_TEMPLATE.format(
        field=ctx.field,
        level=ctx.level,
        technical_count=ctx.question_mix.get("technical", 0),
        soft_count=ctx.question_mix.get("soft", 0),
        total_turns=len(ctx.turns),
        start_difficulty=ctx.start_difficulty,
        end_difficulty=ctx.end_difficulty,
        transcript=transcript,
    )


def _format_transcript(turns: list[dict[str, Any]]) -> str:
   
    lines: list[str] = []
    for t in turns:
        tags_str = ", ".join(t.get("tags", [])) or "—"
        answer = t.get("answer", "").strip() or "(no answer)"
        lines.append(
            f"Turn {t['turn_id']} [{t['category']} | {t['difficulty']} | tags: {tags_str}]\n"
            f"Q: {t['question']}\n"
            f"A: {answer}\n"
        )
    return "\n".join(lines)


def _call_groq(prompt: str) -> str:
    """Call the Groq chat completions endpoint and return the raw assistant text."""
    response = config.groq_client.chat.completions.create(
        model=config.EVALUATOR_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,        # lower temp for more deterministic structured output
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from the model's response."""
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        inner_lines = [
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ]
        text = "\n".join(inner_lines).strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object in evaluator response: {raw[:300]!r}")

    return json.loads(text[start:end])