"""Shared pieces for the LLM stages: prompt loading, clip rendering, evidence checks."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .utils import CLIFF_WINDOW, HOOK_WINDOW, PROMPTS_DIR, Block, Candidate, env, sha

Verdict = Literal["PASS", "UNCERTAIN", "FAIL"]
VERDICT_RANK = {"FAIL": 0, "UNCERTAIN": 1, "PASS": 2}
WEIGHTS = {"hook": 0.25, "cliffhanger": 0.25, "standalone": 0.20, "emotional_impact": 0.15,
           "pacing": 0.10, "clean_cut": 0.05}

DEFAULT_MODELS = {
    # Verified against third-party listings in Nov 2026 - run `python run.py models` to confirm on your key.
    "gemini": {"a": "gemini-3.5-flash-lite", "b": "gemini-3.6-flash"},
    "openai": {"a": "gpt-4.1-mini", "b": "gpt-4.1"},
    "mock": {"a": "mock-lite", "b": "mock-flash"},
}


def model_for(provider: str, stage: str) -> str:
    key = {"a": "MODEL_STAGE_A", "b": "MODEL_STAGE_B", "leak": "MODEL_LEAKAGE"}[stage]
    fallback = DEFAULT_MODELS.get(provider, {}).get("a" if stage == "leak" else stage, "")
    if stage == "leak":
        fallback = env("MODEL_STAGE_A", fallback)
    return env(key, fallback)


def load_prompt(name: str) -> tuple[str, str]:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    return text, f"{name}:{sha(text)[:8]}"


def cap(v: str, ceiling: str) -> str:
    return v if VERDICT_RANK[v] <= VERDICT_RANK[ceiling] else ceiling


def render_blocks(blocks: list[Block], t0: float, prefix: str = "") -> str:
    lines = []
    for b in blocks:
        rel_s, rel_e = b.start - t0, b.end - t0
        if b.type == "dialogue":
            who = b.speaker or b.label_raw
            extra = []
            if b.delivery:
                extra.append(b.delivery)
            if b.offscreen:
                extra.append("off-screen")
            head = f"DIALOGUE {who}" + (f" ({'; '.join(extra)})" if extra else "")
        else:
            head = b.type.upper()
        lines.append(f"{prefix}[{b.block_id}] {rel_s:+.1f}s to {rel_e:+.1f}s  {head}: {b.text}")
    return "\n".join(lines)


def render_clip(c: Candidate, idx: dict[str, Block]) -> str:
    blocks = [idx[i] for i in c.block_ids]
    return (f"CLIP LENGTH: {c.duration:.1f} s. Hook window: +0.0 to +{HOOK_WINDOW:.1f} s. "
            f"Cliffhanger window: +{c.duration - CLIFF_WINDOW:.1f} to +{c.duration:.1f} s.\n"
            f"BLOCKS (time relative to clip start):\n" + render_blocks(blocks, c.start))


def check_evidence(ids: list[str], allowed: set[str]) -> list[str]:
    return [i for i in ids if i not in allowed]


def hook_timing_ok(ids: list[str], idx: dict[str, Block], c: Candidate, slack: float = 0.5) -> bool:
    """At least one cited hook block must be on screen within the first HOOK_WINDOW seconds."""
    for i in ids:
        b = idx.get(i)
        if b is not None and b.start - c.start <= HOOK_WINDOW + slack and b.end > c.start:
            return True
    return False


def cliff_timing_ok(ids: list[str], idx: dict[str, Block], c: Candidate, slack: float = 3.0) -> bool:
    """At least one cited cliffhanger block must still be running in the final CLIFF_WINDOW (+slack) s."""
    for i in ids:
        b = idx.get(i)
        if b is not None and b.end >= c.end - CLIFF_WINDOW - slack:
            return True
    return False


class EvidenceAnswer(BaseModel):
    answer: str
    evidence_block_ids: list[str] = Field(default_factory=list)


def weighted(scores: dict) -> float:
    return round(sum(WEIGHTS[k] * float(scores.get(k, 0)) for k in WEIGHTS) * 10, 2)
