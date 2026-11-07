"""Step 5b - payoff-leakage check (separate from the judges on purpose).

Runs only on the shortlisted promos, after ranking. Unlike Stage A/B, the model here also sees
what comes right AFTER the clip (next ~20 s of the episode, or the opening of the next episode when
the clip ends on the episode's final cut). Keeping this separate means the judges never use future
information to make a clip look better or worse.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .judging import load_prompt, render_blocks, render_clip
from .llm import JsonCaller, make_client, summarize_log
from .utils import MIN_DUR, OUTPUTS, Candidate, load_blocks, load_candidates, read_json, write_json

STAGE = "leakage"
AFTER_SECONDS = 20.0


class LeakFlag(BaseModel):
    present: bool
    evidence_block_ids: list[str] = Field(default_factory=list)
    explanation: str


class Resolution(LeakFlag):
    seconds_until_resolution: Optional[float] = None


class EarlierEnd(BaseModel):
    end_block_id: Optional[str] = None
    explanation: str = ""


class LeakResult(BaseModel):
    answer_revealed_in_clip: LeakFlag
    resolved_immediately_after: Resolution
    stronger_earlier_end: EarlierEnd
    recommendation: Literal["KEEP", "TRIM", "DROP"]
    explanation: str


def after_blocks(c: Candidate, by_ep, durations):
    ep_blocks = by_ep[c.episode]
    after = [b for b in ep_blocks if b.start is not None and c.end - 0.05 <= b.start < c.end + AFTER_SECONDS]
    note = f"next {AFTER_SECONDS:.0f} s of episode {c.episode}"
    if not after and (c.episode + 1) in by_ep:
        after = [b for b in by_ep[c.episode + 1] if b.start is not None and b.start < AFTER_SECONDS]
        note = f"clip ends the episode; opening {AFTER_SECONDS:.0f} s of episode {c.episode + 1}"
        return after, note, 0.0
    return after, note, c.end


def mock_payload(c: Candidate) -> dict:
    drop = c.heuristic["cliff"] < 5
    f = {"present": False, "evidence_block_ids": [], "explanation": "mock"}
    return {"answer_revealed_in_clip": f, "resolved_immediately_after": {**f, "seconds_until_resolution": None},
            "stronger_earlier_end": {"end_block_id": None, "explanation": "mock"},
            "recommendation": "DROP" if drop else "KEEP", "explanation": "mock output"}


def run(provider: str, model: str, top: int = 15, rpm: float | None = None, out_dir=OUTPUTS) -> dict:
    system, version = load_prompt("leakage_v1")
    by_ep = load_blocks(out_dir / "beats.jsonl")
    idx = {b.block_id: b for bl in by_ep.values() for b in bl}
    durations = {m["episode"]: m["duration"] for m in read_json(out_dir / "episodes.json")}
    cands = {c.candidate_id: c for c in load_candidates(out_dir / "candidates.jsonl")}
    promos = read_json(out_dir / "promos.json")
    ids = [p["candidate_id"] for p in promos["quality_ranking"][:top]]
    caller = JsonCaller(make_client(provider, model), STAGE, version, rpm=rpm)
    out = {}
    for cid in ids:
        c = cands[cid]
        after, note, t0 = after_blocks(c, by_ep, durations)
        prompt = ("CLIP\n" + render_clip(c, idx) + f"\n\nAFTER ({note}; times relative to the clip's end):\n"
                  + (render_blocks(after, t0) if after else "(nothing - end of available material)"))
        res = caller.call(cid, system, prompt, LeakResult, mock=lambda c=c: mock_payload(c))
        if res is None:
            out[cid] = {"ok": False}
            continue
        rec = {"ok": True, "result": res.model_dump(), "recommendation": res.recommendation, "after_source": note}
        if res.recommendation == "TRIM":
            eid = res.stronger_earlier_end.end_block_id
            if eid in c.block_ids and idx[eid].end - c.start >= MIN_DUR:
                rec["trim_to"] = idx[eid].end
            else:
                rec["recommendation"] = "KEEP"
                rec["note"] = "TRIM suggestion rejected: end block outside clip or < 20 s"
        out[cid] = rec
        print(f"  leakage {cid}: {rec['recommendation']}")
    write_json(out_dir / "leakage.json", {"prompt_version": version, "provider": provider, "model": model,
                                          "results": out})
    summarize_log()
    return {"checked": len(ids), "drop": sum(1 for r in out.values() if r.get("recommendation") == "DROP"),
            "trim": sum(1 for r in out.values() if r.get("recommendation") == "TRIM")}
