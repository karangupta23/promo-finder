"""Step 3 - Stage A: broad screening of every candidate with a cheap model (Flash-Lite).

The model sees ONLY the clip's blocks (timestamps relative to the clip, original speaker labels).
No episode summary, logline, continuity notes or surrounding scenes.
After the call, code checks the model's claims: cited block ids must exist in the clip, the hook
evidence must sit in the first 3 s and the cliffhanger evidence near the end. Failing a check caps
that verdict at UNCERTAIN, so a confident but unsupported PASS cannot reach the shortlist.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .judging import (Verdict, cap, check_evidence, cliff_timing_ok, hook_timing_ok, load_prompt,
                      render_clip, weighted)
from .llm import JsonCaller, make_client, summarize_log
from .utils import OUTPUTS, Candidate, load_blocks, load_candidates, write_jsonl

STAGE = "stage_a"


class Criterion(BaseModel):
    verdict: Verdict
    score: int = Field(ge=0, le=10)
    evidence_block_ids: list[str] = Field(default_factory=list)
    explanation: str


class CliffCriterion(Criterion):
    unresolved_question: str = ""


class StandaloneCriterion(Criterion):
    missing_context: list[str] = Field(default_factory=list)


class DeadTime(BaseModel):
    score: int = Field(ge=0, le=10, description="10 = no dead time")
    dead_block_ids: list[str] = Field(default_factory=list)
    explanation: str


class StageAResult(BaseModel):
    hook: Criterion
    cliffhanger: CliffCriterion
    standalone: StandaloneCriterion
    emotional_impact: Criterion
    dead_time: DeadTime
    risks: list[str] = Field(default_factory=list)
    one_line_why: str


def _v(x: float) -> str:
    return "PASS" if x >= 7 else "UNCERTAIN" if x >= 4.5 else "FAIL"


def mock_payload(c: Candidate) -> dict:
    h = c.heuristic
    return {
        "hook": {"verdict": _v(h["hook"]), "score": round(h["hook"]), "evidence_block_ids": [h["hook_block"]] if h["hook_block"] else [],
                 "explanation": "mock: lexicon hook signal"},
        "cliffhanger": {"verdict": _v(h["cliff"]), "score": round(h["cliff"]),
                        "evidence_block_ids": [h["cliff_block"]] if h["cliff_block"] else [],
                        "explanation": "mock: lexicon cliff signal", "unresolved_question": "mock"},
        "standalone": {"verdict": _v(h["standalone"]), "score": round(h["standalone"]), "evidence_block_ids": [],
                       "explanation": "mock", "missing_context": []},
        "emotional_impact": {"verdict": _v(h["emotion"]), "score": round(h["emotion"]), "evidence_block_ids": [],
                             "explanation": "mock"},
        "dead_time": {"score": round(h["pacing"]), "dead_block_ids": [], "explanation": "mock"},
        "risks": [], "one_line_why": "mock output - not a real judgement",
    }


def postprocess(c: Candidate, r: StageAResult, idx) -> dict:
    allowed = set(c.block_ids)
    checks = {
        "invalid_evidence_ids": sorted(set(
            check_evidence(r.hook.evidence_block_ids, allowed) + check_evidence(r.cliffhanger.evidence_block_ids, allowed)
            + check_evidence(r.standalone.evidence_block_ids, allowed))),
        "hook_in_first_3s": hook_timing_ok(r.hook.evidence_block_ids, idx, c),
        "cliff_near_end": cliff_timing_ok(r.cliffhanger.evidence_block_ids, idx, c),
    }
    gates = {"hook": r.hook.verdict, "cliffhanger": r.cliffhanger.verdict, "standalone": r.standalone.verdict}
    if not checks["hook_in_first_3s"] or set(r.hook.evidence_block_ids) - allowed:
        gates["hook"] = cap(gates["hook"], "UNCERTAIN")
    if not checks["cliff_near_end"] or set(r.cliffhanger.evidence_block_ids) - allowed:
        gates["cliffhanger"] = cap(gates["cliffhanger"], "UNCERTAIN")
    scores = {"hook": r.hook.score, "cliffhanger": r.cliffhanger.score, "standalone": r.standalone.score,
              "emotional_impact": r.emotional_impact.score, "pacing": r.dead_time.score,
              "clean_cut": c.heuristic.get("clean_cut", 5)}
    return {"gates": gates, "checks": checks, "scores": scores, "score": weighted(scores)}


def run(provider: str, model: str, rpm: float | None = None, episodes: set[int] | None = None,
        limit: int | None = None, out_dir=OUTPUTS) -> dict:
    system, version = load_prompt("stage_a_v1")
    idx = {b.block_id: b for bl in load_blocks(out_dir / "beats.jsonl").values() for b in bl}
    cands = load_candidates(out_dir / "candidates.jsonl")
    if episodes:
        cands = [c for c in cands if c.episode in episodes]
    if limit:
        cands = cands[:limit]
    caller = JsonCaller(make_client(provider, model), STAGE, version, rpm=rpm)
    rows, ok = [], 0
    for n, c in enumerate(cands, 1):
        prompt = f"EPISODE {c.episode}, candidate {c.candidate_id}\n" + render_clip(c, idx)
        res = caller.call(c.candidate_id, system, prompt, StageAResult, mock=lambda c=c: mock_payload(c))
        row = {"candidate_id": c.candidate_id, "episode": c.episode, "ok": res is not None,
               "provider": provider, "model": model, "prompt_version": version}
        if res is not None:
            ok += 1
            row["result"] = res.model_dump()
            row.update(postprocess(c, res, idx))
        rows.append(row)
        if n % 25 == 0 or n == len(cands):
            print(f"  stage A {n}/{len(cands)}  ok={ok}")
    write_jsonl(out_dir / "stage_a.jsonl", rows)
    summarize_log()
    return {"candidates": len(cands), "ok": ok, "failed": len(cands) - ok, "prompt_version": version}
