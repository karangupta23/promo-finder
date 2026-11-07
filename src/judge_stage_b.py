"""Step 4 - Stage B: deeper editorial review with a stronger model (Flash) on a budgeted subset:

  * top candidates from Stage A (all three gates not FAIL, highest Stage A score)
  * UNCERTAIN candidates (a gate is UNCERTAIN, none FAIL)
  * possibly-underrated candidates: strong on the heuristic baseline but rejected/low in Stage A

Default budget is 18 calls because the Gemini free tier allows ~20 Flash requests/day; pass
--budget 40 (or more) on a paid key. Same context restriction as Stage A: clip blocks only.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .judging import (EvidenceAnswer, Verdict, VERDICT_RANK, cap, check_evidence, cliff_timing_ok,
                      hook_timing_ok, load_prompt, render_clip, weighted)
from .llm import JsonCaller, make_client, summarize_log
from .utils import OUTPUTS, Candidate, load_blocks, load_candidates, read_jsonl, write_jsonl

STAGE = "stage_b"


class Check(BaseModel):
    verdict: Verdict
    evidence_block_ids: list[str] = Field(default_factory=list)
    explanation: str


class Flag(BaseModel):
    present: bool
    evidence_block_ids: list[str] = Field(default_factory=list)
    explanation: str


class BetterCut(BaseModel):
    recommended: bool
    start_block_id: Optional[str] = None
    end_block_id: Optional[str] = None
    reason: str = ""


class Scores(BaseModel):
    hook: int = Field(ge=0, le=10)
    cliffhanger: int = Field(ge=0, le=10)
    standalone: int = Field(ge=0, le=10)
    emotional_impact: int = Field(ge=0, le=10)
    pacing: int = Field(ge=0, le=10)
    clean_cut: int = Field(ge=0, le=10)


class StageBResult(BaseModel):
    central_character: EvidenceAnswer
    want: EvidenceAnswer
    change: EvidenceAnswer
    unresolved_question: EvidenceAnswer
    missing_context: list[str] = Field(default_factory=list)
    scroll_stop: Check
    open_question: Check
    comprehension: Check
    reveals_too_much: Flag
    payoff_too_early: Flag
    dead_time: Flag
    better_cut: BetterCut
    scores: Scores
    verdict: Verdict
    hook_description: str
    cliffhanger_description: str
    why_it_works: str
    posting_rationale: str
    risks: list[str] = Field(default_factory=list)


def select(stage_a: list[dict], cands: dict[str, Candidate], budget: int) -> list[tuple[str, str]]:
    """Return [(candidate_id, reason)] in priority order, at most `budget` long."""
    n_top = max(1, round(budget * 2 / 3))
    n_unc = max(0, round(budget * 2 / 9))
    ok = [r for r in stage_a if r.get("ok")]
    no_fail = [r for r in ok if all(v != "FAIL" for v in r["gates"].values())]
    top = sorted([r for r in no_fail if all(v == "PASS" for v in r["gates"].values())], key=lambda r: -r["score"])
    unc = sorted([r for r in no_fail if any(v == "UNCERTAIN" for v in r["gates"].values())], key=lambda r: -r["score"])
    picked: list[tuple[str, str]] = []
    seen = set()

    def add(cid, why):
        if cid not in seen and len(picked) < budget:
            seen.add(cid); picked.append((cid, why))

    for r in top[:n_top]:
        add(r["candidate_id"], "top_stage_a")
    for r in unc[:n_unc]:
        add(r["candidate_id"], "uncertain")
    # underrated: best heuristic windows that Stage A failed or scored low
    a_score = {r["candidate_id"]: r for r in ok}
    by_heur = sorted(cands.values(), key=lambda c: -c.heuristic["score"])
    for c in by_heur[:40]:
        r = a_score.get(c.candidate_id)
        if r is None or any(v == "FAIL" for v in r["gates"].values()):
            add(c.candidate_id, "underrated_by_stage_a")
    # fill remaining budget with the next best (passed + uncertain mixed by score)
    for r in sorted(no_fail, key=lambda r: -r["score"]):
        add(r["candidate_id"], "fill")
    return picked


def mock_payload(c: Candidate) -> dict:
    h = c.heuristic
    v = lambda x: "PASS" if x >= 7 else "UNCERTAIN" if x >= 4.5 else "FAIL"  # noqa: E731
    ea = {"answer": "mock", "evidence_block_ids": [h["hook_block"]] if h["hook_block"] else []}
    flag = {"present": False, "evidence_block_ids": [], "explanation": "mock"}
    overall = min(h["hook"], h["cliff"], h["standalone"])
    return {
        "central_character": ea, "want": ea, "change": ea,
        "unresolved_question": {"answer": "mock", "evidence_block_ids": [h["cliff_block"]] if h["cliff_block"] else []},
        "missing_context": [],
        "scroll_stop": {"verdict": v(h["hook"]), "evidence_block_ids": ea["evidence_block_ids"], "explanation": "mock"},
        "open_question": {"verdict": v(h["cliff"]), "evidence_block_ids": [h["cliff_block"]] if h["cliff_block"] else [],
                          "explanation": "mock"},
        "comprehension": {"verdict": v(h["standalone"]), "evidence_block_ids": [], "explanation": "mock"},
        "reveals_too_much": flag, "payoff_too_early": flag, "dead_time": flag,
        "better_cut": {"recommended": False, "reason": "mock"},
        "scores": {"hook": round(h["hook"]), "cliffhanger": round(h["cliff"]), "standalone": round(h["standalone"]),
                   "emotional_impact": round(h["emotion"]), "pacing": round(h["pacing"]), "clean_cut": round(h["clean_cut"])},
        "verdict": v(overall),
        "hook_description": "mock", "cliffhanger_description": "mock", "why_it_works": "mock output - not a real judgement",
        "posting_rationale": "mock", "risks": [],
    }


def postprocess(c: Candidate, r: StageBResult, idx) -> dict:
    allowed = set(c.block_ids)
    cited = (r.scroll_stop.evidence_block_ids + r.open_question.evidence_block_ids
             + r.central_character.evidence_block_ids + r.want.evidence_block_ids + r.change.evidence_block_ids
             + r.unresolved_question.evidence_block_ids)
    checks = {
        "invalid_evidence_ids": sorted(set(check_evidence(cited, allowed))),
        "hook_in_first_3s": hook_timing_ok(r.scroll_stop.evidence_block_ids, idx, c),
        "cliff_near_end": cliff_timing_ok(r.open_question.evidence_block_ids + r.unresolved_question.evidence_block_ids,
                                          idx, c),
    }
    gates = {"hook": r.scroll_stop.verdict, "cliffhanger": r.open_question.verdict,
             "standalone": r.comprehension.verdict}
    if not checks["hook_in_first_3s"]:
        gates["hook"] = cap(gates["hook"], "UNCERTAIN")
    if not checks["cliff_near_end"]:
        gates["cliffhanger"] = cap(gates["cliffhanger"], "UNCERTAIN")
    if r.reveals_too_much.present:
        gates["cliffhanger"] = cap(gates["cliffhanger"], "UNCERTAIN")
    verdict = r.verdict
    if min(VERDICT_RANK[g] for g in gates.values()) < VERDICT_RANK[verdict]:
        verdict = min(gates.values(), key=lambda g: VERDICT_RANK[g])
    # validate a suggested re-cut: must be inside the clip and keep 20-60 s
    cut = None
    bc = r.better_cut
    if bc.recommended:
        s_id = bc.start_block_id or c.block_ids[0]
        e_id = bc.end_block_id or c.block_ids[-1]
        if s_id in allowed and e_id in allowed:
            s, e = idx[s_id].start, idx[e_id].end
            cut = {"start": s, "end": e, "valid": 20 <= e - s <= 60 and s >= c.start - 0.05 and e <= c.end + 0.05,
                   "reason": bc.reason}
        else:
            cut = {"valid": False, "reason": "cited block not in clip: " + bc.reason}
    scores = r.scores.model_dump()
    return {"gates": gates, "verdict": verdict, "checks": checks, "scores": scores, "score": weighted(scores),
            "suggested_cut": cut}


def run(provider: str, model: str, budget: int = 18, rpm: float | None = None, out_dir=OUTPUTS) -> dict:
    system, version = load_prompt("stage_b_v1")
    idx = {b.block_id: b for bl in load_blocks(out_dir / "beats.jsonl").values() for b in bl}
    cands = {c.candidate_id: c for c in load_candidates(out_dir / "candidates.jsonl")}
    stage_a = read_jsonl(out_dir / "stage_a.jsonl")
    chosen = select(stage_a, cands, budget)
    caller = JsonCaller(make_client(provider, model), STAGE, version, rpm=rpm)
    rows, ok = [], 0
    for n, (cid, why) in enumerate(chosen, 1):
        c = cands[cid]
        prompt = f"EPISODE {c.episode}, candidate {c.candidate_id}\n" + render_clip(c, idx)
        res = caller.call(cid, system, prompt, StageBResult, mock=lambda c=c: mock_payload(c))
        row = {"candidate_id": cid, "episode": c.episode, "selected_because": why, "ok": res is not None,
               "provider": provider, "model": model, "prompt_version": version}
        if res is not None:
            ok += 1
            row["result"] = res.model_dump()
            row.update(postprocess(c, res, idx))
        rows.append(row)
        print(f"  stage B {n}/{len(chosen)} {cid} ({why}) -> {row.get('verdict', 'ERROR')}")
    write_jsonl(out_dir / "stage_b.jsonl", rows)
    summarize_log()
    return {"selected": len(chosen), "ok": ok, "failed": len(chosen) - ok, "prompt_version": version,
            "by_reason": {w: sum(1 for _, x in chosen if x == w) for w in {x for _, x in chosen}}}
