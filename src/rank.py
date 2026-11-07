"""Step 5 - validate, gate, score, de-overlap and rank.

Order (fixed):
  1. validate timestamps / duration       2. reject invalid clips
  3. hook, cliffhanger, standalone gates  4. final score (weighted rubric)
  5. overlap removal (same episode, > 50% of the shorter clip)
  6. diversity-aware POSTING ORDER (kept separate from the QUALITY ranking)

Three ranking methods are produced from the SAME frozen candidate pool so they can be compared:
  baseline   - heuristic score only (no LLM)
  stage_a    - Flash-Lite gates + score
  stage_a_b  - Stage A, overridden by Stage B where Stage B reviewed the clip
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .heuristics import blocks_in
from .judging import WEIGHTS
from .utils import (MAX_DUR, MIN_DUR, OUTPUTS, Candidate, fmt_ts, load_blocks, load_candidates, read_json,
                    read_jsonl, write_json)

OVERLAP_MAX = 0.5
MAX_PER_EPISODE_TOP10 = 2
FUNNEL_BONUS = {5: 4.0, 10: 2.0}   # posting order only: early-episode promos land viewers near episode 1


def overlap_frac(a: Candidate, b: Candidate) -> float:
    if a.episode != b.episode:
        return 0.0   # timestamps of different episodes are not comparable
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / min(a.duration, b.duration)


def valid(c: Candidate, idx) -> list[str]:
    errs = []
    if not (MIN_DUR - 0.05 <= c.end - c.start <= MAX_DUR + 0.05):
        errs.append("duration_out_of_range")
    for bid in c.block_ids:
        b = idx.get(bid)
        if b is None:
            errs.append(f"unknown_block:{bid}")
        elif "malformed_timestamp" in b.flags or "end_before_start" in b.flags:
            errs.append(f"bad_timestamp:{bid}")
    return errs


def _tier_from_gates(gates: dict) -> str:
    vals = list(gates.values())
    if "FAIL" in vals:
        return "REJECT"
    return "SHORTLIST" if all(v == "PASS" for v in vals) else "REVIEW"


def method_table(method: str, cands: dict[str, Candidate], A: dict, B: dict) -> list[dict]:
    rows = []
    for cid, c in cands.items():
        if method == "baseline":
            rows.append({"candidate_id": cid, "status": "SHORTLIST", "score": c.heuristic["score"], "source": "heuristic"})
            continue
        a = A.get(cid)
        b = B.get(cid) if method == "stage_a_b" else None
        if b and b.get("ok"):
            tier = "REJECT" if b["verdict"] == "FAIL" else _tier_from_gates(b["gates"])
            if b["verdict"] == "UNCERTAIN" and tier == "SHORTLIST":
                tier = "REVIEW"
            rows.append({"candidate_id": cid, "status": tier, "score": b["score"], "source": "stage_b",
                         "reviewed_b": True})
        elif a and a.get("ok"):
            rows.append({"candidate_id": cid, "status": _tier_from_gates(a["gates"]), "score": a["score"],
                         "source": "stage_a", "reviewed_b": False})
        else:
            rows.append({"candidate_id": cid, "status": "ERROR", "score": 0.0, "source": "missing"})
    order = {"SHORTLIST": 0, "REVIEW": 1, "REJECT": 2, "ERROR": 3}
    # Stage-B-confirmed clips outrank clips only Stage A has seen, within the same tier.
    rows.sort(key=lambda r: (order[r["status"]], -int(r.get("reviewed_b", False)) if method == "stage_a_b" else 0,
                             -r["score"], r["candidate_id"]))
    return rows


def dedupe(rows: list[dict], cands: dict[str, Candidate]) -> list[dict]:
    kept: list[dict] = []
    for r in rows:
        c = cands[r["candidate_id"]]
        clash = next((k for k in kept if overlap_frac(c, cands[k["candidate_id"]]) > OVERLAP_MAX), None)
        if clash is None:
            kept.append(r)
    return kept


def posting_order(shortlist: list[dict], cands: dict[str, Candidate], n: int = 10) -> list[dict]:
    pool = []
    for r in shortlist:
        ep = cands[r["candidate_id"]].episode
        bonus = next((v for k, v in sorted(FUNNEL_BONUS.items()) if ep <= k), 0.0)
        pool.append({**r, "posting_score": round(r["score"] + bonus, 2), "funnel_bonus": bonus})
    pool.sort(key=lambda r: -r["posting_score"])
    order, per_ep = [], {}
    while pool and len(order) < n:
        last_ep = cands[order[-1]["candidate_id"]].episode if order else None
        pick = next((r for r in pool if per_ep.get(cands[r["candidate_id"]].episode, 0) < MAX_PER_EPISODE_TOP10
                     and cands[r["candidate_id"]].episode != last_ep), None)
        if pick is None:
            pick = pool[0]
        pool.remove(pick)
        ep = cands[pick["candidate_id"]].episode
        per_ep[ep] = per_ep.get(ep, 0) + 1
        order.append(pick)
    return order


def _block_view(b, t0):
    who = b.speaker if b.type == "dialogue" else b.type.upper()
    return {"block_id": b.block_id, "start": fmt_ts(b.start), "end": fmt_ts(b.end), "rel_start": round(b.start - t0, 2),
            "type": b.type, "who": who, "delivery": b.delivery, "text": b.text}


def build_record(r: dict, c: Candidate, idx, A, B, leak) -> dict:
    a, b = A.get(c.candidate_id), B.get(c.candidate_id)
    blocks = blocks_in([idx[i] for i in c.block_ids], c.start, c.end)
    h = c.heuristic
    rec = {
        "candidate_id": c.candidate_id, "status": r["status"], "score": r["score"], "score_source": r["source"],
        "episode": c.episode,
        "segments": [{"episode": c.episode, "start": fmt_ts(c.start), "end": fmt_ts(c.end),
                      "start_s": c.start, "end_s": c.end}],
        "start": fmt_ts(c.start), "end": fmt_ts(c.end), "duration_s": c.duration,
        "generation_methods": c.methods, "heuristic_score": h["score"],
    }
    hook = {"evidence_block_ids": [h["hook_block"]] if h["hook_block"] else [], "description": None}
    cliff = {"evidence_block_ids": [h["cliff_block"]] if h["cliff_block"] else [], "description": None,
             "unresolved_question": None}
    standalone, why, rationale, risks, scores, gates = {}, None, None, [], None, None
    if a and a.get("ok"):
        ar = a["result"]
        hook = {"description": ar["hook"]["explanation"], "evidence_block_ids": ar["hook"]["evidence_block_ids"]}
        cliff = {"description": ar["cliffhanger"]["explanation"], "evidence_block_ids": ar["cliffhanger"]["evidence_block_ids"],
                 "unresolved_question": ar["cliffhanger"]["unresolved_question"]}
        standalone = {"explanation": ar["standalone"]["explanation"], "missing_context": ar["standalone"]["missing_context"]}
        why, risks, scores, gates = ar["one_line_why"], ar["risks"], a["scores"], a["gates"]
        rec["stage_a"] = {"gates": a["gates"], "score": a["score"], "checks": a["checks"]}
    if b and b.get("ok"):
        br = b["result"]
        hook = {"description": br["hook_description"], "evidence_block_ids": br["scroll_stop"]["evidence_block_ids"]}
        cliff = {"description": br["cliffhanger_description"], "evidence_block_ids": br["open_question"]["evidence_block_ids"],
                 "unresolved_question": br["unresolved_question"]["answer"]}
        standalone = {"central_character": br["central_character"], "want": br["want"], "change": br["change"],
                      "missing_context": br["missing_context"], "explanation": br["comprehension"]["explanation"]}
        why, rationale, risks, scores, gates = br["why_it_works"], br["posting_rationale"], br["risks"], b["scores"], b["gates"]
        rec["stage_b"] = {"verdict": b["verdict"], "gates": b["gates"], "score": b["score"], "checks": b["checks"],
                          "selected_because": b["selected_because"], "suggested_cut": b.get("suggested_cut"),
                          "flags": {k: br[k]["present"] for k in ("reveals_too_much", "payoff_too_early", "dead_time")}}
    for part in (hook, cliff):
        part["evidence"] = [{"block_id": i, "at": fmt_ts(idx[i].start), "text": idx[i].text}
                            for i in part["evidence_block_ids"] if i in idx]
    if why is None:   # baseline only
        hb, cb = idx.get(h["hook_block"]), idx.get(h["cliff_block"])
        hook["description"] = f"Baseline lexicon hit: {hb.text}" if hb else None
        cliff["description"] = f"Baseline lexicon hit: {cb.text}" if cb else None
        why = "Heuristic baseline only (no model judgement yet)."
    rec.update({"hook": hook, "cliffhanger": cliff, "standalone": standalone, "why_it_works": why,
                "posting_rationale": rationale, "risks": risks,
                "scores": scores or {"hook": h["hook"], "cliffhanger": h["cliff"], "standalone": h["standalone"],
                                     "emotional_impact": h["emotion"], "pacing": h["pacing"], "clean_cut": h["clean_cut"]},
                "gates": gates, "weights": WEIGHTS,
                "blocks": [_block_view(x, c.start) for x in blocks]})
    if leak and c.candidate_id in leak:
        rec["leakage"] = leak[c.candidate_id]
    return rec


def run(out_dir: Path = OUTPUTS) -> dict:
    by_ep = load_blocks(out_dir / "beats.jsonl")
    idx = {b.block_id: b for bl in by_ep.values() for b in bl}
    all_cands = load_candidates(out_dir / "candidates.jsonl")
    invalid = {c.candidate_id: e for c in all_cands if (e := valid(c, idx))}
    cands = {c.candidate_id: c for c in all_cands if c.candidate_id not in invalid}
    A = {r["candidate_id"]: r for r in read_jsonl(out_dir / "stage_a.jsonl")} if (out_dir / "stage_a.jsonl").exists() else {}
    B = {r["candidate_id"]: r for r in read_jsonl(out_dir / "stage_b.jsonl")} if (out_dir / "stage_b.jsonl").exists() else {}
    leak = read_json(out_dir / "leakage.json")["results"] if (out_dir / "leakage.json").exists() else {}

    methods = ["baseline"] + (["stage_a"] if A else []) + (["stage_a_b"] if A and B else [])
    rankings = {m: dedupe(method_table(m, cands, A, B), cands) for m in methods}
    main = methods[-1]
    ranked = rankings[main]
    shortlist = [r for r in ranked if r["status"] == "SHORTLIST"]
    review = [r for r in ranked if r["status"] == "REVIEW"]

    quality = []
    for i, r in enumerate(shortlist + review, 1):
        rec = build_record(r, cands[r["candidate_id"]], idx, A, B, leak)
        rec["quality_rank"] = i
        quality.append(rec)
    dropped = {cid for cid, l in leak.items() if l.get("recommendation") == "DROP"}
    postable = [r for r in shortlist if r["candidate_id"] not in dropped]
    order = posting_order(postable, cands)
    by_id = {q["candidate_id"]: q for q in quality}
    posting = []
    for i, r in enumerate(order, 1):
        q = by_id[r["candidate_id"]]
        reason = [f"quality score {r['score']:.1f}"]
        if r["funnel_bonus"]:
            reason.append(f"+{r['funnel_bonus']:.0f} early-episode bonus: a viewer who clicks starts the show almost from the beginning")
        posting.append({"post_rank": i, "candidate_id": r["candidate_id"], "episode": q["episode"],
                        "start": q["start"], "end": q["end"], "quality_rank": q["quality_rank"],
                        "posting_score": r["posting_score"], "why_this_slot": "; ".join(reason),
                        "posting_rationale": q.get("posting_rationale"), "trim_to": (q.get("leakage") or {}).get("trim_to")})

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ranking_method": main,
        "note": None if main != "baseline" else "No model results found - this is the heuristic BASELINE ranking.",
        "counts": {"candidates": len(all_cands), "invalid": len(invalid), "after_overlap_removal": len(ranked),
                   "shortlist": len(shortlist), "review": len(review),
                   "rejected": sum(1 for r in ranked if r["status"] in ("REJECT", "ERROR")),
                   "leakage_drop": len(dropped)},
        "invalid_candidates": invalid,
        "posting_order": posting,
        "quality_ranking": quality,
        "method_rankings": {m: [{"candidate_id": r["candidate_id"], "status": r["status"], "score": r["score"],
                                 "episode": cands[r["candidate_id"]].episode} for r in rk]
                            for m, rk in rankings.items()},
    }
    write_json(out_dir / "promos.json", result)
    (out_dir / "promos.md").write_text(to_markdown(result), encoding="utf-8")
    return {k: v for k, v in result["counts"].items()} | {"method": main}


def to_markdown(res: dict, top_detail: int = 20) -> str:
    L = ["# Promo shortlist", "",
         f"Generated {res['generated_at']} · ranking method: **{res['ranking_method']}**", ""]
    if res.get("note"):
        L += [f"> {res['note']}", ""]
    c = res["counts"]
    L += [f"Candidates {c['candidates']} → after overlap removal {c['after_overlap_removal']} → "
          f"shortlist {c['shortlist']}, needs human review {c['review']}, rejected {c['rejected']}.", ""]
    L += ["## Recommended posting order", "", "| # | Episode | Start | End | Quality rank | Why this slot |",
          "|---|---|---|---|---|---|"]
    for p in res["posting_order"]:
        L.append(f"| {p['post_rank']} | {p['episode']} | {p['start']} | {p['end']} | {p['quality_rank']} | {p['why_this_slot']} |")
    L += ["", "## Promos by quality", ""]
    for q in res["quality_ranking"][:top_detail]:
        L += [f"### {q['quality_rank']}. Episode {q['episode']}  {q['start']} → {q['end']}  ({q['duration_s']:.1f} s) — "
              f"{q['status']}, score {q['score']:.1f}", ""]
        L.append(f"- **Hook:** {q['hook'].get('description')}")
        for e in q["hook"].get("evidence", [])[:2]:
            L.append(f"  - `{e['block_id']}` @ {e['at']}: {e['text']}")
        L.append(f"- **Cliffhanger:** {q['cliffhanger'].get('description')}")
        if q["cliffhanger"].get("unresolved_question"):
            L.append(f"  - Unresolved question: {q['cliffhanger']['unresolved_question']}")
        for e in q["cliffhanger"].get("evidence", [])[:2]:
            L.append(f"  - `{e['block_id']}` @ {e['at']}: {e['text']}")
        st = q.get("standalone") or {}
        if st.get("explanation"):
            L.append(f"- **Standalone:** {st['explanation']}")
        if st.get("missing_context"):
            L.append(f"  - Missing context: {'; '.join(st['missing_context'])}")
        L.append(f"- **Why it works:** {q['why_it_works']}")
        if q.get("posting_rationale"):
            L.append(f"- **Posting rationale:** {q['posting_rationale']}")
        if q.get("risks"):
            L.append(f"- **Risks:** {'; '.join(q['risks'])}")
        if q.get("leakage"):
            lk = q["leakage"]
            L.append(f"- **Payoff-leakage check:** {lk.get('recommendation')}" +
                     (f" (trim end to {fmt_ts(lk['trim_to'])})" if lk.get("trim_to") else ""))
        sc = q["scores"]
        L.append("- **Scores:** " + ", ".join(f"{k} {float(v):.1f}" for k, v in sc.items()))
        L += ["", "<details><summary>Clip blocks</summary>", ""]
        for b in q["blocks"]:
            who = b["who"] + (f" ({b['delivery']})" if b.get("delivery") else "")
            L.append(f"- `{b['start']}` **{who}:** {b['text']}")
        L += ["", "</details>", ""]
    return "\n".join(L) + "\n"
