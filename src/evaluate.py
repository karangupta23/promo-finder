"""Evaluation: blind human review sheet + precision@10 per method.

  python run.py review-sheet     -> outputs/review_sheet.csv (blind, shuffled) + outputs/review_key.json
  (fill in the hook / cliffhanger / comprehension / editing columns with Pass / Uncertain / Fail)
  python run.py score-review     -> outputs/evaluation.json + outputs/evaluation.md

Only EVALUATION episodes (6-50) are used; episodes 1-5 were used to develop the prompts.
Every method is ranked over the same frozen candidate pool. A clip picked by several methods
appears once in the sheet and its label counts for each method that picked it.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .judging import render_clip
from .utils import (OUTPUTS, fmt_ts, is_eval_episode, load_blocks, load_candidates, read_json,
                    stable_hash_int, write_json)

CHECKS = ["hook", "cliffhanger", "comprehension", "editing"]
SEED = "kuku-blind-v1"
LOWER_RANKS = (11, 40)
LOWER_PER_METHOD = 4


def _eligible(rows):
    return [r for r in rows if is_eval_episode(r["episode"]) and r["status"] in ("SHORTLIST", "REVIEW")]


def make_sheet(out_dir: Path = OUTPUTS, top: int = 10) -> dict:
    promos = read_json(out_dir / "promos.json")
    idx = {b.block_id: b for bl in load_blocks(out_dir / "beats.jsonl").values() for b in bl}
    cands = {c.candidate_id: c for c in load_candidates(out_dir / "candidates.jsonl")}
    picks: dict[str, dict] = {}   # candidate_id -> {method: rank}
    lower: dict[str, dict] = {}
    for m, rows in promos["method_rankings"].items():
        el = _eligible(rows)
        for rank, r in enumerate(el[:top], 1):
            picks.setdefault(r["candidate_id"], {})[m] = rank
        lo, hi = LOWER_RANKS
        band = [(rank, r) for rank, r in enumerate(el, 1) if lo <= rank <= hi]
        band.sort(key=lambda x: stable_hash_int(f"{SEED}|{m}|{x[1]['candidate_id']}"))
        for rank, r in band[:LOWER_PER_METHOD]:
            lower.setdefault(r["candidate_id"], {})[m] = rank
    all_ids = sorted(set(picks) | set(lower), key=lambda cid: stable_hash_int(f"{SEED}|{cid}"))
    key, rows = {}, []
    for n, cid in enumerate(all_ids, 1):
        rid = f"R{n:03d}"
        c = cands[cid]
        key[rid] = {"candidate_id": cid, "top10_ranks": picks.get(cid, {}), "lower_sample_ranks": lower.get(cid, {})}
        rows.append({"review_id": rid, "episode": c.episode, "start": fmt_ts(c.start), "end": fmt_ts(c.end),
                     "duration_s": c.duration, "clip": render_clip(c, idx),
                     **{k: "" for k in CHECKS}, "overall": "", "notes": ""})
    path = out_dir / "review_sheet.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["review_id"])
        w.writeheader()
        w.writerows(rows)
    write_json(out_dir / "review_key.json", {"seed": SEED, "labels": "Pass / Uncertain / Fail", "items": key})
    return {"clips_to_review": len(rows), "from_top10": len(picks), "lower_sample": len(set(lower) - set(picks)),
            "methods": list(promos["method_rankings"])}


def _norm(v: str) -> str | None:
    v = (v or "").strip().lower()
    if not v:
        return None
    if v[0] in "py✓" or v in ("ok", "yes"):
        return "Pass"
    if v[0] in "u?m":
        return "Uncertain"
    if v[0] in "fn✗x":
        return "Fail"
    return None


def overall_label(row: dict) -> str | None:
    if (o := _norm(row.get("overall", ""))):
        return o
    labels = [_norm(row.get(k, "")) for k in CHECKS]
    if any(x is None for x in labels):
        return None
    if "Fail" in labels:
        return "Fail"
    return "Pass" if all(x == "Pass" for x in labels) else "Uncertain"


def score(sheet: Path | None = None, out_dir: Path = OUTPUTS) -> dict:
    sheet = sheet or out_dir / "review_sheet.csv"
    key = read_json(out_dir / "review_key.json")["items"]
    with open(sheet, newline="", encoding="utf-8") as f:
        labels = {r["review_id"]: r for r in csv.DictReader(f)}
    methods = sorted({m for v in key.values() for m in v["top10_ranks"]} |
                     {m for v in key.values() for m in v["lower_sample_ranks"]})
    res = {"methods": {}, "unlabelled": [], "disagreements": []}
    for rid, row in labels.items():
        if overall_label(row) is None:
            res["unlabelled"].append(rid)
    for m in methods:
        top = sorted([(v["top10_ranks"][m], rid) for rid, v in key.items() if m in v["top10_ranks"]])
        cnt = Counter(overall_label(labels[rid]) for _, rid in top)
        per_check = {k: Counter(_norm(labels[rid].get(k, "")) for _, rid in top) for k in CHECKS}
        low = [rid for rid, v in key.items() if m in v["lower_sample_ranks"]]
        lcnt = Counter(overall_label(labels[rid]) for rid in low)
        n = len(top)
        res["methods"][m] = {
            "top_n": n,
            "pass": cnt["Pass"], "uncertain": cnt["Uncertain"], "fail": cnt["Fail"], "unlabelled": cnt[None],
            "precision_at_10": round(cnt["Pass"] / 10, 2),
            "lenient_precision_at_10": round((cnt["Pass"] + cnt["Uncertain"]) / 10, 2),
            "per_check_pass": {k: v["Pass"] for k, v in per_check.items()},
            "lower_ranked_sample": {"n": len(low), "pass": lcnt["Pass"], "uncertain": lcnt["Uncertain"],
                                    "fail": lcnt["Fail"]},
        }
        for rank, rid in top:
            if overall_label(labels[rid]) == "Fail":
                res["disagreements"].append({"method": m, "rank": rank, "review_id": rid,
                                             "candidate_id": key[rid]["candidate_id"],
                                             "notes": labels[rid].get("notes", "")})
    write_json(out_dir / "evaluation.json", res)
    (out_dir / "evaluation.md").write_text(_md(res), encoding="utf-8")
    return res


def _md(res: dict) -> str:
    L = ["# Blind human review results", "",
         "precision@10 = share of a method's top 10 (evaluation episodes 6-50) labelled **Pass** (postable as-is).", "",
         "| Method | Pass | Uncertain | Fail | precision@10 | lenient (Pass+Uncertain) | lower-ranked sample Pass/n |",
         "|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        lo = r["lower_ranked_sample"]
        L.append(f"| {m} | {r['pass']} | {r['uncertain']} | {r['fail']} | {r['precision_at_10']:.2f} | "
                 f"{r['lenient_precision_at_10']:.2f} | {lo['pass']}/{lo['n']} |")
    if res["unlabelled"]:
        L += ["", f"Unlabelled rows: {', '.join(res['unlabelled'])}"]
    if res["disagreements"]:
        L += ["", "## Top-10 picks a human failed (failure-case candidates)", ""]
        for d in res["disagreements"]:
            L.append(f"- {d['method']} #{d['rank']}: {d['candidate_id']} ({d['review_id']}) {d['notes']}")
    L += ["", "The lower-ranked sample is small; it hints at missed promos but is not a recall estimate."]
    return "\n".join(L) + "\n"
