"""Export frozen pipeline outputs to web/data.json for the static demo.

Deliberately exposes only short excerpts (hook and cliffhanger lines, trimmed) - never full
screenplays, full extracted episode data, API keys or raw model context.
"""
from __future__ import annotations

from pathlib import Path

from .utils import OUTPUTS, ROOT, read_json, write_json

MAX_EXCERPT = 160
TOP = 15


def _cut(s: str | None, n: int = MAX_EXCERPT) -> str | None:
    if not s:
        return s
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _evidence(part: dict) -> list[dict]:
    return [{"at": e["at"], "text": _cut(e["text"])} for e in (part.get("evidence") or [])[:2]]


def _labelled_evaluation(out_dir: Path):
    """Only publish review results once some clips have actually been labelled."""
    p = out_dir / "evaluation.json"
    if not p.exists():
        return None
    ev = read_json(p)
    labelled = sum(m["pass"] + m["uncertain"] + m["fail"] for m in ev.get("methods", {}).values())
    return ev if labelled else None


def run(out_dir: Path = OUTPUTS, web_dir: Path = ROOT / "web") -> dict:
    promos = read_json(out_dir / "promos.json")
    q = promos["quality_ranking"][:TOP]
    items = []
    for p in q:
        st = p.get("standalone") or {}
        items.append({
            "id": p["candidate_id"], "quality_rank": p["quality_rank"], "status": p["status"],
            "episode": p["episode"], "start": p["start"], "end": p["end"], "duration_s": p["duration_s"],
            "score": p["score"], "score_source": p["score_source"], "scores": p["scores"], "gates": p.get("gates"),
            "hook": {"description": _cut(p["hook"].get("description"), 300), "evidence": _evidence(p["hook"])},
            "cliffhanger": {"description": _cut(p["cliffhanger"].get("description"), 300),
                            "question": _cut(p["cliffhanger"].get("unresolved_question"), 200),
                            "evidence": _evidence(p["cliffhanger"])},
            "standalone": {"explanation": _cut(st.get("explanation"), 300),
                           "central_character": _cut((st.get("central_character") or {}).get("answer"), 120),
                           "missing_context": [_cut(x, 120) for x in (st.get("missing_context") or [])[:3]]},
            "why": _cut(p.get("why_it_works"), 240), "posting_rationale": _cut(p.get("posting_rationale"), 300),
            "risks": [_cut(r, 140) for r in (p.get("risks") or [])[:3]],
            "leakage": (p.get("leakage") or {}).get("recommendation"),
            "methods": p["generation_methods"],
        })
    data = {
        "generated_at": promos["generated_at"], "ranking_method": promos["ranking_method"],
        "note": promos.get("note"), "counts": promos["counts"], "posting_order": promos["posting_order"],
        "promos": items,
        "coverage": {k: v for k, v in read_json(out_dir / "coverage_report.json").items() if k != "episodes"
                     and k != "input"},
        "candidates": read_json(out_dir / "candidates_stats.json") if (out_dir / "candidates_stats.json").exists() else None,
        "run_log": read_json(out_dir / "run_log.json") if (out_dir / "run_log.json").exists() else None,
        "evaluation": _labelled_evaluation(out_dir),
        "failure_cases": read_json(ROOT / "docs" / "failure_cases.json") if (ROOT / "docs" / "failure_cases.json").exists() else [],
    }
    for p in data["posting_order"]:
        p["posting_rationale"] = _cut(p.get("posting_rationale"), 300)
    if data["run_log"]:
        for s in data["run_log"].get("stages", []):
            s.pop("errors", None)
    write_json(web_dir / "data.json", data)
    return {"promos": len(items), "posting_order": len(data["posting_order"]),
            "has_evaluation": data["evaluation"] is not None, "failure_cases": len(data["failure_cases"])}
