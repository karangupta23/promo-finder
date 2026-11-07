#!/usr/bin/env python3
"""Kuku promo finder - command line entry point.

    python run.py all --provider gemini              # full pipeline on the input pack
    python run.py all --provider mock                # offline dry run (no API key needed)
    python run.py <step> --help                      # individual steps

Steps: parse, candidates, stage-a, rank, stage-b, leakage, review-sheet, score-review, export-web, models
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src import candidates, evaluate, export_web, judge_stage_a, judge_stage_b, leakage, parse, rank
from src.judging import model_for
from src.llm import FatalLLMError, GeminiClient
from src.utils import OUTPUTS, ROOT, env, load_dotenv_if_present

DEFAULT_RPM = {"gemini": {"a": 9, "b": 4, "leak": 9}}   # just under free-tier limits


def default_input() -> Path:
    pack = ROOT / "pack"
    if pack.is_dir() and any(pack.rglob("*.txt")):
        return pack
    zips = sorted(ROOT.glob("*.zip")) + (sorted(pack.glob("*.zip")) if pack.is_dir() else [])
    if zips:
        return zips[0]
    sys.exit("No input found: put the screenplay .txt files (or the pack .zip) in ./pack, or pass --input")


def parse_episodes(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def rpm_for(provider: str, stage: str, cli: float | None) -> float | None:
    if cli is not None:
        return cli or None
    e = env({"a": "RPM_STAGE_A", "b": "RPM_STAGE_B", "leak": "RPM_LEAKAGE"}[stage])
    if e:
        return float(e)
    return DEFAULT_RPM.get(provider, {}).get(stage)


def show(title: str, obj) -> None:
    print(f"\n== {title}\n{json.dumps(obj, indent=2, default=str)}")


def main(argv=None) -> None:
    load_dotenv_if_present()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def llm_args(p, stage):
        p.add_argument("--provider", default=env("LLM_PROVIDER", "gemini"), help="gemini | openai | mock")
        p.add_argument("--model", default=None, help="model id (default from env / built-in)")
        p.add_argument("--rpm", type=float, default=None, help="max requests per minute (0 = unthrottled)")
        p.set_defaults(stage=stage)

    p = sub.add_parser("parse"); p.add_argument("--input", type=Path)
    sub.add_parser("candidates")
    p = sub.add_parser("stage-a"); llm_args(p, "a")
    p.add_argument("--episodes", help="e.g. 1-5 (development split)"); p.add_argument("--limit", type=int)
    p = sub.add_parser("stage-b"); llm_args(p, "b"); p.add_argument("--budget", type=int, default=int(env("STAGE_B_BUDGET", "18")))
    sub.add_parser("rank")
    p = sub.add_parser("leakage"); llm_args(p, "leak"); p.add_argument("--top", type=int, default=15)
    sub.add_parser("review-sheet")
    p = sub.add_parser("score-review"); p.add_argument("--sheet", type=Path)
    sub.add_parser("export-web")
    p = sub.add_parser("models")
    p = sub.add_parser("all"); p.add_argument("--input", type=Path)
    p.add_argument("--provider", default=env("LLM_PROVIDER", "gemini"))
    p.add_argument("--model-a"); p.add_argument("--model-b")
    p.add_argument("--budget", type=int, default=int(env("STAGE_B_BUDGET", "18")))
    p.add_argument("--skip-b", action="store_true", help="stop after Stage A (e.g. Flash quota used up)")
    p.add_argument("--skip-leakage", action="store_true")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "parse":
            r = parse.run(a.input or default_input())
            show("parse", {k: v for k, v in r.items() if k != "episodes"})
        elif a.cmd == "candidates":
            show("candidates", candidates.run())
        elif a.cmd == "stage-a":
            m = a.model or model_for(a.provider, "a")
            show("stage A", judge_stage_a.run(a.provider, m, rpm_for(a.provider, "a", a.rpm),
                                              parse_episodes(a.episodes), a.limit))
        elif a.cmd == "stage-b":
            m = a.model or model_for(a.provider, "b")
            show("stage B", judge_stage_b.run(a.provider, m, a.budget, rpm_for(a.provider, "b", a.rpm)))
        elif a.cmd == "rank":
            show("rank", rank.run())
        elif a.cmd == "leakage":
            m = a.model or model_for(a.provider, "leak")
            show("leakage", leakage.run(a.provider, m, a.top, rpm_for(a.provider, "leak", a.rpm)))
            show("re-rank", rank.run())
        elif a.cmd == "review-sheet":
            show("review sheet", evaluate.make_sheet())
        elif a.cmd == "score-review":
            r = evaluate.score(a.sheet)
            show("evaluation", r["methods"])
        elif a.cmd == "export-web":
            show("export-web", export_web.run())
        elif a.cmd == "models":
            print("\n".join(GeminiClient("x").list_models()))
        elif a.cmd == "all":
            t0 = time.time()
            pv = a.provider
            r = parse.run(a.input or default_input())
            show("parse", {k: v for k, v in r.items() if k != "episodes"})
            show("candidates", candidates.run())
            show("stage A", judge_stage_a.run(pv, a.model_a or model_for(pv, "a"), rpm_for(pv, "a", None)))
            show("rank (A)", rank.run())
            if not a.skip_b:
                show("stage B", judge_stage_b.run(pv, a.model_b or model_for(pv, "b"), a.budget, rpm_for(pv, "b", None)))
                show("rank (A+B)", rank.run())
            if not a.skip_leakage:
                show("leakage", leakage.run(pv, a.model_a or model_for(pv, "leak"), 15, rpm_for(pv, "leak", None)))
                show("rank (final)", rank.run())
            show("review sheet", evaluate.make_sheet())
            show("export-web", export_web.run())
            print(f"\nDone in {time.time() - t0:.0f} s. See outputs/promos.md")
    except FatalLLMError as e:
        sys.exit(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
