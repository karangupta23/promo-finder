"""Shared helpers: paths, timestamps, JSON I/O and the core data models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"
PROMPTS_DIR = ROOT / "prompts"

MIN_DUR = 20.0
MAX_DUR = 60.0
HOOK_WINDOW = 3.0      # seconds from clip start that count as "the hook"
CLIFF_WINDOW = 5.0     # seconds before clip end that count as "the cliffhanger"

# Development / evaluation split (see README). Prompts and candidate settings are
# tuned on DEV episodes only and frozen before looking at EVAL results.
DEV_EPISODES = set(range(1, 6))


def is_eval_episode(ep: int) -> bool:
    return ep not in DEV_EPISODES


# ----------------------------------------------------------------- data models
BlockType = Literal["camera", "action", "sound", "dialogue"]


class Block(BaseModel):
    episode: int
    block_id: str
    scene: int
    scene_heading: str = ""
    start: Optional[float]
    end: Optional[float]
    type: BlockType
    label_raw: str                      # exactly as written between ** **
    speaker: Optional[str] = None        # original label, parentheticals removed
    delivery: Optional[str] = None       # e.g. "Cold, Commanding"
    offscreen: bool = False              # (O.S.) / (V.O.)
    text: str
    flags: list[str] = Field(default_factory=list)


class EpisodeMeta(BaseModel):
    episode: int
    source_file: str
    title_line: str = ""
    logline: str = ""            # kept for humans / leakage check only, never sent to the judge
    continuity_notes: str = ""   # ditto
    duration: float = 0.0
    n_scenes: int = 0
    n_blocks: int = 0


class Candidate(BaseModel):
    candidate_id: str
    episode: int
    start: float
    end: float
    duration: float
    block_ids: list[str]
    methods: list[str]                       # generation strategies that proposed it
    anchor_block_id: Optional[str] = None
    heuristic: dict = Field(default_factory=dict)   # baseline features + score


# ----------------------------------------------------------------- helpers
def fmt_ts(sec: Optional[float]) -> str:
    if sec is None:
        return "??:??.??"
    m, s = divmod(round(sec, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, BaseModel):
                r = r.model_dump()
            f.write(json.dumps(r, ensure_ascii=False, default=_default) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _default(o):
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def stable_hash_int(s: str) -> int:
    """Deterministic (process-independent) integer hash for sampling/shuffling."""
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def load_blocks(path: Path | None = None) -> dict[int, list[Block]]:
    path = path or OUTPUTS / "beats.jsonl"
    by_ep: dict[int, list[Block]] = {}
    for row in read_jsonl(path):
        b = Block(**row)
        by_ep.setdefault(b.episode, []).append(b)
    return by_ep


def load_candidates(path: Path | None = None) -> list[Candidate]:
    path = path or OUTPUTS / "candidates.jsonl"
    return [Candidate(**r) for r in read_jsonl(path)]


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def load_dotenv_if_present() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
