"""Step 2 - generate continuous 20-60 s candidate windows.

Two independent families of strategies (they can propose the same window; methods are merged):

  A. Shot-aligned     - windows that start/end on camera-shot boundaries
       * cliffhanger-driven : pick the strongest endings, then the best shot-aligned opening 20-60 s earlier
       * coverage           : one lexicon-free window per episode, chosen by a fixed rule (no randomness)
  B. Event-anchored   - find a strong moment first, then open the clip on it (even mid-shot)
       * dialogue-driven    : a sharp line (accusation, threat, reveal, question)
       * action/visual      : a slap, fall, entrance, reveal of an object...

Cut rules: never start or end inside a line of dialogue; prefer camera boundaries ("clean" cuts).
Slots per episode: dialogue 3, action/visual 2, cliffhanger 2, coverage 1  ->  <= 8 per episode.
"""
from __future__ import annotations

import json
from collections import Counter

from .heuristics import blocks_in, block_cliff_score, block_hook_score, window_features
from .utils import MAX_DUR, MIN_DUR, OUTPUTS, Block, Candidate, load_blocks, read_json, write_jsonl

SLOTS = {"dialogue": 3, "action": 2, "cliffhanger": 2, "coverage": 1}
TARGET_PER_EPISODE = 7   # strategies overlap; backfill distinct windows up to this many
ANCHOR_LEAD = 1.5        # an event anchor may sit at most this far after the cut
DEDUPE_IOU = 0.7
SETTINGS_VERSION = "cand-v1"
TOL = 0.05


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


class EpisodeTimeline:
    def __init__(self, ep: int, blocks: list[Block], duration: float):
        self.ep = ep
        self.blocks = [b for b in blocks if b.start is not None and b.end is not None and b.end >= b.start]
        self.duration = duration
        self.dialogue = [b for b in self.blocks if b.type == "dialogue"]
        cam = [b for b in self.blocks if b.type == "camera"]
        self.cam_starts = {round(b.start, 2) for b in cam}
        self.cam_ends = {round(b.end, 2) for b in cam} | {round(duration, 2)}
        starts = self.cam_starts | {round(b.start, 2) for b in self.blocks if b.type != "camera"}
        ends = self.cam_ends | {round(b.end, 2) for b in self.blocks if b.type != "camera"}
        self.starts = sorted(t for t in starts if not self.inside_dialogue(t) and t < duration - MIN_DUR + TOL)
        self.ends = sorted(t for t in ends if not self.inside_dialogue(t) and t <= duration + TOL)

    def inside_dialogue(self, t: float) -> bool:
        return any(d.start + 0.1 < t < d.end - 0.1 for d in self.dialogue)

    def clean(self, t: float, kind: str) -> bool:
        pool = self.cam_starts if kind == "start" else self.cam_ends
        return round(t, 2) in pool

    def feats(self, s: float, e: float) -> dict:
        return window_features(self.blocks, s, e, self.clean(s, "start"), self.clean(e, "end"), self.duration)

    def best_end(self, s: float):
        best = None
        for e in self.ends:
            if MIN_DUR - TOL <= e - s <= MAX_DUR + TOL:
                f = self.feats(s, e)
                if best is None or f["score"] > best[1]["score"]:
                    best = (e, f)
        return best

    def best_start(self, e: float):
        best = None
        for s in self.starts:
            if MIN_DUR - TOL <= e - s <= MAX_DUR + TOL:
                f = self.feats(s, e)
                if best is None or f["score"] > best[1]["score"]:
                    best = (s, f)
        return best

    def start_for_anchor(self, b: Block):
        """Latest legal cut at or before the anchor, no more than ANCHOR_LEAD s earlier.
        Prefer a clean (camera) cut; fall back to cutting exactly on the anchor."""
        options = [s for s in self.starts if b.start - ANCHOR_LEAD - TOL <= s <= b.start + TOL]
        clean = [s for s in options if self.clean(s, "start")]
        if clean:
            return max(clean)
        return max(options) if options else None

    def make(self, s: float, e: float, f: dict, method: str, anchor: str | None) -> Candidate:
        inside = blocks_in(self.blocks, s, e)
        return Candidate(
            candidate_id=f"E{self.ep:02d}_{int(round(s * 10)):04d}_{int(round(e * 10)):04d}",
            episode=self.ep, start=round(s, 2), end=round(e, 2), duration=round(e - s, 2),
            block_ids=[b.block_id for b in inside], methods=[method], anchor_block_id=anchor, heuristic=f)


def _pick(windows: list[Candidate], k: int) -> list[Candidate]:
    """Top-k by heuristic score with no two windows overlapping heavily."""
    chosen: list[Candidate] = []
    for c in sorted(windows, key=lambda c: (-c.heuristic["score"], c.start)):
        if all(_iou((c.start, c.end), (o.start, o.end)) < DEDUPE_IOU for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen


def generate_episode(tl: EpisodeTimeline) -> list[Candidate]:
    out: dict[str, list[Candidate]] = {}
    spare: list[Candidate] = []          # windows a strategy proposed but did not slot

    # ---- B. event-anchored: dialogue
    anchors = sorted(tl.dialogue, key=lambda b: -block_hook_score(b))[:12]
    wins = []
    for a in anchors:
        s = tl.start_for_anchor(a)
        if s is None or (r := tl.best_end(s)) is None:
            continue
        wins.append(tl.make(s, r[0], r[1], "event:dialogue", a.block_id))
    out["dialogue"] = _pick(wins, SLOTS["dialogue"])
    spare += [w for w in wins if w not in out["dialogue"]]

    # ---- B. event-anchored: action / visual
    visual = [b for b in tl.blocks if b.type != "dialogue" and block_hook_score(b) >= 4.0]
    anchors = sorted(visual, key=lambda b: (-block_hook_score(b), b.start))[:10]
    wins = []
    for a in anchors:
        s = tl.start_for_anchor(a)
        if s is None or (r := tl.best_end(s)) is None:
            continue
        wins.append(tl.make(s, r[0], r[1], "event:action", a.block_id))
    out["action"] = _pick(wins, SLOTS["action"])
    spare += [w for w in wins if w not in out["action"]]

    # ---- A. shot-aligned, cliffhanger-driven: strongest endings first
    end_scores = []
    for e in tl.ends:
        ending = [b for b in tl.blocks if abs(b.end - e) < 0.3]
        sc = max((block_cliff_score(b) for b in ending), default=0.0)
        if abs(e - tl.duration) < 0.3:
            sc += 2.0      # the episode's own ending is written as a cliffhanger
        end_scores.append((sc, e))
    wins = []
    for _, e in sorted(end_scores, key=lambda x: (-x[0], -x[1]))[:8]:
        if (r := tl.best_start(e)) is None:
            continue
        s, f = r
        wins.append(tl.make(s, e, f, "shot:cliffhanger", None))
    out["cliffhanger"] = _pick(wins, SLOTS["cliffhanger"])
    spare += [w for w in wins if w not in out["cliffhanger"]]

    # ---- A. shot-aligned coverage: fixed rule, no lexicon. Clean cut nearest (midpoint - 20 s),
    # end on the legal cut closest to 40 s later.  Guards against heuristic blind spots.
    clean_starts = [s for s in tl.starts if tl.clean(s, "start")] or tl.starts
    target = max(0.0, tl.duration / 2 - 20)
    s = min(clean_starts, key=lambda x: (abs(x - target), x))
    ends = [e for e in tl.ends if MIN_DUR - TOL <= e - s <= MAX_DUR + TOL]
    if ends:
        e = min(ends, key=lambda x: (abs((x - s) - 40), x))
        out["coverage"] = [tl.make(s, e, tl.feats(s, e), "shot:coverage", None)]

    # ---- merge identical / near-identical windows across strategies
    merged: list[Candidate] = []
    for c in [c for k in ("dialogue", "action", "cliffhanger", "coverage") for c in out.get(k, [])]:
        dup = next((m for m in merged if _iou((c.start, c.end), (m.start, m.end)) >= DEDUPE_IOU), None)
        if dup is None:
            merged.append(c)
            continue
        keep, drop = (dup, c) if dup.heuristic["score"] >= c.heuristic["score"] else (c, dup)
        keep.methods = sorted(set(keep.methods) | set(drop.methods))
        if keep is c:
            merged[merged.index(dup)] = c
    # backfill: the slot winners often coincide, so top up with the best distinct spare windows
    for c in sorted(spare, key=lambda c: (-c.heuristic["score"], c.start)):
        if len(merged) >= TARGET_PER_EPISODE:
            break
        if all(_iou((c.start, c.end), (m.start, m.end)) < DEDUPE_IOU for m in merged):
            c.methods = [c.methods[0] + "+backfill"]
            merged.append(c)
    return sorted(merged, key=lambda c: c.start)


def run(out_dir=OUTPUTS) -> dict:
    by_ep = load_blocks(out_dir / "beats.jsonl")
    durations = {m["episode"]: m["duration"] for m in read_json(out_dir / "episodes.json")}
    cands: list[Candidate] = []
    for ep in sorted(by_ep):
        tl = EpisodeTimeline(ep, by_ep[ep], durations[ep])
        cands.extend(generate_episode(tl))
    # validate before writing - every candidate must be a legal continuous window
    for c in cands:
        assert MIN_DUR - TOL <= c.duration <= MAX_DUR + TOL, c
        assert c.block_ids, c
    write_jsonl(out_dir / "candidates.jsonl", cands)
    method_counts = Counter(m for c in cands for m in c.methods)
    stats = {
        "settings_version": SETTINGS_VERSION, "slots": SLOTS, "target_per_episode": TARGET_PER_EPISODE,
        "candidates_total": len(cands), "episodes": len(by_ep),
        "per_episode_min": min(Counter(c.episode for c in cands).values()),
        "per_episode_max": max(Counter(c.episode for c in cands).values()),
        "by_method": dict(method_counts),
        "multi_method": sum(1 for c in cands if len(c.methods) > 1),
        "mean_duration_s": round(sum(c.duration for c in cands) / len(cands), 1),
        "clean_both_ends": sum(1 for c in cands if c.heuristic["clean_cut"] >= 10),
    }
    (out_dir / "candidates_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
