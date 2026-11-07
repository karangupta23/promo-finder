"""Step 1 - parse screenplay files into timestamped blocks.

Rules (from the plan):
  * keep original speaker labels - no alias merging ("JULIANA" and "LADY JULIANA" stay distinct)
  * keep original timestamps - never "fix" them, only flag problems
  * every block gets a stable id  E{ep:02d}_B{n:03d}  (order of appearance in the file)
  * malformed timestamps, overlaps and out-of-scene blocks are flagged, not repaired

Input can be a folder of .txt files or the .zip from the input pack (read in memory).
"""
from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path

from .utils import OUTPUTS, Block, EpisodeMeta, write_json, write_jsonl

RE_EP_NUM = re.compile(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
RE_LOGLINE = re.compile(r"^EPISODE LOGLINE:\s*(.*)", re.I)
RE_CONT = re.compile(r"^CONTINUITY NOTES:\s*(.*)", re.I)
RE_SCENE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
RE_HEADING = re.compile(r"^SCENE HEADING:\s*(.*)", re.I)
RE_SC_START = re.compile(r"^SCENE START:\s*(\S+)", re.I)
RE_SC_END = re.compile(r"^SCENE END:\s*(\S+)", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
RE_TIMING = re.compile(r"^Start:\s*\((?P<s>[^)]*)\)\s*\|\s*End:\s*\((?P<e>[^)]*)\)\s*$", re.I)
RE_TS = re.compile(r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
RE_FILE_EP = re.compile(r"episode-(\d+)", re.I)
RE_PAREN = re.compile(r"\(([^)]*)\)")

TYPE_BY_LABEL = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound", "SFX": "sound"}
OFFSCREEN_TAGS = {"O.S.", "V.O.", "O.C.", "OS", "VO", "CONT'D"}
OVERLAP_TOL = 0.05


def parse_ts(s: str):
    m = RE_TS.match(s.strip())
    if not m:
        return None
    return int(m.group(1)) * 60 + float(m.group(2))


def read_input(path: Path) -> list[tuple[str, str]]:
    """Return [(file_name, text)] for every episode file, sorted by name."""
    path = Path(path)
    files: list[tuple[str, str]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                name = info.filename
                if info.is_dir() or not name.lower().endswith(".txt") or "__MACOSX" in name:
                    continue
                files.append((Path(name).name, z.read(info).decode("utf-8", errors="replace")))
    elif path.is_dir():
        for p in sorted(path.rglob("*.txt")):
            files.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    else:
        raise FileNotFoundError(f"Input must be a folder of .txt screenplays or a .zip: {path}")
    return sorted(files)


def split_label(label: str):
    """'MATRIARCH (O.S.) (Firm)' -> speaker 'MATRIARCH', delivery 'Firm', offscreen True."""
    parens = [p.strip() for p in RE_PAREN.findall(label)]
    speaker = RE_PAREN.sub("", label).strip()
    offscreen = any(p.upper() in OFFSCREEN_TAGS for p in parens)
    delivery = [p for p in parens if p.upper() not in OFFSCREEN_TAGS]
    return speaker, (delivery[-1] if delivery else None), offscreen


def parse_episode(file_name: str, text: str) -> tuple[EpisodeMeta, list[Block], list[str]]:
    lines = text.replace("\r\n", "\n").split("\n")
    issues: list[str] = []
    ep_num = None
    meta = {"logline": "", "continuity_notes": "", "title_line": ""}
    for ln in lines[:10]:
        if (m := RE_EP_NUM.match(ln)):
            ep_num = int(m.group(1))
            meta["title_line"] = ln.strip()
        elif (m := RE_LOGLINE.match(ln)):
            meta["logline"] = m.group(1).strip()
        elif (m := RE_CONT.match(ln)):
            meta["continuity_notes"] = m.group(1).strip()
    fm = RE_FILE_EP.search(file_name)
    file_ep = int(fm.group(1)) if fm else None
    if ep_num is None:
        ep_num = file_ep
        issues.append("missing EPISODE NUMBER header; used file name")
    elif file_ep is not None and file_ep != ep_num:
        issues.append(f"header says episode {ep_num} but file name says {file_ep}")
    if ep_num is None:
        raise ValueError(f"cannot determine episode number for {file_name}")

    blocks: list[Block] = []
    scene, heading, sc_range = 0, "", (None, None)
    scene_ranges: dict[int, tuple] = {}
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if (m := RE_SCENE.match(ln)):
            scene = int(m.group(1)); heading = ""; sc_range = (None, None)
        elif (m := RE_HEADING.match(ln)):
            heading = m.group(1).strip()
        elif (m := RE_SC_START.match(ln)):
            sc_range = (parse_ts(m.group(1)), sc_range[1])
        elif (m := RE_SC_END.match(ln)):
            sc_range = (sc_range[0], parse_ts(m.group(1)))
            scene_ranges[scene] = sc_range
        elif (m := RE_LABEL.match(ln)):
            label = m.group(1).strip()
            flags: list[str] = []
            start = end = None
            j = i + 1
            tm = RE_TIMING.match(lines[j].strip()) if j < len(lines) else None
            if tm:
                start, end = parse_ts(tm.group("s")), parse_ts(tm.group("e"))
                if start is None or end is None:
                    flags.append("malformed_timestamp")
                j += 1
            else:
                flags.append("missing_timestamp")
            body = []
            while j < len(lines) and lines[j].strip() and not RE_LABEL.match(lines[j].strip()):
                body.append(lines[j].strip()); j += 1
            if not body:
                flags.append("missing_text")
            up = label.upper()
            btype = TYPE_BY_LABEL.get(up, "dialogue")
            speaker = delivery = None
            offscreen = False
            if btype == "dialogue":
                speaker, delivery, offscreen = split_label(label)
            if start is not None and end is not None:
                if end < start:
                    flags.append("end_before_start")
                elif end == start:
                    flags.append("zero_duration")
                lo, hi = sc_range
                if lo is not None and hi is not None and (start < lo - OVERLAP_TOL or end > hi + OVERLAP_TOL):
                    flags.append("outside_scene_range")
            blocks.append(Block(
                episode=ep_num, block_id=f"E{ep_num:02d}_B{len(blocks) + 1:03d}", scene=scene,
                scene_heading=heading, start=start, end=end, type=btype, label_raw=label,
                speaker=speaker, delivery=delivery, offscreen=offscreen,
                text=" ".join(body), flags=flags))
            i = j
            continue
        i += 1

    # Overlapping dialogue lines (two people talking over each other, or a timing error).
    dlg = [b for b in blocks if b.type == "dialogue" and b.start is not None and b.end is not None]
    for a, b in zip(dlg, dlg[1:]):
        if b.start < a.end - OVERLAP_TOL:
            b.flags.append(f"overlaps_dialogue:{a.block_id}")
    # Non-monotonic starts (blocks listed out of time order)
    timed = [b for b in blocks if b.start is not None]
    for a, b in zip(timed, timed[1:]):
        if b.start < a.start - 1.0:
            b.flags.append("out_of_order")

    ends = [b.end for b in blocks if b.end is not None]
    ends += [r[1] for r in scene_ranges.values() if r[1] is not None]
    em = EpisodeMeta(episode=ep_num, source_file=file_name, duration=max(ends) if ends else 0.0,
                     n_scenes=len(scene_ranges), n_blocks=len(blocks), **meta)
    if not blocks:
        issues.append("no blocks parsed")
    return em, blocks, issues


def run(input_path: Path, out_dir: Path = OUTPUTS) -> dict:
    files = read_input(input_path)
    all_blocks: list[Block] = []
    metas: list[EpisodeMeta] = []
    report_eps = []
    failed = []
    for name, text in files:
        try:
            meta, blocks, issues = parse_episode(name, text)
        except Exception as e:  # noqa: BLE001 - report, don't crash the batch
            failed.append({"file": name, "error": str(e)})
            continue
        metas.append(meta)
        all_blocks.extend(blocks)
        flag_counts = Counter(f.split(":")[0] for b in blocks for f in b.flags)
        report_eps.append({
            "episode": meta.episode, "file": name, "duration_s": round(meta.duration, 2),
            "scenes": meta.n_scenes, "blocks": len(blocks),
            "by_type": dict(Counter(b.type for b in blocks)),
            "flags": dict(flag_counts), "issues": issues,
        })
    eps = sorted(m.episode for m in metas)
    dup = [e for e, c in Counter(eps).items() if c > 1]
    missing = [e for e in range(1, max(eps) + 1) if e not in eps] if eps else []
    total_flags = Counter(f.split(":")[0] for b in all_blocks for f in b.flags)
    report = {
        "input": str(input_path),
        "files_found": len(files),
        "episodes_parsed": len(metas),
        "episodes_failed": failed,
        "missing_episode_numbers": missing,
        "duplicate_episode_numbers": dup,
        "blocks_total": len(all_blocks),
        "blocks_by_type": dict(Counter(b.type for b in all_blocks)),
        "blocks_with_invalid_timestamps": sum(
            1 for b in all_blocks if {"malformed_timestamp", "missing_timestamp", "end_before_start"} & set(b.flags)),
        "flag_counts": dict(total_flags),
        "distinct_speaker_labels": len({b.speaker for b in all_blocks if b.speaker}),
        "total_runtime_s": round(sum(m.duration for m in metas), 1),
        "episodes": sorted(report_eps, key=lambda r: r["episode"]),
    }
    all_blocks.sort(key=lambda b: (b.episode, int(b.block_id.split("_B")[1])))
    write_jsonl(out_dir / "beats.jsonl", all_blocks)
    write_json(out_dir / "episodes.json", [m.model_dump() for m in sorted(metas, key=lambda m: m.episode)])
    write_json(out_dir / "coverage_report.json", report)
    return report
