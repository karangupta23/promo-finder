"""Transparent, lexicon-based signals used for (a) candidate generation and
(b) the *Baseline* ranking method that the LLM judges are compared against.

Everything here is deliberately simple and inspectable. It is NOT meant to be a
good judge - it is meant to (1) propose plausible windows cheaply and (2) give
the evaluation an honest non-LLM baseline.
"""
from __future__ import annotations

import re

from .utils import CLIFF_WINDOW, HOOK_WINDOW, Block

# ---- lexicons (lower-case; matched as word prefixes) --------------------------
PHYSICAL = [
    "slap", "strik", "struck", "punch", "grab", "seiz", "shove", "collaps", "faint", "kneel",
    "knees", "sword", "blade", "knife", "dagger", "pistol", "gun", "blood", "bleed", "burst",
    "slam", "shatter", "crash", "scream", "kiss", "chok", "throw", "throws", "hurl", "rip",
    "kick", "stab", "flame", "fire", "smash", "chain", "handcuff", "arrest", "drag", "yank",
    "tackl", "lung", "pin", "wrench", "fist", "thud", "impact", "storm", "barg", "splinter",
]
REVEAL = [
    "reveal", "unmask", "ring", "seal", "crest", "letter", "document", "decree", "signet",
    "pendant", "locket", "scar", "medal", "uniform", "insignia", "portrait", "pregnan",
]
STRONG_DELIVERY = [
    "shout", "scream", "furious", "outrag", "enrag", "roar", "terrif", "threaten", "menac",
    "venom", "chilling", "sinister", "shock", "stunn", "horrif", "desperat", "crying out",
    "gloat", "mock", "spite", "taunt", "cruel", "accus", "commanding", "demanding", "sobbing",
    "explod", "snarl", "hiss", "seething", "livid", "frantic", "panick",
]
HOOK_WORDS = [
    "married", "marry", "husband", "wife", "duke", "kill", "dead", "die", "lie", "liar", "betray",
    "never", "how dare", "who are you", "get out", "arrest", "guards", "divorce", "kneel", "secret",
    "truth", "traitor", "murder", "worthless", "beggar", "pathetic", "disgrace", "humiliat", "steal",
    "thief", "prison", "hang", "witch", "bastard", "pregnant", "baby", "heir", "your grace",
    "mistake", "nothing", "dare",
]
SHOT_HOOK = ["extreme close", "whip", "crash zoom", "snap zoom", "smash", "slow-motion", "slow motion"]
CLIFF_SOUND = ["cliffhanger"]
CLIFF_SOUND_SOFT = ["ominous", "sting", "swell", "drone", "dramatic", "boom", "sudden silence", "abrupt"]
CLIFF_CAMERA = ["cut to black", "smash to black", "smash cut", "freeze", "hard cut"]
CLIFF_ACTION = [
    "freez", "stunn", "eyes widen", "widen", "turns slowly", "reach", "grips", "door", "steps forward",
    "draws", "raises", "frozen", "gasps", "shock", "pale", "trembl", "hand hover", "looms", "silhouett",
    "approach", "towering", "lunges", "whirls", "spins", "blood drains", "horror",
]
THREAT = ["you will", "or else", "you'll regret", "i will make", "i'll make", "pay for", "before i",
          "if you", "you have no idea", "you don't know who", "do you know who", "watch", "remember this"]
RESOLUTION = ["smiles warmly", "laugh", "hug", "embrac", "relief", "sleep", "peace", "calm", "content",
              "fade-out", "fades out", "fading out", "gentle", "tender", "warm", "rest", "thank you",
              "goodnight", "good night"]
PRONOUN_START = re.compile(r"^\W*(he|she|they|it|that|this|those|them|him|her)\b", re.I)


def _has(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


def _clip(x: float, lo=0.0, hi=10.0) -> float:
    return max(lo, min(hi, x))


def block_hook_score(b: Block) -> float:
    t = b.text
    if b.type == "dialogue":
        s = 2.0
        s += 2.0 if "!" in t else 0
        s += 1.0 if "?" in t else 0
        s += 2.0 if _has(b.delivery or "", STRONG_DELIVERY) else 0
        s += min(3.0, 1.5 * _has(t, HOOK_WORDS))
        if len(t.split()) <= 7:
            s += 1.0
        if t.strip('"\' ').startswith(("...", "…")):
            s -= 3.0    # opens mid-sentence
        return _clip(s)
    if b.type == "action":
        return _clip(1.0 + 3.0 * min(2, _has(t, PHYSICAL)) + 2.0 * min(1, _has(t, REVEAL)))
    if b.type == "sound":
        return _clip(1.0 + 3.0 * min(2, _has(t, PHYSICAL)) + 1.0 * min(1, _has(t, ["gasp", "thunder", "bang"])))
    # camera
    return _clip(0.5 + 1.5 * min(1, _has(t, SHOT_HOOK)) + 2.0 * min(1, _has(t, PHYSICAL)) + 1.0 * min(1, _has(t, REVEAL)))


def block_cliff_score(b: Block) -> float:
    t = b.text
    tl = t.lower()
    if b.type == "sound":
        if _has(t, CLIFF_SOUND):
            return 9.0
        s = 2.0 + 2.5 * min(1, _has(t, CLIFF_SOUND_SOFT))
        return _clip(s - 3.0 * min(1, _has(t, RESOLUTION)))
    if b.type == "camera":
        s = 1.0 + (4.0 if _has(t, CLIFF_CAMERA) else 0) + (1.0 if "close" in tl else 0)
        return _clip(s - 2.0 * min(1, _has(t, RESOLUTION)))
    if b.type == "action":
        return _clip(2.0 + 3.0 * min(1, _has(t, CLIFF_ACTION)) + 1.0 * min(1, _has(t, PHYSICAL))
                     - 3.0 * min(1, _has(t, RESOLUTION)))
    # dialogue
    st = t.strip('"\' ')
    s = 2.0
    if st.endswith("?"):
        s += 3.0
    if st.endswith(("...", "…", "—", "--", "-")):
        s += 3.5   # interrupted / trailing line
    if _has(t, THREAT):
        s += 2.5
    if st.endswith("!"):
        s += 1.5
    s += 1.0 if _has(b.delivery or "", STRONG_DELIVERY) else 0
    s -= 3.0 * min(1, _has(t + " " + (b.delivery or ""), RESOLUTION))
    return _clip(s)


def blocks_in(blocks: list[Block], s: float, e: float) -> list[Block]:
    """Blocks that belong to the clip [s, e]: fully inside, or overlapping it by > 0.5 s."""
    out = []
    for b in blocks:
        if b.start is None or b.end is None:
            continue
        ov = min(b.end, e) - max(b.start, s)
        if (b.start >= s - 0.05 and b.end <= e + 0.05) or ov > 0.5:
            out.append(b)
    return out


def hook_signal(blocks: list[Block], s: float):
    """Best hook evidence among blocks that start inside the first HOOK_WINDOW seconds
    (or are already under way at the cut and continue for > 1 s)."""
    cands = [b for b in blocks if b.start is not None and
             ((s - 0.05 <= b.start < s + HOOK_WINDOW) or (b.start < s < b.end - 1.0))]
    if not cands:
        return 0.0, None
    scored = sorted(((block_hook_score(b), b) for b in cands), key=lambda x: -x[0])
    best, bb = scored[0]
    bonus = 0.5 if len(scored) > 1 and scored[1][0] >= 5 else 0.0
    # A first line that begins mid-sentence is a bad open regardless of other signals.
    first_dlg = next((b for b in sorted(cands, key=lambda b: b.start) if b.type == "dialogue"), None)
    if first_dlg is not None and first_dlg.text.strip('"\' ').startswith(("...", "…")):
        best -= 2.0
    return _clip(best + bonus), bb.block_id


def cliff_signal(blocks: list[Block], e: float):
    cands = [b for b in blocks if b.end is not None and e - CLIFF_WINDOW <= b.end <= e + 0.05]
    if not cands:
        return 0.0, None
    scored = sorted(((block_cliff_score(b), b) for b in cands), key=lambda x: -x[0])
    best, bb = scored[0]
    # the very last line of dialogue matters most: penalise a resolving last line
    last_dlg = max((b for b in cands if b.type == "dialogue"), key=lambda b: b.end, default=None)
    if last_dlg is not None and _has(last_dlg.text + " " + (last_dlg.delivery or ""), RESOLUTION):
        best -= 2.0
    return _clip(best), bb.block_id


def window_features(ep_blocks: list[Block], s: float, e: float, clean_start: bool, clean_end: bool,
                    episode_end: float) -> dict:
    inside = blocks_in(ep_blocks, s, e)
    dur = e - s
    hook, hook_id = hook_signal(inside, s)
    cliff, cliff_id = cliff_signal(inside, e)

    dlg = [b for b in inside if b.type == "dialogue"]
    # dialogue coverage and the longest stretch with no dialogue/action/sound event
    events = sorted([b for b in inside if b.type != "camera"], key=lambda b: b.start)
    covered, cursor, longest_gap = 0.0, s, 0.0
    for b in events:
        st, en = max(b.start, s), min(b.end, e)
        if st > cursor:
            longest_gap = max(longest_gap, st - cursor)
        if b.type == "dialogue":
            covered += max(0.0, en - st)
        cursor = max(cursor, en)
    longest_gap = max(longest_gap, e - cursor)
    coverage = covered / dur if dur else 0.0

    speakers = {b.speaker for b in dlg if b.speaker}
    scenes = {b.scene for b in inside}
    intense = sum(1 for b in dlg if _has(b.delivery or "", STRONG_DELIVERY)) + \
        sum(1 for b in inside if b.type in ("action", "sound") and _has(b.text, PHYSICAL))
    first_dlg = dlg[0] if dlg else None

    # ---- component scores on 0-10 -------------------------------------------
    standalone = 8.0
    if len(speakers) > 4:
        standalone -= 1.5 * (len(speakers) - 4)
    if len(speakers) == 0:
        standalone -= 3
    if first_dlg is not None and PRONOUN_START.match(first_dlg.text.strip('"')):
        standalone -= 1.0
    if first_dlg is not None and first_dlg.text.strip('"\' ').startswith(("...", "…")):
        standalone -= 2.5
    standalone -= 1.5 * (len(scenes) - 1)
    emotion = _clip(2 + 10 * intense / max(1.0, dur / 10) / 3)
    pacing = _clip(10 - max(0.0, longest_gap - 4) * 1.2 - max(0.0, 0.35 - coverage) * 15)
    dur_pref = 10 - abs(dur - 38) / 3.5      # sweet spot ~30-45 s
    clean = _clip(5 + 2.5 * clean_start + 2.5 * clean_end)
    # ending exactly on the episode's own final cliffhanger is a strong prior in microdrama
    if abs(e - episode_end) < 0.3:
        cliff = _clip(cliff + 1.0)
    score = (0.25 * hook + 0.25 * cliff + 0.20 * _clip(standalone) + 0.15 * emotion
             + 0.10 * _clip((pacing + dur_pref) / 2) + 0.05 * clean) * 10
    return {
        "score": round(score, 2),
        "hook": round(hook, 2), "hook_block": hook_id,
        "cliff": round(cliff, 2), "cliff_block": cliff_id,
        "standalone": round(_clip(standalone), 2), "emotion": round(emotion, 2),
        "pacing": round(pacing, 2), "clean_cut": round(clean, 2),
        "dialogue_coverage": round(coverage, 3), "longest_gap_s": round(longest_gap, 2),
        "n_speakers": len(speakers), "n_scenes": len(scenes),
        "ends_at_episode_end": abs(e - episode_end) < 0.3,
    }
