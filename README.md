# Promo Finder — Kuku AI Generalist Assignment, Problem 1

Reads microdrama screenplays and returns a ranked list of **continuous 20–60 s promo clips**: each with episode,
exact start/end timestamps, the full blocks in the clip, the hook (first 3 s), the cliffhanger and the question
it leaves open, why it works for a first-time viewer, scores, risks, and a recommended posting order.

The Python CLI is the product. The `web/` page only displays frozen outputs.

```
screenplays ──► 1 parse ──► 2 candidates ──► 3 Stage A ──► 4 Stage B ──► 5 rank ──► 6 leakage check ──► rank
 (.txt/.zip)    beats.jsonl  candidates.jsonl  cheap model    strong model   gates,      sees what comes
                coverage     shot-aligned +    every clip     ~18-40 clips   score,      AFTER the clip
                report       event-anchored                                  overlap,    (separate on purpose)
                                                                             posting order
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # only pydantic; HTTP uses the standard library
cp .env.example .env                        # add GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)

python run.py models                        # confirm the Gemini model ids your key can use
python run.py all --provider gemini         # full run on the input pack (≈45 min on the free tier, throttled)
open outputs/promos.md
```

No key? `python run.py all --provider mock` runs the whole pipeline offline with a heuristic stand-in judge
(useful for checking the plumbing; its "judgements" are not real).

Input: by default the first `*.zip` in the repo root (the input pack), or `./pack/` if it contains `.txt` files.
Pass `--input path/to/folder_or.zip` for a new show.

## Swapping the model

| Provider | Flag | Needs |
|---|---|---|
| Google Gemini (default) | `--provider gemini` | `GEMINI_API_KEY` |
| Any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Together, vLLM, Ollama…) | `--provider openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| Offline mock | `--provider mock` | nothing |

Model ids: `--model` on any step, or `MODEL_STAGE_A` / `MODEL_STAGE_B` / `MODEL_LEAKAGE` in `.env`.
Defaults are `gemini-3.5-flash-lite` (Stage A, leakage) and `gemini-3.6-flash` (Stage B) — check them with
`python run.py models`; Google renames models often.

## Step by step

```bash
python run.py parse                      # outputs/beats.jsonl, episodes.json, coverage_report.json
python run.py candidates                 # outputs/candidates.jsonl, candidates_stats.json
python run.py stage-a --episodes 1-5     # development split only, while tuning prompts
python run.py stage-a                    # all candidates (cached calls are free on re-runs)
python run.py rank
python run.py stage-b --budget 18        # 18 fits the free Flash quota (~20/day); use 40+ on a paid key
python run.py rank
python run.py leakage --top 15           # re-ranks automatically
python run.py review-sheet               # outputs/review_sheet.csv (blind) + review_key.json
#   ...label the sheet: hook / cliffhanger / comprehension / editing = Pass | Uncertain | Fail
python run.py score-review               # outputs/evaluation.json + evaluation.md
python run.py export-web                 # web/data.json for the demo page
python -m unittest discover -s tests -v  # offline tests (parser, candidates, Gemini client vs a fake server)
```

## How it works

**1. Parse** (`src/parse.py`). Every `**LABEL:**` block becomes a record with a stable id (`E01_B034`),
original timestamps, type (camera / action / sound / dialogue), the speaker label exactly as written and the
delivery note. Speaker labels are *not* merged (`JULIANA` and `LADY JULIANA` stay distinct). Malformed
timestamps, overlapping dialogue and out-of-scene blocks are flagged, never fixed. Result on the pack: 50/50
episodes, 3,727 blocks, 0 invalid timestamps, 10 overlapping dialogue lines and 1 out-of-scene block flagged.

**2. Candidates** (`src/candidates.py`, `src/heuristics.py`). A promo can't be picked if it was never
generated, so two independent families of strategies propose windows:

- *Shot-aligned*: choose the strongest endings first (the script often marks them — e.g. a sound cue "swells to
  a cliffhanger peak"), then the best camera-cut opening 20–60 s earlier; plus one lexicon-free
  "coverage" window per episode chosen by a fixed rule (guards against blind spots in the keyword lists).
- *Event-anchored*: find a sharp line (accusation, threat, reveal) or a physical/visual event (slap, fall,
  entrance), then open the clip on it — even mid-shot — and pick the best ending.

Never cut inside a line of dialogue. Slots per episode: dialogue 3, action/visual 2, cliffhanger 2, coverage 1;
near-duplicates (IoU ≥ 0.7) are merged and the pool topped up to 7 distinct windows per episode.
Result: **333 candidates** (4–7 per episode, mean 32.9 s, 244 with clean camera cuts at both ends).
Settings are frozen as `cand-v1`.

**3. Stage A** (`src/judge_stage_a.py`, `prompts/stage_a_v1.md`). A cheap model judges every candidate seeing
**only the clip's own blocks** with clip-relative times — no loglines, summaries, continuity notes or later
scenes. It returns PASS / UNCERTAIN / FAIL with 0–10 scores and evidence block ids for hook, cliffhanger,
standalone comprehension, emotional impact, dead time and risks. Code then checks the model's claims: cited
ids must be in the clip, the hook evidence must be on screen in the first 3 s and the cliffhanger evidence
near the end. A failed check caps that verdict at UNCERTAIN.

**4. Stage B** (`src/judge_stage_b.py`, `prompts/stage_b_v1.md`). A stronger model does a skeptical editorial
pass on the top Stage A clips, the UNCERTAIN ones, and clips the heuristic liked but Stage A rejected
(possible misses). It must name the central character, what they want, what changes, the unresolved
question and the missing context, all with evidence ids; flag clips that reveal too much or peak too early;
and may propose a tighter cut (validated to stay inside the clip and within 20–60 s).

**5. Rank** (`src/rank.py`). Fixed order: validate timestamps and duration → reject invalid → hard gates
(hook, cliffhanger, standalone must all PASS for the shortlist; any FAIL rejects; otherwise "needs human
review") → weighted score (hook 25, cliffhanger 25, standalone 20, emotion 15, pacing 10, clean cut 5) →
remove clips sharing > 50 % of their duration with a better clip from the same episode → quality ranking.
The **posting order** is separate: no more than 2 per episode in the first 10, never the same episode twice
in a row, and a small bonus for episodes 1–10 because a viewer who clicks starts near the beginning.

**6. Payoff-leakage check** (`src/leakage.py`, `prompts/leakage_v1.md`). Only here does a model see the ~20 s
after the clip (or the next episode's opening when the clip ends the episode). It flags clips that answer
their own question or would be stronger cut earlier: DROP removes the clip from the posting order, TRIM
records the suggested end. Kept separate so the judges never use future information.

## Evaluation

- **Split**: episodes 1–5 for prompt development, 6–50 for evaluation. Freeze prompts and `cand-v1` before
  looking at evaluation results.
- **Same candidate pool** for all three methods — Baseline (heuristic score), Stage A, Stage A + B — so
  differences come from judging, not from different candidates.
- **Blind review**: `review-sheet` pools each method's top 10 (evaluation episodes only) plus a deterministic
  sample of 4 clips ranked 11–40 per method, shuffles them and hides which method picked what
  (`review_key.json` holds the mapping). A clip picked by several methods is reviewed once.
- **Metric**: precision@10 = share of top 10 labelled Pass (postable as-is); also reported with Uncertain
  counted as usable. The lower-ranked sample hints at misses; it is not a recall estimate.
- `score-review` also lists the top-10 picks a reviewer failed — the starting point for the three
  documented failure cases (`docs/failure_cases.json`, shown on the demo page).

## Reliability and cost

- Every model reply is validated with Pydantic; one automatic repair attempt with the validation error.
- Responses are cached under `cache/` keyed by provider + model + prompt version + full prompt, so an
  interrupted run resumes and re-runs cost nothing.
- Exponential backoff with jitter on 429/5xx; server retry hints (`retryDelay`, `Retry-After`) are honoured;
  per-stage throttle (defaults 9 / 4 / 9 requests per minute for Gemini, just under free-tier limits).
- Failed calls are logged, never silently dropped (`outputs/run_log.jsonl`, summary in `run_log.json`:
  calls, cache hits, retries, input/output/thinking tokens, latency, errors).
- Temperature 0 where supported — this reduces but does not guarantee run-to-run variation.

Rough budget for the 50-episode pack: Stage A 333 calls × ~1.4k input tokens ≈ 0.45 M input tokens;
Stage B 18–40 calls; leakage 15 calls. On the Gemini free tier (late 2026: Flash-Lite ~10 RPM / 500 per
day, Flash ~5 RPM / 20 per day — check your AI Studio quota page) this fits in one day; Stage A takes ~40 min
at 9 RPM. On paid Flash-Lite pricing the whole run is well under US$1.

## Outputs

| File | What |
|---|---|
| `outputs/promos.json` / `promos.md` | ranked promos with all fields, posting order, per-method rankings |
| `outputs/coverage_report.json` | parse coverage and validation flags per episode |
| `outputs/candidates.jsonl`, `candidates_stats.json` | the frozen candidate pool |
| `outputs/stage_a.jsonl`, `stage_b.jsonl`, `leakage.json` | raw model judgements + code checks |
| `outputs/run_log.json` (+ `.jsonl`) | calls, tokens, latency, retries, errors |
| `outputs/review_sheet.csv`, `review_key.json` | blind human review sheet and its key |
| `outputs/evaluation.json` / `.md` | precision@10 per method |

The committed outputs are from the full Gemini run (Stage A: 333 calls, Stage B: 18, leakage: 15, 0 failures).
The write-up is in `docs/SOLUTION_DOC.md`; real failure cases are in `docs/failure_cases.json`.

## Demo site (Vercel)

`web/` is a static page (no build step) reading `web/data.json`. Deploy: import the repo in Vercel and set
**Root Directory = `web`**, framework "Other". Run `python run.py export-web` before each deploy. It shows
only short excerpts — no full screenplays, extracted episode data, keys or raw model context.

## Assumptions and limits

- Single continuous clips only (no stitching across moments or episodes), 20–60 s.
- The screenplay stands in for the video: the tool cannot see faces, performances or pacing, so "first
  3 seconds" means what the script places there. A human should watch the shortlisted cuts.
- Speaker labels are kept as written; the judges are told viewers do not see names unless spoken.
- The 3 s hook and 5 s cliffhanger windows are practical engineering rules for this assignment, not universal
  definitions of a good promo.
- The heuristic keyword lists were written by looking at this one show; another genre would need new lists
  (the LLM stages do not depend on them, but candidate generation partly does).
