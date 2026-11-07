"""Offline tests: python -m unittest discover -s tests -v

They run the real parser/candidate generator on the input pack, and exercise the Gemini REST client
against a local fake server (429 + retry hint, bad JSON repair, caching) - no API key needed.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import candidates, judge_stage_a, llm, parse  # noqa: E402
from src.llm import inline_schema  # noqa: E402
from src.utils import Candidate, load_blocks, load_candidates  # noqa: E402

PACK = next(iter(sorted(ROOT.glob("*.zip"))), None) or ROOT / "pack"


class PipelineOnPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.report = parse.run(PACK, cls.tmp)
        cls.stats = candidates.run(cls.tmp)

    def test_parse_all_episodes(self):
        self.assertEqual(self.report["episodes_parsed"], 50)
        self.assertEqual(self.report["episodes_failed"], [])
        self.assertEqual(self.report["blocks_with_invalid_timestamps"], 0)

    def test_block_ids_stable_and_unique(self):
        ids = [b.block_id for bl in load_blocks(self.tmp / "beats.jsonl").values() for b in bl]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "E01_B001")

    def test_original_speaker_labels_preserved(self):
        speakers = {b.speaker for bl in load_blocks(self.tmp / "beats.jsonl").values() for b in bl if b.speaker}
        # the script uses both labels for the protagonist - we must not merge them
        self.assertIn("JULIANA", speakers)
        self.assertIn("LADY JULIANA", speakers)

    def test_candidates_are_legal_windows(self):
        by_ep = load_blocks(self.tmp / "beats.jsonl")
        cands = load_candidates(self.tmp / "candidates.jsonl")
        self.assertTrue(250 <= len(cands) <= 400, len(cands))
        for c in cands:
            self.assertGreaterEqual(c.duration, 19.95)
            self.assertLessEqual(c.duration, 60.05)
            for d in by_ep[c.episode]:
                if d.type != "dialogue":
                    continue
                for t in (c.start, c.end):
                    self.assertFalse(d.start + 0.1 < t < d.end - 0.1, f"{c.candidate_id} cuts through {d.block_id}")

    def test_candidates_deterministic(self):
        first = (self.tmp / "candidates.jsonl").read_text()
        candidates.run(self.tmp)
        self.assertEqual(first, (self.tmp / "candidates.jsonl").read_text())


class FakeGemini(BaseHTTPRequestHandler):
    script: list = []
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeGemini.seen.append(body)
        status, payload = FakeGemini.script.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *a):
        pass


def _ok(obj):
    return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}],
                 "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50}, "modelVersion": "fake-1"}


class GeminiClientOffline(unittest.TestCase):
    def setUp(self):
        self.srv = HTTPServer(("127.0.0.1", 0), FakeGemini)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        llm.GeminiClient.BASE = f"http://127.0.0.1:{self.srv.server_port}/v1beta"
        import os
        os.environ["GEMINI_API_KEY"] = "test"
        [os.environ.pop(k, None) for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")]
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        self.tmp = Path(tempfile.mkdtemp())
        llm.CACHE_DIR = self.tmp / "cache"
        FakeGemini.seen.clear()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_retry_hint_repair_and_cache(self):
        c = Candidate(candidate_id="E01_0000_0300", episode=1, start=0, end=30, duration=30,
                      block_ids=["E01_B001"], methods=["x"],
                      heuristic={"hook": 8, "cliff": 8, "standalone": 8, "emotion": 6, "pacing": 7,
                                 "clean_cut": 10, "hook_block": "E01_B001", "cliff_block": "E01_B001"})
        good = judge_stage_a.mock_payload(c)
        FakeGemini.script = [
            (429, {"error": {"details": [{"retryDelay": "0s"}]}}),     # rate limited, server says retry now
            _ok({"hook": "not the right shape"}),                       # schema violation -> repair prompt
            _ok(good),
        ]
        caller = llm.JsonCaller(llm.make_client("gemini", "fake-model"), "test", "v1",
                                log=llm.CallLog(self.tmp / "log.jsonl"))
        r = caller.call("x", "sys", "prompt", judge_stage_a.StageAResult)
        self.assertIsNotNone(r)
        self.assertEqual(len(FakeGemini.seen), 3)
        self.assertIn("rejected by the validator", FakeGemini.seen[2]["contents"][0]["parts"][0]["text"])
        self.assertEqual(FakeGemini.seen[0]["generationConfig"]["temperature"], 0)
        # second identical call is served from cache: no HTTP
        r2 = caller.call("x", "sys", "prompt", judge_stage_a.StageAResult)
        self.assertEqual(r2, r)
        self.assertEqual(len(FakeGemini.seen), 3)
        log = [json.loads(x) for x in (self.tmp / "log.jsonl").read_text().splitlines()]
        self.assertEqual([x["cached"] for x in log], [False, True])
        self.assertEqual(log[0]["retries"], 2)
        self.assertEqual(log[0]["input_tokens"], 200)

    def test_bad_key_is_fatal(self):
        FakeGemini.script = [(403, {"error": "denied"})]
        caller = llm.JsonCaller(llm.make_client("gemini", "m"), "test", "v1", log=llm.CallLog(self.tmp / "l.jsonl"))
        with self.assertRaises(llm.FatalLLMError):
            caller.call("x", "sys", "p", judge_stage_a.StageAResult)


class JudgeChecks(unittest.TestCase):
    def test_schema_inlined(self):
        s = json.dumps(inline_schema(judge_stage_a.StageAResult))
        self.assertNotIn("$ref", s)

    def test_unsupported_pass_is_capped(self):
        from src.utils import Block
        b1 = Block(episode=1, block_id="E01_B001", scene=1, start=0, end=2, type="camera", label_raw="CAMERA", text="x")
        b2 = Block(episode=1, block_id="E01_B002", scene=1, start=10, end=12, type="dialogue", label_raw="A",
                   speaker="A", text="y")
        idx = {b.block_id: b for b in (b1, b2)}
        c = Candidate(candidate_id="c", episode=1, start=0, end=30, duration=30, block_ids=list(idx), methods=["x"],
                      heuristic={"clean_cut": 10})
        crit = {"verdict": "PASS", "score": 9, "explanation": "e"}
        r = judge_stage_a.StageAResult(
            hook={**crit, "evidence_block_ids": ["E01_B002"]},            # at +10 s: not a 3-second hook
            cliffhanger={**crit, "evidence_block_ids": ["E99_B999"]},     # hallucinated id
            standalone={**crit}, emotional_impact={**crit},
            dead_time={"score": 8, "explanation": "e"}, one_line_why="w")
        out = judge_stage_a.postprocess(c, r, idx)
        self.assertEqual(out["gates"]["hook"], "UNCERTAIN")
        self.assertEqual(out["gates"]["cliffhanger"], "UNCERTAIN")
        self.assertIn("E99_B999", out["checks"]["invalid_evidence_ids"])


if __name__ == "__main__":
    unittest.main()
