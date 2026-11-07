"""Model-agnostic LLM layer.

Providers (choose with --provider or LLM_PROVIDER):
  gemini  - Google Gemini REST API (generateContent). Needs GEMINI_API_KEY.
  openai  - any OpenAI-compatible /chat/completions endpoint (OpenAI, Groq, OpenRouter, Together,
            vLLM, Ollama, Gemini's OpenAI endpoint...). Needs OPENAI_API_KEY and OPENAI_BASE_URL.
  mock    - offline, deterministic, heuristic-driven. For tests and dry runs only - NOT a judge.

Only the standard library is used for HTTP, so swapping models needs no new dependencies.

Reliability: response cache (key = provider + model + prompt version + full prompt), Pydantic
validation with one repair attempt, exponential backoff with jitter, server retry hints honoured,
a per-stage RPM throttle, and a JSONL call log (tokens, latency, retries, errors).
"""
from __future__ import annotations

import copy
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Type

from pydantic import BaseModel, ValidationError

from .utils import CACHE_DIR, OUTPUTS, env, read_jsonl, sha, write_json

MAX_ATTEMPTS = 6
HTTP_TIMEOUT = 120


class FatalLLMError(RuntimeError):
    """Configuration problems that retrying cannot fix (bad key, unknown model...)."""


class RetryableLLMError(RuntimeError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    model_version: str = ""


# ------------------------------------------------------------------ JSON schema helpers
def inline_schema(model: Type[BaseModel]) -> dict:
    """Pydantic schema with $refs inlined and 'title' noise removed (portable across providers)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                return walk(copy.deepcopy(defs[ref]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def extract_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _parse_retry_hint(body: str, headers) -> float | None:
    if headers is not None:
        ra = headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body or "")
    return float(m.group(1)) if m else None


def _post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code in (408, 429, 500, 502, 503, 504):
            raise RetryableLLMError(f"HTTP {e.code}: {body[:300]}", _parse_retry_hint(body, e.headers))
        if e.code in (401, 403):
            raise FatalLLMError(f"HTTP {e.code} (check your API key): {body[:300]}")
        if e.code == 404:
            raise FatalLLMError(f"HTTP 404 (unknown model id? run `python run.py models`): {body[:300]}")
        raise RuntimeError(f"HTTP {e.code}: {body[:500]}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise RetryableLLMError(f"network error: {e}")


# ------------------------------------------------------------------ providers
class LLMClient:
    provider = "base"

    def __init__(self, model: str):
        self.model = model

    def generate(self, system: str, prompt: str, schema: dict | None,
                 mock: Callable[[], dict] | None = None) -> LLMResult:
        raise NotImplementedError


class GeminiClient(LLMClient):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str):
        super().__init__(model)
        self.key = env("GEMINI_API_KEY") or env("GOOGLE_API_KEY")
        if not self.key:
            raise FatalLLMError("GEMINI_API_KEY is not set (see .env.example)")
        self.use_schema = True

    def generate(self, system, prompt, schema, mock=None):
        gen = {"temperature": 0, "responseMimeType": "application/json"}
        if schema is not None and self.use_schema:
            gen["responseJsonSchema"] = schema
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
        }
        try:
            data = _post(f"{self.BASE}/models/{self.model}:generateContent", payload, {"x-goog-api-key": self.key})
        except RuntimeError as e:
            # Older/smaller models may reject a JSON schema; fall back to prompt-only JSON once.
            if self.use_schema and "HTTP 400" in str(e) and "schema" in str(e).lower():
                self.use_schema = False
                return self.generate(system, prompt, schema, mock)
            raise
        cands = data.get("candidates") or []
        if not cands:
            raise RetryableLLMError(f"no candidates returned: {json.dumps(data)[:300]}")
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata", {})
        return LLMResult(text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0),
                         u.get("thoughtsTokenCount", 0), data.get("modelVersion", self.model))

    def list_models(self) -> list[str]:
        req = urllib.request.Request(f"{self.BASE}/models?pageSize=200", headers={"x-goog-api-key": self.key})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read().decode())
        return [m["name"].split("/")[-1] for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]


class OpenAICompatClient(LLMClient):
    provider = "openai"

    def __init__(self, model: str):
        super().__init__(model)
        self.key = env("OPENAI_API_KEY")
        self.base = (env("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not self.key and "localhost" not in self.base and "127.0.0.1" not in self.base:
            raise FatalLLMError("OPENAI_API_KEY is not set (see .env.example)")

    def generate(self, system, prompt, schema, mock=None):
        sys_msg = system
        if schema is not None:
            sys_msg += "\n\nReturn ONLY a JSON object matching this JSON Schema:\n" + json.dumps(schema)
        payload = {
            "model": self.model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}],
        }
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        data = _post(f"{self.base}/chat/completions", payload, headers)
        u = data.get("usage", {}) or {}
        return LLMResult(data["choices"][0]["message"]["content"] or "", u.get("prompt_tokens", 0),
                         u.get("completion_tokens", 0), 0, data.get("model", self.model))


class MockClient(LLMClient):
    """Deterministic offline stand-in. Each judge supplies a heuristic `mock` payload."""
    provider = "mock"

    def generate(self, system, prompt, schema, mock=None):
        if mock is None:
            raise FatalLLMError("mock provider needs a mock payload")
        return LLMResult(json.dumps(mock()), len(prompt) // 4, 200, 0, f"mock-{self.model}")


def make_client(provider: str, model: str) -> LLMClient:
    provider = provider.lower()
    if provider == "gemini":
        return GeminiClient(model)
    if provider in ("openai", "openai-compatible", "oai"):
        return OpenAICompatClient(model)
    if provider == "mock":
        return MockClient(model)
    raise FatalLLMError(f"unknown provider {provider!r} (gemini | openai | mock)")


# ------------------------------------------------------------------ rate limiting / logging
class Throttle:
    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.last = 0.0

    def wait(self):
        if self.interval:
            dt = time.monotonic() - self.last
            if dt < self.interval:
                time.sleep(self.interval - dt)
        self.last = time.monotonic()


class CallLog:
    """Append-only JSONL so an interrupted run keeps its accounting."""

    def __init__(self, path: Path = OUTPUTS / "run_log.jsonl"):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, rec: dict):
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **rec}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize_log(path: Path = OUTPUTS / "run_log.jsonl", out: Path = OUTPUTS / "run_log.json") -> dict:
    rows = read_jsonl(path) if path.exists() else []
    by: dict[str, dict] = {}
    for r in rows:
        k = f'{r.get("stage")}|{r.get("provider")}|{r.get("model")}'
        s = by.setdefault(k, {"stage": r.get("stage"), "provider": r.get("provider"), "model": r.get("model"),
                              "calls": 0, "api_calls": 0, "cache_hits": 0, "failures": 0, "retries": 0,
                              "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "latency_s": 0.0,
                              "errors": []})
        s["calls"] += 1
        s["cache_hits"] += int(bool(r.get("cached")))
        s["api_calls"] += r.get("api_calls", 0)
        s["failures"] += int(not r.get("ok"))
        s["retries"] += r.get("retries", 0)
        for t in ("input_tokens", "output_tokens", "thinking_tokens"):
            s[t] += r.get(t, 0) or 0
        s["latency_s"] = round(s["latency_s"] + (r.get("latency_s") or 0), 2)
        if r.get("error") and len(s["errors"]) < 20:
            s["errors"].append({"id": r.get("item_id"), "error": r["error"][:300]})
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "stages": list(by.values()), "raw_log": path.name, "total_records": len(rows)}
    write_json(out, summary)
    return summary


# ------------------------------------------------------------------ the one entry point judges use
@dataclass
class JsonCaller:
    client: LLMClient
    stage: str
    prompt_version: str
    rpm: Optional[float] = None
    log: CallLog = field(default_factory=CallLog)
    use_cache: bool = True

    def __post_init__(self):
        self.throttle = Throttle(self.rpm)

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / self.stage / key[:2] / f"{key}.json"

    def call(self, item_id: str, system: str, prompt: str, schema_model: Type[BaseModel],
             mock: Callable[[], dict] | None = None) -> Optional[BaseModel]:
        key = sha(self.client.provider, self.client.model, self.prompt_version, system, prompt)
        cpath = self._cache_path(key)
        base = {"stage": self.stage, "item_id": item_id, "provider": self.client.provider,
                "model": self.client.model, "prompt_version": self.prompt_version, "cache_key": key}
        if self.use_cache and cpath.exists():
            cached = json.loads(cpath.read_text())
            try:
                obj = schema_model.model_validate(cached["response"])
                self.log.add({**base, "ok": True, "cached": True, "api_calls": 0})
                return obj
            except ValidationError:
                pass  # schema changed - recompute

        schema = inline_schema(schema_model)
        attempts, api_calls, retries = 0, 0, 0
        tokens = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
        user_prompt, last_err, model_version = prompt, None, ""
        t0 = time.monotonic()
        repaired = False
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            self.throttle.wait()
            try:
                api_calls += 1
                res = self.client.generate(system, user_prompt, schema, mock)
                tokens["input_tokens"] += res.input_tokens
                tokens["output_tokens"] += res.output_tokens
                tokens["thinking_tokens"] += res.thinking_tokens
                model_version = res.model_version
                raw = extract_json(res.text)
                obj = schema_model.model_validate(raw)
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps({"meta": {**base, "model_version": model_version},
                                             "response": obj.model_dump()}, ensure_ascii=False, indent=1))
                self.log.add({**base, "ok": True, "cached": False, "api_calls": api_calls, "retries": retries,
                              "latency_s": round(time.monotonic() - t0, 2), "model_version": model_version,
                              **tokens})
                return obj
            except FatalLLMError:
                raise
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = f"invalid JSON/schema: {str(e)[:400]}"
                if repaired:
                    break
                repaired = True
                retries += 1
                user_prompt = (prompt + "\n\nYour previous reply was rejected by the validator:\n"
                               + str(e)[:800] + "\nReturn corrected JSON only.")
            except RetryableLLMError as e:
                last_err = str(e)
                retries += 1
                delay = e.retry_after if e.retry_after else min(64.0, 2.0 ** attempts)
                time.sleep(delay + random.uniform(0, 1.5))
            except RuntimeError as e:
                last_err = str(e)
                break
        self.log.add({**base, "ok": False, "cached": False, "api_calls": api_calls, "retries": retries,
                      "latency_s": round(time.monotonic() - t0, 2), "error": last_err, **tokens})
        return None
