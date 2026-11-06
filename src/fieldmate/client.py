"""Thin, dependency-light Ollama HTTP client.

We talk to Ollama's REST API directly with httpx (instead of the `ollama` SDK) so that:
  * every timing field Ollama returns is surfaced for benchmarking, and
  * the client is trivial to mock in tests (no network needed).
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings, get_settings


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error payload."""


@dataclass
class GenerationMetrics:
    """Timing/throughput numbers for one generation (Ollama reports durations in ns)."""

    model: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0
    total_s: float = 0.0  # server-side total_duration
    wall_s: float = 0.0  # client-side wall clock
    ttft_s: float | None = None  # client-side, only measured when streaming

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s > 0 else 0.0

    @property
    def prompt_tokens_per_s(self) -> float:
        return self.prompt_tokens / self.prompt_eval_s if self.prompt_eval_s > 0 else 0.0

    @classmethod
    def from_response(cls, data: dict[str, Any], wall_s: float, ttft_s: float | None = None):
        ns = 1e9
        return cls(
            model=data.get("model", ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            load_s=(data.get("load_duration", 0) or 0) / ns,
            prompt_eval_s=(data.get("prompt_eval_duration", 0) or 0) / ns,
            eval_s=(data.get("eval_duration", 0) or 0) / ns,
            total_s=(data.get("total_duration", 0) or 0) / ns,
            wall_s=wall_s,
            ttft_s=ttft_s,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "load_s": round(self.load_s, 4),
            "prompt_eval_s": round(self.prompt_eval_s, 4),
            "eval_s": round(self.eval_s, 4),
            "total_s": round(self.total_s, 4),
            "wall_s": round(self.wall_s, 4),
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 4),
            "tokens_per_s": round(self.tokens_per_s, 3),
            "prompt_tokens_per_s": round(self.prompt_tokens_per_s, 3),
        }


@dataclass
class ChatResult:
    content: str
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)


class OllamaClient:
    """Minimal client for /api/chat, /api/tags, /api/version and /api/embed."""

    def __init__(
        self,
        settings: Settings | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http = http or httpx.Client(
            base_url=self.settings.ollama_host,
            timeout=httpx.Timeout(self.settings.request_timeout_s, connect=5.0),
            # Offline-first: never route local inference traffic through a corporate proxy.
            trust_env=False,
        )

    # ------------------------------------------------------------------ helpers
    def _options(self, temperature: float | None, options: dict | None) -> dict:
        opts = {
            "temperature": self.settings.temperature if temperature is None else temperature,
            "num_ctx": self.settings.num_ctx,
        }
        if options:
            opts.update(options)
        return opts

    def _payload(self, messages, model, fmt, temperature, options, stream, keep_alive):
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": messages,
            "stream": stream,
            "options": self._options(temperature, options),
        }
        if fmt is not None:
            payload["format"] = fmt
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return payload

    # --------------------------------------------------------------- endpoints
    def version(self) -> str:
        try:
            r = self._http.get("/api/version")
            r.raise_for_status()
            return r.json().get("version", "unknown")
        except httpx.HTTPError as exc:  # pragma: no cover - exercised via health tests
            raise OllamaError(
                f"Ollama not reachable at {self.settings.ollama_host}: {exc}"
            ) from exc

    def list_models(self) -> list[str]:
        try:
            r = self._http.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(str(exc)) from exc
        return [m["name"] for m in r.json().get("models", [])]

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        fmt: dict | str | None = None,
        temperature: float | None = None,
        options: dict | None = None,
        keep_alive: str | int | None = None,
    ) -> ChatResult:
        """Blocking chat call. `fmt` may be "json" or a JSON-schema dict (constrained decoding)."""
        payload = self._payload(messages, model, fmt, temperature, options, False, keep_alive)
        t0 = time.perf_counter()
        try:
            r = self._http.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"request failed: {exc}") from exc
        wall = time.perf_counter() - t0
        data = r.json() if r.content else {}
        if r.status_code >= 400 or "error" in data:
            raise OllamaError(data.get("error", f"HTTP {r.status_code}"))
        content = data.get("message", {}).get("content", "")
        return ChatResult(content=content, metrics=GenerationMetrics.from_response(data, wall))

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        fmt: dict | str | None = None,
        temperature: float | None = None,
        options: dict | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[str | GenerationMetrics]:
        """Yield text deltas as they arrive, then a final GenerationMetrics object.

        Time-to-first-token (TTFT) is measured client-side: from request start to the first
        non-empty content chunk. It therefore includes model load + prompt processing.
        """
        payload = self._payload(messages, model, fmt, temperature, options, True, keep_alive)
        t0 = time.perf_counter()
        ttft: float | None = None
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as r:
                if r.status_code >= 400:
                    r.read()
                    raise OllamaError(r.text or f"HTTP {r.status_code}")
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise OllamaError(chunk["error"])
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        yield delta
                    if chunk.get("done"):
                        wall = time.perf_counter() - t0
                        yield GenerationMetrics.from_response(chunk, wall, ttft)
                        return
        except httpx.HTTPError as exc:
            raise OllamaError(f"stream failed: {exc}") from exc

    def embed(self, texts: list[str], model: str = "all-minilm") -> list[list[float]]:
        r = self._http.post("/api/embed", json={"model": model, "input": texts})
        data = r.json()
        if r.status_code >= 400 or "error" in data:
            raise OllamaError(data.get("error", f"HTTP {r.status_code}"))
        return data["embeddings"]

    def unload(self, model: str) -> None:
        """Evict a model from memory (keep_alive=0) so the next call measures a cold load."""
        with contextlib.suppress(httpx.HTTPError):
            self._http.post("/api/generate", json={"model": model, "keep_alive": 0})

    def close(self) -> None:
        self._http.close()
