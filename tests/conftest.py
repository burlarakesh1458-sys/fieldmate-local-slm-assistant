"""Offline test fixtures: a fake Ollama server built on httpx.MockTransport."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from fieldmate.client import OllamaClient
from fieldmate.config import Settings

NS = 1_000_000_000


def done_payload(content: str, model: str = "test-model", eval_count: int = 20) -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 50,
        "prompt_eval_duration": int(0.5 * NS),
        "eval_count": eval_count,
        "eval_duration": int(2 * NS),
        "load_duration": int(0.1 * NS),
        "total_duration": int(2.7 * NS),
    }


class FakeOllama:
    """Scriptable fake. `replies` is consumed in order by /api/chat calls."""

    def __init__(self, replies: list[str] | None = None, models: list[str] | None = None):
        self.replies = list(replies or [])
        self.models = models or ["test-model"]
        self.requests: list[dict] = []
        self.fail_with: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if self.fail_with:
            return httpx.Response(self.fail_with, json={"error": "boom"})
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.0-test"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
        if path == "/api/embed":
            body = json.loads(request.content)
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * len(body["input"])})
        if path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if path == "/api/chat":
            body = json.loads(request.content)
            self.requests.append(body)
            content = self.replies.pop(0) if self.replies else "ok"
            if body.get("stream"):
                words = content.split(" ")
                lines = [
                    json.dumps({"message": {"content": (" " if i else "") + w}, "done": False})
                    for i, w in enumerate(words)
                ]
                final = done_payload("", body["model"])
                final["message"]["content"] = ""
                lines.append(json.dumps(final))
                return httpx.Response(200, content="\n".join(lines).encode())
            return httpx.Response(200, json=done_payload(content, body["model"]))
        return httpx.Response(404, json={"error": f"no route {path}"})


@pytest.fixture
def fake() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def make_client(fake: FakeOllama) -> Callable[..., OllamaClient]:
    def _make(**overrides) -> OllamaClient:
        settings = Settings(ollama_host="http://fake", model="test-model", **overrides)
        http = httpx.Client(base_url="http://fake", transport=httpx.MockTransport(fake.handler))
        return OllamaClient(settings=settings, http=http)

    return _make
