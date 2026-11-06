import json

import pytest
from fastapi.testclient import TestClient

from fieldmate.api import create_app, get_client
from fieldmate.client import GenerationMetrics, OllamaError

from .test_extractor import RULE_BREAKING_DEFECT, VALID_DEFECT


@pytest.fixture
def api(make_client):
    app = create_app()
    client = make_client()
    app.dependency_overrides[get_client] = lambda: client
    return TestClient(app)


def test_client_metrics(fake, make_client):
    fake.replies = ["hello there"]
    res = make_client().chat([{"role": "user", "content": "hi"}])
    assert res.content == "hello there"
    assert res.metrics.tokens_per_s == pytest.approx(10.0)  # 20 tokens / 2 s
    assert res.metrics.prompt_tokens_per_s == pytest.approx(100.0)


def test_stream_yields_deltas_then_metrics_with_ttft(fake, make_client):
    fake.replies = ["three word answer"]
    items = list(make_client().chat_stream([{"role": "user", "content": "q"}]))
    assert "".join(i for i in items if isinstance(i, str)) == "three word answer"
    assert isinstance(items[-1], GenerationMetrics)
    assert items[-1].ttft_s is not None and items[-1].ttft_s >= 0


def test_client_raises_on_server_error(fake, make_client):
    fake.fail_with = 500
    with pytest.raises(OllamaError):
        make_client().chat([{"role": "user", "content": "hi"}])


def test_options_include_temperature_and_ctx(fake, make_client):
    make_client(num_ctx=1024).chat([{"role": "user", "content": "x"}], temperature=0.7)
    opts = fake.requests[0]["options"]
    assert opts == {"temperature": 0.7, "num_ctx": 1024}


def test_health_ok(api):
    r = api.get("/health")
    assert r.status_code == 200
    assert r.json()["default_model_available"] is True


def test_health_503_when_ollama_down(api, fake):
    fake.fail_with = 500
    assert api.get("/health").status_code == 503


def test_chat_endpoint_with_context(api, fake):
    fake.replies = ["Q4-ALPHA and MAPPER-5"]
    r = api.post("/chat", json={"message": "Which are grounded?", "context": "log..."})
    assert r.status_code == 200
    assert r.json()["answer"] == "Q4-ALPHA and MAPPER-5"
    assert "FLEET LOG" in fake.requests[0]["messages"][-1]["content"]


def test_chat_stream_endpoint_ndjson(api, fake):
    fake.replies = ["a b c"]
    r = api.post("/chat/stream", json={"message": "hi"})
    lines = [json.loads(x) for x in r.text.strip().splitlines()]
    assert "".join(x.get("delta", "") for x in lines) == "a b c"
    assert lines[-1]["done"] is True and "tokens_per_s" in lines[-1]["metrics"]


def test_extract_endpoint_retry_and_graceful_failure(api, fake):
    fake.replies = [RULE_BREAKING_DEFECT, VALID_DEFECT]
    r = api.post("/extract", json={"kind": "defect", "text": "prop chipped"})
    assert r.status_code == 200 and r.json()["ok"] is True and len(r.json()["attempts"]) == 2

    fake.replies = ["{bad", "{bad"]
    r = api.post("/extract", json={"kind": "defect", "text": "prop chipped"})
    assert r.status_code == 200 and r.json()["ok"] is False


def test_extract_endpoint_rejects_unknown_kind(api):
    assert api.post("/extract", json={"kind": "weather", "text": "x"}).status_code == 422


def test_extract_endpoint_502_when_ollama_down(api, fake):
    fake.fail_with = 500
    assert api.post("/extract", json={"kind": "defect", "text": "x"}).status_code == 502
