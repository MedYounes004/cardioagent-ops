import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.clients import NullRetriever, OllamaClient
from app.graph import build_graph
from app.main import app

BODY = {
    "audio": {
        "label": "abnormal",
        "probabilities": {"normal": 0.2, "pathological": 0.8},
        "features": {"rms": 0.1, "peak": 0.7, "zero_crossing_rate": 0.04},
        "model_loaded": True,
    },
    "patient_id": "patient-1",
    "metadata": "routine check",
    "request_id": "req-42",
}


class FixedLLM:
    async def generate(self, prompt):
        return "The pattern may be consistent with an abnormality; clinical review is advised."


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "agent-orchestrator"
    assert body["llm_configured"] is False


def test_analyze_fallback_returns_valid_report(client):
    response = client.post("/analyze", json=BODY)
    assert response.status_code == 200
    report = response.json()
    assert report["request_id"] == "req-42"
    assert report["patient_id"] == "patient-1"
    assert report["risk_level"] == "high"
    assert report["llm_used"] is False
    assert report["audit"]["status"] == "approved"
    assert response.headers["x-request-id"] == "req-42"


def test_analyze_uses_injected_llm(client):
    app.state.graph = build_graph(FixedLLM(), NullRetriever())
    response = client.post("/analyze", json=BODY)
    assert response.status_code == 200
    assert response.json()["llm_used"] is True


def test_request_id_taken_from_header_then_generated(client):
    body = {key: value for key, value in BODY.items() if key != "request_id"}
    response = client.post("/analyze", json=body, headers={"x-request-id": "from-header"})
    assert response.json()["request_id"] == "from-header"
    assert response.headers["x-request-id"] == "from-header"

    response = client.post("/analyze", json=body)
    generated = response.json()["request_id"]
    assert generated and response.headers["x-request-id"] == generated


def test_invalid_body_returns_422(client):
    assert client.post("/analyze", json={"patient_id": "x"}).status_code == 422
    bad = {"audio": {"probabilities": {"normal": 0.0, "pathological": 3.0}}}
    assert client.post("/analyze", json=bad).status_code == 422


def test_pipeline_failure_returns_500(client):
    class Boom:
        async def ainvoke(self, state):
            raise RuntimeError("boom")

    app.state.graph = Boom()
    response = client.post("/analyze", json=BODY)
    assert response.status_code == 500
    assert response.json()["detail"] == "The analysis pipeline failed"


def ollama(handler, url="http://llm:11434"):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaClient(url, "cardio-qwen", http)


def test_ollama_returns_text():
    client = ollama(lambda request: httpx.Response(200, json={"response": "  hello  "}))
    assert asyncio.run(client.generate("hi")) == "hello"


def test_ollama_unconfigured_returns_none():
    client = ollama(lambda request: httpx.Response(200, json={"response": "x"}), url="")
    assert client.configured is False
    assert asyncio.run(client.generate("hi")) is None


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(500, json={}),
        lambda request: httpx.Response(200, content=b"not json"),
        lambda request: httpx.Response(200, json=["unexpected"]),
        lambda request: httpx.Response(200, json={"response": "   "}),
    ],
)
def test_ollama_bad_responses_return_none(handler):
    assert asyncio.run(ollama(handler).generate("hi")) is None


def test_ollama_network_error_returns_none():
    def handler(request):
        raise httpx.ConnectError("down")

    assert asyncio.run(ollama(handler).generate("hi")) is None
