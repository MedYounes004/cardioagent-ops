import io
import wave
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


def wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_invalid_wav_returns_422(client):
    response = client.post("/analyze_pcg", files={"file": ("bad.wav", b"not a wav", "audio/wav")})
    assert response.status_code == 422


def test_oversized_file_returns_413(client, monkeypatch):
    monkeypatch.setattr("app.main.MAX_UPLOAD_BYTES", 10)
    response = client.post("/analyze_pcg", files={"file": ("r.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 413


def test_audio_engine_unreachable_returns_502(client, monkeypatch):
    monkeypatch.setattr(app.state.http_client, "post", AsyncMock(side_effect=httpx.ConnectError("down")))
    response = client.post("/analyze_pcg", files={"file": ("r.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 502


def test_audio_engine_error_skips_orchestrator(client, monkeypatch):
    post = AsyncMock(return_value=httpx.Response(500, json={}))
    monkeypatch.setattr(app.state.http_client, "post", post)
    response = client.post("/analyze_pcg", files={"file": ("r.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 502
    assert post.await_count == 1


def test_orchestrator_invalid_json_returns_502(client, monkeypatch):
    post = AsyncMock(side_effect=[httpx.Response(200, json={"features": {}}), httpx.Response(200, content=b"not json")])
    monkeypatch.setattr(app.state.http_client, "post", post)
    response = client.post("/analyze_pcg", files={"file": ("r.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 502


def test_liveness_and_generated_request_id(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.headers["x-request-id"]


def test_root_serves_analysis_console(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "CardioAgent / API gateway" in response.text


def test_readiness_ok(client, monkeypatch):
    monkeypatch.setattr(app.state.http_client, "get", AsyncMock(return_value=httpx.Response(200)))
    assert client.get("/health/ready").status_code == 200


def test_readiness_unavailable(client, monkeypatch):
    monkeypatch.setattr(app.state.http_client, "get", AsyncMock(side_effect=httpx.ConnectError("down")))
    assert client.get("/health/ready").status_code == 503