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


def test_rejects_non_wav(client):
    response = client.post("/analyze_pcg", files={"file": ("note.mp3", b"data", "audio/mpeg")})
    assert response.status_code == 415


def test_analyze_pcg_forwards_audio_result(client):
    audio_response = httpx.Response(200, json={"features": {"heart_rate": 72}})
    orchestrator_response = httpx.Response(200, json={"prediction": "normal"})
    client_app = app.state.http_client
    client_app.post = AsyncMock(side_effect=[audio_response, orchestrator_response])

    response = client.post(
        "/analyze_pcg",
        files={"file": ("recording.wav", wav_bytes(), "audio/wav")},
        data={"patient_id": "patient-1"},
        headers={"x-request-id": "request-1"},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": "normal"}
    assert response.headers["x-request-id"] == "request-1"
    assert client_app.post.await_count == 2