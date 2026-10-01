import io
import wave

import pytest
from fastapi.testclient import TestClient

from app.main import app


def wav_bytes(channels: int = 1, sample_rate: int = 2000, sample_width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * 160 * channels)
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False  # no model file present in CI


def test_process_returns_expected_shape(client):
    response = client.post("/process", files={"file": ("recording.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert "probabilities" in body
    assert "pathological" in body["probabilities"]
    assert set(body["features"].keys()) == {"rms", "peak", "zero_crossing_rate"}


def test_process_rejects_non_wav_extension(client):
    response = client.post("/process", files={"file": ("note.mp3", b"data", "audio/mpeg")})
    assert response.status_code == 415


def test_process_rejects_invalid_wav_header(client):
    response = client.post("/process", files={"file": ("bad.wav", b"not a real wav file", "audio/wav")})
    assert response.status_code == 422


def test_process_rejects_8bit_wav(client):
    data = wav_bytes(sample_width=1)
    response = client.post("/process", files={"file": ("8bit.wav", data, "audio/wav")})
    assert response.status_code == 422


def test_process_handles_stereo(client):
    data = wav_bytes(channels=2)
    response = client.post("/process", files={"file": ("stereo.wav", data, "audio/wav")})
    assert response.status_code == 200


def test_extract_features_on_empty_samples():
    from app.inference import extract_features
    features = extract_features([], 2000)
    assert features == {"rms": 0.0, "peak": 0.0, "zero_crossing_rate": 0.0}