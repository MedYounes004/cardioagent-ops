"""End-to-end test: client -> api-gateway -> audio-engine -> agent-orchestrator.

Run against a running stack:  docker compose up --build -d api-gateway
Then:                         pytest integration_tests
"""

import io
import math
import os
import struct
import time
import wave

import httpx
import pytest

BASE_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")


def make_wav(seconds: float = 2.0, sample_rate: int = 2000) -> bytes:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 60 * n / sample_rate)))
        for n in range(int(seconds * sample_rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


@pytest.fixture(scope="session", autouse=True)
def wait_for_stack():
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/health/ready", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    pytest.fail("The stack did not become ready within 120 seconds")


def test_full_pipeline_returns_validated_report():
    response = httpx.post(
        f"{BASE_URL}/analyze_pcg",
        files={"file": ("recording.wav", make_wav(), "audio/wav")},
        data={"patient_id": "it-patient"},
        headers={"x-request-id": "it-1"},
        timeout=60,
    )
    assert response.status_code == 200
    report = response.json()
    assert report["request_id"] == "it-1"
    assert report["patient_id"] == "it-patient"
    assert set(report["probabilities"]) == {"normal", "pathological"}
    assert report["risk_level"] in {"low", "moderate", "high", "indeterminate"}
    assert report["audit"]["status"] in {"approved", "corrected"}
    assert report["recommendations"]
    assert report["disclaimer"]
    assert response.headers["x-request-id"] == "it-1"


def test_gateway_rejects_non_wav_before_reaching_services():
    response = httpx.post(
        f"{BASE_URL}/analyze_pcg",
        files={"file": ("note.mp3", b"data", "audio/mpeg")},
        timeout=10,
    )
    assert response.status_code == 415
