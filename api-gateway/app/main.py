"""Public HTTP gateway for phonocardiogram analysis."""

from __future__ import annotations

import io
import logging
import os
import uuid
import wave
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

LOGGER = logging.getLogger("api_gateway")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
AUDIO_ENGINE_URL = os.getenv("AUDIO_ENGINE_URL", "http://audio-engine:8001/process")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://agent-orchestrator:8002/analyze")
DOWNSTREAM_TIMEOUT_SECONDS = float(os.getenv("DOWNSTREAM_TIMEOUT_SECONDS", "60"))
HEALTH_TIMEOUT_SECONDS = float(os.getenv("HEALTH_TIMEOUT_SECONDS", "5"))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(DOWNSTREAM_TIMEOUT_SECONDS),
        follow_redirects=False,
    )
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="CardioAgent API Gateway",
    description="Single entry point for phonocardiogram analysis.",
    version=os.getenv("SERVICE_VERSION", "1.0.0"),
    lifespan=lifespan,
)

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=ALLOWED_ORIGINS != ["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def _is_wav(filename: str | None, content_type: str | None) -> bool:
    extension_ok = bool(filename and filename.lower().endswith(".wav"))
    content_type_ok = not content_type or content_type.lower() in {"audio/wav", "audio/x-wav", "application/octet-stream"}
    return extension_ok and content_type_ok


def _validate_wav(data: bytes) -> None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            if wav_file.getnchannels() < 1 or wav_file.getframerate() < 1:
                raise ValueError("invalid WAV stream properties")
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The uploaded file is not a valid WAV audio stream") from exc


def _downstream_error(service: str, exc: Exception, request_id: str) -> HTTPException:
    LOGGER.warning("%s request failed request_id=%s error=%s", service, request_id, exc)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=f"{service} is unavailable",
        headers={"x-request-id": request_id},
    )


async def _json_response(response: httpx.Response, service: str, request_id: str) -> dict[str, Any]:
    if response.is_error:
        LOGGER.warning("%s returned %s request_id=%s", service, response.status_code, request_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{service} rejected the analysis request",
            headers={"x-request-id": request_id},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{service} returned an invalid response",
            headers={"x-request-id": request_id},
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"{service} returned an invalid response")
    return payload


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = _request_id(request)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health/live", tags=["health"])
async def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "api-gateway"}


@app.get("/health/ready", tags=["health"])
async def readiness(request: Request) -> JSONResponse:
    client: httpx.AsyncClient = request.app.state.http_client
    checks: dict[str, str] = {}
    for service, url in (("audio-engine", AUDIO_ENGINE_URL), ("agent-orchestrator", ORCHESTRATOR_URL)):
        health_url = url.rsplit("/", 1)[0] + "/health"
        try:
            response = await client.get(health_url, timeout=HEALTH_TIMEOUT_SECONDS)
            checks[service] = "ok" if response.is_success else "unhealthy"
        except httpx.HTTPError:
            checks[service] = "unreachable"
    ready = all(value == "ok" for value in checks.values())
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ok" if ready else "unavailable", "dependencies": checks})


@app.post("/analyze_pcg", tags=["analysis"])
async def analyze_pcg(
    request: Request,
    file: UploadFile = File(..., description="Phonocardiogram recording in WAV format"),
    patient_id: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
) -> JSONResponse:
    request_id = _request_id(request)
    if not _is_wav(file.filename, file.content_type):
        raise HTTPException(status_code=415, detail="Only WAV files are accepted", headers={"x-request-id": request_id})

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file is too large", headers={"x-request-id": request_id})
    _validate_wav(data)

    client: httpx.AsyncClient = request.app.state.http_client
    multipart = {"file": (file.filename or "recording.wav", data, "audio/wav")}
    try:
        audio_response = await client.post(
            AUDIO_ENGINE_URL,
            files=multipart,
            headers={"x-request-id": request_id},
        )
        audio_result = await _json_response(audio_response, "audio-engine", request_id)
        orchestrator_response = await client.post(
            ORCHESTRATOR_URL,
            json={"audio": audio_result, "patient_id": patient_id, "metadata": metadata, "request_id": request_id},
            headers={"x-request-id": request_id},
        )
        result = await _json_response(orchestrator_response, "agent-orchestrator", request_id)
    except HTTPException:
        raise
    except (httpx.HTTPError, TimeoutError) as exc:
        raise _downstream_error("Downstream service", exc, request_id) from exc

    return JSONResponse(content=result, headers={"x-request-id": request_id})