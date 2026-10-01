"""FastAPI service for phonocardiogram inference."""

from __future__ import annotations

import io
import wave

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import InferenceEngine

app = FastAPI(title="CardioAgent Audio Engine", version="1.0.0")
engine = InferenceEngine()


def validate_wav(audio_bytes: bytes) -> None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            if wav_file.getnchannels() < 1 or wav_file.getframerate() < 1:
                raise ValueError("invalid WAV properties")
    except (EOFError, ValueError, wave.Error) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid WAV audio stream") from exc


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "service": "audio-engine", "model_loaded": engine.loaded}


@app.post("/process")
async def process(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=415, detail="Only WAV files are accepted")
    audio_bytes = await file.read()
    validate_wav(audio_bytes)
    try:
        return engine.predict(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
