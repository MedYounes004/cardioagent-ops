"""ONNX-backed phonocardiogram inference and lightweight audio features."""

from __future__ import annotations

import io
import math
import os
import struct
import wave
from pathlib import Path
from typing import Any

class InferenceEngine:
    """Load an exported ONNX model when available and expose one prediction API."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/cardio_model.onnx"))
        self._session: Any = None
        if self.model_path.exists():
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required when a model file is present") from exc
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def predict(self, audio_bytes: bytes) -> dict[str, Any]:
        sample_rate, samples = decode_wav(audio_bytes)
        features = extract_features(samples, sample_rate)
        if self._session is None:
            return {
                "label": "unknown",
                "probabilities": {"normal": 0.5, "pathological": 0.5},
                "features": features,
                "model_loaded": False,
            }

        input_name = self._session.get_inputs()[0].name
        import numpy as np

        output = self._session.run(None, {input_name: np.asarray([list(features.values())], dtype=np.float32)})[0]
        score = float(np.asarray(output).reshape(-1)[0])
        return {
            "label": "abnormal" if score >= 0.5 else "normal",
            "probabilities": {"normal": 1.0 - score, "pathological": score},
            "features": features,
            "model_loaded": True,
        }


def decode_wav(audio_bytes: bytes) -> tuple[int, list[float]]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise ValueError("only 16-bit PCM WAV files are supported")
    samples = [value / 32768.0 for (value,) in struct.iter_unpack("<h", frames)]
    if channels > 1:
        samples = [
            sum(samples[index : index + channels]) / channels
            for index in range(0, len(samples), channels)
        ]
    return sample_rate, samples


def extract_features(samples: list[float], sample_rate: int) -> dict[str, float]:
    if not samples:
        return {"rms": 0.0, "peak": 0.0, "zero_crossing_rate": 0.0}
    mean = sum(samples) / len(samples)
    centered = [sample - mean for sample in samples]
    rms = math.sqrt(sum(sample**2 for sample in centered) / len(centered))
    peak = max(abs(sample) for sample in centered)
    zero_crossings = sum(
        left * right < 0 for left, right in zip(centered, centered[1:])
    ) / max(len(centered) - 1, 1)
    return {"rms": rms, "peak": peak, "zero_crossing_rate": zero_crossings}
