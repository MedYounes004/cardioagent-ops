"""Shared preprocessing helpers for model training."""

from __future__ import annotations

import numpy as np


def normalize(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return samples / peak if peak else samples


def resample(samples: np.ndarray, source_rate: int, target_rate: int = 16_000) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    duration = len(samples) / source_rate
    target_length = max(1, round(duration * target_rate))
    source_positions = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(source_positions, np.arange(len(samples)), samples).astype(np.float32)


def extract_features(samples: np.ndarray) -> np.ndarray:
    normalized = normalize(samples)
    centered = normalized - normalized.mean() if normalized.size else normalized
    rms = np.sqrt(np.mean(centered**2)) if centered.size else 0.0
    peak = np.max(np.abs(centered)) if centered.size else 0.0
    crossings = np.count_nonzero(np.diff(np.signbit(centered))) / max(len(centered) - 1, 1)
    return np.asarray([rms, peak, crossings], dtype=np.float32)
