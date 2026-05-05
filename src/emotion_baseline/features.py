from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def extract_features(audio_path: str | Path, sample_rate: int = 22050) -> dict[str, float]:
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    y = librosa.util.normalize(y)

    features: dict[str, float] = {
        "duration_sec": float(librosa.get_duration(y=y, sr=sr)),
    }

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    features.update(_safe_stats(zcr, "zcr"))
    features.update(_safe_stats(rms, "rms"))
    features.update(_safe_stats(centroid, "spectral_centroid"))
    features.update(_safe_stats(bandwidth, "spectral_bandwidth"))
    features.update(_safe_stats(rolloff, "spectral_rolloff"))
    features.update(_safe_stats(flatness, "spectral_flatness"))

    for index, row in enumerate(contrast, start=1):
        features.update(_safe_stats(row, f"spectral_contrast_{index}"))

    for index, row in enumerate(mfcc, start=1):
        features[f"mfcc_{index}_mean"] = float(np.mean(row))
        features[f"mfcc_{index}_std"] = float(np.std(row))

    for index, row in enumerate(delta_mfcc, start=1):
        features[f"delta_mfcc_{index}_mean"] = float(np.mean(row))
        features[f"delta_mfcc_{index}_std"] = float(np.std(row))

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        features.update(_safe_stats(f0, "pitch"))
        voiced = np.isfinite(f0)
        features["pitch_voiced_ratio"] = float(np.mean(voiced))
    except Exception:
        features.update(_safe_stats(np.array([]), "pitch"))
        features["pitch_voiced_ratio"] = 0.0

    return features

