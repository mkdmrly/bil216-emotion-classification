from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


EMOTIONS = ("neutral", "happy", "angry", "sad", "surprised")
ALIASES = {
    "neutral": "neutral",
    "happy": "happy",
    "angry": "angry",
    "sad": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class AudioSample:
    path: Path
    label: str


def normalize_label(label: str) -> str:
    key = label.strip().lower()
    if key not in ALIASES:
        raise ValueError(
            f"Unsupported label '{label}'. Expected one of: {', '.join(EMOTIONS)}"
        )
    return ALIASES[key]


def _infer_label_from_path(path: Path, data_dir: Path) -> str | None:
    parts = [part.lower() for part in path.relative_to(data_dir).parts[:-1]]
    for part in reversed(parts):
        if part in ALIASES:
            return ALIASES[part]

    tokens = re.split(r"[^a-zA-Z]+", path.stem.lower())
    for token in tokens:
        if token in ALIASES:
            return ALIASES[token]
    return None


def _find_audio_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def load_samples(data_dir: str | Path, metadata_csv: str | Path | None = None) -> list[AudioSample]:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if metadata_csv is not None:
        metadata_path = Path(metadata_csv).expanduser().resolve()
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
        return _load_samples_from_csv(data_dir, metadata_path)

    samples: list[AudioSample] = []
    for audio_path in _find_audio_files(data_dir):
        label = _infer_label_from_path(audio_path, data_dir)
        if label is None:
            continue
        samples.append(AudioSample(path=audio_path, label=label))

    if not samples:
        raise ValueError(
            "No labeled audio files were found. Use folder names like "
            "'neutral/happy/angry/sad/surprised' or provide --metadata-csv."
        )
    return samples


def _load_samples_from_csv(data_dir: Path, metadata_path: Path) -> list[AudioSample]:
    frame = pd.read_csv(metadata_path)
    lower_columns = {column.lower(): column for column in frame.columns}

    if "path" not in lower_columns:
        raise ValueError("Metadata CSV must include a 'path' column.")

    label_column = None
    for candidate in ("label", "emotion", "class"):
        if candidate in lower_columns:
            label_column = lower_columns[candidate]
            break
    if label_column is None:
        raise ValueError("Metadata CSV must include one of: label, emotion, class.")

    path_column = lower_columns["path"]
    samples: list[AudioSample] = []
    for _, row in frame.iterrows():
        audio_path = (data_dir / str(row[path_column])).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file from CSV not found: {audio_path}")
        samples.append(
            AudioSample(
                path=audio_path,
                label=normalize_label(str(row[label_column])),
            )
        )
    return samples

