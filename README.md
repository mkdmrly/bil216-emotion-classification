# BIL216 Final Project Starter

This repository is a Phase 1 baseline for speech emotion classification using Python.

Supported target emotions:

- `neutral`
- `happy`
- `angry`
- `sad`
- `surprised`

## 1. Recommended folder layout

Put your audio dataset under `data/raw/`.

Two layouts are supported out of the box:

### Option A: Folder-per-emotion

```text
data/raw/
  neutral/
    sample_001.wav
  happy/
    sample_002.wav
  angry/
    sample_003.wav
  sad/
    sample_004.wav
  surprised/
    sample_005.wav
```

### Option B: CSV metadata

Create a CSV with at least these columns:

- `path`: relative path to audio file
- `label`: emotion label

Example:

```csv
path,label
speaker01/file001.wav,neutral
speaker01/file002.wav,happy
speaker02/file003.wav,angry
```

Then place the audio files under `data/raw/` and pass the CSV path with `--metadata-csv`.

## 2. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Train and evaluate

If your dataset uses emotion folders:

```bash
python3 train.py --data-dir data/raw --model auto
```

If your dataset uses a metadata CSV:

```bash
python3 train.py --data-dir data/raw --metadata-csv data/metadata.csv --model auto
```

Outputs are written to `results/`:

- `metrics.json`
- `confusion_matrix.png`
- `classification_report.txt`
- `feature_table.csv`

## 4. What this baseline does

The pipeline extracts a compact but strong classical feature set:

- zero-crossing rate
- RMS energy
- pitch statistics
- spectral centroid, bandwidth, rolloff, flatness, contrast
- MFCC mean and standard deviation
- delta MFCC mean and standard deviation

Then it compares several traditional models:

- SVM
- Random Forest
- KNN
- Logistic Regression

The best model is selected by cross-validation on the training split.

## 5. Notes for your report

This starter is intended to help with Phase 1 quickly. For later phases, strong next steps are:

- speaker-aware splits to avoid leakage
- class balancing
- hyperparameter tuning
- feature selection
- more detailed error analysis
- comparing classical ML against deep learning baselines

