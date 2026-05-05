from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(y_true, y_pred, labels: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        cmap="Blues",
        normalize=None,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Emotion Classification Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_feature_overview(feature_table, output_path: str | Path) -> None:
    output_path = Path(output_path)
    selected_columns = [
        column
        for column in ("zcr_mean", "rms_mean", "pitch_mean", "mfcc_1_mean", "mfcc_2_mean")
        if column in feature_table.columns
    ]
    if not selected_columns:
        return

    fig, axes = plt.subplots(len(selected_columns), 1, figsize=(9, 3 * len(selected_columns)))
    axes = np.atleast_1d(axes)

    for ax, column in zip(axes, selected_columns):
        feature_table.boxplot(column=column, by="label", ax=ax)
        ax.set_title(column)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Value")

    fig.suptitle("Selected Feature Distributions by Emotion")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

