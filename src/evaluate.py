"""
evaluate.py
-----------
Loads saved model artifacts and produces a full evaluation report:
  • Accuracy score
  • Classification report (precision, recall, F1 per class)
  • Confusion matrix (printed + saved as PNG)

Usage (as a script):
    python -m src.evaluate --data-dir data/ --model-dir models/ --output-dir results/

Usage (as a module):
    from src.evaluate import evaluate_model
    metrics = evaluate_model(clf, X_test_vec, y_test)
"""

import argparse
import logging
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.preprocess import prepare_data
from src.train import load_artifacts

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Label names for display ────────────────────────────────────────────────────
CLASS_NAMES = ["REAL", "FAKE"]   # index 0 = REAL, index 1 = FAKE

# ── Output file names ──────────────────────────────────────────────────────────
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
METRICS_REPORT_FILE   = "metrics_report.txt"


# ── Core evaluation function ───────────────────────────────────────────────────

def evaluate_model(clf, X_test_vec, y_test) -> Dict[str, Any]:
    """
    Run predictions and compute all evaluation metrics.

    Parameters
    ----------
    clf         : fitted classifier (PassiveAggressiveClassifier or any sklearn clf)
    X_test_vec  : sparse matrix — TF-IDF test features
    y_test      : array-like — true labels (0=REAL, 1=FAKE)

    Returns
    -------
    dict with keys:
        accuracy          (float)
        classification_report (str)
        confusion_matrix  (np.ndarray)
        y_pred            (np.ndarray)
    """
    y_pred   = clf.predict(X_test_vec)
    acc      = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    cm       = confusion_matrix(y_test, y_pred)

    # ── Print to console ───────────────────────────────────────────────────────
    separator = "─" * 55
    logger.info(separator)
    logger.info("  Accuracy : %.4f  (%.2f%%)", acc, acc * 100)
    logger.info(separator)
    print("\nClassification Report:\n")
    print(report)
    logger.info(separator)

    return {
        "accuracy"               : acc,
        "classification_report"  : report,
        "confusion_matrix"       : cm,
        "y_pred"                 : y_pred,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: str,
    show: bool = False,
) -> str:
    """
    Render and save a styled confusion matrix heatmap.

    Parameters
    ----------
    cm         : 2×2 numpy array from confusion_matrix()
    output_dir : folder where the PNG is written
    show       : if True, display the plot interactively (useful in notebooks)

    Returns
    -------
    Full path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, CONFUSION_MATRIX_FILE)

    # ── Build figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#1a1a2e")      # dark background
    ax.set_facecolor("#16213e")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap="Blues",
        values_format="d",
    )

    # ── Style ──────────────────────────────────────────────────────────────────
    ax.set_title("Confusion Matrix — Fake News Detector",
                 color="white", fontsize=13, pad=14)
    ax.set_xlabel("Predicted Label", color="#a0a0c0", fontsize=11)
    ax.set_ylabel("True Label",      color="#a0a0c0", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    # White text on cells for readability
    for text in disp.text_.ravel():
        text.set_color("white")
        text.set_fontsize(14)
        text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Confusion matrix saved → %s", output_path)

    if show:
        plt.show()
    plt.close()

    return output_path


def save_text_report(report: str, accuracy: float, output_dir: str) -> str:
    """
    Write the classification report + accuracy to a plain-text file.

    Parameters
    ----------
    report     : string from sklearn.metrics.classification_report
    accuracy   : float
    output_dir : folder where the .txt is written

    Returns
    -------
    Full path to the saved report.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, METRICS_REPORT_FILE)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("  Fake News Detector — Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)\n\n")
        f.write("Classification Report:\n\n")
        f.write(report)

    logger.info("Text report saved → %s", output_path)
    return output_path


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Fake News Detector on the test set."
    )
    parser.add_argument("--data-dir",   default="data",    help="Folder with Fake.csv / True.csv")
    parser.add_argument("--model-dir",  default="models",  help="Folder with saved .joblib files")
    parser.add_argument("--output-dir", default="results", help="Folder to save report + plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    data = prepare_data(args.data_dir)

    # ── Load saved model + vectorizer ─────────────────────────────────────────
    clf, _ = load_artifacts(args.model_dir)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics = evaluate_model(clf, data["X_test_vec"], data["y_test"])

    # ── Save outputs ───────────────────────────────────────────────────────────
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir=args.output_dir)
    save_text_report(
        metrics["classification_report"],
        metrics["accuracy"],
        output_dir=args.output_dir,
    )
