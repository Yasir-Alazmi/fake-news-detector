"""
run_all.py
----------
Improvement #2: One-Command Pipeline Runner

Chains the full pipeline in a single script:
  1. Load & vectorize data
  2. Train the model
  3. Save artifacts
  4. Evaluate on the test set
  5. Run a sample prediction with explanation

Usage:
    python run_all.py --data-dir data/ --model-dir models/ --output-dir results/
"""

import argparse
import logging
import time

from src.preprocess import prepare_data
from src.train import train_model, save_artifacts
from src.evaluate import evaluate_model, plot_confusion_matrix, save_text_report
from src.predict import FakeNewsPredictor
from src.explain import explain_prediction

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Demo headlines for the end-of-run showcase ─────────────────────────────────
DEMO_HEADLINES = [
    "Federal Reserve raises interest rates amid inflation concerns",
    "SHOCKING: Local man discovers government has been hiding free energy for decades",
    "NASA confirms new findings on the surface composition of Mars",
    "Scientists reveal that drinking bleach cures all known diseases instantly",
]


def _banner(text: str) -> None:
    width = 58
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def run_pipeline(data_dir: str, model_dir: str, output_dir: str) -> None:
    t_start = time.perf_counter()

    # ── Phase 1: Data ──────────────────────────────────────────────────────────
    _banner("Phase 1 / 4 — Loading & Vectorizing Data")
    data = prepare_data(data_dir)

    # ── Phase 2: Training ──────────────────────────────────────────────────────
    _banner("Phase 2 / 4 — Training Model")
    clf = train_model(data["X_train_vec"], data["y_train"])
    save_artifacts(clf, data["vectorizer"], model_dir)

    # ── Phase 3: Evaluation ────────────────────────────────────────────────────
    _banner("Phase 3 / 4 — Evaluating on Test Set")
    metrics = evaluate_model(clf, data["X_test_vec"], data["y_test"])
    plot_confusion_matrix(metrics["confusion_matrix"], output_dir=output_dir)
    save_text_report(
        metrics["classification_report"],
        metrics["accuracy"],
        output_dir=output_dir,
    )

    # ── Phase 4: Demo Predictions + Explanations ───────────────────────────────
    _banner("Phase 4 / 4 — Demo Predictions with Explanations")
    predictor = FakeNewsPredictor(model_dir=model_dir)
    for headline in DEMO_HEADLINES:
        report = explain_prediction(clf, data["vectorizer"], headline)
        print(f"  Text: {headline[:70]}")
        print(report)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    _banner(f"Pipeline Complete — {elapsed:.1f}s total")
    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Model     : {model_dir}/pac_model.joblib")
    print(f"  Report    : {output_dir}/metrics_report.txt")
    print(f"  CM Plot   : {output_dir}/confusion_matrix.png")
    print("═" * 58 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Fake News Detector pipeline end-to-end."
    )
    parser.add_argument("--data-dir",   default="data",    help="Folder with Fake.csv / True.csv")
    parser.add_argument("--model-dir",  default="models",  help="Where to save model artifacts")
    parser.add_argument("--output-dir", default="results", help="Where to save evaluation outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.data_dir, args.model_dir, args.output_dir)
