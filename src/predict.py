"""
predict.py
----------
Inference module for the Fake News Detector.

Loads persisted model artifacts and classifies raw text as REAL or FAKE,
returning both a label and a confidence score.

Usage (as a script):
    python -m src.predict --model-dir models/ --text "Breaking: Scientists discover water on Mars"

Usage (as a module):
    from src.predict import FakeNewsPredictor
    predictor = FakeNewsPredictor(model_dir="models/")
    result = predictor.predict("Your headline here")
    print(result)  # {"label": "FAKE", "confidence": 0.91, "raw_score": 1.43}
"""

import argparse
import logging
import math
from typing import Dict, Union

from src.preprocess import _clean_text
from src.train import load_artifacts

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Label map ──────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "REAL", 1: "FAKE"}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Sigmoid function: maps any real number → (0, 1).

    Used to convert the classifier's raw decision score into a
    human-readable confidence percentage.

    sigmoid(0)   = 0.50  → completely uncertain
    sigmoid(3)   ≈ 0.95  → very confident
    sigmoid(-3)  ≈ 0.05  → very confident in the opposite class
    """
    return 1.0 / (1.0 + math.exp(-x))


# ── Main predictor class ───────────────────────────────────────────────────────

class FakeNewsPredictor:
    """
    A reusable inference object that wraps the saved model and vectorizer.

    Design choice: class (not a bare function) so we load artifacts once
    and reuse them across many predictions — important for production/APIs.

    Parameters
    ----------
    model_dir : str
        Path to the folder containing pac_model.joblib and
        tfidf_vectorizer.joblib.
    """

    def __init__(self, model_dir: str = "models") -> None:
        self.clf, self.vectorizer = load_artifacts(model_dir)
        logger.info("FakeNewsPredictor ready.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Classify a single piece of text.

        Parameters
        ----------
        text : str
            Raw news article text or headline.

        Returns
        -------
        dict with keys:
            label      (str)   — "REAL" or "FAKE"
            confidence (float) — 0.0 – 1.0 probability-like score
            raw_score  (float) — raw decision function output
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        # ── 1. Clean text the same way we cleaned training data ───────────────
        cleaned = _clean_text(pd.Series([text])).iloc[0]

        # ── 2. Vectorize using the SAME fitted vectorizer ─────────────────────
        X = self.vectorizer.transform([cleaned])

        # ── 3. Predict label ──────────────────────────────────────────────────
        label_int = int(self.clf.predict(X)[0])
        label_str = LABEL_MAP[label_int]

        # ── 4. Compute confidence via decision function + sigmoid ─────────────
        # decision_function returns the signed distance from the hyperplane:
        #   positive  → leans toward class 1 (FAKE)
        #   negative  → leans toward class 0 (REAL)
        raw_score  = float(self.clf.decision_function(X)[0])
        # For REAL predictions the raw score is negative; flip it so confidence
        # always reflects certainty in the *predicted* class.
        if label_int == 0:
            confidence = _sigmoid(-raw_score)
        else:
            confidence = _sigmoid(raw_score)

        return {
            "label"     : label_str,
            "confidence": round(confidence, 4),
            "raw_score" : round(raw_score,  4),
        }

    def predict_batch(self, texts: list) -> list:
        """
        Classify a list of texts. Returns a list of result dicts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        list of dicts (same structure as predict())
        """
        return [self.predict(t) for t in texts]


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether a news article is REAL or FAKE."
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Folder with saved .joblib files  (default: models/)",
    )
    parser.add_argument(
        "--text",
        required=True,
        help='News text to classify. Wrap in quotes, e.g. --text "Headline here"',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    predictor = FakeNewsPredictor(model_dir=args.model_dir)
    result    = predictor.predict(args.text)

    print("\n" + "─" * 50)
    print(f"  Text      : {args.text[:80]}{'…' if len(args.text) > 80 else ''}")
    print(f"  Prediction: {result['label']}")
    print(f"  Confidence: {result['confidence'] * 100:.1f}%")
    print(f"  Raw score : {result['raw_score']}")
    print("─" * 50 + "\n")
