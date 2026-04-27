"""
explain.py
----------
Improvement #1: Model Explainability

Extracts the top words that pushed a prediction toward FAKE or REAL,
using the classifier's learned feature weights (coefficients).

Why this matters:
  A model you can't explain is a model you can't trust.
  This module surfaces the exact vocabulary the classifier relies on,
  giving transparency to every prediction.

Usage (as a script):
    python -m src.explain --model-dir models/ --text "Your article text here"

Usage (as a module):
    from src.explain import explain_prediction
    explanation = explain_prediction(clf, vectorizer, "Some news text")
    print(explanation)
"""

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np

from src.preprocess import _clean_text
from src.train import load_artifacts

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TOP_N = 10   # number of influential words to surface per prediction


def get_top_words(
    clf,
    vectorizer,
    text: str,
    top_n: int = TOP_N,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identify the words in *text* that most strongly influenced the prediction.

    How it works:
      • The Passive Aggressive Classifier stores a weight (coef_) for every
        word in the vocabulary.
      • Positive weights → push toward FAKE (class 1).
      • Negative weights → push toward REAL (class 0).
      • We multiply the TF-IDF score of each word in *this specific text* by
        its learned weight, then rank them.

    Parameters
    ----------
    clf        : fitted PassiveAggressiveClassifier
    vectorizer : fitted TfidfVectorizer
    text       : raw input string
    top_n      : how many words to return per direction

    Returns
    -------
    dict with keys:
        "fake_words"  : list of (word, score) tuples — pushed toward FAKE
        "real_words"  : list of (word, score) tuples — pushed toward REAL
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")

    cleaned   = _clean_text(pd.Series([text])).iloc[0]
    X         = vectorizer.transform([cleaned])

    # Feature names (vocabulary list)
    vocab      = np.array(vectorizer.get_feature_names_out())

    # Model weights — shape: (1, n_features) for binary classification
    weights    = clf.coef_[0]

    # Element-wise: tfidf_score * model_weight for words present in this text
    tfidf_row  = np.array(X.todense())[0]
    influence  = tfidf_row * weights

    # Words actually present in this text (non-zero TF-IDF)
    present_mask = tfidf_row > 0
    present_vocab     = vocab[present_mask]
    present_influence = influence[present_mask]

    # Sort by influence score
    sorted_idx = np.argsort(present_influence)[::-1]   # descending
    sorted_words  = present_vocab[sorted_idx]
    sorted_scores = present_influence[sorted_idx]

    # Top-N push toward FAKE (highest positive scores)
    fake_words = [
        (w, round(float(s), 4))
        for w, s in zip(sorted_words[:top_n], sorted_scores[:top_n])
        if s > 0
    ]

    # Top-N push toward REAL (highest negative scores → flip sign for display)
    real_words = [
        (w, round(float(-s), 4))
        for w, s in zip(sorted_words[::-1][:top_n], sorted_scores[::-1][:top_n])
        if s < 0
    ]

    return {"fake_words": fake_words, "real_words": real_words}


def explain_prediction(clf, vectorizer, text: str, top_n: int = TOP_N) -> str:
    """
    Build a human-readable explanation string for a single prediction.

    Parameters
    ----------
    clf        : fitted classifier
    vectorizer : fitted TfidfVectorizer
    text       : raw news text
    top_n      : words to show per direction

    Returns
    -------
    Formatted multi-line string explanation.
    """
    from src.predict import FakeNewsPredictor, LABEL_MAP
    import math

    # Get label + confidence
    X       = vectorizer.transform([_clean_text(pd.Series([text])).iloc[0]])
    label   = LABEL_MAP[int(clf.predict(X)[0])]
    raw     = float(clf.decision_function(X)[0])
    conf    = 1.0 / (1.0 + math.exp(-abs(raw)))

    words   = get_top_words(clf, vectorizer, text, top_n=top_n)

    lines = [
        "",
        "═" * 56,
        f"  PREDICTION  : {label}",
        f"  CONFIDENCE  : {conf * 100:.1f}%",
        "─" * 56,
        f"  Top words pushing → FAKE:",
    ]
    if words["fake_words"]:
        for word, score in words["fake_words"]:
            lines.append(f"    {word:<25} +{score:.4f}")
    else:
        lines.append("    (none found in this text)")

    lines.append(f"  Top words pushing → REAL:")
    if words["real_words"]:
        for word, score in words["real_words"]:
            lines.append(f"    {word:<25} +{score:.4f}")
    else:
        lines.append("    (none found in this text)")

    lines.append("═" * 56)
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a fake news prediction — show influential words."
    )
    parser.add_argument("--model-dir", default="models",
                        help="Folder with saved .joblib files")
    parser.add_argument("--text", required=True,
                        help="News text to explain")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                        help=f"Words to show per direction (default: {TOP_N})")
    return parser.parse_args()


if __name__ == "__main__":
    args      = _parse_args()
    clf, vec  = load_artifacts(args.model_dir)
    report    = explain_prediction(clf, vec, args.text, top_n=args.top_n)
    print(report)
