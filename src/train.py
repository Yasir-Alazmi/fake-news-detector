"""
train.py
--------
Trains a Passive Aggressive Classifier on TF-IDF features and persists
both the model and the vectorizer to disk for later inference.

Algorithm spotlight — Passive Aggressive Classifier
----------------------------------------------------
• Designed for streaming / large-scale text classification.
• "Passive"   → if prediction is correct, do nothing (keep weights).
• "Aggressive"→ if prediction is wrong, update weights just enough to
                correct the mistake (controlled by parameter C).
• Converges fast and works exceptionally well on text data.

Usage (as a script):
    python -m src.train --data-dir data/ --model-dir models/

Usage (as a module):
    from src.train import train_model, save_artifacts
"""

import argparse
import logging
import os
from typing import Tuple

import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.base import BaseEstimator

from src.preprocess import prepare_data

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
PAC_C            = 0.5   # regularisation strength  (smaller → stronger regularisation)
PAC_MAX_ITER     = 50    # max training passes over the data
PAC_RANDOM_STATE = 42
PAC_TOL          = 1e-3  # convergence tolerance

# ── File names for saved artifacts ────────────────────────────────────────────
MODEL_FILENAME      = "pac_model.joblib"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"


# ── Core functions ─────────────────────────────────────────────────────────────

def train_model(X_train_vec, y_train) -> PassiveAggressiveClassifier:
    """
    Instantiate and fit a Passive Aggressive Classifier.

    Parameters
    ----------
    X_train_vec : sparse matrix  (output of TfidfVectorizer.transform)
    y_train     : array-like of int labels (1=FAKE, 0=REAL)

    Returns
    -------
    Fitted PassiveAggressiveClassifier
    """
    logger.info(
        "Training PassiveAggressiveClassifier  "
        "(C=%.2f, max_iter=%d) …", PAC_C, PAC_MAX_ITER
    )

    clf = PassiveAggressiveClassifier(
        C=PAC_C,
        max_iter=PAC_MAX_ITER,
        random_state=PAC_RANDOM_STATE,
        tol=PAC_TOL,
        class_weight="balanced",   # handles slight class imbalance automatically
        n_jobs=-1,                 # use all CPU cores
    )
    clf.fit(X_train_vec, y_train)

    logger.info("Training complete.")
    return clf


def save_artifacts(
    clf: BaseEstimator,
    vectorizer,
    model_dir: str,
) -> Tuple[str, str]:
    """
    Persist the fitted model and vectorizer to *model_dir*.

    Why save both?
    → At inference time we need the *same* vectorizer that was used during
      training, otherwise the feature indices won't match the model's weights.

    Parameters
    ----------
    clf         : fitted classifier
    vectorizer  : fitted TfidfVectorizer
    model_dir   : directory to write .joblib files into

    Returns
    -------
    (model_path, vectorizer_path) as strings
    """
    os.makedirs(model_dir, exist_ok=True)

    model_path      = os.path.join(model_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(model_dir, VECTORIZER_FILENAME)

    joblib.dump(clf,        model_path)
    joblib.dump(vectorizer, vectorizer_path)

    logger.info("Model saved      → %s", model_path)
    logger.info("Vectorizer saved → %s", vectorizer_path)

    return model_path, vectorizer_path


def load_artifacts(model_dir: str) -> Tuple[BaseEstimator, object]:
    """
    Load a previously saved model + vectorizer from *model_dir*.

    Parameters
    ----------
    model_dir : str — directory containing the .joblib files

    Returns
    -------
    (clf, vectorizer)
    """
    model_path      = os.path.join(model_dir, MODEL_FILENAME)
    vectorizer_path = os.path.join(model_dir, VECTORIZER_FILENAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"No saved vectorizer found at: {vectorizer_path}")

    clf        = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    logger.info("Loaded model from      %s", model_path)
    logger.info("Loaded vectorizer from %s", vectorizer_path)

    return clf, vectorizer


# ── CLI entry-point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Fake News Detector and save artifacts."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing Fake.csv and True.csv  (default: data/)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Folder where model + vectorizer are saved  (default: models/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # ── Step 1: load & vectorize data ─────────────────────────────────────────
    data = prepare_data(args.data_dir)

    # ── Step 2: train ─────────────────────────────────────────────────────────
    clf = train_model(data["X_train_vec"], data["y_train"])

    # ── Step 3: save everything ────────────────────────────────────────────────
    save_artifacts(clf, data["vectorizer"], args.model_dir)

    logger.info("Done! Run evaluate.py next to see metrics.")
