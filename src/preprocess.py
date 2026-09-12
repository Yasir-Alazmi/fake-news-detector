"""
preprocess.py
-------------
Handles all data loading, cleaning, and feature extraction for the
Fake News Detector pipeline.

Dataset expected layout (inside data/):
    Fake.csv   — articles labeled as FAKE
    True.csv   — articles labeled as REAL

Each CSV must have at least these columns: title, text
"""

import os
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_FAKE = 1
LABEL_REAL = 0

# TF-IDF hyperparameters (easily tunable in one place)
TFIDF_MAX_FEATURES = 5_000   # vocabulary cap
TFIDF_NGRAM_RANGE  = (1, 2)  # unigrams + bigrams
TEST_SIZE          = 0.20    # 80/20 split
RANDOM_STATE       = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_text(series: pd.Series) -> pd.Series:
    """
    Basic text normalization:
      • lowercase
      • strip leading/trailing whitespace
      • collapse multiple spaces
    """
    return (
        series.fillna("")
              .str.lower()
              .str.strip()
              .str.replace(r"\s+", " ", regex=True)
    )


def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv from *data_dir*, attach labels,
    and return a single shuffled DataFrame with columns:
        text (str), label (int: 1=FAKE / 0=REAL)

    Parameters
    ----------
    data_dir : str
        Path to the folder containing Fake.csv and True.csv.

    Returns
    -------
    pd.DataFrame
    """
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        sample_fake = os.path.join(data_dir, "sample_Fake.csv")
        sample_true = os.path.join(data_dir, "sample_True.csv")
        if os.path.exists(sample_fake) and os.path.exists(sample_true):
            logger.warning(
                "Full dataset (Fake.csv / True.csv) not found in '%s'. "
                "Running in DEMO mode using built-in sample datasets.",
                data_dir,
            )
            fake_path = sample_fake
            true_path = sample_true
        else:
            raise FileNotFoundError(
                f"Missing dataset files in {data_dir}. Expected Fake.csv / True.csv or sample_Fake.csv / sample_True.csv."
            )

    logger.info("Loading Fake dataset from %s ...", os.path.basename(fake_path))
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = LABEL_FAKE

    logger.info("Loading True dataset from %s ...", os.path.basename(true_path))
    true_df = pd.read_csv(true_path)
    true_df["label"] = LABEL_REAL

    # Combine title + text for richer features
    for df in (fake_df, true_df):
        df["text"] = _clean_text(df["title"] + " " + df["text"])

    combined = (
        pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]])
          .sample(frac=1, random_state=RANDOM_STATE)   # shuffle
          .reset_index(drop=True)
    )

    logger.info(
        "Dataset loaded — total rows: %d  |  FAKE: %d  |  REAL: %d",
        len(combined),
        (combined["label"] == LABEL_FAKE).sum(),
        (combined["label"] == LABEL_REAL).sum(),
    )
    return combined


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split DataFrame into train/test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pd.Series)
    """
    X = df["text"]
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_vectorizer(max_features: int = TFIDF_MAX_FEATURES,
                     ngram_range: tuple = TFIDF_NGRAM_RANGE) -> TfidfVectorizer:
    """
    Construct (but do NOT fit) a TF-IDF vectorizer with project defaults.

    TF-IDF recap
    ------------
    • TF  (Term Frequency)  = how often a word appears in *this* doc
    • IDF (Inverse Doc Freq)= penalises words that appear in *every* doc
    • Result: common words (the, a) get low scores; distinctive words get high scores
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # apply log(1 + tf) to dampen extreme counts
        strip_accents="unicode",
        analyzer="word",
        stop_words="english",    # remove English stopwords automatically
    )


def vectorize(
    vectorizer: TfidfVectorizer,
    X_train: pd.Series,
    X_test: pd.Series,
) -> Tuple[spmatrix, spmatrix, TfidfVectorizer]:
    """
    Fit the vectorizer on training data, then transform both splits.

    IMPORTANT: We fit ONLY on X_train to avoid data leakage.

    Returns
    -------
    X_train_vec, X_test_vec, fitted_vectorizer
    """
    logger.info("Fitting TF-IDF vectorizer on training data …")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    logger.info(
        "Vocabulary size: %d | Train matrix: %s | Test matrix: %s",
        len(vectorizer.vocabulary_),
        X_train_vec.shape,
        X_test_vec.shape,
    )
    return X_train_vec, X_test_vec, vectorizer


def prepare_data(data_dir: str) -> dict:
    """
    Full pipeline convenience function.
    Returns a dict with keys:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """
    df = load_raw_data(data_dir)
    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer = build_vectorizer()
    X_train_vec, X_test_vec, vectorizer = vectorize(vectorizer, X_train, X_test)

    return {
        "X_train_vec": X_train_vec,
        "X_test_vec" : X_test_vec,
        "y_train"    : y_train,
        "y_test"     : y_test,
        "vectorizer" : vectorizer,
    }
