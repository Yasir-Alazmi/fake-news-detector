"""
tests/test_pipeline.py
----------------------
Unit tests for the Fake News Detector pipeline.

Run with:
    pytest tests/ -v

Coverage targets:
    • preprocess.py  → text cleaning, vectorizer, data splitting
    • train.py       → model training produces correct output shape
    • predict.py     → valid labels, confidence range, edge cases
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Module imports ─────────────────────────────────────────────────────────────
from src.preprocess import (
    _clean_text,
    build_vectorizer,
    split_data,
    vectorize,
)
from src.train import train_model, save_artifacts, load_artifacts
from src.predict import FakeNewsPredictor, _sigmoid


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — shared test data
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sample_texts() -> list:
    """A small set of labelled texts for smoke-testing the pipeline."""
    return [
        "breaking news the president signed a new law today",
        "scientists discover water on mars nasa confirms",
        "shocking truth the government is hiding aliens from us",
        "fake story about miracle cure doctors hate this trick",
        "economy grows by three percent according to official report",
        "local team wins championship in overtime thriller game",
    ]


@pytest.fixture()
def sample_labels() -> list:
    return [0, 0, 1, 1, 0, 0]   # 0=REAL, 1=FAKE


@pytest.fixture()
def fitted_vectorizer_and_matrix(sample_texts):
    """Return a fitted TF-IDF vectorizer and the transformed matrix."""
    vec = build_vectorizer(max_features=100)
    X   = vec.fit_transform(sample_texts)
    return vec, X


@pytest.fixture()
def trained_clf(fitted_vectorizer_and_matrix, sample_labels):
    """Return a fitted classifier trained on the sample corpus."""
    _, X = fitted_vectorizer_and_matrix
    clf  = train_model(X, sample_labels)
    return clf


@pytest.fixture()
def saved_artifacts_dir(trained_clf, fitted_vectorizer_and_matrix, tmp_path):
    """Persist model + vectorizer to a temp directory and return its path."""
    vec, _ = fitted_vectorizer_and_matrix
    save_artifacts(trained_clf, vec, str(tmp_path))
    return str(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# 1. preprocess.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCleanText:
    def test_lowercases_input(self):
        result = _clean_text(pd.Series(["HELLO WORLD"]))
        assert result.iloc[0] == "hello world"

    def test_strips_whitespace(self):
        result = _clean_text(pd.Series(["  spaces  "]))
        assert result.iloc[0] == "spaces"

    def test_collapses_multiple_spaces(self):
        result = _clean_text(pd.Series(["too   many    spaces"]))
        assert result.iloc[0] == "too many spaces"

    def test_handles_empty_string(self):
        result = _clean_text(pd.Series([""]))
        assert result.iloc[0] == ""

    def test_handles_nan(self):
        result = _clean_text(pd.Series([None]))
        assert result.iloc[0] == ""

    def test_preserves_digits(self):
        result = _clean_text(pd.Series(["price is 99 dollars"]))
        assert "99" in result.iloc[0]


class TestBuildVectorizer:
    def test_returns_tfidf_instance(self):
        vec = build_vectorizer()
        assert isinstance(vec, TfidfVectorizer)

    def test_max_features_set(self):
        vec = build_vectorizer(max_features=200)
        assert vec.max_features == 200

    def test_ngram_range(self):
        vec = build_vectorizer(ngram_range=(1, 2))
        assert vec.ngram_range == (1, 2)

    def test_not_fitted_before_transform(self):
        vec = build_vectorizer()
        with pytest.raises(Exception):
            # transform() before fit_transform() must raise
            vec.transform(["some text"])


class TestVectorize:
    def test_output_shapes(self, sample_texts):
        train_texts = sample_texts[:4]
        test_texts  = sample_texts[4:]
        vec = build_vectorizer(max_features=50)
        X_train, X_test, _ = vectorize(vec, pd.Series(train_texts), pd.Series(test_texts))
        # Rows must match the number of documents
        assert X_train.shape[0] == len(train_texts)
        assert X_test.shape[0]  == len(test_texts)
        # Both matrices must share the same feature width
        assert X_train.shape[1] == X_test.shape[1]

    def test_returns_fitted_vectorizer(self, sample_texts):
        vec = build_vectorizer(max_features=50)
        _, _, fitted = vectorize(
            vec,
            pd.Series(sample_texts[:4]),
            pd.Series(sample_texts[4:]),
        )
        assert hasattr(fitted, "vocabulary_")


class TestSplitData:
    def test_split_sizes(self, sample_texts, sample_labels):
        df = pd.DataFrame({"text": sample_texts, "label": sample_labels})
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.33)
        total = len(sample_texts)
        assert len(X_train) + len(X_test) == total
        assert len(y_train) + len(y_test) == total

    def test_no_overlap(self, sample_texts, sample_labels):
        df = pd.DataFrame({"text": sample_texts, "label": sample_labels})
        X_train, X_test, _, _ = split_data(df)
        assert set(X_train.index).isdisjoint(set(X_test.index))


# ══════════════════════════════════════════════════════════════════════════════
# 2. train.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainModel:
    def test_returns_pac_instance(self, fitted_vectorizer_and_matrix, sample_labels):
        _, X = fitted_vectorizer_and_matrix
        clf = train_model(X, sample_labels)
        assert isinstance(clf, PassiveAggressiveClassifier)

    def test_predict_returns_correct_shape(self, trained_clf, fitted_vectorizer_and_matrix):
        _, X = fitted_vectorizer_and_matrix
        preds = trained_clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_only_valid_labels(self, trained_clf, fitted_vectorizer_and_matrix):
        _, X = fitted_vectorizer_and_matrix
        preds = trained_clf.predict(X)
        assert set(preds).issubset({0, 1})


class TestSaveLoadArtifacts:
    def test_files_created(self, saved_artifacts_dir):
        assert os.path.exists(os.path.join(saved_artifacts_dir, "pac_model.joblib"))
        assert os.path.exists(os.path.join(saved_artifacts_dir, "tfidf_vectorizer.joblib"))

    def test_loaded_clf_is_fitted(self, saved_artifacts_dir):
        clf, _ = load_artifacts(saved_artifacts_dir)
        assert isinstance(clf, PassiveAggressiveClassifier)
        assert hasattr(clf, "coef_")

    def test_loaded_vectorizer_is_fitted(self, saved_artifacts_dir):
        _, vec = load_artifacts(saved_artifacts_dir)
        assert hasattr(vec, "vocabulary_")

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(str(tmp_path))


# ══════════════════════════════════════════════════════════════════════════════
# 3. predict.py tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmoid:
    def test_zero_returns_half(self):
        assert _sigmoid(0) == pytest.approx(0.5)

    def test_large_positive_near_one(self):
        assert _sigmoid(10) > 0.99

    def test_large_negative_near_zero(self):
        assert _sigmoid(-10) < 0.01

    def test_output_in_zero_one(self):
        for x in [-5, -1, 0, 1, 5]:
            val = _sigmoid(x)
            assert 0.0 <= val <= 1.0


class TestFakeNewsPredictor:
    @pytest.fixture()
    def predictor(self, saved_artifacts_dir):
        return FakeNewsPredictor(model_dir=saved_artifacts_dir)

    def test_label_is_real_or_fake(self, predictor):
        result = predictor.predict("The president held a press conference today")
        assert result["label"] in {"REAL", "FAKE"}

    def test_confidence_in_valid_range(self, predictor):
        result = predictor.predict("miracle cure discovered doctors hate it")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_result_has_required_keys(self, predictor):
        result = predictor.predict("Economy grows steadily in Q3 report")
        assert {"label", "confidence", "raw_score"} == set(result.keys())

    def test_empty_text_raises_value_error(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict("")

    def test_whitespace_only_raises_value_error(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict("   ")

    def test_batch_predict_length_matches(self, predictor):
        texts = [
            "Economy report shows steady growth",
            "Aliens secretly control world governments",
            "Local school wins science competition",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == len(texts)

    def test_batch_all_have_label(self, predictor):
        results = predictor.predict_batch(["news one", "news two"])
        for r in results:
            assert r["label"] in {"REAL", "FAKE"}
