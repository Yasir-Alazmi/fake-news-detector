"""Repeated stratified 5-fold cross-validation over ALL articles in data/, with and without the
'(Reuters)' source tag, plus the words that drive the model. Writes results/cv_report.md.

    python scripts/evaluate_cv.py
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
true = pd.read_csv(ROOT / "data" / "True.csv")
fake = pd.read_csv(ROOT / "data" / "Fake.csv")
df = pd.concat([true.assign(y=0), fake.assign(y=1)]).drop_duplicates("text").reset_index(drop=True)


def strip_source_tag(text: str) -> str:
    text = re.sub(r"^[^-\u2013\u2014]{0,60}\(Reuters\)\s*[-\u2013\u2014]\s*", "", text)
    return re.sub(r"\(?Reuters\)?", "", text, flags=re.I)


def make():
    return make_pipeline(
        TfidfVectorizer(stop_words="english", max_df=0.7),
        PassiveAggressiveClassifier(C=0.5, max_iter=50, random_state=42, tol=1e-3),
    )


def cv_scores(texts, repeats=5):
    return [
        accuracy_score(df.y, cross_val_predict(make(), texts, df.y, cv=StratifiedKFold(5, shuffle=True, random_state=s)))
        for s in range(repeats)
    ]


rows = []
for name, texts in (("raw text", df.text), ("'(Reuters)' tag removed", df.text.map(strip_source_tag))):
    a = cv_scores(texts)
    rows.append((name, np.mean(a), min(a), max(a)))
vec = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vec.fit_transform(df.text)
w = PassiveAggressiveClassifier(C=0.5, max_iter=50, random_state=42).fit(X, df.y).coef_[0]
names = np.array(vec.get_feature_names_out())
lines = [
    "# Cross-validated evaluation",
    "",
    f"{len(df)} articles ({int((df.y == 0).sum())} real, {int((df.y == 1).sum())} fake), repeated stratified 5-fold CV (5 shuffles).",
    "",
    "| Input | Mean accuracy | Min | Max |",
    "|---|---|---|---|",
    *[f"| {n} | {m:.3f} | {lo:.3f} | {hi:.3f} |" for n, m, lo, hi in rows],
    "",
    f"Words most associated with REAL: {', '.join(names[np.argsort(w)[:8]])}",
    f"Words most associated with FAKE: {', '.join(names[np.argsort(w)[-8:]])}",
    "",
    "Reading: removing the '(Reuters)' tag barely changes the score, because other wire-style cues remain",
    "(datelines, weekday and city names on one side; 'image', 'com', 'mr' on the other). The classifier",
    "mostly separates wire-service writing from fringe-site writing; it does not check whether a claim is true.",
]
out = ROOT / "results" / "cv_report.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
