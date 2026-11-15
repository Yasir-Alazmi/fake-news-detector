# 🕵️ Fake News Detector

[![Tests](https://img.shields.io/badge/Tests-32%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)](https://scikit-learn.org)
[![CI](https://github.com/Yasir-Alazmi/fake-news-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/Yasir-Alazmi/fake-news-detector/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small learning project: a TF-IDF + Passive Aggressive text classifier that separates news articles labelled **REAL** from articles labelled **FAKE**, with feature explainability. It learns the *writing style of the source*, not whether a claim is true (see Results).

> **Honest status:** the repository ships 198 articles (100 real from a wire service, 98 fake). Cross-validated accuracy on them is about 0.89. An earlier "~99%" claim was for a different, larger dataset and is not reproducible from this repository.

---

## 📁 Project Structure

```
fake-news-detector/
├── data/                  # Place Fake.csv and True.csv here (not tracked by git)
├── models/                # Saved model artifacts (generated after training)
├── results/               # Confusion matrix PNG + metrics report (generated)
├── src/
│   ├── preprocess.py      # Data loading, cleaning, TF-IDF vectorization
│   ├── train.py           # Model training and artifact persistence
│   ├── evaluate.py        # Metrics, classification report, confusion matrix
│   └── predict.py         # Inference — FakeNewsPredictor class
├── tests/
│   └── test_pipeline.py   # 25 unit tests covering all modules
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/Yasir-Alazmi/fake-news-detector.git
cd fake-news-detector
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place both files inside the `data/` folder:
```
data/
├── Fake.csv
└── True.csv
```

---

## 🚀 Usage

### Train the model
```bash
python -m src.train --data-dir data/ --model-dir models/
```

### Evaluate on the test set
```bash
python -m src.evaluate --data-dir data/ --model-dir models/ --output-dir results/
```

### Predict a single article
```bash
python -m src.predict --model-dir models/ --text "Breaking: Scientists confirm water on Mars"
```

**Example output:**
```
──────────────────────────────────────────────────
  Text      : Breaking: Scientists confirm water on Mars
  Prediction: REAL
  Confidence: 87.3%
  Raw score : 1.98
──────────────────────────────────────────────────
```

### Use as a Python module
```python
from src.predict import FakeNewsPredictor

predictor = FakeNewsPredictor(model_dir="models/")
result = predictor.predict("Miracle cure discovered, doctors hate this!")
print(result)
# {"label": "FAKE", "confidence": 0.95, "raw_score": 2.91}

# Batch prediction
headlines = ["Economy grows 3%", "Aliens secretly run NASA"]
results = predictor.predict_batch(headlines)
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

To also see coverage:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 🧠 How It Works

| Stage | Tool | Purpose |
|---|---|---|
| Text cleaning | `pandas` + `re` | Lowercase, strip, collapse whitespace |
| Vectorization | `TfidfVectorizer` | Convert text → numerical feature matrix |
| Training | `PassiveAggressiveClassifier` | Online, linear classifier — ideal for text |
| Evaluation | `sklearn.metrics` | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Persistence | `joblib` | Save/load model + vectorizer without retraining |

### Why Passive Aggressive Classifier?
- Designed for **large-scale text classification**
- **Online learning** — can update with new data without full retraining
- Extremely fast on sparse TF-IDF matrices
- Competitive accuracy with far less compute than deep learning

---

## 📊 Results

Repeated stratified 5-fold cross-validation over all 198 articles (`python scripts/evaluate_cv.py`, full output in [`results/cv_report.md`](results/cv_report.md)):

| Input | Mean accuracy | Range over 5 shuffles |
|---|---|---|
| raw text | 0.893 | 0.884 to 0.904 |
| "(Reuters)" tag removed | 0.886 | 0.864 to 0.899 |

**What the model learns.** The most REAL-leaning words are `reuters, minister, tuesday, moscow, washington`; the most FAKE-leaning are `mr, right, like, com, image`. Removing the "(Reuters)" tag barely lowers accuracy because datelines and city names remain, so the classifier separates wire-service writing from fringe-site writing. It does not verify facts and would not generalise to other sources.

The single hold-out run in `results/metrics_report.txt` uses only 40 test articles, so its accuracy has a very wide interval; prefer the cross-validated figures above.

Confusion matrix saved to `results/confusion_matrix.png` after evaluation.

---

## 🔧 Configuration

All hyperparameters are defined as named constants at the top of each module — no digging through code required:

**`preprocess.py`**
```python
TFIDF_MAX_FEATURES = 5_000   # vocabulary size cap
TFIDF_NGRAM_RANGE  = (1, 2)  # unigrams + bigrams
TEST_SIZE          = 0.20    # 80/20 train/test split
```

**`train.py`**
```python
PAC_C        = 0.5   # regularisation strength
PAC_MAX_ITER = 50    # max training iterations
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | ≥ 1.3 | ML algorithms and metrics |
| pandas | ≥ 2.0 | Data loading and manipulation |
| numpy | ≥ 1.24 | Numerical operations |
| matplotlib | ≥ 3.7 | Confusion matrix visualization |
| joblib | ≥ 1.3 | Model serialization |
| pytest | ≥ 7.4 | Unit testing |

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Yasir Al-Azmi** · AI Student  
GitHub: [@Yasir-Alazmi](https://github.com/Yasir-Alazmi)
