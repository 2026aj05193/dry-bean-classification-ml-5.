"""Train five classifiers on the UCI Dry Bean dataset.

Run:
    pip install -r requirements.txt
    python train_models.py

The script fetches the official UCI dataset (ID 602) using ucimlrepo.
It creates data/training_data.csv, test_data.csv, saved models, and metrics.
"""
from pathlib import Path
import json
import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
TARGET = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20

def load_dataset():
    """Load official UCI dataset. Keep all original rows; do not silently drop duplicates."""
    bean = fetch_ucirepo(id=602)
    X = bean.data.features.copy()
    y = bean.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df[TARGET] = y.astype(str).values
    return df

def main():
    df = load_dataset()
    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")
    df = df.dropna().copy()
    X = df.drop(columns=[TARGET])
    y_text = df[TARGET].astype(str)

    # Keep a consistent feature order for training and Streamlit inference.
    feature_names = list(X.columns)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Keep readable target labels in the test CSV for evaluation in Streamlit.
    test_output = X_test.copy()
    test_output[TARGET] = encoder.inverse_transform(y_test)
    test_output.to_csv(ROOT / "test_data.csv", index=False)
    train_output = X_train.copy()
    train_output[TARGET] = encoder.inverse_transform(y_train)
    train_output.to_csv(DATA_DIR / "training_data.csv", index=False)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight=None
        ),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=RANDOM_STATE))
        ]),
    }

    results = {}
    for display_name, estimator in models.items():
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        proba = estimator.predict_proba(X_test)
        scores = {
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro",
                labels=list(range(len(encoder.classes_)))
            ),
            "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        }
        results[display_name] = {
            "metrics": {k: float(v) for k, v in scores.items()},
            "classification_report": classification_report(
                y_test, pred, labels=list(range(len(encoder.classes_))),
                target_names=list(encoder.classes_), output_dict=True,
                zero_division=0
            ),
        }
        filename_map = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "K-Nearest Neighbors": "k_nearest_neighbors",
            "Gaussian Naive Bayes": "gaussian_naive_bayes",
            "Support Vector Machine": "support_vector_machine",
        }
        joblib.dump(estimator, MODEL_DIR / f"{filename_map[display_name]}.joblib")
        print(f"{display_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    metadata = {
        "dataset": "UCI Dry Bean (ID 602)",
        "target": TARGET,
        "features": feature_names,
        "classes": list(encoder.classes_),
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "auc_method": "One-vs-Rest, macro average",
        "precision_recall_f1_average": "macro",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\nArtifacts saved. Review model/metrics.json and update your README observations.")
    print("Test file created at:", ROOT / "test_data.csv")

if __name__ == "__main__":
    main()
