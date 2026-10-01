"""Train candidate classifiers and log runs to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def train(features: np.ndarray, labels: np.ndarray, output_dir: str = "artifacts") -> Path:
    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_name = ""
    best_score = -1.0
    best_model = None
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(features, labels)
            score = accuracy_score(labels, model.predict(features))
            mlflow.log_metric("accuracy", float(score))
            if score > best_score:
                best_name, best_score, best_model = name, float(score), model
    if best_model is None:
        raise RuntimeError("no model was trained")
    import joblib

    destination = output_path / f"{best_name}.joblib"
    joblib.dump(best_model, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="NPZ file containing features and labels arrays")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    dataset = np.load(args.dataset)
    train(dataset["features"], dataset["labels"], args.output_dir)


if __name__ == "__main__":
    main()
