"""
train_pipeline.py

Simple model pipeline: data preprocessing -> training -> evaluation.
Dataset: Iris (public dataset built into scikit-learn).
Task: classify iris flower species from 4 measured features.

Run:
    python train_pipeline.py

Produces:
    model.joblib          - trained pipeline (scaler + classifier), ready for the API
    evaluation_report.txt - accuracy / precision / recall / F1 on the held-out test set
"""

import json
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def load_data():
    """Step 1a: Load the public dataset."""
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    target_names = data.target_names
    return X, y, target_names


def build_pipeline():
    """
    Step 1b: Define preprocessing + model as a single sklearn Pipeline.
    Bundling preprocessing and the model together means the exact same
    transformations used in training are guaranteed to run at inference
    time - no risk of train/serve skew.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # preprocessing: standardize features
            ("classifier", LogisticRegression(max_iter=200)),  # training
        ]
    )
    return pipeline


def train_and_evaluate():
    X, y, target_names = load_data()

    # Preprocessing step: train/test split (stratified so class balance is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Training
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )

    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Evaluation ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("\nFull classification report:\n", report)
    print("Confusion matrix:\n", cm)

    # Save evaluation results
    with open("evaluation_report.txt", "w") as f:
        f.write("=== Evaluation ===\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 score:  {f1:.4f}\n\n")
        f.write("Full classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))

    # Save baseline feature statistics from the TRAINING data.
    # The API/monitoring step will compare future live traffic against
    # these baseline stats to help detect data drift.
    baseline_stats = {
        "feature_means": X_train.mean().to_dict(),
        "feature_stds": X_train.std().to_dict(),
        "class_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))
        },
    }
    with open("baseline_stats.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    # Save the trained pipeline (preprocessing + model bundled together)
    joblib.dump(
        {"pipeline": pipeline, "target_names": list(target_names), "feature_names": list(X.columns)},
        "model.joblib",
    )
    print("\nSaved: model.joblib, evaluation_report.txt, baseline_stats.json")


if __name__ == "__main__":
    train_and_evaluate()
