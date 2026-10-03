from __future__ import annotations

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import load_and_split_data


def evaluate_registered_model(model_name="WineClassifier", alias="champion"):
    """Load the champion model from MLflow Model Registry and evaluate it."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias)
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    final_macro_f1 = f1_score(y_test, predictions, average="macro")
    final_accuracy = accuracy_score(y_test, predictions)
    final_log_loss = log_loss(y_test, probabilities, labels=np.unique(y_train))

    return {
        "model_name": model_name,
        "version": model_version.version,
        "macro_f1": final_macro_f1,
        "accuracy": final_accuracy,
        "log_loss": final_log_loss,
        "predictions": np.asarray(predictions, dtype=int),
        "X_test": X_test,
        "predictor": model,
    }
