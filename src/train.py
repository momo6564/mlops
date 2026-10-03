
from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.data import load_and_split_data


def _compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics for training and validation outputs."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=sorted(set(y_true))),
    }


def _model_family_configurations():
    return {
        "RandomForestClassifier": [
            {
                "n_estimators": 200,
                "max_depth": 8,
                "random_state": 42,
            },
            {
                "n_estimators": 300,
                "max_depth": 10,
                "random_state": 42,
                "min_samples_leaf": 1,
            },
            {
                "n_estimators": 450,
                "max_depth": 6,
                "random_state": 42,
                "min_samples_leaf": 2,
            },
        ],
        "GradientBoostingClassifier": [
            {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 2,
                "random_state": 42,
            },
            {
                "n_estimators": 250,
                "learning_rate": 0.05,
                "max_depth": 2,
                "random_state": 42,
                "subsample": 0.9,
            },
            {
                "n_estimators": 350,
                "learning_rate": 0.03,
                "max_depth": 3,
                "random_state": 42,
                "subsample": 0.8,
            },
        ],
    }


def run_hyperparameter_experiment():
    """Train each candidate model family configuration and log all results to MLflow."""
    tracking_uri = (Path(__file__).resolve().parents[1] / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Wine-Cultivar-Classification")

    X_train, X_test, y_train, y_test = load_and_split_data()
    model_configs = _model_family_configurations()
    all_runs = []

    for family_name, configs in model_configs.items():
        for index, params in enumerate(configs, start=1):
            if family_name == "RandomForestClassifier":
                model_class = RandomForestClassifier
            else:
                model_class = GradientBoostingClassifier
            model = model_class(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            with mlflow.start_run(run_name=f"{family_name}-{index}") as run:
                val_pred = cross_val_predict(model, X_train, y_train, cv=cv, method="predict")
                val_proba = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")
                validation_metrics = _compute_metrics(y_train, val_pred, val_proba)

                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                train_proba = model.predict_proba(X_train)
                train_metrics = _compute_metrics(y_train, train_pred, train_proba)

                input_example = X_train[:5]
                signature = mlflow.models.infer_signature(X_train, model.predict(X_train[:5]))

                mlflow.log_params(params)
                mlflow.log_metric("validation_macro_f1", validation_metrics["macro_f1"])
                mlflow.log_metric("validation_accuracy", validation_metrics["accuracy"])
                mlflow.log_metric("validation_log_loss", validation_metrics["log_loss"])
                mlflow.log_metric("train_macro_f1", train_metrics["macro_f1"])
                mlflow.log_metric("train_accuracy", train_metrics["accuracy"])
                mlflow.log_metric("train_log_loss", train_metrics["log_loss"])
                mlflow.set_tag("model_family", family_name)
                mlflow.set_tag("config_index", str(index))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                )

                run_payload = {
                    "run_id": run.info.run_id,
                    "model_family": family_name,
                    "config_index": index,
                    "params": params,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_log_loss": validation_metrics["log_loss"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_log_loss": train_metrics["log_loss"],
                }
                all_runs.append(run_payload)

    best_run = max(all_runs, key=lambda item: item["validation_macro_f1"])
    best_run_id = best_run["run_id"]
    registered_model = mlflow.register_model(f"runs:/{best_run_id}/model", "WineClassifier")

    client = MlflowClient()
    client.set_registered_model_alias(
        name="WineClassifier",
        alias="champion",
        version=registered_model.version,
    )

    return {
        "best_run": best_run,
        "all_runs": all_runs,
        "model_name": "WineClassifier",
        "champion_version": registered_model.version,
        "X_test": X_test,
        "y_test": y_test,
    }


def train_model():
    """Backwards-compatible training entry point returning test accuracy."""
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    experiment = run_hyperparameter_experiment()
    print(f"Best validation macro F1: {experiment['best_run']['validation_macro_f1']:.4f}")
