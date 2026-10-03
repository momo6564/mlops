import time

import numpy as np

from src.data import load_and_split_data, validate_dataset
from src.evaluate import evaluate_registered_model
from src.train import run_hyperparameter_experiment


def test_wine_data_pipeline():
    X_train, X_test, y_train, y_test = load_and_split_data()

    validate_dataset(X_train, y_train)
    validate_dataset(X_test, y_test)

    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13
    assert set(np.unique(y_train)).issubset({0, 1, 2})
    assert set(np.unique(y_test)).issubset({0, 1, 2})


def test_model_gate():
    experiment = run_hyperparameter_experiment()
    best_run = experiment["best_run"]

    assert best_run["validation_macro_f1"] >= 0.88

    metrics = evaluate_registered_model()
    assert metrics["macro_f1"] >= 0.88
    assert set(metrics["predictions"]).issubset({0, 1, 2})

    start = time.perf_counter()
    for _ in range(20):
        metrics["predictor"].predict(np.asarray(metrics["X_test"]))
    elapsed_ms = (time.perf_counter() - start) * 1000 / 20
    assert elapsed_ms <= 30
