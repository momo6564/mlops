from __future__ import annotations

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def validate_dataset(X, y=None):
    """Validate the dataset before training or evaluation."""
    if X is None or len(X) == 0:
        raise ValueError("Dataset is empty.")

    if np.isnan(X).any():
        raise ValueError("Feature matrix contains null values.")

    if X.shape[1] != 13:
        raise ValueError(f"Expected 13 features, received {X.shape[1]}.")

    if y is not None and np.isnan(y).any():
        raise ValueError("Target vector contains null values.")

    return True


def load_wine_dataset():
    """Load the Wine dataset and validate it."""
    X, y = load_wine(return_X_y=True)
    validate_dataset(X, y)
    return X, y


def load_and_split_data(test_size=0.2, random_state=42):
    """Load the Wine dataset and perform a stratified train/test split."""
    X, y = load_wine_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    validate_dataset(X_train, y_train)
    validate_dataset(X_test, y_test)
    return X_train, X_test, y_train, y_test
