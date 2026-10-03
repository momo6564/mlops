
from src.train import train_model


def test_train_model():
    accuracy = train_model()

    assert 0.0 <= accuracy <= 1.0
