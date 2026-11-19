import numpy as np
import pytest


@pytest.fixture
def toy_split():
    """Small multivariate dataset in aeon's (n_cases, n_channels, n_timepoints) layout.

    Three classes with string labels, like the .ts loader returns.
    """
    rng = np.random.default_rng(0)

    def make(n_per_class):
        X, y = [], []
        for k, label in enumerate(["a", "b", "c"]):
            t = np.linspace(0, 1, 60)
            base = np.sin(2 * np.pi * (k + 1) * t)
            X.append(base + 0.1 * rng.standard_normal((n_per_class, 2, 60)))
            y += [label] * n_per_class
        return np.concatenate(X), np.array(y)

    X_train, y_train = make(6)
    X_test, y_test = make(4)
    return X_train, y_train, X_test, y_test


class MajorityClassifier:
    """Stand-in for TabPFN in tests that do not need the real weights."""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.label_ = values[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.label_)


@pytest.fixture
def majority_factory():
    return lambda random_state: MajorityClassifier(random_state)
