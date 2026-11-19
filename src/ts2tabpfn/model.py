"""TabPFN 2.5 classifier, configured as in references/TS2TabPFN/code/utils/utils_train.py:136-180.

TabPFNClassifier(random_state=seed+run, ignore_pretraining_limits=True, device=...)
wrapped in ManyClassClassifier(alphabet_size=10) for datasets with more than 10 classes.
Everything else stays at the library defaults (paper §4.3: no hyperparameter tuning).
"""

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def make_classifier(random_state: int, device: str):
    from tabpfn import TabPFNClassifier
    from tabpfn_extensions.many_class import ManyClassClassifier

    from .fast_dtypes import install

    install()  # same output, much faster with ~50k MultiROCKET columns (decision #18)
    estimator = TabPFNClassifier(
        random_state=random_state,
        ignore_pretraining_limits=True,
        device=device,
    )
    return ManyClassClassifier(estimator=estimator, alphabet_size=10)


def fit_predict(classifier, X_train, y_train, X_test):
    """Returns predictions and the fit/predict times in seconds."""
    start = time.time()
    classifier.fit(X_train, y_train)
    fit_s = time.time() - start

    start = time.time()
    y_pred = classifier.predict(X_test)
    predict_s = time.time() - start
    return y_pred, fit_s, predict_s


def classification_metrics(y_test, y_pred) -> dict:
    """Same metrics as get_results_clf (utils_train.py:201-218)."""
    y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted"),
    }
