import os

import numpy as np
import pytest

from ts2tabpfn.model import classification_metrics, fit_predict, make_classifier


def test_classification_metrics_perfect_and_wrong():
    y = np.array(["a", "b", "b", "c"])
    assert classification_metrics(y, y)["accuracy"] == 1.0
    wrong = classification_metrics(y, np.array(["b", "a", "c", "b"]))
    assert wrong["accuracy"] == 0.0 and wrong["f1"] == 0.0


def test_make_classifier_matches_reference_configuration():
    classifier = make_classifier(random_state=7, device="cpu")
    assert type(classifier).__name__ == "ManyClassClassifier"
    assert classifier.alphabet_size == 10
    estimator = classifier.estimator
    assert estimator.random_state == 7
    assert estimator.ignore_pretraining_limits is True
    assert estimator.device == "cpu"


@pytest.mark.tabpfn
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN for the gated TabPFN 2.5 weights")
def test_tabpfn_learns_an_easy_problem():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 5))
    y = np.where(X[:, 0] > 0, "pos", "neg")
    y_pred, fit_s, predict_s = fit_predict(make_classifier(0, "cpu"), X[:60], y[:60], X[60:])
    assert classification_metrics(y[60:], y_pred)["accuracy"] >= 0.9
    assert fit_s >= 0 and predict_s >= 0
