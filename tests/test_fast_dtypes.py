"""Decision #18: the fast helpers must return exactly what tabpfn's originals return."""

import numpy as np
import pandas as pd
import pytest
from tabpfn.preprocessing import clean

from ts2tabpfn import fast_dtypes

ORIGINAL = fast_dtypes.ORIGINAL_FIX_DTYPES
ORIGINAL_INFER = fast_dtypes.ORIGINAL_INFER


def _assert_same(X, cat_indices, numeric_dtype="float64"):
    expected = ORIGINAL(X.copy() if hasattr(X, "copy") else X, cat_indices, numeric_dtype)
    got = fast_dtypes.fix_dtypes(X.copy() if hasattr(X, "copy") else X, cat_indices, numeric_dtype)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    assert list(got.dtypes) == list(expected.dtypes)


@pytest.fixture
def mixed_array():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 30))
    X[:, 3] = rng.integers(0, 3, 40)   # low-cardinality columns that TabPFN tags as categorical
    X[:, 10] = rng.integers(0, 2, 40)
    X[5, 7] = np.nan
    return X


@pytest.mark.parametrize("cat_indices", [None, [], [3, 10]])
@pytest.mark.parametrize("numeric_dtype", ["float64", "float32"])
def test_numeric_array(mixed_array, cat_indices, numeric_dtype):
    _assert_same(mixed_array, cat_indices, numeric_dtype)


def test_integer_and_float32_arrays(mixed_array):
    _assert_same(mixed_array.astype(np.float32), [3])
    _assert_same(np.arange(60).reshape(10, 6), [1, 4])


def test_object_array_and_dataframe(mixed_array):
    obj = mixed_array.astype(object)
    obj[0, 2] = None
    _assert_same(obj, [3])
    df = pd.DataFrame(mixed_array, columns=[f"f{i}" for i in range(30)])
    df["f5"] = df["f5"].astype(str)
    _assert_same(df, ["f3", "f10"])
    _assert_same(df, [3, 10])  # numeric indices with named columns -> iloc rule


def test_wide_multirocket_like_input_is_identical():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5000))
    X[:, ::7] = (X[:, ::7] > 0)  # PPV-like features with two values
    _assert_same(X, list(range(0, 5000, 7)))


def test_install_replaces_every_import_site():
    import tabpfn.base
    import tabpfn.classifier
    import tabpfn.preprocessing.initialization

    fast_dtypes.install()
    try:
        for module in (clean, tabpfn.base, tabpfn.classifier, tabpfn.preprocessing.initialization):
            assert module.fix_dtypes is fast_dtypes.fix_dtypes
        assert tabpfn.preprocessing.initialization.infer_categorical_features is fast_dtypes.infer_categorical_features
    finally:
        fast_dtypes.uninstall()
    assert clean.fix_dtypes is ORIGINAL


@pytest.mark.tabpfn
def test_tabpfn_predictions_identical_with_and_without_patch():
    """End to end: same probabilities from TabPFN with the original and the fast function."""
    import tabpfn.base
    import tabpfn.classifier
    import tabpfn.preprocessing.initialization
    from tabpfn import TabPFNClassifier

    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 800))
    X[:, ::10] = rng.integers(0, 3, (60, 80))
    y = (X[:, 1] + X[:, 2] > 0).astype(int)

    def proba():
        clf = TabPFNClassifier(random_state=0, ignore_pretraining_limits=True, device="cpu", n_estimators=2)
        try:
            clf.fit(X[:40], y[:40])
        except Exception as error:  # weights not downloadable here
            pytest.skip(f"TabPFN weights unavailable: {error}")
        return clf.predict_proba(X[40:])

    fast_dtypes.uninstall()
    expected = proba()
    fast_dtypes.install()
    try:
        got = proba()
    finally:
        fast_dtypes.uninstall()
    np.testing.assert_array_equal(got, expected)


INFER_ARGS = dict(min_samples_for_inference=20, max_unique_for_category=30, min_unique_for_numerical=4)


@pytest.mark.parametrize("provided", [None, [0, 3, 5]])
def test_infer_categorical_features_matches_original(provided):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 40))
    X[:, 0] = rng.integers(0, 3, 60)          # categorical
    X[:, 1] = 1.0                              # constant
    X[:, 2] = np.nan                           # all NaN
    X[:5, 3] = np.nan                          # NaN counted as a value
    X[:, 3] = np.where(np.isnan(X[:, 3]), np.nan, rng.integers(0, 2, 60))
    X[:, 4] = np.where(rng.random(60) > 0.5, 0.0, -0.0)   # +0 and -0
    X[:, 5] = np.where(rng.random(60) > 0.5, np.inf, -np.inf)
    fast = fast_dtypes.infer_categorical_features(X, provided=provided, **INFER_ARGS)
    assert fast == ORIGINAL_INFER(X, provided=provided, **INFER_ARGS)
    small = X[:10]  # below min_samples_for_inference
    fast_small = fast_dtypes.infer_categorical_features(small, provided=provided, **INFER_ARGS)
    assert fast_small == ORIGINAL_INFER(small, provided=provided, **INFER_ARGS)


def test_count_distinct_matches_pandas_nunique():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 5, (30, 50)).astype(float)
    X[rng.random(X.shape) < 0.2] = np.nan
    X[:, 7] = np.nan
    expected = [pd.Series(col).nunique(dropna=False) for col in X.T]
    assert list(fast_dtypes.count_distinct_per_column(X)) == expected
    ints = rng.integers(0, 4, (30, 6))
    assert list(fast_dtypes.count_distinct_per_column(ints)) == [pd.Series(c).nunique(dropna=False) for c in ints.T]


def test_infer_falls_back_for_object_arrays():
    X = np.array([["a", 1], ["b", 2], ["a", 3]] * 10, dtype=object)
    assert fast_dtypes.infer_categorical_features(X, **INFER_ARGS) == ORIGINAL_INFER(X, **INFER_ARGS)
