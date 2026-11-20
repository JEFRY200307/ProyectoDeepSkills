import numpy as np
import pandas as pd
import pytest
from tsfresh.feature_extraction import MinimalFCParameters

from ts2tabpfn.data import get_resample
from ts2tabpfn.features import (
    assemble_chunks,
    cache_path,
    chunk_bounds,
    chunk_path,
    save_chunk,
    extract_tsfresh_chunked,
    extract_catch22,
    extract_full,
    extract_multirocket,
    extract_tsfresh,
    load_feature_cache,
    save_feature_cache,
    to_tsfresh_df,
)


def test_catch22_gives_22_features_per_channel(toy_split):
    X_train = toy_split[0]
    features = extract_catch22(X_train)
    assert features.shape == (len(X_train), 22 * X_train.shape[1])


def test_to_tsfresh_df_long_format(toy_split):
    X = toy_split[0][:3]
    df = to_tsfresh_df(X)
    assert len(df) == 3 * 2 * 60
    assert set(df["kind"]) == {"ch0", "ch1"}


def test_tsfresh_has_no_nans_after_impute(toy_split):
    features = extract_tsfresh(toy_split[0], n_jobs=1, fc_parameters=MinimalFCParameters())
    assert features.shape[0] == len(toy_split[0])
    assert np.isfinite(features).all()


def test_multirocket_is_deterministic_with_seed(toy_split):
    X_train, _, X_test, _ = toy_split
    a_train, a_test = extract_multirocket(X_train, X_test, random_state=3, n_jobs=1)
    b_train, b_test = extract_multirocket(X_train, X_test, random_state=3, n_jobs=1)
    assert a_train.shape[0] == len(X_train) and a_test.shape[0] == len(X_test)
    assert a_train.shape[1] == a_test.shape[1]
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_test, b_test)


@pytest.mark.parametrize("extractor", ["c22", "tsfresh"])
def test_cached_features_equal_per_resample_extraction(toy_split, extractor, monkeypatch):
    """Decision #12: extracting once and resampling rows == extracting after resampling."""
    if extractor == "tsfresh":
        import ts2tabpfn.features as features_module

        original = features_module.extract_tsfresh
        monkeypatch.setattr(
            features_module, "extract_tsfresh",
            lambda X, n_jobs=1: original(X, n_jobs=1, fc_parameters=MinimalFCParameters()),
        )
    X_train, y_train, X_test, y_test = toy_split
    full = extract_full(X_train, X_test, extractor, n_jobs=1)
    F_train, F_test = full[: len(X_train)], full[len(X_train):]

    for resample_id in (0, 4):
        cached = get_resample(F_train, y_train, F_test, y_test, resample_id)
        Xr_train, yr_train, Xr_test, yr_test = get_resample(X_train, y_train, X_test, y_test, resample_id)
        direct = extract_full(Xr_train, Xr_test, extractor, n_jobs=1)
        np.testing.assert_allclose(cached[0], direct[: len(Xr_train)], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(cached[2], direct[len(Xr_train):], rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(cached[1], yr_train)
        np.testing.assert_array_equal(cached[3], yr_test)


def test_chunk_bounds_cover_all_rows():
    assert chunk_bounds(10, 3) == [(0, 3), (3, 7), (7, 10)]
    bounds = chunk_bounds(259, 10)
    assert bounds[0][0] == 0 and bounds[-1][1] == 259
    assert all(a < b for a, b in bounds)
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(9))


def test_chunked_tsfresh_equals_single_call_and_resumes(tmp_path, toy_split):
    """Decision #19: per-chunk extraction + one impute == one extraction over all rows."""
    X_train, _, X_test, _ = toy_split
    X_train = X_train.copy()
    X_train[0, 0, :] = 0.0  # constant series: features with NaN that impute must fill
    fc = MinimalFCParameters()
    expected = extract_tsfresh(np.concatenate([X_train, X_test]), n_jobs=1, fc_parameters=fc)

    parts = tmp_path / "parts"
    first = extract_tsfresh_chunked(X_train, X_test, parts, 3, chunk_ids=[0], n_jobs=1, fc_parameters=fc,
                                    log=lambda _: None)
    assert first is None  # chunks 1 and 2 still missing
    full = extract_tsfresh_chunked(X_train, X_test, parts, 3, n_jobs=1, fc_parameters=fc, log=lambda _: None)
    np.testing.assert_allclose(full, expected, rtol=1e-12, atol=1e-12)
    assert len(list(parts.glob("part_*.npz"))) == 3


def test_assemble_chunks_aligns_columns_by_name(tmp_path, toy_split):
    """With n_jobs > 1, tsfresh can return each chunk's columns in a different order."""
    X_train, _, X_test, _ = toy_split
    fc = MinimalFCParameters()
    expected = extract_tsfresh(np.concatenate([X_train, X_test]), n_jobs=1, fc_parameters=fc)
    parts = tmp_path / "parts"
    extract_tsfresh_chunked(X_train, X_test, parts, 2, n_jobs=1, fc_parameters=fc, log=lambda _: None)

    # Rewrite chunk 1 with its columns reversed, as a different arrival order would.
    with np.load(chunk_path(parts, 1, 2)) as data:
        shuffled = pd.DataFrame(data["values"], columns=list(data["columns"])).iloc[:, ::-1]
    save_chunk(chunk_path(parts, 1, 2), shuffled)
    np.testing.assert_allclose(assemble_chunks(parts, 2), expected, rtol=1e-12, atol=1e-12)

    # A chunk with a different set of features is still an error.
    save_chunk(chunk_path(parts, 1, 2), shuffled.iloc[:, 1:])
    with pytest.raises(ValueError, match="different features"):
        assemble_chunks(parts, 2)


def test_feature_cache_roundtrip(tmp_path, toy_split):
    X_train, y_train, X_test, y_test = toy_split
    full = np.arange(len(y_train) + len(y_test) * 3, dtype=float).reshape(-1, 1)[: len(y_train) + len(y_test)]
    path = cache_path(tmp_path, "c22", "Toy")
    save_feature_cache(path, full, np.concatenate([y_train, y_test]), len(y_train))
    F_train, f_y_train, F_test, f_y_test = load_feature_cache(path)
    np.testing.assert_array_equal(F_train, full[: len(y_train)])
    np.testing.assert_array_equal(f_y_test, y_test)
    assert not list(path.parent.glob("*.tmp*"))
