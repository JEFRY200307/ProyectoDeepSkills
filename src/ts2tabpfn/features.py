"""Feature extractors (paper §3) and the on-disk feature cache.

Adapted from references/TS2TabPFN/code/utils/utils_train.py:22-82 (commit 70543e4).

- catch22 and tsfresh compute features per series with nothing fitted, so they
  are extracted once on the official train+test and the rows are resampled
  afterwards (decision #12). tsfresh concatenates train+test and imputes over
  both, as in the paper.
- MultiROCKET is fitted on the training set of every resample (decision #10),
  with random_state = seed + run (decision #11).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

N_JOBS_DEFAULT = os.cpu_count() or 1


def extract_catch22(X) -> np.ndarray:
    """22 features per channel, concatenated (utils_train.py:42-46)."""
    from aeon.transformations.collection.feature_based import Catch22

    return np.asarray(Catch22().fit_transform(X))


def to_tsfresh_df(X) -> pd.DataFrame:
    """Verbatim copy of utils_train.py:22-37."""
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, T = X.shape
    dfs = []
    for i in range(N):
        for c in range(C):
            dfs.append(
                pd.DataFrame({
                    "id": i,
                    "time": np.arange(T),
                    "value": X[i, c],
                    "kind": f"ch{c}"
                })
            )
    return pd.concat(dfs, ignore_index=True)


def extract_tsfresh(X, n_jobs: int = N_JOBS_DEFAULT, fc_parameters=None) -> np.ndarray:
    """EfficientFCParameters + impute (utils_train.py:59-80).

    `fc_parameters` exists only so tests can use a cheaper profile.
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    from tsfresh.utilities.dataframe_functions import impute

    features = extract_features(
        to_tsfresh_df(X),
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters if fc_parameters is not None else EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True,
    )
    impute(features)
    return features.to_numpy()


def extract_multirocket(X_train, X_test, random_state: int, n_jobs: int = N_JOBS_DEFAULT):
    """Default MultiROCKET, fitted on X_train (utils_train.py:48-57).

    n_jobs only sets the number of threads; it does not change the features.
    """
    from aeon.transformations.collection.convolution_based import MultiRocket

    transformer = MultiRocket(random_state=random_state, n_jobs=n_jobs)
    transformer.fit(X_train)
    return np.asarray(transformer.transform(X_train)), np.asarray(transformer.transform(X_test))


def extract_full(X_train, X_test, extractor: str, n_jobs: int = N_JOBS_DEFAULT) -> np.ndarray:
    """Features for train+test in the official order (train rows first)."""
    X_full = np.concatenate([X_train, X_test], axis=0)
    if extractor == "c22":
        return extract_catch22(X_full)
    if extractor == "tsfresh":
        return extract_tsfresh(X_full, n_jobs=n_jobs)
    raise ValueError(f"No full-set cache for extractor {extractor!r}")


def cache_path(features_dir: str | Path, extractor: str, dataset: str) -> Path:
    return Path(features_dir) / extractor / f"{dataset}.npz"


def save_feature_cache(path: str | Path, X_full, y_full, n_train: int) -> None:
    """Atomic write, so an interrupted session never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, X=X_full, y=np.asarray(y_full).astype(str), n_train=np.array([n_train]))
    os.replace(tmp, path)


def load_feature_cache(path: str | Path):
    """Returns the official split of the cached features: F_train, y_train, F_test, y_test."""
    with np.load(path, allow_pickle=False) as data:
        X_full, y_full, n_train = data["X"], data["y"], int(data["n_train"][0])
    return X_full[:n_train], y_full[:n_train], X_full[n_train:], y_full[n_train:]
