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


def extract_tsfresh_raw(X, n_jobs: int = N_JOBS_DEFAULT, fc_parameters=None) -> pd.DataFrame:
    """EfficientFCParameters features, one row per series, before imputation.

    `fc_parameters` exists only so tests can use a cheaper profile.
    """
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters

    return extract_features(
        to_tsfresh_df(X),
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_parameters if fc_parameters is not None else EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True,
    )


def impute_features(features: pd.DataFrame) -> np.ndarray:
    """tsfresh's `impute` over all rows (train+test, as the paper), then a plain array."""
    from tsfresh.utilities.dataframe_functions import impute

    impute(features)
    return features.to_numpy()


def extract_tsfresh(X, n_jobs: int = N_JOBS_DEFAULT, fc_parameters=None) -> np.ndarray:
    """EfficientFCParameters + impute (utils_train.py:59-80)."""
    return impute_features(extract_tsfresh_raw(X, n_jobs=n_jobs, fc_parameters=fc_parameters))


# --- Chunked tsfresh extraction (decision #19) -------------------------------------
# Every EfficientFCParameters feature is computed from one series alone, so extracting
# blocks of rows separately and concatenating them gives the same table as one call.
# Only `impute` uses all rows, so it runs once, after the blocks are joined. This lets
# EigenWorms (~33 h for the authors) be extracted in pieces shorter than a Colab session.


def chunk_bounds(n_rows: int, n_chunks: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) row ranges of near-equal size."""
    edges = np.linspace(0, n_rows, n_chunks + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def chunk_path(parts_dir: str | Path, chunk: int, n_chunks: int) -> Path:
    return Path(parts_dir) / f"part_{chunk:03d}_of_{n_chunks:03d}.npz"


def save_chunk(path: str | Path, features: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, values=features.to_numpy(), columns=np.asarray(features.columns, dtype=str))
    os.replace(tmp, path)


def assemble_chunks(parts_dir: str | Path, n_chunks: int):
    """Imputed full feature matrix once all chunks exist; None while some are missing."""
    paths = [chunk_path(parts_dir, c, n_chunks) for c in range(n_chunks)]
    if not all(p.exists() for p in paths):
        return None
    frames = []
    for p in paths:
        with np.load(p, allow_pickle=False) as data:
            frames.append(pd.DataFrame(data["values"], columns=list(data["columns"])))
    # tsfresh orders its columns by the order in which parallel results arrive
    # (imap_unordered in tsfresh/utilities/distribution.py), so with n_jobs > 1 each chunk
    # can come back with the same features in a different order. Align by name to the
    # first chunk's order; only a different set of features is an error.
    columns = frames[0].columns
    for i, frame in enumerate(frames[1:], 1):
        if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
            raise ValueError(f"Chunk {i} has different features than chunk 0 in {parts_dir}")
        frames[i] = frame[columns]
    full = pd.concat(frames, ignore_index=True)
    return impute_features(full)


def extract_tsfresh_chunked(X_train, X_test, parts_dir, n_chunks: int, chunk_ids=None,
                            n_jobs: int = N_JOBS_DEFAULT, fc_parameters=None, log=print):
    """Extract the missing chunks among `chunk_ids` (default: all), then try to assemble.

    Returns the full imputed matrix (train rows first) or None if chunks are still missing.
    Chunks already on disk are skipped, so the call can be repeated after an interruption,
    or split across machines with different `chunk_ids`.
    """
    X_full = np.concatenate([X_train, X_test], axis=0)
    bounds = chunk_bounds(len(X_full), n_chunks)
    for chunk in (range(n_chunks) if chunk_ids is None else chunk_ids):
        path = chunk_path(parts_dir, chunk, n_chunks)
        if path.exists():
            continue
        start, end = bounds[chunk]
        log(f"  chunk {chunk + 1}/{n_chunks}: rows {start}-{end}")
        save_chunk(path, extract_tsfresh_raw(X_full[start:end], n_jobs=n_jobs, fc_parameters=fc_parameters))
    return assemble_chunks(parts_dir, n_chunks)


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
