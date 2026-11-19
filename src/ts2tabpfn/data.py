"""Data loading and resampling.

Adapted from references/TS2TabPFN/code/utils/utils_data.py (commit 70543e4).
`stratified_resample_data` is copied verbatim so the 30 resamples match the
authors' splits exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from aeon.datasets import load_from_ts_file
from sklearn.utils import check_random_state


def load_split(data_dir: str | Path, dataset: str):
    """Official train/test split, as in `load_experiment_data` (utils_data.py:20-26)."""
    base = Path(data_dir) / dataset
    X_train, y_train = load_from_ts_file(str(base / f"{dataset}_TRAIN.ts"))
    X_test, y_test = load_from_ts_file(str(base / f"{dataset}_TEST.ts"))
    return X_train, y_train, X_test, y_test


def stratified_resample_data(X_train, y_train, X_test, y_test, random_state=None):
    """Verbatim copy of utils_data.py:67-138."""
    if isinstance(X_train, np.ndarray):
        is_array = True
    elif isinstance(X_train, list):
        is_array = False
    else:
        raise ValueError(
            "X_train must be a np.ndarray array or list of np.ndarray arrays"
        )

    # add both train and test to a single dataset
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = (
        np.concatenate([X_train, X_test], axis=0) if is_array else X_train + X_test
    )

    # shuffle data indices
    rng = check_random_state(random_state)

    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # ensure same classes exist in both train and test
    assert list(unique_train) == list(unique_test)

    if is_array:
        shape = list(X_train.shape)
        shape[0] = 0

    X_train = np.zeros(shape) if is_array else []
    y_train = np.zeros(0)
    X_test = np.zeros(shape) if is_array else []
    y_test = np.zeros(0)

    # for each class
    for label_index in range(len(unique_train)):
        # get the indices of all instances with this class label and shuffle them
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        rng.shuffle(indices)

        train_indices = indices[: counts_train[label_index]]
        test_indices = indices[counts_train[label_index] :]

        # extract data from corresponding indices
        train_cases = (
            all_data[train_indices]
            if is_array
            else [all_data[i] for i in train_indices]
        )
        train_labels = all_labels[train_indices]
        test_cases = (
            all_data[test_indices] if is_array else [all_data[i] for i in test_indices]
        )
        test_labels = all_labels[test_indices]

        # concat onto current data from previous loop iterations
        X_train = (
            np.concatenate([X_train, train_cases], axis=0)
            if is_array
            else X_train + train_cases
        )
        y_train = np.concatenate([y_train, train_labels], axis=None)
        X_test = (
            np.concatenate([X_test, test_cases], axis=0)
            if is_array
            else X_test + test_cases
        )
        y_test = np.concatenate([y_test, test_labels], axis=None)

    return X_train, y_train, X_test, y_test


def get_resample(X_train, y_train, X_test, y_test, resample_id: int):
    """Resample 0 is the official split; others are stratified (utils_data.py:155-165).

    Works on raw series and on feature matrices alike: the row selection only
    depends on the labels and the random state.
    """
    if resample_id == 0:
        return X_train, y_train, X_test, y_test
    return stratified_resample_data(X_train, y_train, X_test, y_test, random_state=resample_id)


def describe_split(X_train, y_train, X_test, y_test) -> dict:
    """Facts that matter for the pipeline: equal length, missing values, classes."""
    equal_length = isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray)
    arrays = [X_train, X_test] if equal_length else list(X_train) + list(X_test)
    n_nan = int(sum(np.isnan(np.asarray(a, dtype=float)).sum() for a in arrays))
    shape = X_train.shape[1:] if equal_length else (len(X_train[0]), None)
    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_channels": int(shape[0]),
        "length": int(shape[1]) if shape[1] is not None else None,
        "equal_length": equal_length,
        "n_nan": n_nan,
        "n_classes": len(np.unique(y_train)),
        "same_classes": set(np.unique(y_train)) == set(np.unique(y_test)),
    }


def read_dataset_table(path: str | Path) -> pd.DataFrame:
    """configs/datasets_tsc.csv: the 134 datasets and the authors' timings."""
    return pd.read_csv(path)
