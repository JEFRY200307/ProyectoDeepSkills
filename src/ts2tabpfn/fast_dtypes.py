"""Faster drop-ins for two tabpfn 6.3.2 preprocessing helpers. Decision #18.

fix_dtypes (tabpfn/preprocessing/clean.py):

The original assigns casted columns back with `X[cols] = X[cols].astype(...)`.
With MultiROCKET's ~50k columns, pandas splits its internal blocks once per
column, which costs ~80% of TabPFN's fit time (profiled on ECG200: 116 of 145 s),
and the second cast is a no-op because the columns are already float64.

This version follows the same steps and returns an identical DataFrame
(same values, dtypes, categories and column order; checked in
tests/test_fast_dtypes.py against the original). It only builds the result in
one pass instead of column by column.

infer_categorical_features (tabpfn/preprocessing/type_detection.py): counts distinct
values of all columns at once instead of one pandas Series per column.

Nothing in the model changes; predictions are bit-identical (tested).
"""

from typing import Literal, Sequence

import numpy as np
import pandas as pd
from tabpfn.preprocessing import clean as _clean
from tabpfn.preprocessing import type_detection as _type_detection

# Captured at import, before install() replaces them.
ORIGINAL_FIX_DTYPES = _clean.fix_dtypes
ORIGINAL_INFER = _type_detection.infer_categorical_features


def fix_dtypes(
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int | str] | None,
    numeric_dtype: Literal["float32", "float64"] = "float64",
) -> pd.DataFrame:
    # Same input handling as the original (clean.py:35-55).
    if isinstance(X, pd.DataFrame):
        convert_dtype = True
    elif isinstance(X, np.ndarray):
        if X.dtype.kind in _clean.NUMERIC_DTYPE_KINDS:
            X = pd.DataFrame(X, copy=False, dtype=numeric_dtype)
            convert_dtype = False
        elif X.dtype.kind in _clean.OBJECT_DTYPE_KINDS:
            X = pd.DataFrame(X, copy=True)
            convert_dtype = True
        elif X.dtype.kind in _clean.STRING_DTYPE_KINDS:
            raise ValueError(f"String dtypes are not supported. Got dtype: {X.dtype}")
        else:
            raise ValueError(f"Invalid dtype for X: {X.dtype}")
    else:
        raise ValueError(f"Invalid type for X: {type(X)}")

    columns = list(X.columns)
    if cat_indices is not None:
        # Same column selection rule as the original (clean.py:57-74).
        is_numeric_indices = all(isinstance(i, (int, np.integer)) for i in cat_indices)
        columns_are_numeric = all(isinstance(col, (int, np.integer)) for col in columns)
        use_iloc = is_numeric_indices and not columns_are_numeric
        cat_columns = [columns[i] for i in cat_indices] if use_iloc else list(cat_indices)
    else:
        cat_columns = []

    if convert_dtype:
        if cat_columns:
            X = _rebuild(X, {c: X[c].astype("category") for c in cat_columns})
        X = X.convert_dtypes()
    elif cat_columns:
        X = _rebuild(X, {c: X[c].astype("category") for c in cat_columns})

    number_columns = X.select_dtypes(include=["number"]).columns
    dtypes = X.dtypes
    to_cast = [c for c in number_columns if dtypes[c] != np.dtype(numeric_dtype)]
    if to_cast:
        X = _rebuild(X, {c: X[c].astype(numeric_dtype) for c in to_cast})
    return X


def infer_categorical_features(
    X: np.ndarray,
    *,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
    provided: Sequence[int] | None = None,
) -> list[int]:
    """Vectorized tabpfn.preprocessing.type_detection.infer_categorical_features.

    Same rule; the per-column `pd.Series(col).nunique(dropna=False)` (NaN counts as
    one value) is computed for all columns at once by sorting. Non-numeric input
    falls back to the original.
    """
    if not (isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype.kind in "biuf"):
        return ORIGINAL_INFER(
            X,
            min_samples_for_inference=min_samples_for_inference,
            max_unique_for_category=max_unique_for_category,
            min_unique_for_numerical=min_unique_for_numerical,
            provided=provided,
        )
    num_distinct = count_distinct_per_column(X)
    maybe_categoricals = () if provided is None else provided
    large_enough = X.shape[0] > min_samples_for_inference
    indices = []
    for ix, n in enumerate(num_distinct):
        if ix in maybe_categoricals:
            if n <= max_unique_for_category:
                indices.append(ix)
        elif large_enough and n < min_unique_for_numerical:
            indices.append(ix)
    return indices


def count_distinct_per_column(X: np.ndarray) -> np.ndarray:
    """Distinct values per column, NaN counted as one extra value (pandas nunique(dropna=False))."""
    if X.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=int)
    S = np.sort(X, axis=0)  # NaN sorts last
    is_nan = np.isnan(S) if S.dtype.kind == "f" else np.zeros(S.shape, dtype=bool)
    valid = ~is_nan
    changes = (S[1:] != S[:-1]) & valid[1:]  # a new value among the non-NaN rows
    n_valid_distinct = valid[0].astype(int) + changes.sum(axis=0)
    return n_valid_distinct + is_nan.any(axis=0)


def _rebuild(X: pd.DataFrame, replaced: dict) -> pd.DataFrame:
    """X with some columns replaced, built in one pass and in the original column order."""
    data = {c: replaced[c] if c in replaced else X[c] for c in X.columns}
    out = pd.DataFrame(data, index=X.index, copy=False)
    out.columns = X.columns
    return out


def install() -> None:
    """Replace both functions in every tabpfn module that imported them by name."""
    import tabpfn.base
    import tabpfn.classifier
    import tabpfn.preprocessing.initialization
    import tabpfn.regressor

    for module in (_clean, tabpfn.base, tabpfn.classifier,
                   tabpfn.preprocessing.initialization, tabpfn.regressor):
        module.fix_dtypes = fix_dtypes
    for module in (_type_detection, tabpfn.preprocessing.initialization):
        module.infer_categorical_features = infer_categorical_features


def uninstall() -> None:
    """Restore tabpfn's original functions (used by the tests)."""
    import tabpfn.base
    import tabpfn.classifier
    import tabpfn.preprocessing.initialization
    import tabpfn.regressor

    for module in (_clean, tabpfn.base, tabpfn.classifier,
                   tabpfn.preprocessing.initialization, tabpfn.regressor):
        module.fix_dtypes = ORIGINAL_FIX_DTYPES
    for module in (_type_detection, tabpfn.preprocessing.initialization):
        module.infer_categorical_features = ORIGINAL_INFER
