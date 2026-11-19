import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ts2tabpfn.data import (
    describe_split,
    get_resample,
    load_split,
    read_dataset_table,
    stratified_resample_data,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_UTILS = ROOT / "references" / "TS2TabPFN" / "code" / "utils" / "utils_data.py"


def test_resample_zero_is_official_split(toy_split):
    out = get_resample(*toy_split, resample_id=0)
    for got, expected in zip(out, toy_split):
        np.testing.assert_array_equal(got, expected)


def test_resample_keeps_sizes_and_class_counts(toy_split):
    X_train, y_train, X_test, y_test = toy_split
    Xr_train, yr_train, Xr_test, yr_test = get_resample(*toy_split, resample_id=3)
    assert Xr_train.shape == X_train.shape and Xr_test.shape == X_test.shape
    for label in np.unique(y_train):
        assert (yr_train == label).sum() == (y_train == label).sum()
        assert (yr_test == label).sum() == (y_test == label).sum()


def test_resample_is_deterministic_and_changes_with_seed(toy_split):
    a = get_resample(*toy_split, resample_id=5)
    b = get_resample(*toy_split, resample_id=5)
    c = get_resample(*toy_split, resample_id=6)
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[0], c[0])


def test_resample_is_a_partition_of_all_cases(toy_split):
    X_train, _, X_test, _ = toy_split
    Xr_train, _, Xr_test, _ = get_resample(*toy_split, resample_id=2)
    original = np.sort(np.concatenate([X_train, X_test]).reshape(len(X_train) + len(X_test), -1), axis=0)
    resampled = np.sort(np.concatenate([Xr_train, Xr_test]).reshape(len(Xr_train) + len(Xr_test), -1), axis=0)
    np.testing.assert_array_equal(original, resampled)


@pytest.mark.skipif(not REFERENCE_UTILS.exists(), reason="references/ not cloned")
def test_matches_reference_implementation(toy_split):
    spec = importlib.util.spec_from_file_location("ref_utils_data", REFERENCE_UTILS)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for seed in (1, 7, 29):
        ours = stratified_resample_data(*toy_split, random_state=seed)
        theirs = ref.stratified_resample_data(*toy_split, random_state=seed)
        for got, expected in zip(ours, theirs):
            np.testing.assert_array_equal(got, expected)


def test_load_split_reads_ts_files(tmp_path):
    folder = tmp_path / "Toy"
    folder.mkdir()
    header = "@problemName Toy\n@timeStamps false\n@univariate true\n@equalLength true\n@seriesLength 4\n@classLabel true a b\n@data\n"
    (folder / "Toy_TRAIN.ts").write_text(header + "1,2,3,4:a\n4,3,2,1:b\n")
    (folder / "Toy_TEST.ts").write_text(header + "1,2,3,5:a\n5,3,2,1:b\n")
    X_train, y_train, X_test, y_test = load_split(tmp_path, "Toy")
    assert X_train.shape == (2, 1, 4)
    assert list(y_test) == ["a", "b"]


def test_describe_split_reports_shape_nans_and_classes(toy_split):
    X_train, y_train, X_test, y_test = toy_split
    X_test = X_test.copy()
    X_test[0, 0, :3] = np.nan
    info = describe_split(X_train, y_train, X_test, y_test)
    assert info == {
        "n_train": 18, "n_test": 12, "n_channels": 2, "length": 60, "equal_length": True,
        "n_nan": 3, "n_classes": 3, "same_classes": True,
    }


def test_dataset_table_has_134_datasets():
    table = read_dataset_table(ROOT / "configs" / "datasets_tsc.csv")
    assert len(table) == 134
    assert table["dataset"].is_unique
    assert set(table["archive"]) == {"UCR", "UEA"}
    assert {"ECG200", "ECG5000", "ECGFiveDays", "TwoLeadECG"} <= set(table["dataset"])
