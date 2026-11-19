import json

import numpy as np
import pytest

from ts2tabpfn import runner
from ts2tabpfn.features import cache_path, save_feature_cache
from ts2tabpfn.runner import (
    balance_shards,
    collect_results,
    is_done,
    result_path,
    run_dataset,
    to_wide,
    write_json_atomic,
)


@pytest.fixture
def c22_cache(tmp_path, toy_split):
    """A pre-built feature cache, so the runner does not need .ts files."""
    X_train, y_train, X_test, y_test = toy_split
    features = np.concatenate([X_train, X_test]).reshape(len(y_train) + len(y_test), -1)
    save_feature_cache(cache_path(tmp_path / "features", "c22", "Toy"), features,
                       np.concatenate([y_train, y_test]), len(y_train))
    return tmp_path


def _run(base, factory, resamples, **kwargs):
    return run_dataset(
        "c22", "Toy", resamples,
        data_dir=base / "data", features_dir=base / "features", results_dir=base / "results",
        make_classifier=factory, allow_extract=False, log=lambda _: None, **kwargs,
    )


def test_runner_writes_one_file_per_resample(c22_cache, majority_factory):
    assert _run(c22_cache, majority_factory, range(3)) == 3
    for r in range(3):
        record = json.loads(result_path(c22_cache / "results", "c22", "Toy", r).read_text())
        assert record["resample"] == r and 0.0 <= record["accuracy"] <= 1.0
        assert record["n_train"] == 18 and record["n_test"] == 12


def test_runner_resumes_after_interruption(c22_cache, majority_factory, monkeypatch):
    """Simulates a dropped session in the middle of resample 2."""
    original = runner.fit_predict
    calls = {"n": 0}

    def crash_on_third(*args):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("session dropped")
        return original(*args)

    monkeypatch.setattr(runner, "fit_predict", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        _run(c22_cache, majority_factory, range(5))
    done = [r for r in range(5) if is_done(result_path(c22_cache / "results", "c22", "Toy", r))]
    assert done == [0, 1]

    monkeypatch.setattr(runner, "fit_predict", original)
    assert _run(c22_cache, majority_factory, range(5)) == 3  # only the missing ones
    assert _run(c22_cache, majority_factory, range(5)) == 0  # nothing left
    results = collect_results(c22_cache / "results")
    assert sorted(results["resample"]) == [0, 1, 2, 3, 4]


def test_corrupt_result_file_is_redone(tmp_path):
    path = result_path(tmp_path, "c22", "Toy", 0)
    path.parent.mkdir(parents=True)
    path.write_text('{"accuracy": 0.')  # truncated write
    assert not is_done(path)
    write_json_atomic(path, {"accuracy": np.float64(0.5)})
    assert is_done(path)


def test_missing_cache_raises_without_extraction(tmp_path, majority_factory):
    with pytest.raises(FileNotFoundError):
        _run(tmp_path, majority_factory, [0])


def test_parse_range():
    assert runner.parse_range("0-29") == list(range(30))
    assert runner.parse_range("0,3,5-6") == [0, 3, 5, 6]


def test_dataset_cost_adds_multirocket_extraction():
    row = {"tabpfn_s_per_run_multirocket": 10.0, "extract_s_multirocket": 2.0, "tabpfn_s_per_run_c22": 1.0}
    assert runner.dataset_cost(row, "multirocket") == 360.0
    assert runner.dataset_cost(row, "c22") == 30.0


def test_balance_shards_splits_cost_evenly():
    costs = {"a": 10, "b": 9, "c": 5, "d": 4, "e": 1, "f": 1}
    shards = balance_shards(costs, 2)
    assert sorted(sum(shards, [])) == sorted(costs)
    loads = [sum(costs[d] for d in shard) for shard in shards]
    assert max(loads) - min(loads) <= 2


def test_to_wide_matches_authors_layout(c22_cache, majority_factory):
    _run(c22_cache, majority_factory, range(2))
    wide = to_wide(collect_results(c22_cache / "results"), "c22")
    assert list(wide.columns) == ["dataset", "0", "1"]
    assert wide.loc[0, "dataset"] == "Toy"
