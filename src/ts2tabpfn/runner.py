"""Resumable experiment loop.

The unit of work is (extractor, dataset, resample). Each finished unit is
written as one JSON file, atomically, so a dropped Colab session loses at most
the unit in progress. Re-running the same command skips finished units.

Mirrors the loop in references/TS2TabPFN/code/main.py (runs x datasets x models),
with resample_id = seed + run.
"""

import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import SEED
from .data import get_resample, load_split
from .features import (
    cache_path,
    extract_full,
    extract_multirocket,
    load_feature_cache,
    save_feature_cache,
)
from .model import classification_metrics, fit_predict


def result_path(results_dir: str | Path, extractor: str, dataset: str, resample: int) -> Path:
    return Path(results_dir) / extractor / dataset / f"resample_{resample:02d}.json"


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except (json.JSONDecodeError, OSError):
        return False


def write_json_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=_to_builtin))
    os.replace(tmp, path)


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def parse_range(text: str) -> list[int]:
    """'0-29' -> [0..29]; '0,3,5' -> [0, 3, 5]."""
    values = []
    for part in text.split(","):
        if "-" in part:
            start, end = part.split("-")
            values += list(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def dataset_cost(row, extractor: str, n_resamples: int = 30) -> float:
    """Expected seconds for all resamples on the authors' RTX 3060 (configs/datasets_tsc.csv)."""
    cost = row[f"tabpfn_s_per_run_{extractor}"] * n_resamples
    if extractor == "multirocket":
        cost += row["extract_s_multirocket"] * n_resamples
    return float(cost)


def balance_shards(costs: dict[str, float], n_shards: int) -> list[list[str]]:
    """Greedy longest-processing-time split of datasets into n_shards of similar cost."""
    shards = [[] for _ in range(n_shards)]
    loads = [0.0] * n_shards
    for dataset, cost in sorted(costs.items(), key=lambda kv: (-kv[1], kv[0])):
        target = loads.index(min(loads))
        shards[target].append(dataset)
        loads[target] += cost
    return shards


def environment_info() -> dict:
    info = {"host": socket.gethostname(), "python": platform.python_version()}
    for name in ("tabpfn", "aeon", "tsfresh", "torch", "numpy", "sklearn"):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            info[name] = None
    try:
        import torch

        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["gpu"] = None
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def _official_split(extractor, dataset, data_dir, features_dir, allow_extract, n_jobs, log):
    """Raw series for MultiROCKET; cached feature matrices for catch22 and tsfresh."""
    if extractor == "multirocket":
        return load_split(data_dir, dataset)

    path = cache_path(features_dir, extractor, dataset)
    if not path.exists():
        if not allow_extract:
            raise FileNotFoundError(f"Missing feature cache {path}")
        log(f"  extracting {extractor} features for {dataset}")
        X_train, y_train, X_test, y_test = load_split(data_dir, dataset)
        start = time.time()
        features = extract_full(X_train, X_test, extractor, n_jobs=n_jobs)
        save_feature_cache(path, features, np.concatenate([y_train, y_test]), len(y_train))
        log(f"  cached {features.shape} in {time.time() - start:.1f}s")
    return load_feature_cache(path)


def run_dataset(
    extractor: str,
    dataset: str,
    resamples: Iterable[int],
    *,
    data_dir: str | Path,
    features_dir: str | Path,
    results_dir: str | Path,
    make_classifier: Callable[[int], object],
    allow_extract: bool = True,
    n_jobs: int = 1,
    env: dict | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Runs the pending resamples of one dataset. Returns how many were run."""
    pending = [r for r in resamples if not is_done(result_path(results_dir, extractor, dataset, r))]
    if not pending:
        return 0

    X_train, y_train, X_test, y_test = _official_split(
        extractor, dataset, data_dir, features_dir, allow_extract, n_jobs, log
    )
    env = env or {}

    for resample in pending:
        resample_id = SEED + resample
        Xr_train, yr_train, Xr_test, yr_test = get_resample(X_train, y_train, X_test, y_test, resample_id)

        extract_s = 0.0
        if extractor == "multirocket":
            start = time.time()
            Xr_train, Xr_test = extract_multirocket(Xr_train, Xr_test, random_state=resample_id, n_jobs=n_jobs)
            extract_s = time.time() - start

        classifier = make_classifier(resample_id)
        y_pred, fit_s, predict_s = fit_predict(classifier, Xr_train, yr_train, Xr_test)

        record = {
            "extractor": extractor,
            "dataset": dataset,
            "resample": resample,
            **classification_metrics(yr_test, y_pred),
            "n_train": len(yr_train),
            "n_test": len(yr_test),
            "n_features": int(np.asarray(Xr_train).reshape(len(yr_train), -1).shape[1]),
            "extract_s": round(extract_s, 3),
            "fit_s": round(fit_s, 3),
            "predict_s": round(predict_s, 3),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **env,
        }
        write_json_atomic(result_path(results_dir, extractor, dataset, resample), record)
        log(f"  {extractor} {dataset} r{resample:02d} acc={record['accuracy']:.4f} "
            f"({extract_s + fit_s + predict_s:.1f}s)")
    return len(pending)


def collect_results(results_dir: str | Path) -> pd.DataFrame:
    """All finished units as one long table."""
    records = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*/*/resample_*.json"))]
    return pd.DataFrame(records)


def to_wide(results: pd.DataFrame, extractor: str) -> pd.DataFrame:
    """Same layout as references/TS2TabPFN/tsc/results/Tab_*.csv: dataset x resample accuracy."""
    subset = results[results["extractor"] == extractor]
    wide = subset.pivot(index="dataset", columns="resample", values="accuracy")
    wide.columns = [str(c) for c in wide.columns]
    return wide.reset_index()
