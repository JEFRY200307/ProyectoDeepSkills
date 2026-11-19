import numpy as np
import pandas as pd
import pytest

from ts2tabpfn.compare import RESAMPLE_COLUMNS, load_wide, per_dataset, plot_scatter, summarize


def _wide(values: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(values, orient="index", columns=RESAMPLE_COLUMNS)


def test_per_dataset_keeps_complete_common_datasets():
    ours = _wide({"A": [0.9] * 30, "B": [0.5] * 30, "C": [0.7] * 30})
    ours.loc["B", "29"] = np.nan  # incomplete
    authors = _wide({"A": [0.8] * 30, "B": [0.5] * 30, "D": [0.1] * 30})
    table = per_dataset(ours, authors)
    assert list(table.index) == ["A"]
    assert table.loc["A", "diff"] == pytest.approx(0.1)
    assert list(per_dataset(ours, authors, complete_only=False).index) == ["A", "B"]


def test_summarize_counts_and_wilcoxon():
    rng = np.random.default_rng(0)
    authors = rng.uniform(0.5, 0.9, 20)
    table = pd.DataFrame({"ours": authors + 0.01, "authors": authors})
    table["diff"] = table["ours"] - table["authors"]
    stats = summarize(table)
    assert stats["wins"] == 20 and stats["losses"] == 0 and stats["ties"] == 0
    assert stats["wilcoxon_p"] < 0.001
    same = table.assign(ours=table["authors"], diff=0.0)
    stats_same = summarize(same)
    assert stats_same["ties"] == 20 and stats_same["wilcoxon_p"] == 1.0


def test_load_wide_and_plot(tmp_path):
    path = tmp_path / "Tab_x.csv"
    _wide({"A": [0.9] * 30}).rename_axis("dataset").reset_index().to_csv(path, index=False)
    table = load_wide(path)
    assert list(table.columns) == RESAMPLE_COLUMNS
    comparison = per_dataset(table, table)
    plot_scatter(comparison, "test", tmp_path / "plot.png")
    assert (tmp_path / "plot.png").stat().st_size > 0
