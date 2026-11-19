"""Comparison against the authors' published accuracies (Phase 2 verification).

The authors' tables are references/TS2TabPFN/tsc/results/Tab_{c22,multi,tsfresh}.csv:
one row per dataset, columns "0".."29" with the accuracy of each resample.
The paper compares methods by mean accuracy over the 30 resamples and Wilcoxon
signed-rank tests over datasets (§4.2).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

AUTHOR_FILES = {"c22": "Tab_c22.csv", "multirocket": "Tab_multi.csv", "tsfresh": "Tab_tsfresh.csv"}
RESAMPLE_COLUMNS = [str(i) for i in range(30)]


def load_wide(path: str | Path) -> pd.DataFrame:
    """A Tab_*.csv file (ours or the authors') indexed by dataset."""
    table = pd.read_csv(path).set_index("dataset")
    table.columns = [str(c) for c in table.columns]
    return table


def per_dataset(ours: pd.DataFrame, authors: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    """Mean accuracy per dataset, ours vs. the authors', on the datasets both have.

    With complete_only, only datasets where we have all 30 resamples are kept.
    """
    ours = ours.reindex(columns=RESAMPLE_COLUMNS)
    n_ours = ours.notna().sum(axis=1)
    common = ours.index.intersection(authors.index)
    if complete_only:
        common = common[n_ours.loc[common] == len(RESAMPLE_COLUMNS)]
    table = pd.DataFrame({
        "ours": ours.loc[common].mean(axis=1),
        "authors": authors.loc[common, RESAMPLE_COLUMNS].mean(axis=1),
        "n_resamples": n_ours.loc[common],
    })
    table["diff"] = table["ours"] - table["authors"]
    return table.sort_index()


def summarize(table: pd.DataFrame, tie_tol: float = 1e-4) -> dict:
    """Mean accuracies, win/tie/loss for us vs. the authors and the Wilcoxon p-value."""
    diff = table["diff"].to_numpy()
    wins, ties = int((diff > tie_tol).sum()), int((np.abs(diff) <= tie_tol).sum())
    nonzero = diff[np.abs(diff) > tie_tol]
    p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 1 else 1.0
    return {
        "n_datasets": len(table),
        "mean_ours": float(table["ours"].mean()),
        "mean_authors": float(table["authors"].mean()),
        "mean_diff": float(diff.mean()) if len(diff) else 0.0,
        "max_abs_diff": float(np.abs(diff).max()) if len(diff) else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": len(diff) - wins - ties,
        "wilcoxon_p": p_value,
    }


def plot_scatter(table: pd.DataFrame, title: str, path: str | Path) -> None:
    """Our mean accuracy against the authors' per dataset, with the identity line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="y = x")
    ax.scatter(table["authors"], table["ours"], s=14, alpha=0.8)
    ax.set_xlabel("Autores: accuracy media (30 resamples)")
    ax.set_ylabel("Nuestra: accuracy media (30 resamples)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
