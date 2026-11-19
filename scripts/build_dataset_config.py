"""Build configs/datasets_tsc.csv from the authors' published artifacts.

Source (read-only): references/TS2TabPFN/tsc/
- results/Tab_tsfresh.csv: the 134 datasets common to the three extractors.
- results/{Univariate,Multivariate}/HC2_accuracy.csv: archive of each dataset.
- outputs/{c22,multi,tsfresh}.csv: per-run TabPFN time (fit + predict) on an RTX 3060.
- outputs/time_{tsfresh,multi}.csv: feature extraction time (16 threads).

The per-run times are used only to balance work across sessions.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "references" / "TS2TabPFN" / "tsc"
OUT = ROOT / "configs" / "datasets_tsc.csv"

EXTRACTOR_FILES = {"c22": "c22.csv", "multirocket": "multi.csv", "tsfresh": "tsfresh.csv"}


def main() -> None:
    datasets = pd.read_csv(REF / "results" / "Tab_tsfresh.csv")["dataset"].tolist()
    multivariate = set(pd.read_csv(REF / "results" / "Multivariate" / "HC2_accuracy.csv").iloc[:, 0])

    table = pd.DataFrame({"dataset": datasets})
    table["archive"] = ["UEA" if d in multivariate else "UCR" for d in datasets]

    for name, filename in EXTRACTOR_FILES.items():
        runs = pd.read_csv(REF / "outputs" / filename)
        mean_seconds = runs.groupby("dataset")["time"].mean()
        table[f"tabpfn_s_per_run_{name}"] = table["dataset"].map(mean_seconds).round(2)

    for name, filename in {"tsfresh": "time_tsfresh.csv", "multirocket": "time_multi.csv"}.items():
        times = pd.read_csv(REF / "outputs" / filename).drop_duplicates("dataset", keep="last")
        table[f"extract_s_{name}"] = table["dataset"].map(times.set_index("dataset")["time_seconds"]).round(2)

    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"{len(table)} datasets -> {OUT}")
    print(table["archive"].value_counts().to_string())
    print("missing values per column:")
    print(table.isna().sum()[table.isna().sum() > 0].to_string() or "none")


if __name__ == "__main__":
    main()
