"""Check that the authors' TSC data has the 134 datasets and describe each one.

Writes results/data_check.csv: sizes, channels, length, equal length, NaN count, classes.
The paper (§4.1) says missing values were linearly interpolated, but the reference
code has no interpolation step; this check tells whether the data needs it.

python scripts/check_data.py --data-dir data/raw/tsc
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts2tabpfn.data import describe_split, load_split, read_dataset_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default=str(ROOT / "results" / "data_check.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in read_dataset_table(ROOT / "configs" / "datasets_tsc.csv")["dataset"]:
        try:
            rows.append({"dataset": dataset, **describe_split(*load_split(args.data_dir, dataset)), "error": None})
        except Exception as error:
            rows.append({"dataset": dataset, "error": f"{type(error).__name__}: {error}"})
        print(rows[-1], flush=True)

    report = pd.DataFrame(rows)
    report.to_csv(args.out, index=False)
    ok = report["error"].isna()
    print(f"\nloaded {ok.sum()}/{len(report)}")
    print(f"unequal length: {report.loc[ok & ~report['equal_length'].astype(bool), 'dataset'].tolist()}")
    print(f"with NaN: {report.loc[ok & (report['n_nan'] > 0), 'dataset'].tolist()}")
    print(f"classes differ train/test: {report.loc[ok & ~report['same_classes'].astype(bool), 'dataset'].tolist()}")
    print(f"errors: {report.loc[~ok, 'dataset'].tolist()}")


if __name__ == "__main__":
    main()
