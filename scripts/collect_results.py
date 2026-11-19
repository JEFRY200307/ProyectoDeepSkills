"""Gather the per-resample JSON results into CSV tables under results/.

Writes results/tsc_long.csv (one row per unit) and results/Tab_<extractor>.csv
(dataset x resample accuracy, the layout of references/TS2TabPFN/tsc/results/Tab_*.csv).

python scripts/collect_results.py --results-dir /content/drive/MyDrive/ts2tabpfn/results
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts2tabpfn.runner import collect_results, to_wide  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    results = collect_results(args.results_dir)
    if results.empty:
        sys.exit("No results found.")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    results.to_csv(out / "tsc_long.csv", index=False)
    for extractor in sorted(results["extractor"].unique()):
        to_wide(results, extractor).to_csv(out / f"Tab_{extractor}.csv", index=False)
    summary = results.groupby("extractor").agg(datasets=("dataset", "nunique"), units=("accuracy", "size"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
