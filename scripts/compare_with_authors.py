"""Compare our results with the authors' (references/TS2TabPFN/tsc/results/Tab_*.csv).

Input: results/Tab_<extractor>.csv produced by scripts/collect_results.py.
Output, per extractor: results/compare_<extractor>.csv (per dataset),
results/compare_<extractor>.png (scatter) and a summary in results/compare_summary.csv.

python scripts/compare_with_authors.py --extractors c22
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts2tabpfn.compare import AUTHOR_FILES, load_wide, per_dataset, plot_scatter, summarize  # noqa: E402

AUTHORS_DIR = ROOT / "references" / "TS2TabPFN" / "tsc" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractors", nargs="+", default=list(AUTHOR_FILES))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    out = Path(args.results_dir)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    rows = []
    for extractor in args.extractors:
        ours_path = out / f"Tab_{extractor}.csv"
        if not ours_path.exists():
            print(f"{extractor}: {ours_path} not found, skipping")
            continue
        table = per_dataset(load_wide(ours_path), load_wide(AUTHORS_DIR / AUTHOR_FILES[extractor]))
        table.round(4).to_csv(out / f"compare_{extractor}.csv")
        stats = summarize(table)
        plot_scatter(table, f"{extractor}-TabPFN: {stats['n_datasets']} datasets", out / f"compare_{extractor}.png")
        rows.append({"extractor": extractor, **stats, "commit": commit})
        print(f"\n{extractor}: {stats['n_datasets']} datasets | mean ours {stats['mean_ours']:.4f} "
              f"vs authors {stats['mean_authors']:.4f} (diff {stats['mean_diff']:+.4f}) | "
              f"W/T/L {stats['wins']}/{stats['ties']}/{stats['losses']} | Wilcoxon p = {stats['wilcoxon_p']:.3f}")
        print("largest differences:")
        print(table.reindex(table["diff"].abs().sort_values(ascending=False).index).head(5).round(4).to_string())

    if rows:
        pd.DataFrame(rows).round(6).to_csv(out / "compare_summary.csv", index=False)


if __name__ == "__main__":
    main()
