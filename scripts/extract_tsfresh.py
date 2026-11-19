"""Extract tsfresh features once per dataset (official train+test) for the cache.

Meant for a CPU machine (AWS c6i.8xlarge, decision #8). Resumable: datasets whose
cache file already exists are skipped. Cheap datasets go first so the GPU sessions
can start early; EigenWorms (~33 h for the authors) goes last.

python scripts/extract_tsfresh.py --data-dir data/raw/tsc --features-dir data/processed/features \
    --n-jobs 32 --shutdown
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts2tabpfn.data import load_split, read_dataset_table  # noqa: E402
from ts2tabpfn.features import (  # noqa: E402
    cache_path,
    extract_full,
    extract_tsfresh_chunked,
    save_feature_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--datasets", nargs="*", help="default: all 134, cheapest first")
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--shutdown", action="store_true", help="power off the machine when finished")
    parser.add_argument("--chunks", type=int, default=1,
                        help="split each dataset into this many row blocks (decision #19); "
                             "blocks are saved under <features-dir>/tsfresh_parts/<dataset>/")
    parser.add_argument("--chunk-ids", type=int, nargs="*",
                        help="only these blocks (0-based), e.g. to split the work across machines")
    args = parser.parse_args()

    table = read_dataset_table(ROOT / "configs" / "datasets_tsc.csv")
    if args.datasets:
        table = table[table["dataset"].isin(args.datasets)]
    datasets = table.sort_values("extract_s_tsfresh")["dataset"].tolist()

    failed = []
    for i, dataset in enumerate(datasets, 1):
        path = cache_path(args.features_dir, "tsfresh", dataset)
        if path.exists():
            print(f"[{i}/{len(datasets)}] {dataset}: cached, skipping", flush=True)
            continue
        print(f"[{i}/{len(datasets)}] {dataset}: extracting", flush=True)
        start = time.time()
        try:
            X_train, y_train, X_test, y_test = load_split(args.data_dir, dataset)
            if args.chunks > 1:
                parts_dir = Path(args.features_dir) / "tsfresh_parts" / dataset
                features = extract_tsfresh_chunked(
                    X_train, X_test, parts_dir, args.chunks, args.chunk_ids, n_jobs=args.n_jobs,
                    log=lambda msg: print(msg, flush=True),
                )
                if features is None:
                    done = len(list(parts_dir.glob("part_*.npz")))
                    print(f"  {done}/{args.chunks} chunks ready; run again for the rest", flush=True)
                    continue
            else:
                features = extract_full(X_train, X_test, "tsfresh", n_jobs=args.n_jobs)
            save_feature_cache(path, features, np.concatenate([y_train, y_test]), len(y_train))
            print(f"  {features.shape} in {time.time() - start:.0f}s", flush=True)
        except Exception as error:
            failed.append(dataset)
            print(f"  [ERROR] {type(error).__name__}: {error}", flush=True)

    print(f"\nFinished. Failed: {failed or 'none'}", flush=True)
    if args.shutdown:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    main()
