"""Run TS2TabPFN on the TSC datasets. Resumable: finished units are skipped.

Examples
--------
# Session 1 of 2, MultiROCKET, all 134 datasets balanced by the authors' timings:
python scripts/run_tsc.py --extractor multirocket --shard 0 --num-shards 2 \
    --data-dir /content/drive/MyDrive/ts2tabpfn/data/tsc \
    --features-dir /content/drive/MyDrive/ts2tabpfn/features \
    --results-dir /content/drive/MyDrive/ts2tabpfn/results

# A few datasets and resamples only (calibration):
python scripts/run_tsc.py --extractor c22 --datasets ECG200 ECG5000 --resamples 0-2 ...
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ts2tabpfn import EXTRACTORS, N_RESAMPLES  # noqa: E402
from ts2tabpfn.data import read_dataset_table  # noqa: E402
from ts2tabpfn.model import default_device, make_classifier  # noqa: E402
from ts2tabpfn.runner import (  # noqa: E402
    balance_shards,
    dataset_cost,
    environment_info,
    parse_range,
    run_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractor", required=True, choices=EXTRACTORS)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--datasets", nargs="*", help="default: all 134 in configs/datasets_tsc.csv")
    parser.add_argument("--resamples", default=f"0-{N_RESAMPLES - 1}")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--no-extract", action="store_true",
                        help="fail instead of extracting missing catch22/tsfresh caches")
    args = parser.parse_args()

    table = read_dataset_table(ROOT / "configs" / "datasets_tsc.csv").set_index("dataset")
    datasets = args.datasets or list(table.index)
    unknown = sorted(set(datasets) - set(table.index))
    if unknown:
        parser.error(f"not in the 134 datasets: {unknown}")

    costs = {d: dataset_cost(table.loc[d], args.extractor) for d in datasets}
    shard = balance_shards(costs, args.num_shards)[args.shard]
    shard.sort(key=lambda d: costs[d])  # cheap datasets first: results show up early

    device = args.device or default_device()
    env = environment_info()
    print(f"extractor={args.extractor} shard={args.shard}/{args.num_shards} datasets={len(shard)} "
          f"device={device} gpu={env.get('gpu')} est={sum(costs[d] for d in shard) / 3600:.1f}h (RTX 3060)")

    resamples = parse_range(args.resamples)
    failed = []
    for i, dataset in enumerate(shard, 1):
        print(f"[{i}/{len(shard)}] {dataset}", flush=True)
        start = time.time()
        try:
            n = run_dataset(
                args.extractor, dataset, resamples,
                data_dir=args.data_dir, features_dir=args.features_dir, results_dir=args.results_dir,
                make_classifier=lambda rs: make_classifier(rs, device),
                allow_extract=not args.no_extract, n_jobs=args.n_jobs, env=env,
                log=lambda msg: print(msg, flush=True),
            )
            print(f"  done: {n} new resamples in {time.time() - start:.0f}s", flush=True)
        except Exception as error:  # keep going; the failure is reported at the end
            failed.append(dataset)
            print(f"  [ERROR] {dataset}: {type(error).__name__}: {error}", flush=True)

    print(f"\nFinished. Failed datasets: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
