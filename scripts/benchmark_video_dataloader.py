"""Benchmark DROID RLDS data loading: MEM (K-frame video) vs single-frame.

Usage:
    uv run python scripts/benchmark_video_dataloader.py --config pi0_mem_droid_local --steps 100

Reports samples/sec and mean per-step wall time. Use the result to decide whether
the lazy-K decode is a bottleneck (and thus whether decode-once is worth building).
Requires DROID RLDS data to be available at the config's rlds_data_dir.
"""

import argparse
import time

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main():
    p = argparse.ArgumentParser(description="Benchmark DROID RLDS video data loading.")
    p.add_argument("--config", required=True, help="TrainConfig registry name (e.g. pi0_mem_droid_local)")
    p.add_argument("--steps", type=int, default=100, help="Number of benchmark steps (after warmup)")
    p.add_argument("--warmup", type=int, default=5, help="Number of warmup steps (not timed)")
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True)
    it = iter(loader)

    # warmup
    print(f"Warming up ({args.warmup} steps)...")
    for _ in range(args.warmup):
        next(it)

    print(f"Benchmarking {args.steps} steps...")
    t0 = time.perf_counter()
    n = 0
    for _ in range(args.steps):
        next(it)
        n += cfg.batch_size
    dt = time.perf_counter() - t0

    samples_per_sec = n / dt
    ms_per_step = dt / args.steps * 1000
    print(
        f"{args.config}: {samples_per_sec:.1f} samples/s, "
        f"{ms_per_step:.1f} ms/step "
        f"over {args.steps} steps"
    )


if __name__ == "__main__":
    main()
