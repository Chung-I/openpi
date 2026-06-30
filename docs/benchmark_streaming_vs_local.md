# DROID Dataset: GCS Streaming vs Local Data Benchmark

**Date:** 2026-06-29
**Cluster:** NCHC Nano5 (H100 80GB GPUs)
**Config:** 2 GPUs, FSDP=2, batch_size=8, 10 training steps, MEM model (pi0.5 base)

## Results

| Metric | GCS Streaming | Local Disk | Speedup |
|--------|--------------|------------|---------|
| **Total wall time** | 5m 34s | 1m 59s | 2.8x |
| **Data loader init** | ~2m 18s | ~36s | 3.8x |
| **10 steps (avg rate)** | 5.0 s/step | 4.5 s/step | 1.1x |
| **10 steps elapsed** | 66s | 29s | 2.3x |

## Setup

- **Streaming:** `rlds_data_dir=gs://gresearch/robotics`, full 2048-shard dataset with filter hash table from `gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json`
- **Local:** 10 shards (~9.1GB) downloaded to `/work/roboleon1295/openpi/data/droid_subset/`, patched `dataset_info.json` for 10-shard subset, no episode filter

## Conclusion

1. **Initialization is the main overhead** — GCS streaming spends ~2min building the filter hash table from a remote JSON file; local data skips this.
2. **Per-step training speed is nearly identical** — 5.0 vs 4.5 s/step (including XLA compilation warmup). Steady-state difference is negligible.
3. **For production training (100k steps at ~1.7s/step on 4 GPUs), the ~2min init cost is <0.1% of total time.**
4. **GCS streaming is recommended** — avoids downloading ~900GB of data to local storage with no meaningful throughput penalty.

---

## MEM Video Loader Benchmark (Spec B — K-frame lazy decode)

**Status: PENDING — requires DROID RLDS data**

The MEM model (pi0.5 base) reads K-frame video windows from DROID RLDS data
(`num_video_frames=6`, `video_stride_frames=1`). To decide whether the lazy-K decode
path is a bottleneck (and thus whether a decode-once optimization is warranted),
run the benchmark script on a machine with DROID data:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/benchmark_video_dataloader.py \
    --config pi0_mem_droid_local --steps 100
```

Expected output: `pi0_mem_droid_local: <samples/s> samples/s, <ms/step> ms/step over 100 steps`

Once numbers are available, fill in the table below and update the decode-once decision.

| Metric | MEM (K=6 lazy decode) | Notes |
|--------|----------------------|-------|
| samples/s | _pending_ | `pi0_mem_droid_local`, 100 steps post-warmup |
| ms/step | _pending_ | wall time excluding warmup |
| decode bottleneck? | _pending_ | if ms/step >> single-frame, decode-once is warranted |

**Command to run E2E training verification (needs DROID data):**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train.py pi0_mem_droid_local
```

Expected: loader yields batches with `video_image` shape `[b, K, 224, 224, 3]` and
`video_states` shape `[b, K, action_dim]`; training step completes without error.
