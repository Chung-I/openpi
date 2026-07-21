# Accelerating pi0.5 by Layer Truncation

**Date:** 2026-07-21
**Branch:** `worktree-pi05-layer-truncation` (off `origin/main`)
**Status:** design approved, ready for implementation planning

## Problem

pi0.5 inference is dominated by an 18-layer Gemma stack that runs once over the
prefix and again on every flow-denoising step. Cutting the stack to 6 layers
would cut that work threefold. The question is what it costs in task success.

This project builds the truncation mechanism, measures the real speedup, and
measures the accuracy cost on the RoboLab DROID suite already used for
`docs/eval/RESULTS.md`.

## Scope

In scope: truncating the Gemma stack of `pi05_droid_jointpos`, a LoRA finetune
to recover accuracy, and a two-arm RoboLab evaluation.

Out of scope, deliberately: teacher distillation, truncating SigLIP, full-rank
finetuning, and applying truncation to the MEM video-encoder model. Each is a
follow-up that this project's results should inform, not precede.

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | vanilla `pi05_droid_jointpos` | Layer count is the only variable; MEM code stays out of the picture entirely. |
| Layers kept | `(0, 3, 7, 11, 14, 17)`, configurable | Strided and inclusive of layer 17, so `final_norm` and the flow head still see the activations they were trained on. The field takes any index list, so alternatives need no code change. |
| Objective | plain pi0.5 flow loss | Reuses `scripts/train.py` untouched. Distillation is the escalation, not the starting point. |
| Trainable | LoRA adapters only | Matches the validated d-eval recipe. |
| Control | depth-18 LoRA arm | Isolates truncation from finetuning. |

## Architecture

### How truncation works

The 18 blocks are one `nn.scan` shared by both experts (`gemma.py:376`), and
`gemma.py:352` asserts both experts have equal depth. Parameters therefore carry
a leading depth axis. Building the model at `depth=6` and gathering 6 slices out
of the checkpoint yields a genuinely 6-layer model: no masking, no wasted
compute, and both towers truncated in one step.

Verified against a local checkpoint, the parameter layout is:

```
(4, 8, 16, 64)   PaliGemma/llm/layers/attn/attn_vec_einsum/w      scanned (axis 0 = depth)
(4, 8, 16, 64)   PaliGemma/llm/layers/attn/attn_vec_einsum_1/w    scanned, action expert
(4, 2, 64, 128)  PaliGemma/llm/layers/mlp/gating_einsum           scanned
(4, 64)          PaliGemma/llm/layers/pre_attention_norm/scale    scanned
(64,)            PaliGemma/llm/final_norm/scale                   not scanned
(257152, 64)     PaliGemma/llm/embedder/input_embedding           not scanned
```

The gather rule is exact rather than heuristic: **every flat key under
`PaliGemma/llm/layers/` is indexed `arr[list(keep_layers)]` on axis 0; every
other key passes through unchanged.** `_1`-suffixed parameters are the action
expert, so a single pass truncates both towers.

### Components

**1. `Pi0Config.keep_layers: tuple[int, ...] | None = None`**
(`src/openpi/models/pi0_config.py`)

When `None`, behaviour is unchanged. When set, `Pi0.__init__`
(`src/openpi/models/pi0.py:69-70`) builds both gemma configs with
`depth=len(keep_layers)` via `dataclasses.replace`, which satisfies the
equal-depth assert by construction. Indices are into the original 18-layer
stack, 0-based. `__post_init__` validates: sorted ascending, unique, non-empty,
and every index within `[0, base_depth)`, where `base_depth` is read from
`_gemma.get_config(self.paligemma_variant).depth` rather than hard-coded.

Scope note: the JAX `Pi0` path only. `pi0_fast.py` and the PyTorch path are
untouched.

**2. `weight_loaders.LayerSubsetWeightLoader(params_path, keep_layers)`**
(`src/openpi/training/weight_loaders.py`)

Restores the checkpoint, applies the gather rule above, then delegates to the
existing `_merge_params(..., missing_regex=".*lora.*")` so freshly initialised
LoRA adapters — which already carry the depth-6 leading axis — fill in.

**3. `TrainConfig` entries** (`src/openpi/training/config.py`)

`pi05_droid_jointpos_trunc6` and `pi05_droid_jointpos_trunc18`, differing only
in `keep_layers`.

**4. `scripts/bench_inference.py`**

Times `policy.infer()` on synthetic DROID observations, reporting p50/p95 over N
calls after warmup, plus a SigLIP-only timing. Takes a config name so it can time
any arm.

### Freeze filter — a deliberate deviation from stock

`Pi0Config.get_freeze_filter()` freezes `.*llm.*` except LoRA
(`pi0_config.py:87-119`), which leaves SigLIP **trainable**. Project memory
records that this exact filter fully trained SigLIP on DROID and collapsed the
policy to 0% with a timid arm. So both arms set the filter explicitly:

```python
freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*"))
```

Trainable: LoRA adapters only — rank 16 attn/ffn on the backbone, rank 32 on the
action expert. Frozen: SigLIP `img`, all LLM and expert base weights, the final
norms, and the action in/out projections. This is the d-eval recipe minus
`video_img`.

### Serving needs no new code

`RLDSDroidDataConfig` with `action_space=JOINT_POSITION` already pushes
`AbsoluteActions(make_bool_mask(7, -1))` onto the output side
(`config.py:403-408`), so a trained checkpoint served through the stock
`scripts/serve_policy.py` emits absolute joint targets. RoboLab evaluates it with
its stock `run.py` and `Pi0DroidJointposClient` — no `MemSessionPolicy`, no
`run_mem.py`.

## Experiment protocol

### Gate 0 — measure the speedup ceiling before any training

Truncating the LLM does nothing to SigLIP, which is unchanged and not free. Run
`bench_inference.py` on the sliced-but-untrained depth-6 model against depth-18,
and report the SigLIP-only fraction that bounds any achievable speedup.

No training required. A 3B bf16 model needs roughly 6 GB for inference, so this
runs on the local RTX 5090; the recorded OOM applies to 3B *gradients*.

**Gate:** if end-to-end speedup is under 1.5x, stop and re-scope. Spending 2 x
20k GPU-hours to buy a 1.3x speedup is not worth it, and the finding itself is
the useful result.

### Gate 1 — coherence check on the untrained slice

Compute the pi0.5 flow loss on one real DROID batch for depth-18 versus the
untrained depth-6 model. Seconds to run, and it catches a wrong gather axis or a
misordered index list before a 20k-step run. The untrained slice is expected to
be much worse than depth-18; the check is that the number is finite, stable, and
in a sane range, not that it is good.

### Training

Two arms, identical in every respect except `keep_layers`:

- **Arm A** `pi05_droid_jointpos_trunc6` — `keep_layers=(0, 3, 7, 11, 14, 17)`
- **Arm B** `pi05_droid_jointpos_trunc18` — `keep_layers=tuple(range(18))`

Arm B doubles as the identity-path test on real hardware: if the loader is
correct, arm B is mathematically the ordinary model.

| Setting | Value |
|---|---|
| base weights | `gs://openpi-assets-simeval/pi05_droid_jointpos/params` |
| assets | `gs://openpi-assets-simeval/pi05_droid_jointpos/assets`, `asset_id="droid"` |
| data | DROID RLDS v1.0.1, `gs://gresearch/robotics`, `JOINT_POSITION`, filtered by `droid_sample_ranges_v1_0_1.json` |
| model | `pi05=True`, `action_dim=32`, `action_horizon=16`, both towers `_lora` |
| schedule | 20k steps, batch 128, `fsdp_devices=4`, cosine warmup 1k, peak 5e-5, `decay_lr` 5e-5 |
| checkpoints | `save_interval=2500`, `keep_period=5000` — retains 5k/10k/15k/20k permanently, so the two arms can be read at matched steps |
| data loading | `num_workers=0` — required by the RLDS path, which parallelizes internally via `tf.data`; forking torch workers on top of it conflicts (see `config.py:859`) |
| tracking | wandb, project `layer-truncation` |
| host | NCHC `8gpus`, account MST114563, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` |

The norm-stats path must be the jointpos assets. Reusing velocity stats on
position targets distorts the flow loss, which project memory records as a
diagnosed cause of a timid policy.

20k steps rather than 10k: memory records 10k as a "too few steps" root cause,
and the successful d-eval runs used 20k.

Both arms run concurrently on one 8-GPU allocation, 4 GPUs each, following the
pattern in `scripts/nchc/train_mem_video_eval.sbatch`.

### Evaluation

RoboLab, the same 5 tasks at 16 episodes each used for `docs/eval/RESULTS.md`:
BananaInBowl, BagelsOnPlate, BowlInBin, MarkerInMug, MustardInRightBin. Results
extend that table alongside the published vanilla 40%, K=1 42%, K=6 52%.

Two deltas are reported:

- **A minus B** — the effect of truncation alone, with finetuning held constant.
- **A minus vanilla** — the end-to-end practical question, confounded with
  finetuning and labelled as such.

Latency is reported next to success rate. Neither number means much alone.

### Success criteria

The deliverable is the (speedup, success-rate) pair, reported honestly whatever
it shows. A clear win is **at least 2x measured end-to-end speedup while
retaining at least 80% of arm B's overall success rate** — relative, not
percentage points: if arm B scores 45%, arm A must score at least 36%. Falling
short is a
publishable negative result about how much depth pi0.5's DROID competence needs,
not a failed project.

At 16 episodes per task the per-task confidence intervals are wide, exactly as
noted in `RESULTS.md`. Consistency of direction across tasks carries more weight
than the aggregate. If the aggregate delta is small and the direction is mixed,
re-run the two tasks with the most headroom at 50-100 episodes.

## Testing

Unit tests, run locally, no GPU required:

- `LayerSubsetWeightLoader` gathers the correct indices from synthetic params,
  and leaves non-scanned keys untouched.
- Identity guard: `keep_layers=tuple(range(18))` reproduces
  `CheckpointWeightLoader` exactly. This is the single most valuable test —
  it catches axis and ordering bugs.
- A model built with `keep_layers` of length 6 reports depth 6 on both towers,
  and a forward pass produces correctly shaped actions.
- `keep_layers=None` leaves the model bit-identical to today's behaviour.
- `__post_init__` rejects unsorted, duplicated, and out-of-range indices.

Integration checks are Gates 0 and 1 above, plus arm B's identity property on
real hardware.

## Risks

**SigLIP bounds the speedup.** The LLM is not the only cost, and truncation does
not touch the vision encoder. Gate 0 exists to surface this before any training
spend.

**LoRA underfits a truncated backbone.** Rank-16 adapters on 6 frozen blocks may
not bridge a threefold depth cut. Keeping layer 17 preserves the head interface,
which is the main mitigation. Escalation order if arm A trains poorly: unfreeze
the action in/out projections, then full-rank finetune the 6 kept blocks, then
add teacher distillation.

**Norm-stats mismatch.** Pinned explicitly to the jointpos assets above.

**NCHC queue contention.** Memory records another account holding 32-56 GPUs on
MST114563, causing long waits. Both arms share one 8-GPU allocation to need a
single scheduling slot rather than two.

## References

- `docs/eval/RESULTS.md` on `mem-video-encoder-d-eval` — baseline numbers and
  eval methodology.
- Project memory `project_droid_jointpos_action_space.md` — the jointpos action
  space, the SigLIP freeze failure, and the norm-stats failure.
- `src/openpi/models/gemma.py:352,376` — the shared-depth scan.
- `src/openpi/training/config.py:403-408` — the absolute-actions output
  transform.
