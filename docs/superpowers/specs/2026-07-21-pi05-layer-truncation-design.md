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

### Gate 0 — speedup: already measured, so confirm rather than gate

A prior latency profile of `pi05_libero` on an RTX 5090 (artifact "pi0.5 Latency
Profile", 40-cell depth x denoise sweep) already timed truncated-depth models
with dropped trained weights. Its own footnote states the case exactly: "timing
is faithful, task success is not measured." That is this gate, already run.

Fitted latency model, R^2 = 0.999:

```
L(K, S) = 16.21 + 1.719*K + 0.039*S + 0.1024*(K*S)   ms
```

At the deployed operating point (`num_steps=10`, `pi0.py:222`; 968 prefix
tokens):

| K | latency | speedup |
|---|---|---|
| 18 | 66.0 ms | 1.00x |
| **6** | **33.1 ms** | **1.996x** |
| 3 | 24.8 ms | 2.66x |
| 1 | 19.3 ms | 3.41x |

The profile transfers from LIBERO to DROID essentially unchanged, because both
present the model with a 968-token prefix.

The DROID dataset itself has three cameras — `exterior_image_1_left`,
`exterior_image_2_left`, `wrist_image_left` — but the loader samples only **one
of the two exteriors per trajectory**, pairing it with the wrist
(`droid_rlds_dataset.py:129-136`; the random choice acts as a viewpoint
augmentation). Those 2 images are then padded to 3 model slots, the third
zero-filled and masked (`droid_policy.py:49-51`), giving 3 x 256 + 200 = 968
prefix tokens — the same as LIBERO. `action_horizon=16` versus LIBERO's 10 sits inside the flat region of
the measured action-token curve (22.9 ms at L=2 rising only to 24.5 ms at L=50),
so it is within noise.

**Why 6 layers is a defensible stopping point.** The fixed floor F = 16.21 ms —
SigLIP encode, embedding, host overhead — is untouchable by layer truncation. At
K=6 it is already 49% of the 33.1 ms total. Truncation's ceiling is therefore
~4.0x as K approaches 0, and cutting a further 6 -> 3 layers buys only another
1.33x for presumably far more accuracy damage. Six layers captures roughly half
the available headroom.

**What remains to do here:** re-run `bench_inference.py` on the actual
`pi05_droid_jointpos` arms to confirm the transfer, and record the measured
numbers next to the predictions above. This is a cheap confirmation on the local
RTX 5090, not a stop/go gate — the 1.5x viability threshold is already cleared
with margin.

Note for cross-reading eval logs: RoboLab's 4-env arena reports `infer_ms` at
roughly 4x these single-observation numbers, at the same ratio.

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

Speedup is no longer an open variable — it is predicted at 1.996x and merely
confirmed (Gate 0 above). The success criterion is therefore purely about
accuracy:

**A clear win is arm A retaining at least 80% of arm B's overall success rate**
— relative, not percentage points: if arm B scores 45%, arm A must score at
least 36%. That buys a 2x speedup for a fifth of the task competence.

Falling short is a publishable negative result about how much depth pi0.5's
DROID competence needs, not a failed project. Report the (speedup, success-rate)
pair either way; neither number means much alone.

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

**~~SigLIP bounds the speedup~~ — retired, now quantified.** This was the main
risk before the latency profile. SigLIP does bound the speedup, at a 16.21 ms
floor, but 6 layers still clears 2x. The residual risk is only that the
LIBERO-measured profile fails to transfer to DROID, which the Gate 0
confirmation run checks directly and cheaply.

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
- Artifact "pi0.5 Latency Profile" (`12417096-394b-47a9-9c51-8d0f4e5d326b`) —
  the depth x denoise sweep, token-speed curve, and fitted cost models this
  spec's latency predictions come from.

## Deferred, informed by the latency profile

Two orthogonal levers surfaced and are deliberately not pursued here. Keeping
them out preserves this experiment's single variable.

**The dead third image slot — ~6.9 ms, ~10%, no retraining.** DROID has three
cameras, but only two ever reach the model, in *both* code paths: training
samples one of the two exteriors per trajectory
(`droid_rlds_dataset.py:129-136`), and inference pins one via an operator CLI
flag, with upstream asserting it and commenting "we only use one external camera
for the policy" (`examples/droid/main.py:74-77`). The third slot is therefore
zero-filled and masked on every DROID sample ever served, yet still costs a full
SigLIP encode of 256 zero-pixel tokens.

Open question before acting on it: whether dropping those 256 masked tokens is
an exact no-op on outputs, or whether it shifts the position indices of the
prompt tokens that follow. If the former, it is provable offline by a numerical
equivalence test with zero GPU-hours. Settle that first; the answer determines
whether this is free or needs its own finetune.

**Longer action chunks.** The denoise loop is flat in chunk length out to ~50
tokens, so predicting more actions per inference is nearly free (7.1 -> 1.5 ms
per action across the measured range). At `action_horizon=16` we sit well inside
that flat region. A throughput lever independent of depth, bounded by how long
the policy stays accurate open-loop.

Stacking all three would ceiling around 2.2x plus the chunk-length throughput
gain, but each is independently testable and they should stay that way.
