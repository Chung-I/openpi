# pi0.5 Layer Truncation — Reproduction

Cuts the shared 18-layer gemma stack to 6 layers `(0, 3, 7, 11, 14, 17)` for a ~2x
inference speedup, then recovers accuracy with a LoRA finetune on DROID joint-position
data. Design: `docs/superpowers/specs/2026-07-21-pi05-layer-truncation-design.md`.

## Arms

| config | layers | role |
|---|---|---|
| `pi05_droid_jointpos_trunc6` | 6 of 18 | the truncated model under test |
| `pi05_droid_jointpos_trunc18` | all 18 | control; isolates truncation from finetuning |

Everything except `keep_layers` is identical, enforced by
`src/openpi/training/config_test.py::test_arms_share_every_hyperparameter_but_depth`.

## 1. Latency (Gate 0)

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py \
  --config-name pi05_droid_jointpos_trunc18
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py \
  --config-name pi05_droid_jointpos_trunc6
```

Predicted at 10 denoise steps and 968 prefix tokens: 66.0 ms -> 33.1 ms, 1.996x.
Runs on a single RTX 5090 — inference of a 3B bf16 model needs ~6 GB.

| config | predicted p50 | measured p50 |
|---|---|---|
| `pi05_droid_jointpos_trunc18` | 66.0 ms | _fill in_ |
| `pi05_droid_jointpos_trunc6` | 33.1 ms | _fill in_ |

## 2. Train

```bash
sbatch scripts/nchc/train_truncation.sbatch
```

Both arms, 4 GPUs each, on one 8-GPU allocation. 20k steps, batch 128. The script gates
on `scripts/check_truncation.py` first, so a loader/model depth mismatch fails in seconds
rather than after the checkpoint downloads.

Watch the first 100 steps in wandb (project `layer-truncation`). Arm A starting far above
arm B and failing to descend means LoRA is not bridging the depth cut — see Escalation.

## 3. Serve

Stock serving; no MEM session server is involved. `RLDSDroidDataConfig` with
`JOINT_POSITION` already appends `AbsoluteActions(make_bool_mask(7, -1))`, so the served
policy emits absolute joint targets.

```bash
uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi05_droid_jointpos_trunc6 \
  --policy.dir=checkpoints/pi05_droid_jointpos_trunc6/pi05_droid_jointpos_trunc6/19999
```

Note the argument order: `--port` is top-level and must precede the `policy:checkpoint`
subcommand.

## 4. Evaluate

RoboLab, 5 tasks x 16 episodes, matching `docs/eval/RESULTS.md` so results are comparable.

```bash
python policies/pi0_family/run.py --remote-host <host> --remote-port 8000 \
  --policy pi05 --num-envs 16 --num-episodes-adaptive 16 --video-mode none --headless \
  --task BananaInBowlTask BagelsOnPlateTask BowlInBinTask MarkerInMugTask MustardInRightBinTask
```

## 5. Report

Two deltas, both stated explicitly:

- **A minus B** — truncation alone, finetuning held constant.
- **A minus vanilla (40%)** — the practical question, confounded with finetuning; label it.

Success: arm A retains at least 80% of arm B's overall success rate (relative — if B
scores 45%, A must clear 36%). At 16 episodes per task the per-task confidence intervals
are wide; consistency of direction across the 5 tasks matters more than the aggregate. If
the aggregate is small and directions are mixed, re-run the two tasks with the most
headroom at 50-100 episodes.

## Escalation if arm A underfits

In order, stopping as soon as accuracy recovers:

1. Unfreeze the action in/out projections.
2. Full-rank finetune the 6 kept blocks instead of LoRA.
3. Add teacher distillation from the depth-18 model.

## Deliberately out of scope

The dead third image slot (~6.9 ms, ~10%, no retraining), longer action chunks, SigLIP
truncation, and truncating the MEM K=6 model. Each is independently testable and should
stay that way. See the spec's "Deferred" section.
