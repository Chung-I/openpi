# pi0.5 Layer Truncation — Reproduction

Cuts the shared 18-layer gemma stack to 6 layers `(0, 3, 7, 11, 14, 17)` for a ~2x
inference speedup, then recovers accuracy with a LoRA finetune on DROID joint-position
data. Design: `docs/superpowers/specs/2026-07-21-pi05-layer-truncation-design.md`.

## Arms

| config | layers | role |
|---|---|---|
| `pi05_droid_jointpos_trunc6` | 6 of 18 | the truncated model under test |
| `pi05_droid_jointpos_trunc18` | all 18 | control; removes finetuning as a confound (but see the caveat in §5 — it does not isolate depth alone) |

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

**Result (2026-07-21, RTX 5090, 12 repeats after 3 warmup, random weights): PASSED.**

| config | predicted p50 | measured p50 | p95 | SigLIP-only p50 |
|---|---|---|---|---|
| `pi05_droid_jointpos_trunc18` | 66.0 ms | **68.6 ms** (+3.9%) | 71.1 ms | 17.0 ms |
| `pi05_droid_jointpos_trunc6` | 33.1 ms | **32.0 ms** (-3.3%) | 32.6 ms | 16.1 ms |

**Measured speedup 2.14x**, against a 1.996x prediction and a 1.5x abort threshold. The
LIBERO-fitted cost model transferred to DROID within 4% on both arms, so the 968-token
prefix argument holds.

Two independent confirmations that the truncation is real rather than masked:

- **The SigLIP floor landed where predicted.** 16.1-17.0 ms measured against 16.21 ms
  predicted, and it is 50.3% of the 6-layer arm's total against 49% predicted. The
  spread between the two runs is cross-process measurement noise, not a real difference —
  neither arm changes the vision encoder.
- **The truncatable remainder scaled by the depth ratio.** Subtracting SigLIP leaves
  51.6 ms at 18 layers and 15.9 ms at 6, a ratio of **3.25x** against a depth ratio of
  exactly 3.00x. Had the layers been masked rather than removed, or had the gather
  silently kept more blocks than requested, this would not hold.

The 2.14x slightly beats the 2.0x prediction because the per-layer cost on this DROID
config is marginally higher than the LIBERO fit, so removing 12 of 18 layers recovers a
little more than the model expected.

## 2. Train

Before the first `sbatch`, create the directory Slurm's `--output` writes to — Slurm opens
that path *before* the script body runs, so an in-script `mkdir` cannot create it and the
job fails immediately with no log to explain why:

```bash
mkdir -p /work/roboleon1295/openpi/logs
```

The script also overrides `HOME` to keep caches off the quota-limited `/home`, which moves
where wandb looks for credentials. Copy the credential into the job's `HOME` once, or the
run trains untracked:

```bash
mkdir -p /work/roboleon1295/jobhome
cp -p ~/.netrc /work/roboleon1295/jobhome/.netrc
```

The script asserts this file exists and aborts if it does not.

The script also points TF's GCS client at the venv's certifi bundle. These nodes are a
RHEL-family image carrying `/etc/ssl/certs/ca-bundle.crt`, but libcurl defaults to the
Debian path `/etc/ssl/certs/ca-certificates.crt`, which does not exist — so `tf.data`
streaming the DROID RLDS shards dies about 60 s in with `libcurl code 77 ... Problem with
the SSL CA cert`. The checkpoint download uses a different code path and succeeds, which
makes this look like a training bug rather than a certificate one.

To run from a git worktree instead of the main checkout — useful when the main checkout has
unrelated uncommitted work — set `OPENPI_ROOT` and override `--output`, since Slurm opens
that path before the script body runs and so cannot expand the variable:

```bash
OPENPI_ROOT=/work/roboleon1295/openpi-trunc \
  sbatch --output=/work/roboleon1295/openpi-trunc/logs/trunc_%j.log \
    scripts/nchc/train_truncation.sbatch
```

Note the worktree needs its own venv (`uv venv --python 3.11 && uv sync --group rlds`):
openpi is installed editable, so a worktree sharing the main checkout's venv would silently
import the main checkout's code. The `rlds` group is required — DROID training reads RLDS.

```bash
sbatch scripts/nchc/train_truncation.sbatch
```

Both arms, 4 GPUs each, on one 8-GPU allocation. 20k steps, batch 128. The script gates
on `scripts/check_truncation.py` first, so a loader/model depth mismatch fails in seconds
rather than after the checkpoint downloads.

Watch the first 100 steps in wandb (project `layer-truncation`). Arm A starting far above
arm B and failing to descend means LoRA is not bridging the depth cut — see Escalation.

**If the job hangs with GPUs at 0% while Slurm still reports RUNNING, it is memory.** Each
arm's `tf.data` shuffle buffer holds 250k DROID timesteps and fills *progressively*, so the
job trains normally for ~30 minutes and only then thrashes. A 400G allocation was not
enough for two arms (they reached 418G RSS combined); the script now requests 1200G, within
NCHC's 200G-per-GPU rule. Diagnose from the login node — you cannot ssh to compute nodes:

```bash
srun --jobid=<id> --overlap -n1 -N1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
srun --jobid=<id> --overlap -n1 -N1 bash -c "ps -eo pid,etime,pcpu,rss,stat,cmd | grep [t]rain.py"
```

If memory is still the binding constraint, shrink the buffer without a code change:
`DROID_SHUFFLE_BUFFER=100000 sbatch scripts/nchc/train_truncation.sbatch`. Keep it at or
above ~100k; below that, shuffling is not sufficiently random.

Neither arm's `TrainConfig` sets `resume` or `overwrite`. If the 48h wall-clock limit is hit
and Slurm requeues the job, training restarts at step 0 by default and then fails on the
already-existing checkpoint directory from the previous attempt. If requeuing after a
timeout, the operator should set `resume=True` (and not `overwrite`) before resubmitting.

### Training result (2026-07-22)

Job 201730, NCHC `8gpus` node `25a-hgpn142`, both arms in one allocation, 4 GPUs each.
**State COMPLETED, exit 0:0, elapsed 5h55m, zero errors in either arm.** Checkpoints at
5000 / 10000 / 15000 / 19999 on both, so every matched-step comparison is available.

| | trunc6 (6 layers) | trunc18 (control) |
|---|---|---|
| final loss @ 19900 | **0.0291** | **0.0105** |
| grad_norm @ 19900 | 0.1117 | 0.0575 |
| param_norm @ 19900 | 1209.4 | 1834.9 |
| training rate | 2.5 it/s | 1.0 s/it |

Both converged: small, stable gradient norms, no divergence or collapse.

Matched-step flow loss:

| step | trunc6 | trunc18 | ratio A/B |
|---|---|---|---|
| 1000 | 0.04490 | 0.01487 | 3.02x |
| 5000 | 0.03340 | 0.01111 | 3.01x |
| 10000 | 0.03147 | 0.01082 | 2.91x |
| 15000 | 0.02919 | 0.01053 | 2.77x |
| 19900 | 0.02911 | 0.01046 | 2.78x |

**The gap narrows but does not close** — 3.02x to 2.78x over 20k steps. Arm A was still
improving at the end (0.0334 at 5k to 0.0291 at 20k), so it is not saturated, but the trend
is far too slow to reach parity with more steps. Read that as a capacity ceiling rather than
undertraining.

Incidental corroboration of the latency result: arm A trained at **2.5x** the control's
step rate (2.5 it/s vs 1.0 s/it), on a forward+backward workload, against the 2.14x
forward-only speedup measured in Gate 0.

**Flow loss is not task success.** The mapping is nonlinear — the published baselines have
MEM K=1 at 42% and vanilla at 40% despite quite different training. Section 4 is what
decides this.

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
Note: `docs/eval/RESULTS.md` — the source of the vanilla 40% / K=1 42% / K=6 52% baselines
below — lives on the `mem-video-encoder-d-eval` branch, not on this branch or `main`; a
reader working only from this branch will not find it in their checkout.

```bash
python policies/pi0_family/run.py --remote-host <host> --remote-port 8000 \
  --policy pi05 --num-envs 16 --num-episodes-adaptive 16 --video-mode none --headless \
  --task BananaInBowlTask BagelsOnPlateTask BowlInBinTask MarkerInMugTask MustardInRightBinTask
```

### Evaluation result (2026-07-22) — FAILED the success bar

Both 19999 checkpoints served from NCHC (jobs 203650 / 203651) over tailscale, RoboLab sim
client local, 5 tasks x 16 episodes, `open_loop_horizon` left at the pi05 default of 15 to
match how the published baselines were evaluated.

| task | trunc6 (6 layers) | trunc18 (control) | published vanilla |
|---|---|---|---|
| BananaInBowl | 0/16 = 0% | **15/16 = 93.8%** | 88% |
| BowlInBin | 0/16 = 0% | 9/16 = 56.2% | 31% |
| MarkerInMug | 0/16 = 0% | 2/16 = 12.5% | 6% |
| MustardInRightBin | 0/16 = 0% | 2/16 = 12.5% | 75% |
| BagelsOnPlate | 0/16 = 0% | 0/16 = 0% | 0% |
| **OVERALL** | **0/80 = 0.0%** | **28/80 = 35.0%** | 32/80 = 40% |

**A − B = −35.0 pp. Arm A retains 0% of the control's success rate, against an 80% bar.**
The 6-layer model fails every episode of every task.

**The control validates the instrument.** 93.8% on BananaInBowl reproduces the documented
94% vanilla zero-shot result, and BagelsOnPlate at 0% matches every published arm. An
18-layer model through the *identical* truncation code path, weight loader, training recipe,
serving stack and eval harness comes out at full strength — so the pipeline is correct, the
LoRA recipe is sound, and arm A's failure is a real property of truncation rather than a bug.

The control lands 5 pp below published vanilla (35% vs 40%), driven almost entirely by
MustardInRightBin (12.5% vs 75%). That task is known to be volatile across arms — the MEM
runs recorded 56% / 81% / 75% on it — so the aggregate gap is within the noise this suite
produces at 16 episodes, but it is a real per-task discrepancy and is not explained here.

**Flow loss badly understated the damage.** Arm A's loss was ~2.8x the control's, which
reads as degraded-but-working. Task success went 93.8% to 0%. Treat flow loss as a poor
proxy for competence when comparing architectures of different depth.

**Serving plumbing was verified independently**, before the control finished, by probing the
trunc6 server directly: actions came back correctly shaped (16, 8), anchored at the current
pose (`max|action[0] − state|` = 0.062 rad) and smooth (max per-step delta 0.018 rad). So
`AbsoluteActions`, normalization and the flow head are all correct — the model emits
well-formed actions, it just cannot do the tasks. (Caveat: that probe used random-noise
images, so it establishes format correctness, not competence.)

**What this does NOT establish.** Arm A carries one third of the control's LoRA parameters,
since adapters live inside the scanned block. This result therefore shows that *6 layers with
1/3 the adapter capacity, recovered by LoRA alone* fails — not that 6 layers is inherently
incapable. The escalation ladder below is untested and is the direct response to this
outcome.

### Escalation results (2026-07-23) — full rank does not rescue it either

After the LoRA arms scored 0/80, two escalations were run. Because the escalation uses
batch 256 against the LoRA arms' 128, **step 10k here equals the LoRA arms' full 20k run in
samples seen (2.56M)** — so this is a data-matched comparison, not a step-matched one.

| arm @ 2.56M samples | trainable | SigLIP | flow loss | RoboLab |
|---|---|---|---|---|
| 6L LoRA | 16.7 M | frozen | 0.02911 | **0/80 = 0%** |
| 6L full rank | 1749 M | **trains** | **0.01656** | **0/80 = 0%** |
| 18L LoRA control | 16.7 M | frozen | 0.01046 | **28/80 = 35%** |

**80x the trainable capacity and 43% lower loss produced exactly zero change in task
success.** The gap to the control closed from 2.78x to 1.58x in loss terms and not at all in
success terms. Per-task, the full-rank arm scored 0/16 on every task including BananaInBowl
(control 15/16) and BowlInBin (control 9/16).

Trajectory metrics say the arm is not frozen but wandering: EE SPARC -7.68 against the
control's -5.03, path length 2.03 m against 1.47 m. It moves more, less smoothly, and
completes nothing.

**Flow loss is not a usable proxy for competence across depths.** This is the second and
more emphatic demonstration: the first was 2.8x loss reading as "degraded but working" while
success went 93.8% -> 0%.

**Two explanations remain open, and the experiment that separates them was destroyed.** An
earlier full-rank arm with SigLIP FROZEN was cancelled at 11.6k and its checkpoints deleted
during a storage cleanup; that arm was the control for this question.

1. *Depth is the wall* — 6 layers cannot represent these tasks under any recipe.
2. *Training SigLIP broke visual grounding* — project memory records exactly this failure
   ("SigLIP fully trained on DROID -> wrecked pretrained visual features"). It fits: flow
   loss improves because action sequences remain predictable from proprioception, while task
   success stays at zero because the policy can no longer localise the object.

Re-running full rank with SigLIP frozen to a data-matched 10k costs ~3h and distinguishes
them. Until that is done, this result does NOT establish that 6 layers is inherently
incapable.

### Data scaling flips the conclusion (2026-07-23)

The 0/80 results above were a red herring about DEPTH. They were all measured at ~10% of a
DROID epoch (2.56M samples). The best-effort 100k run (batch 256, 100k steps, nothing
frozen) was evaluated at two checkpoints from the SAME run:

| checkpoint | samples | Banana | Bowl | Marker | Mustard | Bagels | OVERALL |
|---|---|---|---|---|---|---|---|
| 10k | 2.56M | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 | **0/80 = 0%** |
| 20k | 5.12M | 7/16 = 44% | 0/16 | 0/16 | 1/16 | 0/16 | **8/80 = 10%** |
| 18L control | 2.56M | 15/16 = 94% | 9/16 | 2/16 | 2/16 | 0/16 | 28/80 = 35% |

**Doubling the data took the 6-layer model from 0% to 10% overall, and BananaInBowl from 0%
to 44%.** The earlier conclusion -- reinforced by the full-rank arm also scoring 0/80 at
2.56M -- that "6 layers cannot do these tasks" was premature. At 2.56M the truncated model
was DATA-STARVED, not necessarily depth-limited. The 18-layer control succeeded at the same
data because more depth needs less data to become useful, not because 6 layers is incapable.

The recovery is task-ordered so far: the easiest task (Banana, control 94%) crosses first;
harder tasks (Bowl 56%, Mustard/Marker ~12% on the control) are still at or near zero at 20k.
Whether they follow with more data is what the remaining 80k steps (up to 25.6M samples, 5x
the 20k checkpoint) will show.

**Process note, recorded deliberately:** at the 20k eval the running assistant had twice
recommended KILLING this run based on the 0/80 results, reasoning "more data rarely rescues a
policy scoring zero." That reasoning was wrong here -- more data was exactly what it needed.
The run was allowed to continue only because a cheap 20k-checkpoint eval was run before
acting on the recommendation. Lesson: at 10% of an epoch, a 0% RoboLab score is not evidence
of a capability ceiling.

### 40k checkpoint: recovery stalls at ~10% (2026-07-23)

| task | 10k (2.56M) | 20k (5.12M) | 40k (10.24M) | 18L ctrl (2.56M) |
|---|---|---|---|---|
| BananaInBowl | 0% | 44% | 25% | 94% |
| MustardInRightBin | 0% | 6% | 25% | 12% |
| BowlInBin | 0% | 0% | **0%** | 56% |
| MarkerInMug | 0% | 0% | **0%** | 12% |
| BagelsOnPlate | 0% | 0% | 0% | 0% |
| **OVERALL** | **0/80 = 0%** | **8/80 = 10%** | **8/80 = 10%** | **28/80 = 35%** |

**The aggregate is flat from 20k to 40k (both 10%), despite doubling the data again (5.12M ->
10.24M).** The 0% -> 10% jump between 10k and 20k was real; the curve then stalls.

Read the per-task detail, not the aggregate:
- Only the two EASIEST tasks (Banana and Mustard, control 94%/12%) ever leave zero. They
  trade places between 20k and 40k -- Banana 44->25, Mustard 6->25 -- but at n=16 those swings
  are within noise (Wilson 95% CIs on 25% and 44% overlap heavily). Do not read a trend into
  either single number.
- **BowlInBin and MarkerInMug are 0/16 at every checkpoint** (control 56%/12%). Four data
  doublings have not moved them off zero. This is the signal: the harder tasks are NOT
  following Banana's recovery, which argues against a pure data-scaling story and for a
  per-task capability that 6 layers has not reached.

So the corrected picture after three checkpoints: data-starvation explained the 10k 0/80,
but data scaling alone plateaus the 6-layer model around 10% -- roughly a third of the
control's 35% -- with competence confined to the easiest one or two tasks. Whether the
remaining 60k steps (up to 25.6M) break BowlInBin/Marker off zero is the open question; the
40k result makes a large further gain look unlikely but not excluded.

Statistical honesty: every cell is n=16 (Wilson 95% CI on a single task spans ~20 points),
and the 20k vs 40k aggregates (both 10%, CI [5-19]%) are statistically indistinguishable.
Treat per-checkpoint task numbers as directional, not precise.

### 60k checkpoint: plateau breaks, recovery resumes (2026-07-23)

| task | 10k | 20k | 40k | 60k | 18L ctrl |
|---|---|---|---|---|---|
| samples | 2.56M | 5.12M | 10.24M | 15.4M | 2.56M |
| BananaInBowl | 0% | 44% | 25% | 38% | 94% |
| MustardInRightBin | 0% | 6% | 25% | 25% | 12% |
| BowlInBin | 0% | 0% | 0% | **12%** | 56% |
| MarkerInMug | 0% | 0% | 0% | 0% | 12% |
| BagelsOnPlate | 0% | 0% | 0% | 0% | 0% |
| **OVERALL** | **0%** | **10%** | **10%** | **15%** | **35%** |
| non-zero tasks | 0 | 2 | 2 | **3** | 4 |

The 20k->40k plateau was NOT the ceiling. At 60k the aggregate rises to 15% and **BowlInBin
breaks off a zero it held for three straight checkpoints** (0/0/0 -> 2/16). Non-zero task
count goes 2 -> 2 -> 3. So the earlier "data scaling plateaus at ~10%" read (written at 40k)
was itself premature -- the plateau was a pause, not a wall.

The corrected trajectory: the 6-layer model recovers with data, but SLOWLY and
TASK-BY-TASK, each task crossing off zero at a different data budget (Banana ~5M, Mustard
~5M, BowlInBin ~15M). MarkerInMug is still at zero at 15.4M (control only 12%, so it has the
least headroom and may need the most data). MustardInRightBin at 60k (25%) already exceeds
the control's 12% -- on the tasks it can do, the truncated model is competitive.

Trend, not plateau, and not yet converged: with 40k steps still to run (up to 25.6M samples)
the aggregate is rising, not flat. The open question is now where it tops out relative to the
control's 35%, not whether 6 layers can work at all -- it demonstrably can, on 3 of 5 tasks.

Statistical honesty unchanged: every cell is n=16. The 12/80 vs 8/80 step (15% vs 10%) has
overlapping CIs ([9-24]% vs [5-19]%), so the aggregate rise is suggestive; the BowlInBin
qualitative change (three zeros then non-zero) is the firmer signal.

## 5. Report

Two deltas, both stated explicitly:

- **A minus B** — same seed, data, steps, schedule, base params and norm stats in both arms,
  so this isolates depth. But it does not isolate depth *alone*: LoRA adapters live inside
  the scanned block, so arm A trains on 6 blocks' worth of adapters and arm B on 18 — a
  third the trainable adapter capacity. A minus B therefore measures depth reduction and
  adapter-capacity reduction jointly, not truncation with finetuning held perfectly constant.
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
