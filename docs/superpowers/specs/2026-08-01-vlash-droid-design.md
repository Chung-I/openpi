# VLASH on DROID: Training `pi05_droid_jointpos` with Future-State-Aware Async, Evaluated in RoboLab

**Date:** 2026-08-01
**Status:** Approved design (brainstormed with user; approach, venues, baselines, and feature scope all user-selected)

## Goal

Demonstrate VLASH's asynchronous-inference benefit on a dynamic manipulation task in simulation: a VLASH-fine-tuned `pi05_droid_jointpos` should **beat naive async** (and ideally match or exceed sync) on RoboLab's `rolling_ball_in_bowl_task` (fast reaction required), while **matching the baseline** on the byte-identical `static_ball_in_bowl_task` (accuracy control). This is the simulator analog of the VLASH paper's ping-pong/whack-a-mole results, and a direct alternative to layer truncation for hiding π0.5's inference latency on DROID (truncation to 6 layers halved latency but collapsed task success; VLASH hides latency instead of shrinking the model).

## Why these pieces

- **`pi05_droid_jointpos`** (not `pi05_droid`): its state (7 joint positions + 1 gripper) and action (7 absolute joint targets + gripper, served after the `AbsoluteActions` output transform) live in the same 8-dim joint space, so VLASH's future-state rollforward is *exact*: the last commanded action of the executing chunk IS the estimated execution-time state. The velocity variant would need integration plus ground-truth-state training and carries a silent-dimension-match trap.
- **RoboLab ball-in-bowl pair**: purpose-built dynamic/static contrast with byte-identical spawns, in-distribution for DROID policies, built-in success conditionals, and already served via the openpi websocket protocol (`policies/pi0_family/client.py` sends `observation/joint_position` + gripper).
- **Approach A — VLASH implemented inside openpi/JAX** (user-selected over porting DROID into the vlash/torch stack): reuses the proven DROID RLDS pipeline, the released checkpoint, the nano4 training recipe (truncation-work precedent), and RoboLab's serving path. Lesson from the LIBERO reproduction: train and serve in the same stack — cross-stack checkpoint ports fail silently (the `lerobot/pi05_libero` normalization incident).

## Architecture

### A. Training (nano4, 4×H200, openpi JAX, branch `vlash-droid` on fork `Chung-I/openpi`)

New `TrainConfig` **`pi05_droid_jointpos_vlash`**, derived from the truncation worktree's jointpos config (DROID RLDS `droid:1.0.1` + `droid_sample_ranges_v1_0_1.json` filter; init from the local copy of `gs://openpi-assets-simeval/pi05_droid_jointpos` at `/work/roboleon1295/checkpoints/pi05_droid_jointpos`).

**A1. Temporal-offset augmentation (the core VLASH mechanism, always on):**
- RLDS loader emits action windows of length `H + Δmax` (H = action_horizon = 15; **Δmax = 3**, covering ≥200 ms at DROID's 15 Hz).
- A new transform samples δ ~ U{0..Δmax} per example, slices the action targets to `[δ : δ+H]`, and computes the rolled-forward state `s_{t+δ} = s_t + Σ_{i<δ} aᵢ` (exact in delta-action space — the jointpos recipe converts absolute actions to deltas via `DeltaActions(make_bool_mask(7,-1))` before the model, and commanded deltas integrate to commanded positions). Gripper dim: rolled forward by taking the (δ−1)-th commanded gripper value, not summed (the mask excludes it from delta conversion).
- Images and prompt stay at time `t` (the paper's "fix the environment observation" rule).

**A2. AdaRMS state conditioning (user-selected: from the start, not a fallback):**
- Port VLASH's state pathway (vlash `modeling_pi05.py`, `state_cond=true` branch is the reference): a lightweight state-projection MLP feeding the action expert's **existing** AdaRMSNorm conditioning (openpi's π0.5 already conditions on flow time via AdaRMS — this extends the cond vector, not a new mechanism).
- Prompt building switches to the no-state-tokens variant ("Task: …; Action:") — the continuous state's only channel is AdaRMS, making the future-state signal strong.
- Init: released weights + fresh state-proj/AdaRMS-extension params; all trainable.
- Health metric: per-offset validation loss logged separately; δ>0 branches must separate from δ=0 (evidence the state channel is used).

**A3. Opt-in features (both default OFF; implemented + unit-tested regardless):**
- `shared_obs`: pack one observation prefix + all Δmax+1 offset branches into a single sequence with a block-sparse attention mask (branches attend to the shared prefix and themselves, never each other); loss averaged over branches. Requires the AdaRMS state path (A2) since each branch carries its own state. Reference: vlash `SharedObservationVLASHDataset` + its block-mask forward.
- `lora`: openpi's existing LoRA machinery, with the freeze-filter amended so the fresh AdaRMS/state-proj params are always trainable alongside adapters.

**A4. Headline run policy:** defaults = full fine-tune, per-sample offsets. Shared-obs may be enabled for the headline run only if it passes an equivalence gate (first ~2K steps: shared vs non-shared loss trajectories match); otherwise fall back. LoRA reserved for cheap ablations. ~30K steps, effective batch 32, wandb project **`vlash-droid`** (wandb mandatory per user policy).

**A5. Pre-flight gates (LIBERO lessons):** unit test of offset slicing + rollforward against hand-computed RLDS samples; 12-step training repro job before the full run; quota check (`df -h /work/roboleon1295`) before submission — wekafs silently drops writes at full quota.

### B. Serving (nano4 compute node → local, openpi websocket)

- Stock `scripts/serve_policy.py` (+ the state_cond-aware input transform for the VLASH checkpoint) on a 1-GPU job; both checkpoints served by the same code path: released jointpos (baseline arms, unmodified path) and the VLASH finetune (AdaRMS path).
- SSH tunnel `local → login → compute node` (`ssh -L 8000:<node>:8000 nano4`). The login master is bandwidth-policed (~1.7 MB/s): the client resizes images to 224 before sending. Wall-clock cost (~200 ms/call) is harmless — delay is emulated in sim steps, not wall time.
- Known failure mode: if the SSH master dies, re-auth needs password+OTP (user action).

### C. Evaluation (local RTX 5090: Isaac Lab + RoboLab)

New runner in `~/Codes/RoboLab/policies/pi0_family/` embedding the delay-emulating chunk executor semantics proven in the LIBERO reproduction (chunk starting at sim step T was requested with obs from T−Δ; per-episode fresh executor).

**Arms (5) × tasks (3 — `rolling_ball_in_bowl_task`, `static_ball_in_bowl_task`, `banana_in_bowl_task`):**

| Arm | Model | Obs at request | State input |
|---|---|---|---|
| sync d=0 | released | fresh | fresh (in-prompt, stock path) |
| naive-async Δ=1, Δ=2 | released | stale (T−Δ) | stale (in-prompt) |
| VLASH-async Δ=1, Δ=2 | VLASH finetune | stale (T−Δ) | **rolled forward** = last commanded action (exact for jointpos), via AdaRMS |

- Δ values confirmed against measured end-to-end latency before the sweep (expected ~1–2 steps at 15 Hz).
- 50 trials/arm/task = 750 episodes total; success via the tasks' built-in conditionals; per-episode incremental results JSON (resume-capable, as in LIBERO).
- Predicted outcome: static ≈ parity everywhere; rolling: VLASH ≫ naive-async, ≈ sync or better.

## Success criteria

1. Training converges with per-offset loss separation (A2 metric).
2. Static task: VLASH-async within ~5 points of sync baseline.
3. Rolling task: VLASH-async beats naive-async by a clear margin (target: ≥15 points) at Δ≥1.
4. All runs logged to wandb (`vlash-droid`); results reproducible from committed configs.

## Risks

- **Fresh AdaRMS params on a released checkpoint** may need LR warmup care (new params at full LR vs pretrained body) — expose separate LR groups if instability appears.
- **JAX shared-obs mask bugs corrupt training silently** — hence default-off + equivalence gate + unit tests.
- **Local 5090 currently hosts a vLLM process** — Isaac needs it freed (user prerequisite).
- **DROID RLDS location on nano4** assumed present from truncation training; verify path at plan time (Task 0 of the implementation plan).
- **NCHC tunnel stability** (documented master-death/OTP failure mode).
- RoboLab tasks are DROID-in-distribution by design, but the released policy's absolute success rate on ball-in-bowl is unknown until the sync-arm smoke run; if sync d=0 success is near floor (<20%), reaction-time contrasts lose power — surface early via a 10-episode smoke before full sweeps.

## Explicit non-goals

- No velocity-variant (`pi05_droid`) support.
- No real-robot DROID evaluation.
- No action quantization arm (separate follow-up; mechanism is client-side and can reuse this infra).
- No lerobot/torch DROID port.

## Test plan

- JAX unit tests: offset slice + rollforward correctness (hand-computed fixtures); shared-obs mask structure (branch isolation); LoRA param partition (AdaRMS params trainable).
- RoboLab-side unit tests: executor timing semantics (port of the LIBERO test suite, including stale-state arm); resume bookkeeping.
- Smoke gates: 12-step train job; 2-episode eval smoke per arm before 50-trial sweeps; early-success gate on the sync arm.
