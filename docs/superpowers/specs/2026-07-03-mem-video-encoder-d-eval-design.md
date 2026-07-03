# MEM Video-Encoder Eval — Sub-project D: 3-arm RoboLab evaluation (pi0.5 vs MEM K=1 vs MEM K=6)

**Date:** 2026-07-03
**Status:** Design (approved for spec)
**Part of:** "Verify MEM video-encoder effectiveness (pi0.5, DROID → RoboLab)" — the fourth of five
sub-projects (A gate → B training → C serving → **D eval** → E wandb comparison).
**Depends on:** B (two checkpoints), C (serving + tailscale, merged as PR #12).

## Goal

Run the actual Isaac-Sim evaluation and get success numbers for **three arms** — vanilla pi0.5
(reference), MEM K=1 (video off), MEM K=6 (video on) — on **5 simple-tier benchmark tasks**, each
adaptively to the RoboLab/TRI standard, so we can (a) sanity-check the MEM implementation against
vanilla pi0.5 and (b) measure the video-encoder effect (K=6 vs K=1).

## Context and key findings (drive the design)

### F1 — RoboLab eval machinery (built-in success + credible intervals)

`robolab/eval/runner.py:run_evaluation(args, policy, client_factory)` drives it: `--task-dirs`
(default `['benchmark']`, 120 tasks), `--tag` (filters by computed difficulty/attribute), `--task`
(explicit names), `--num-envs`, `--num-runs` (total = runs×envs), and **adaptive mode**
(`--num-episodes-adaptive MAX_N` + `--ci-pp-width`). It writes `output/<ts>_<folder>/<task_env>/` and
**summarizes per-task success rate with a 95% Beta credible interval**. Difficulty is *computed*
(`constants.py` `DIFFICULTY_THRESHOLDS = (2, 4)`: simple ≤2 subtasks) — so **`--tag simple`** selects
the simple tier (benchmark tasks don't hard-code the label).

### F2 — Episode budget: adaptive 200-max (RoboLab/TRI standard)

`docs/statistical_significance.md`: sim standard is **200 rollouts/task** (TRI LBM,
[arxiv 2507.05331]); RoboLab default `--num-episodes-adaptive 200 --ci-pp-width 0.14` runs batches of
`--num-envs` until each task's 95% Beta CI ≤ 14pp (≈±7%) **or** n≥200. Tasks near 0%/100% settle
early; only ~50% tasks burn the full 200. This is unbiased (stops on interval *width*, not value).

### F3 — Two serve paths, two eval clients (one arm at a time on the same endpoint)

- **Vanilla pi0.5** = the released `pi05_droid` (no MEM): served by the **stock
  `scripts/serve_policy.py --env DROID`** (stateless, single-frame), driven by RoboLab's **existing
  `Pi0DroidJointposClient`** via `policies/pi0_family/run.py` — no new eval code.
- **MEM K=1/K=6**: served by `scripts/serve_mem_session.py` (C), driven by `MemDroidJointposClient`
  (C) via a **new `policies/pi0_family/run_mem.py`**.
- All three serve on the same `nchc-mem-serve:8000` tailscale endpoint (one arm at a time). The
  `serve_policy.py` server binds `0.0.0.0` which includes `127.0.0.1`, so userspace-tailscale's
  inbound→localhost forwarding reaches it (verify in pre-check).

### F4 — Throughput (dev-box RTX 5090, from the Isaac report)

32 envs native ≈ 0.14 ep/s (~4 min per 32-episode batch), the 5090's native-res ceiling (64 crashes);
optimal sim knobs `num_threads=1`, TGS solver, `render_interval=8`. Adaptive-200 → up to ~7 batches
(~25 min) per task worst case; ~2 h/arm worst case × 3 arms, likely less (early-stop).

## Design

### Change 1 — `run_mem.py` (the only new code; RoboLab repo)

`RoboLab/policies/pi0_family/run_mem.py` — a thin sibling of `run.py`: same argparse + `import cv2`
(before isaaclab) + `run_evaluation`, but `make_client` builds **`MemDroidJointposClient`** (per-env
`session_id`, from C) instead of `Pi0DroidJointposClient`, defaulting `--remote-host nchc-mem-serve
--remote-port 8000 --policy pi05`. New file in the sanctioned `policies/pi0_family/` — no RoboLab core
change.

### Change 2 — Vanilla-pi0.5 serve behind tailscale (openpi repo)

`scripts/nchc/serve_pi05_droid_nano4.sbatch` — reuses C's **validated userspace-tailscale bring-up**
(`tailscaled --tun=userspace-networking`; `tailscale up --authkey=… --hostname=nchc-mem-serve
--reset`, no `--ephemeral`) and then runs the **stock** `scripts/serve_policy.py --env DROID` (serves
`pi05_droid` from `gs://openpi-assets/checkpoints/pi05_droid`) on port 8000, foreground. 4 GPUs
overridable to 1 for a single-request-throughput serve (like C's smokes used).

### Change 3 — Eval config (identical for all three arms = fair)

- **Tasks:** `--task-dirs benchmark --tag simple`, restricted to **5 tasks** (the concrete 5 pinned in
  pre-check #1 by enumerating simple-tagged benchmark envs — single-object pick-and-place). The same
  5 for every arm.
- **Budget:** `--num-envs 32 --num-episodes-adaptive 200 --ci-pp-width 0.14` (F2).
- **Sim knobs (F4):** native resolution, `num_threads=1`, TGS, `render_interval=8`.
- **Flags:** no `--enable-subtask` (MEM is LL-only → task-level success only); `--video-mode fail`
  (record only failures, for debugging, minimal I/O); a distinct `--output-folder-name <arm>` per arm.
- `open_loop_horizon=15` for all (pi05 default; our `action_horizon=16 ≥ 15`).

### Change 4 — Orchestration (controller-run, 3 serve→eval cycles)

For `arm ∈ {pi05_vanilla, mem_k1, mem_k6}`: submit that arm's serve sbatch on NCHC → wait for
`nchc-mem-serve` up + model loaded → on the dev box run the arm's eval (`run.py` for vanilla,
`run_mem.py` for MEM) with the Change-3 config → RoboLab writes `output/<ts>_<arm>/` → stop the serve
job. Sequential (same endpoint). ~2 h/arm worst case (adaptive; likely less).

## Data flow (one arm)

```
NCHC compute node: <serve sbatch> → userspace tailscale (nchc-mem-serve) + server on 127.0.0.1:8000
dev box (RTX 5090): run[_mem].py --tag simple --num-envs 32 --num-episodes-adaptive 200
   → RoboLab: 5 tasks × (batches of 32 until CI≤14pp or n=200)
      → per step: client → nchc-mem-serve:8000 → actions → sim step
   → output/<ts>_<arm>/<task>/  (per-task k/n success + 95% Beta CI)
```

## Pre-checks (de-risk before the full run)

1. **Enumerate** the 5 simple benchmark tasks: list what `--tag simple --task-dirs benchmark` selects;
   pick the 5 (record their names). Fail early if <5 simple tasks exist.
2. **Fidelity check (from C's final review):** confirm the DROID-jointpos env control rate ≈ **15 Hz**
   (inspect env `dt`/decimation) so the serving video-frame stride (`open_loop_horizon=15`) matches
   training (`video_stride_frames=15`). If it differs, note it as a caveat on the K=6 result.
3. **1-env / few-step full-loop smoke** per serve path: `--num-envs 1 --num-runs 1` for a couple steps
   against the live NCHC serve (vanilla via `run.py`, MEM via `run_mem.py`) before the 5×adaptive run —
   confirms Isaac + client + serve + tailscale end-to-end and that `serve_policy.py` is reachable at
   `nchc-mem-serve:8000`.

## Testing / definition of done

- `run_mem.py` produces a `MemDroidJointposClient` and runs `run_evaluation` (verified by the 1-env
  smoke returning a completed episode).
- All three arms complete the 5-task adaptive eval; each writes an `output/<ts>_<arm>/` with per-task
  `k/n` success + 95% Beta CI.
- **Aggregate success across the 5 tasks** is computed per arm (the get-a-feel signal; per-task is
  ±~7–17% depending on early-stop).
- **Diagnostic read recorded:** vanilla vs MEM K=1 (sanity — should be comparable; a large K=1≪vanilla
  gap flags a MEM impl/serving bug to debug before trusting K=6), and MEM K=6 vs K=1 (the video-encoder
  effect).

## Non-goals (later sub-projects)

- Parsing the three `output/` dirs → a wandb comparison (**E**).
- RoboLab core changes; RTC; enabling HL; retraining; sim-knob tuning beyond the report's optimum.

## Risks / open items

- **Sim-to-real gap:** `pi05_droid` was trained on real DROID → all three arms may have modest absolute
  success on RoboLab sim; the **relative** vanilla/K1/K6 pattern is what's diagnostic, not absolutes.
- **Simple tier diversity:** the simple benchmark tasks are mostly single-object pick-and-place — the
  5 tasks have limited skill diversity; the aggregate reflects those skills.
- **Serve uptime:** each arm's adaptive eval can approach ~2 h; C's serve sbatch `--time=4:00:00`
  covers it. If an arm's eval outlives the serve job, resubmit the serve and resume.
- **Fidelity (stride/sim-rate):** pre-check #2; if mismatched, the K=6 temporal signal differs from
  training and the K=6 number is a caveat, not a clean measurement.
- **Adaptive runtime variance:** total wall-clock is data-dependent (early-stop); could be ~2–6 h
  across three arms.

## Relation to the umbrella experiment

A proved the K=1 baseline; B trained the two checkpoints; C serves them; **D** (this spec) runs the
3-arm Isaac-Sim eval and records per-task success + CIs. **E** parses the three `output/` dirs into
wandb for the pi0.5-vs-K1-vs-K6 comparison. Each is its own spec → plan → implementation cycle.
