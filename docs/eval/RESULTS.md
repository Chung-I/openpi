# MEM Video-Encoder Evaluation — Results

**Question:** Does the MEM `VideoViTEncoder` (multi-frame temporal memory) improve a
pi0.5-based DROID policy, evaluated on NVIDIA RoboLab (Isaac Sim)?

**Answer (2026-07-06): Yes.** At full training (20k steps), across 5 RoboLab DROID
tasks, the 6-frame model (`K=6`) beats the single-frame model (`K=1`) by **+10 pp
overall (52% vs 42%)** and is **≥ K=1 on every task** — with the largest gains on the
mid-difficulty tasks that have headroom to improve.

---

## Setup

- **Base:** `pi05_droid_jointpos` (joint-position DROID checkpoint;
  `gs://openpi-assets-simeval/pi05_droid_jointpos`). Serve the **flow** path.
- **Recipe (flow-video):** train the video encoder directly via the flow expert.
  - Loss: flow-only — `ll_loss_weight=1.0, fast_loss_weight=0.0, hl_loss_weight=0.0`.
  - `insulate_flow_prefix=False` — the flow pass does a single joint forward over
    `[prefix, suffix]` (like vanilla pi0), so flow gradients reach the video encoder +
    backbone (verified on the real model: `dL_flow/d(video_img)` abs-sum = 27070 > 0).
  - Trainable: `video_img` (video encoder, full) + LoRA (backbone + action expert).
    Frozen: single-frame SigLIP `img`, LLM/expert base, `state_proj` (dead).
  - State: pi0.5 text-state in the prompt (`discrete_state_input=True`); image-only memory.
- **Arms** (identical except `num_video_frames`):
  - `K=1` — single frame (temporal attention is a no-op; the "no memory" MEM baseline).
  - `K=6` — 6-frame rolling video memory (the encoder under test).
  - `vanilla` — `pi05_droid_jointpos` unmodified (no MEM at all); reference.
- **Training:** 20k steps, batch 32, LoRA, CosineDecay (warmup 1000, peak 2.5e-5).
  Full runs on NCHC (8×H200). Serving on 1 GPU via `serve_mem_session.py`
  (`MemSessionPolicy`, per-env K-frame buffer keyed by `session_id`).
- **Eval:** RoboLab `run_mem.py` (`MemDroidJointposClient`), 16 episodes/task,
  BananaInBowl + BagelsOnPlate + BowlInBin + MarkerInMug + MustardInRightBin.
- `K=6 − K=1` isolates the video memory: everything else (recipe, base, step, eval) is
  matched, so the gap is attributable to the temporal frames.

---

## Headline result — full training (20k), 5 tasks

| Task | vanilla (no MEM) | K=1 (1 frame) | K=6 (6-frame memory) | **K6 − K1** | **K6 − vanilla** |
|---|---|---|---|---|---|
| BananaInBowl      | 14/16 =  88% | 16/16 = 100% | 16/16 = 100% | +0 (ceiling) | +12 pp |
| MustardInRightBin | 12/16 =  75% |  9/16 =  56% | 13/16 =  81% | **+25 pp**   | +6 pp  |
| BowlInBin         |  5/16 =  31% |  8/16 =  50% | 11/16 =  69% | **+19 pp**   | **+38 pp** |
| MarkerInMug       |  1/16 =   6% |  1/16 =   6% |  2/16 =  12% | +6 pp        | +6 pp  |
| BagelsOnPlate     |  0/16 =   0% |  0/16 =   0% |  0/16 =   0% | +0 (floor)   | +0     |
| **OVERALL**       | **32/80 = 40%** | **34/80 = 42%** | **42/80 = 52%** | **+10 pp** | **+12 pp** |

**Ranking: `K=6 (52%) > K=1 (42%) > vanilla (40%)`.**

- **K=6 ≥ K=1 on every task** — the video memory never hurts; biggest gains where there's
  room (Mustard **+25 pp**, BowlInBin **+19 pp**). No change where saturated (Banana 100%)
  or floored (Bagels 0% — a task neither policy can do yet).
- **K=6 ≥ vanilla on every task, +12 pp overall** — the video-memory model beats the
  *original, un-wrapped* policy too (BowlInBin 31% → 69%, **+38 pp**).
- **K=1 ≈ vanilla** (42% vs 40%) — MEM-wrapping + flow-video finetuning is roughly neutral
  at single-frame (per-task: +12 Banana, −19 Mustard, +19 BowlInBin), so the improvement
  comes from the **6-frame memory**, not the wrapping. (At K=1, finetuning regressed Mustard
  vs vanilla 75% → 56%; K=6 recovers *and* exceeds it at 81%.)

## Early read — dev checkpoints, BananaInBowl (matched steps)

Before the full runs finished, matched-step dev checkpoints already showed the effect:

| Step | K=1 | K=6 | K6 − K1 |
|---|---|---|---|
|  5000 | 12/16 = 75% | 15/16 = 94% | +19 pp |
| 10000 | 14/16 = 88% | 16/16 = 100% | +12 pp |

(At 20k, K=1 on Banana catches up to 100%, so the effect on this saturated task
disappears and migrates to the harder tasks — see above.)

---

## Statistical caveats

- **16 episodes/task** → per-task 95% CIs are wide. The aggregate +10 pp (34 vs 42 of 80)
  is **suggestive, not conclusive** in isolation (2-proportion test p ≈ 0.2).
- The **consistency** is the real signal: monotonic `K=6 ≥ K=1` across all 5 tasks, with
  two independent tasks at +19/+25 pp, plus the same direction at every dev-run checkpoint.
- **To make it airtight:** re-run **Mustard + BowlInBin** (where the effect lives) at
  50–100 episodes/task.

---

## Why it works (recipe notes)

The path to a working experiment required several fixes (see
`memory/project_droid_jointpos_action_space.md` for the full chain):

1. **Action space** — RoboLab sim needs the **joint-position** DROID checkpoint, not the
   released velocity `pi05_droid` (velocity → 0% for all arms).
2. **Freeze SigLIP + text-state, image-only memory** — untrained `K=1` then equals vanilla
   by construction (100% on BananaInBowl).
3. **FAST is a dead end here** — the DROID checkpoints are flow-only finetunes with no
   usable FAST head (FAST loss ≈ 15, worse than random, on both velocity and jointpos
   targets; a real FAST-DROID model, `pi0_fast_droid_jointpos`, scores 1.2). Training FAST
   from that state corrupts the flow backbone.
4. **Flow *can* train the video encoder** — pi0.5's Knowledge-Insulation "insulation" is
   only an explicit `stop_gradient`, not a forward disconnect. With `insulate_flow_prefix=
   False` the flow loss reaches `video_img` on the real trained model. This is the
   flow-video recipe used here.
5. **Stability** — full-training `video_img` via flow does **not** degrade the base
   (`K=1` holds ~88–100% vs vanilla 94%); no crater (unlike FAST/co-training, which hit 0%).

---

## Reproduce

```bash
# Train (NCHC, per arm): pi0_mem_droid_k1_verify (1 GPU) / pi0_mem_droid_k6_verify (2 GPU)
sbatch scripts/nchc/train_mem_arm_1gpu.sbatch pi0_mem_droid_k1_verify
# Serve a checkpoint (MEM session server):
sbatch scripts/nchc/serve_mem_session_nano4.sbatch <config> <ckpt_dir> 8000
# Eval (RoboLab), 5 tasks:
python policies/pi0_family/run_mem.py --remote-host nchc-mem-serve --remote-port 8000 \
  --policy pi05 --num-envs 16 --num-episodes-adaptive 16 --video-mode none --headless \
  --task BananaInBowlTask BagelsOnPlateTask BowlInBinTask MarkerInMugTask MustardInRightBinTask
```
