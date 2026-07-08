# HL-only Training — Experiment Results

- **Date:** 2026-07-08
- **Branch:** `mem-hl-training`
- **wandb project:** `mem-hl-training` (training-time metrics + `samples/*` tables per run)
- **Raw eval logs (nano4):** `/work/roboleon1295/openpi-hl/logs/eval*.log`, `zs_*.log`
- **Checkpoints (nano4):** `/work/roboleon1295/openpi-hl/checkpoints/mem-hl-training/<config>/{5000,10000,15000,20000}`

## TL;DR

- We train the Pi0MEM high-level policy (`compute_loss_hl`) to predict the next subtask + memory
  from a single frame + goal, LoRA-finetuning from a pi0.5 checkpoint on RoboMIND franka_3rgb
  (test = franka_1rgb, a held-out bread-manipulation embodiment).
- **The aggregate metrics were misleading.** ~80% of samples are `update=False` (target memory ==
  input memory), so the model can score well by **copying the input**. Splitting metrics by the
  `update` flag exposed this: `memory_exact_match` is ~0.8 on `update=False` but ~0.3 on the real
  transitions (`update=True`).
- **On the transition (`update=True`) cases, the winner is `B+upsample`** (unfrozen SigLIP **+**
  update-upsampling). It nearly doubles the baseline on the OOD test set (subtask EM 0.109 → 0.203)
  and is best on dev_unseen (0.203 → 0.516). The two levers are individually weak/harmful on OOD but
  **complementary together**.
- **Upsampling** `update=True` is the strongest *single* lever. **Unfreezing SigLIP (B) alone helps
  in-distribution but hurts OOD** (overfits the encoder to franka_3rgb). **The pi0.5-style question
  prompt washed out after finetuning** (≈ baseline) despite a striking zero-shot effect.
- All runs **overfit** (~10.6 epochs at 20k steps / batch 32; dev metrics peak ~step 3k ≈ 1.6 epochs).

## Setup

- **Model:** Pi0MEM, PaliGemma (`gemma_2b`) backbone, single-frame SigLIP encoder for HL. LoRA
  (rank 16) on the LLM. Init from `pi05_base` (or `pi05_droid` for the droid arm).
- **HL objective:** PaliGemma prefix-LM. Prefix = `[image] BOS memory goal SEP(\n)`; target
  `Subtask: {l} Memory: {m}` + EOS; the SEP position predicts the first target token (no BOS
  injected). See `pi0_mem.embed_prefix_hl` / `compute_loss_hl` / `predict_subtask_and_memory*`.
- **Data:** `robomind_hl_fr3` (franka_3rgb). Episode-based, task-level split (`hl_splits.py`):
  train = 60,233 samples (12,058 `update=True` / 48,175 `update=False`, i.e. 20% transitions),
  `dev_seen` = held-out episodes of trained tasks, `dev_unseen` = 9 held-out tasks. Test =
  `robomind_hl` (franka_1rgb, held-out bread embodiment).
- **Training:** batch 32, 20k steps ⇒ **≈10.6 epochs** (no-upsample). Cosine LR peak 5e-5,
  warmup 200. Strict LoRA-only freeze unless noted (SigLIP + base LLM frozen).
- **Eval:** `eval_hl_metrics.py` — per-`update` bucketed generation metrics on a **balanced**,
  fixed-seed-shuffled subset (up to `gen_examples` per bucket; below: 64 update + 64 no-update
  per split). Greedy KV-cache decode; exact-match after `Subtask:/Memory:` parse.

## The copy shortcut (why we bucket by `update`)

`update=False` (≈80% of data): target = `(current subtask l_i, previous memory m_{i-1})` — the
memory target **equals the input memory**, so copying it is correct. `update=True` (subtask
boundary): target = `(next subtask l_{i+1}, updated memory m_i)` — copying fails. Aggregate
`memory_exact_match` ≈ 0.8 is almost entirely the copy shortcut; on transitions it is ~0.3.
Mitigation: `upsample_update` oversamples `update=True` to ~50/50 in the training pool (dataset
unchanged; `make_hl_batch_iterator`).

## Experiment matrix (all init from pi05_base unless noted; all 20k steps)

| config | SigLIP | upsample `update` | prompt | isolates |
|---|---|---|---|---|
| `pi0_mem_hl_fr3_base` (**A**) | frozen | no | raw goal | baseline |
| `pi0_mem_hl_fr3_base_vis` (**B**) | **unfrozen** | no | raw goal | visual grounding |
| `pi0_mem_hl_fr3_base_upsample` | frozen | **yes** | raw goal | shortcut fix |
| `pi0_mem_hl_fr3_base_vis_upsample` | **unfrozen** | **yes** | raw goal | both levers |
| `pi0_mem_hl_fr3_base_q` | frozen | no | **question** | pi0.5 format alignment |

(`pi0_mem_hl_fr3_droid` = A with pi05_droid init, used earlier for a base-vs-droid init check;
droid init ≳ base on dev_seen, ~equal on test.)

## Results — all configs @ step 15000, bucketed (n=64 per bucket)

**`update=True` (transitions — the meaningful cases):**

| | dev_seen |  |  | dev_unseen |  |  | test |  |  |
|---|---|---|---|---|---|---|---|---|---|
| config | sub | mem | tok | sub | mem | tok | sub | mem | tok |
| A | 0.156 | 0.359 | 0.382 | 0.203 | 0.250 | 0.392 | 0.109 | 0.344 | 0.358 |
| B (unfroze) | 0.312 | 0.422 | 0.494 | 0.375 | 0.328 | 0.507 | 0.047 | 0.312 | 0.351 |
| upsample | 0.344 | 0.484 | 0.532 | 0.375 | 0.344 | 0.516 | 0.125 | 0.406 | 0.434 |
| base_q | 0.219 | 0.312 | 0.411 | 0.188 | 0.203 | 0.428 | 0.078 | 0.406 | 0.371 |
| **B+upsample** | **0.375** | 0.422 | 0.523 | **0.516** | **0.406** | **0.643** | **0.203** | 0.359 | 0.423 |

**`update=False` (easy / copy cases):**

| | dev_seen |  |  | dev_unseen |  |  | test |  |  |
|---|---|---|---|---|---|---|---|---|---|
| config | sub | mem | tok | sub | mem | tok | sub | mem | tok |
| A | 0.359 | 0.844 | 0.563 | 0.422 | 0.891 | 0.663 | 0.297 | 0.859 | 0.602 |
| B | 0.438 | 0.891 | 0.600 | 0.406 | 0.859 | 0.631 | 0.250 | 0.703 | 0.544 |
| upsample | 0.188 | 0.812 | 0.435 | 0.375 | 0.859 | 0.579 | 0.328 | 0.656 | 0.586 |
| base_q | 0.422 | 0.875 | 0.550 | 0.406 | 0.891 | 0.654 | 0.328 | 0.688 | 0.589 |
| B+upsample | 0.344 | 0.812 | 0.499 | 0.328 | 0.797 | 0.578 | 0.297 | 0.734 | 0.570 |

(sub = subtask exact-match, mem = memory exact-match, tok = token accuracy. n=64/bucket ⇒
≈±6% per cell; small deltas are noise.)

## Analysis

- **Upsampling breaks the shortcut and lifts transitions.** vs A on `update=True`: dev_seen subtask
  0.156→0.344, dev_unseen 0.203→0.375. Cost: `update=False` memory drops (test 0.859→0.656) — the
  model relies less on copying. Net: more real work.
- **Unfreezing SigLIP (B) overfits the encoder → hurts OOD.** In-distribution transitions improve
  (dev_seen subtask 0.156→0.312) but **test subtask drops 0.109→0.047** — finetuning SigLIP on
  franka_3rgb degrades transfer to the franka_1rgb test embodiment.
- **B+upsample is the winner, incl. OOD.** Best `update=True` subtask on every split
  (dev_seen 0.375, dev_unseen 0.516, **test 0.203** — ~2× baseline). Upsampling appears to
  regularize the SigLIP unfreezing so it generalizes rather than overfits. The interaction is
  clearly positive (better than either lever alone).
- **Question format washed out post-finetuning.** `base_q ≈ A` (test subtask 0.109→0.078, mixed
  deltas). token_accuracy is consistently ~+0.01–0.04 vs A — a weak positive — but exact-match does
  not move. See the zero-shot section for why this is surprising.

**Transition ranking (OOD/test):** `B+upsample > upsample > A ≈ base_q > B`.

## pi0.5 format-alignment diagnosis (why base_q was expected to help)

From the pi0.5 paper (arXiv 2504.16054, Fig 3) and `openpi/models/tokenizer.py`:
- pi0.5's **high-level** prompt phrases the goal as a **question** ("How would you clean the
  bedroom?") and outputs `Bounding boxes: <loc..>obj` then `Subtask: <command>`. So **`Subtask:` is
  pi0.5's format** (ours is aligned), there is **no `State:`** and **no memory** in the HL prompt
  (the `Task: {text}, State: {256-bin state};\nAction:` format is the *low-level* head).
- **Zero-shot proof** (`eval_hl_samples.py --zero-shot`, pi05_droid, no training):
  - **question format** → `Subtask: pick up bread` (correct HL mode, on-task).
  - **raw-goal format** → `The image shows a plate with a slice of bread…` (VQA captioning) or
    degenerate `yes`/`no` — the wrong head.
- So the raw-goal prompt was out-of-distribution; the question format unlocks pi0.5's pretrained
  goal→subtask capability **zero-shot**. **But** LoRA finetuning on our (differently-phrased)
  targets overwrites that prior, and exact-match is scored against our vocabulary ("move towards the
  bread" vs pi0.5's "pick up bread"), so the format advantage does not survive to exact-match.
- Not yet tried: pi0.5's **bounding-box localization CoT** (detect the target object, then decide) —
  its built-in grounding step, and the most plausible remaining lever for OOD. Needs bbox
  pseudo-labels (RoboMIND HL data has none).

## Caveats

- **n=64 per bucket ⇒ ≈±6% noise.** Treat sub-0.05 deltas as ties.
- **Exact-match is strict** and scored against our target phrasing; token_accuracy is a softer view.
- **Memory is our own MEM addition**, not in pi0.5's distribution — the `Memory:` field is off-
  distribution for the checkpoint regardless of prompt format.
- **All runs overfit** (~10.6 epochs; dev peaks ~step 3k). A shorter schedule (4–6k steps ≈ 2–3
  epochs, finer eval) would likely generalize better and is worth trying.
- 15k (not 20k) chosen as the common comparison step because A's 20k checkpoint was lost to an early
  `train_hl.py` bug (no final save; fixed in `bfd805d` — later runs keep 5k/10k/15k/20k).

## Reproduce

```bash
# bucketed per-update eval of any config/checkpoint
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval_hl_metrics.py \
    <config> --splits dev_seen dev_unseen test --step 15000 --gen-examples 64
# generated-vs-target samples (add --zero-shot / --question-prompt for the format diagnostics)
uv run python scripts/eval_hl_samples.py <config> --split test --n 6 --step 15000
```

Configs: `src/openpi/training/config_hl.py`. Sbatches: `scripts/nchc/train_hl*.sbatch`.
