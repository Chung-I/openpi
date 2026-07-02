# MEM Video-Encoder Eval — Sub-project B: Two short-run K=1/K=6 LoRA arms on NCHC

**Date:** 2026-07-02
**Status:** Design (approved for spec)
**Part of:** "Verify MEM video-encoder effectiveness (pi0.5, DROID → RoboLab)" — the second of five
sub-projects (A gate → **B training** → C serving/tunnel → D RoboLab eval → E wandb comparison).
**Depends on:** Sub-project A (merged, PR #10) — the K=1 video path is validated and numerically
equivalent to the `origin/main` pi0.5 single-frame encoder.

## Goal

Produce two short-run (10k-step) **LoRA**-finetuned `Pi0MEM` checkpoints on DROID —

- **K=6** (`pi0_mem_droid_k6_verify`): video encoder **on** (space-time attention every 4th ViT layer).
- **K=1** (`pi0_mem_droid_k1_verify`): video encoder **off** — the A-validated single-frame baseline.

initialized from the DROID-finetuned pi0.5 (`pi05_droid`), launched **in parallel** on NCHC
(4 GPUs each), tracked in one wandb project. The two runs give an early feel for whether the video
encoder helps, and become the checkpoints that C serves and D evaluates on RoboLab.

This is a **"get a feel" run**, not a full training. 10k steps is a fraction of one DROID epoch
(~100k steps at effective batch 256); the design deliberately trades statistical power for fast
turnaround, and mitigates the two things that would otherwise make a short run *uninterpretable*
(random vision encoder; horizon mismatch — see below).

## Context and key findings (drive the design)

The proven `pi0_mem_droid_stream` config (`config.py:1046`) already trains `Pi0MEM` on DROID
end-to-end on NCHC (checkpoints exist under
`/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_stream/`): full finetune from `pi05_base`,
streaming DROID from GCS, `fsdp_devices=4`, `batch_size=128`, `grad_accum_steps=2`, 100k steps.
B derives its two arms from this config; the decisions below are the deliberate *departures* from it.

### F1 — LoRA, not full FT (and it is unvalidated on real DROID)

`Pi0MEMConfig` defaults `lora=False`; the stream config is full finetuning. LoRA **is** wired
(`lora=True` → `get_freeze_filter()` freezes base `.*llm.*` in both experts, excludes `.*lora.*`),
but has only ever run with **dummy** variants (`pi0_mem_lora_debug`) — never on real DROID. B uses
LoRA (cheaper/faster for a "feel" run) and therefore must add validation before spending GPU-hours
(see Testing).

Under the LoRA freeze filter, **`video_img`, `state_proj`, and MEM heads stay fully trainable**
(they are not under `.*llm.*`; `PaliGemma = nnx.Dict(llm, img, video_img)`, `pi0_mem.py:119`). So
the video encoder can learn in both arms, and the only train-time difference between arms is
whether temporal attention runs.

### F2 — Init from `pi05_droid`, not `pi05_base`

From `pi05_base`, 10k steps barely learns DROID → K1≈K6≈weak (ambiguous). From the DROID-finetuned
`pi05_droid` (`gs://openpi-assets/checkpoints/pi05_droid/params`), the model is already
DROID-competent, so 10k steps only adapt the new video/MEM pathway and any K1-vs-K6 gap is cleanly
attributable to the video encoder.

### F3 — `video_img` currently initializes to RANDOM weights (the critical fix)

`video_img` (`VideoViTEncoder`) is a separate SigLIP-architecture module. It is in the weight
loader's `missing_regex` → it initializes **random**. Meanwhile pi0.5's *pretrained* SigLIP loads
into the `img` fallback (`pi0_mem.py:92`, `scan=True`), which is **unused** during K≥1 training.
Result today: the LL/video vision encoder trains from scratch. Acceptable over 100k steps (stream),
but **fatal to signal at 10k** — both arms would spend the budget learning to see, and it breaks
A's premise that K=1 *is* the pi0.5 baseline (A's equivalence required loading pi0.5's SigLIP
*into* `video_img`).

**Fix:** remap the pretrained SigLIP (already downloaded into `img`) into `video_img` at load, so
both arms start from a trained vision encoder and only *adapt* (K=6 additionally learning
temporal). Because temporal attention **reuses spatial params** (`video_vit.py:55-60`, "no new
learnable parameters"; temporal layers selected by `(lyr+1) % temporal_attn_every_n_layers == 0` at
`video_vit.py:306`, default 4 — exactly the MEM paper's "modify every 4th ViT layer for spatial +
temporal context"), copying SigLIP → `video_img` fills **all** learnable params; nothing is left
random. Cost: `img` is `scan=True` (stacked transformer blocks); `video_img` is `scan=False`
(per-layer, required for its temporal-every-N Python loop) — so the remap must **un-stack** the
`encoderblock` params (`Transformer/encoderblock` → `Transformer/encoderblock_{i}`). A already
analyzed this exact scan-layout issue.

### F4 — `action_horizon = 16` (DROID-native), not 50 (paper) or 15 (serving)

pi0.5's *paper* uses H=50; that is the *base* recipe. On DROID, openpi **re-specialized the
prediction horizon down**: the training target length **is** `action_horizon`
(`data_loader.py:167` sets `action_chunk_size=action_horizon`; `droid_rlds_dataset.py:60` defaults
`action_chunk_size=16`), the `pi05_droid_finetune` recipe uses 16, and deployment executes only
`open_loop_horizon=8` before re-querying (`examples/droid/main.py:42,114,137`). So the DROID-native
prediction horizon of the checkpoint we load is **16**; 50 would train MEM to predict a 3.3×
longer chunk than `pi05_droid` ever learned (pure adaptation cost, no DROID benefit), and 15 is
only an inference-time serving detail. **No weight is horizon-shaped** (action in/out projections
are per-timestep; the pi0.5 time signal is a computed MLP, not a horizon-length table), so the
change is free at load time. Both arms use `action_horizon=16`.

### F5 — `state_proj` is not a new param *type*

Stock pi0.5 has the identical `state_proj = nnx.Linear(action_dim, width)` (`pi0.py:97`). It is in
`missing_regex` only because the released `pi05_droid` feeds state as **discrete tokens** (no
continuous `state_proj` weight to copy), so MEM inits it fresh. It is tiny and learns quickly — no
special handling. (Documented here to close the "MEM adds no new params" question: the paper's
no-new-params claim is about *temporal reusing spatial* inside the encoder, which holds; it is not
a claim about MEM-vs-pi0.5 module count.)

## Design

### Change 1 — Two training configs in `config.py`

Two `TrainConfig`s derived from `pi0_mem_droid_stream`, **differing only in `num_video_frames`**:

| field | `pi0_mem_droid_k6_verify` | `pi0_mem_droid_k1_verify` |
|---|---|---|
| `model.num_video_frames` | 6 (encoder on) | 1 (encoder off; A-proven ≡ single-frame SigLIP) |
| everything else | identical | identical |

Shared settings:

- **model:** `Pi0MEMConfig(pi05=True, action_dim=32, action_horizon=16, num_video_frames=K, lora=True)`
- **freeze_filter:** `Pi0MEMConfig(...same..., lora=True).get_freeze_filter()`
- **weight_loader:** a checkpoint loader over `gs://openpi-assets/checkpoints/pi05_droid/params`
  that (a) loads pi0.5-DROID params by matching path, (b) treats `.*(lora|state_proj|video_img).*`
  as expected-missing (fresh init), **and (c) remaps the loaded SigLIP (`img`) into `video_img`,
  un-stacking `scan=True` → `scan=False`** (Change 2).
- **data:** `RLDSDroidDataConfig(repo_id="droid", rlds_data_dir="gs://gresearch/robotics",
  action_space=JOINT_POSITION, datasets=(droid 1.0.1, filter droid_sample_ranges_v1_0_1.json),
  assets=AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
  asset_id="droid"))` — streams DROID from GCS with the DROID-finetuned norm stats.
- **lr_schedule:** same cosine as stream (`warmup_steps=1_000, peak_lr=5e-5, decay_steps=1_000_000,
  decay_lr=5e-5`) — near-constant over 10k; identical for both arms.
- **compute:** `fsdp_devices=4`, `batch_size=128`, `grad_accum_steps=1`, `num_train_steps=10_000`,
  `seed=42` (same seed both arms for fairness), `num_workers=0` (RLDS requirement).
- **checkpoints:** `save_interval=2_000`, `keep_period=10_000` (final step-10000 checkpoint +
  assets retained for C/D).
- **wandb:** `project_name="video-encoder-eval"`, `wandb_enabled=True`; run names are the two
  `exp_name`s (`wandb.init` uses `project=config.project_name`, `name=config.exp_name`;
  `train.py:62-65`). "One project" = both arms share `project_name`; the two `exp_name`s
  distinguish the runs for side-by-side loss curves. (No native wandb `group`; a formal group is a
  one-line `wandb.init(group=...)` addition if later desired — out of scope here.)

### Change 2 — SigLIP → `video_img` weight remap (the F3 fix)

Extend/replace the checkpoint weight loader so that, after the normal path-matched load, the
loaded pi0.5 SigLIP params (the `img` subtree, `scan=True`) are **copied into the `video_img`
subtree** (`scan=False`), un-stacking the stacked `Transformer/encoderblock` array into per-layer
`Transformer/encoderblock_{i}` params. All other `video_img` param paths (`embedding`,
`pos_embedding`, `encoder_norm`, `head`) map directly (identical to A's fixture alignment).

Realization options (decided at plan time): (a) a new `WeightLoader` subclass wrapping
`CheckpointWeightLoader` that performs the img→video_img remap on the loaded tree **before**
`_merge_params`; or (b) a small remap transform composed with the existing loader.

`missing_regex` is kept **unchanged** (`.*(lora|state_proj|video_img).*`). Because the remap runs
before `_merge_params`, the `video_img` spatial keys are already present in the loaded tree and get
matched by path (so the regex branch never fires for them); only genuinely-fresh keys (`lora`,
`state_proj`) actually fall through to fresh init. Keeping the regex broad is deliberately robust:
if the remap silently no-ops, the keys fall back to fresh init rather than being dropped — and that
failure is caught loudly by T-B4 and the NCHC key-diff (which assert `video_img` values came from
the remapped SigLIP, not random init), not left to a silent bad run.

The remap must be **layout-correct**: SigLIP `scan=True` stores encoder blocks as a single stacked
array of shape `[depth, ...]`; `video_img` `scan=False` expects `depth` separate
`encoderblock_{i}` params. The transform slices the stacked array along axis 0. This is the one
non-trivial piece; it is guarded by T-B4.

### Change 3 — NCHC launch script

`scripts/nchc/train_mem_video_eval.sbatch`, mirroring existing NCHC conventions
(`scripts/nchc/run_bakeoff_full.sbatch`): `--account=MST114563`, `cd /work/roboleon1295/openpi`,
`export HF_HOME=/work/roboleon1295/hf_cache`, `uv run --no-sync`, logs to
`/work/roboleon1295/openpi/logs/`.

- Requests the full node on the **8-GPU partition** (`--gres=gpu:8`, appropriate `--cpus-per-task`,
  `--mem`, `--time` sized for two parallel 10k LoRA runs — LoRA + frozen backbone is light; budget
  generously, e.g. 12h).
- Launches **two backgrounded** `uv run scripts/train.py` processes:
  - K6: `CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 ... --config pi0_mem_droid_k6_verify`
  - K1: `CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 ... --config pi0_mem_droid_k1_verify`
- `wait`s for both, tails both logs, reports both exit codes.

(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` per user preference; the two JAX processes see disjoint GPU
sets so 0.8 each is safe.)

## Data flow (per arm)

```
DROID RLDS (stream from GCS)
  → gather_history (K window)                     [droid_rlds_dataset.py, gated >=1 by A]
  → decode video frames [T,K,h,w,3]               [droid_rlds_dataset.py, gated >=1 by A]
  → RepackTransform video_* keys                  [config.py:412, gated >=1 by A]
  → DroidInputs / DeltaActions (jointpos)         [droid_policy.py]
  → obs.video_images [b,K,h,w,3]
  → Pi0MEM (LoRA): video_img (SigLIP-init, temporal every 4th layer; skipped at K=1)
  → LL/HL/action losses (flow matching, action_horizon=16 target)
  → wandb: project=video-encoder-eval, name=pi0_mem_droid_k{1,6}_verify
```

## Testing / definition of done

### Committed CPU unit tests (dummy/tiny variants, alongside `pi0_mem_test.py`)

- **T-B1 — LoRA freeze partition.** For `Pi0MEMConfig(lora=True)` at K=1 and K=6, the trainable set
  `nnx.All(Param, Not(freeze_filter))` **includes** all `video_img.*`, `state_proj.*`, `.*lora.*`,
  and **excludes** base `.*llm.*` (both experts).
- **T-B2 — `missing_regex` coverage invariant.** Build the MEM reference param tree; synthesize a
  "pi0.5-like" loaded tree (reference minus the MEM-new keys, and *without* the img→video_img remap
  applied); run `_merge_params` with `.*(lora|state_proj|video_img).*`; assert
  `result.keys() == reference.keys()` (nothing silently dropped), base keys took loaded values,
  `lora`/`state_proj`/`video_img` took fresh init. Pins the regex to the actual MEM key names and
  proves the fresh-init fallback for `video_img` is safe when the remap is absent.
- **T-B3 — LoRA forward+grad smoke.** Realized as the **NCHC 10-step smoke** (below), not a
  committed CPU unit test: `dummy` gemma variants create **no** LoRA params (only `gemma_*_lora`
  do), and instantiating `gemma_2b` for a real fwd/bwd is too heavy for CI. The 10-step smoke on
  the real `gemma_2b_lora` model checks a **finite, non-increasing** loss (and, by construction,
  the optimizer is built over `trainable_filter` only, so frozen base gets no update — T-B1 locks
  that partition). T-B1 covers the trainable/frozen partition at unit scale via `eval_shape`.
- **T-B4 — SigLIP → `video_img` remap equivalence.** Apply the Change-2 remap from a SigLIP
  (`scan=True`) param tree into a `video_img` (`scan=False`) tree; assert per-layer params equal
  the un-stacked source, and (tying to A) a K=1 forward through the remapped `video_img` equals a
  forward through the source SigLIP on the same frame (`atol=1e-5`).

### NCHC pre-launch check (real `pi05_droid` checkpoint — where the true path-mismatch footgun lives)

- A small script loads `pi05_droid` params + builds the MEM init keys; asserts
  **unmatched-reference-keys ⊆ `missing_regex`** (no base weight left random) and that the
  img→video_img remap covered the `video_img` spatial keys; prints the missing/dropped sets.
- **10-step smoke per arm** (bench-style, `overwrite`, `wandb_enabled=False`): builds under LoRA,
  `video_*` populated, `video_img` present in trainable params, finite **decreasing** loss.

### Done when

1. A-suite green on this branch (`video_vit_test.py`, `pi0_mem_test.py`) — the K=1 arm still routes
   through the A-validated VideoViT encoder.
2. T-B1..T-B4 pass; NCHC pre-launch key-diff + 10-step smoke pass for both arms.
3. Both 10k arms complete in parallel on NCHC; two wandb runs under `video-encoder-eval` with loss
   curves; final step-10000 checkpoints + assets saved on `/work` ready for C.

## Non-goals (deferred to later sub-projects)

- MEM→RoboLab jointpos serving contract + NCHC↔cml18 SSH tunnel, incl. any horizon-N→DROID serving
  adapter (**C**).
- RoboLab simple-tier subset selection and eval runs (**D**).
- Parsing RoboLab output → wandb comparison (**E**).
- Full-length training; training-time RTC (deferred D2 from the fidelity roadmap).

## Risks / open items

- **LoRA-on-real-DROID is unvalidated** (only dummy debug so far) → T-B1..T-B3 + the 10-step
  loss-decreasing smoke gate catch build/shape/grad/convergence issues before GPU-hours.
- **SigLIP→`video_img` scan un-stack** is the one non-trivial transform → guarded by T-B4 and the
  NCHC key-diff; if it regresses, `video_img` silently falls back to random (caught by the key-diff
  asserting the remap covered the spatial keys).
- **`pi05_droid` param compatibility** (horizon 16 vs its native regime; discrete-state → fresh
  `state_proj`) → the NCHC key-diff asserts exactly the MEM-new keys are missing; no weight is
  horizon-shaped.
- **10k steps may be too short** for a clear K6>K1 gap → accepted; this is a "get a feel" run and D
  interprets accordingly. F3 (pretrained `video_img`) is what makes even 10k interpretable.
- **Two JAX processes on one node** → disjoint `CUDA_VISIBLE_DEVICES`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` each.

## Relation to the umbrella experiment

A proved the K=1 baseline is clean. **B** (this spec) produces the two short LoRA checkpoints +
wandb loss curves. **C** builds the MEM→RoboLab jointpos serve contract + SSH tunnel (and any
horizon→DROID serving adapter). **D** runs the RoboLab simple-tier subset per arm. **E** parses
RoboLab output into wandb for the K=1-vs-K=6 comparison. Each is its own spec → plan →
implementation cycle.
