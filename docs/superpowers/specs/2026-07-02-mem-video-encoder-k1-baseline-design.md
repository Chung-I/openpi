# MEM Video-Encoder Eval — Sub-project A: K=1 as a Validated Video-Encoder-Off Baseline

**Date:** 2026-07-02
**Status:** Design (approved for spec)
**Part of:** "Verify MEM video-encoder effectiveness (pi0.5, DROID → RoboLab)" — the first of five
sub-projects (A gate → B training → C serving/tunnel → D RoboLab eval → E wandb eval).

## Context

We want to empirically test whether the MEM **video encoder** (`VideoViTEncoder`,
`src/openpi/models/video_vit.py`, wired into `Pi0MEM`) improves a pi0.5-based policy on DROID,
evaluated on RoboLab's easiest tasks. The comparison is a **K=1 vs K=6 ablation** on
`num_video_frames`: same model, video encoder off (K=1) vs on (K=6).

That comparison is only valid if the **K=1 arm is a genuine, provable "video-encoder-off"
baseline** — i.e. numerically the original pi0.5 single-frame image encoder, with no temporal
information. This sub-project establishes and **proves** that guarantee before any GPU-days are
spent training. It is the cheapest step and it de-risks the premise of the whole experiment.

### Key architectural finding (drives the design)

Reading `VideoEncoder1DBlock` (`video_vit.py:54-132`):

- Temporal attention **reuses the same `LayerNorm` + `MultiHeadDotProductAttention` parameters**
  as spatial attention (instantiated once at `:85-91`; used for temporal at `:108-109`, spatial
  at `:118-119`). The docstring states this explicitly (`:57-62`).
- The temporal positional embedding (`temporal_posemb_sincos`, `:26-51`) is **sinusoidal, not
  learned**, and satisfies `e(0)=0` (current frame gets no temporal bias).
- Therefore **K>1 introduces zero new learnable parameters**. The K=1 and K=6 arms are
  **parameter-identical**; the only difference is whether the space-time attention *operation*
  and temporal posemb are applied.

Consequences:
1. A "gradient isolation" test on temporal-only parameters is **impossible** (there are none) —
   it is explicitly **not** part of this spec.
2. The ablation is cleaner than a separate-params design: K=1 vs K=6 differ by an operation, not
   by parameter count or capacity.
3. At K=1, `VideoViTEncoder` skips all temporal ops (`:95`, `:283`) and its parameter paths are
   structurally identical to SigLIP's — so the same weights produce identical output.

## Goal

Guarantee, with code + tests, that a `Pi0MEM` model at `num_video_frames=1`:
1. Routes single-frame input through the **same `VideoViTEncoder`** used at K=6 (not the
   fallback `img` SigLIP), so the two ablation arms share one code path; **and**
2. Produces image-token features **numerically identical to the original pi0.5 image encoder on
   `origin/main`** (the canonical baseline RoboLab would compare against).

## Non-goals (deferred to later sub-projects)

- Short K=1/K=6 training runs, NCHC sbatch, wandb training config (**B**).
- MEM→RoboLab jointpos serving contract and NCHC↔cml18 SSH tunnel (**C**).
- RoboLab simple-tier subset selection and eval runs (**D**).
- Parsing RoboLab output → wandb comparison (**E**).

## Design

### Change 1 — Route K=1 through the video path (approach A1)

Today three gates assume `num_video_frames > 1`, so K=1 silently falls back to the single-frame
`img` SigLIP (`pi0_mem.py:166-174`), a *different* tensor than K=6's `video_img`. Flip the gates
to `>= 1` so a MEM model **always** takes the video path:

- `src/openpi/training/config.py:412` — RLDS repack map (populate `observation/video_*` keys).
- `src/openpi/training/droid_rlds_dataset.py:220` — `gather_history` (build `[T,K]` window).
- `src/openpi/training/droid_rlds_dataset.py:257` — per-frame decode of the video window.

`droid_policy.py:24` already uses `> 0`, so the transform side is K=1-ready. At K=1,
`_video_window_indices(traj_len, 1, stride)` yields a `[T,1]` current-frame window → gather →
`[T,1]` encoded → decode → `[T,1,h,w,3]`; `obs.video_images` becomes `[b,1,h,w,3]` and
`embed_prefix_ll` takes the `video_img` branch (`pi0_mem.py:152-165`).

**Semantics:** "a PI0_MEM model always receives video frames, even if K=1." Only affects K=1 MEM
(no existing config uses it; debug configs use K=2, unaffected by `>=1`). Loses a trivial K=1
decode fast-path — negligible. The two `droid_rlds_dataset.py` gates above are model-agnostic, so
non-MEM DROID configs are kept off by passing `num_video_frames=0` from `data_loader.py`
(`_num_video_frames_for`).

**Rationale over alternatives:** a dedicated `unify_video_path` flag (A2) adds config surface for
a distinction we always want here; model-side wrapping of `obs.images` into a K=1 video (A3)
duplicates frame handling across two code paths. A1 is the smallest correct change.

Keep the `img` fallback path intact — it still serves inference/serving cases where no video
history is provided.

### Change 2 — T1: origin/main equivalence test (the gate's core)

Prove: **K=1 `VideoViTEncoder` output == the original pi0.5 image encoder output on `origin/main`**,
for the same single frame and same weights.

Realization — a committed **golden fixture** generated from an `origin/main` checkout:

1. **Fixture generator** (run once, in an `origin/main` git worktree): build the pi0.5 image
   encoder exactly as `Pi0.__init__` constructs it (`pi0.py:82-88`: `_siglip.Module`,
   `variant="So400m/14"`, `pool_type="none"`, `num_classes=<paligemma width>`) — but at a **tiny
   variant** (e.g. `mu/2`, as the existing `video_vit_test.py:13-20` does) to keep the committed
   fixture small, and with **`scan=False`** so the encoder-block parameter layout is per-layer
   (`encoderblock_{i}`) and thus directly loadable into `VideoViTEncoder` (`scan=True` produces a
   stacked layout; it is functionally identical, only the storage differs). Feed one fixed,
   deterministic image and dump `{params, image, output}` to
   `tests/fixtures/pi05_encoder_ref_origin_main.npz` (or msgpack). Commit it.
2. **T1 test** (on this branch): load the fixture; build a K=1 `VideoViTEncoder` with matching
   `siglip_kwargs`; **load the fixture params into it** (param paths match by design —
   `embedding`, `pos_embedding`, `Transformer/encoderblock_*`, `encoder_norm`, `head`); feed the
   fixture image as `[1,1,h,w,3]`; assert `np.testing.assert_allclose(video_out, fixture.output,
   atol=1e-5)`.

This proves equivalence to `origin/main`'s *actual* encoder function (not merely "some SigLIP in
this branch"), because the reference output is produced by `origin/main` code and the same weights
reproduce it through the K=1 video path.

**Supporting guard** (cheap, catches drift): a test asserting `src/openpi/models/siglip.py` is
unchanged vs `origin/main` (e.g. `git diff --quiet origin/main -- src/openpi/models/siglip.py`,
or a content hash). `VideoViTEncoder` reuses `_siglip.MlpBlock`, `get_posemb`, `decode_variant`
from this branch's `siglip.py`; if those drift from `origin/main`, the fixture must be
regenerated. The guard makes that requirement explicit and self-documenting.

Retain the existing `video_vit_test.py:23 test_single_frame_matches_siglip` (module-level,
same-branch) as complementary coverage.

### Change 3 — T3: data→model K=1 smoke

Confirm the A1 change flows end-to-end at K=1:

- Add a tiny debug config `pi0_mem_k1_debug` in `config.py` (mirroring `pi0_mem_debug` but
  `num_video_frames=1`, `paligemma_variant="dummy"`, `action_expert_variant="dummy"`,
  `wandb_enabled=False`).
- A test (or the config's smoke path) drives one data→model step and asserts: `video_*` keys are
  populated, `obs.video_images` has shape `[b,1,h,w,3]`, `video_image_masks` is `[b,1]`, and
  `embed_prefix_ll` runs the `video_img` branch with a finite LL loss.

## Data flow (K=1 arm, after A1)

```
DROID RLDS
  → gather_history (K=1 window, [T,1])        [droid_rlds_dataset.py:220-231, gated >=1]
  → decode video frames [T,1,h,w,3]            [droid_rlds_dataset.py:257-266, gated >=1]
  → RepackTransform video_* keys               [config.py:412-418, gated >=1]
  → DroidInputs                                [droid_policy.py, already >0]
  → obs.video_images [b,1,h,w,3]
  → Pi0MEM.embed_prefix_ll → self.PaliGemma.video_img  (VideoViTEncoder, temporal skipped at K=1)
  → LL prefix tokens ≡ origin/main pi0.5 single-frame SigLIP features
```

## Testing / definition of done

A is "done" when all of the following hold:

1. `uv run pytest src/openpi/models/video_vit_test.py src/openpi/models/pi0_mem_test.py -q`
   passes, including the new **T1** (origin/main golden-fixture equivalence, atol 1e-5), the
   **siglip-unchanged guard**, and **T3** (K=1 data→model smoke).
2. A ~10-step K=1 MEM smoke on a tiny local RLDS subset (or a bench-style config) completes with
   `video_*` populated, no shape errors, and finite loss.
3. Existing MEM tests remain green (no regression from the gate flips); K=2 debug configs
   unaffected.

## Risks / open items

- **`scan` layout mismatch:** production pi0.5 uses `scan=True` (stacked block params);
  `VideoViTEncoder` uses per-layer blocks. The fixture must be generated with `scan=False`
  (functionally identical) or include an unstack remap. Verify param-path alignment during
  implementation.
- **Fixture size:** use a tiny variant (`mu/2`) so the committed `.npz` stays small while the
  equivalence (an architectural property independent of width/depth) still holds.
- **`_video_window_indices` at K=1:** confirm it returns a well-formed `[T,1]` window (current
  frame) without off-by-one or empty-window edge cases.
- **Serving fallback preserved:** ensure flipping the training-data gates does not remove the
  `img` fallback used when serving without video history.

## Relation to the umbrella experiment

After A proves the baseline is clean, **B** adds two short-run configs
(`pi0_mem_droid_k1_verify`, `pi0_mem_droid_k6_verify`) + an NCHC sbatch with wandb; **C** builds
the MEM→RoboLab jointpos serve contract + SSH tunnel; **D** runs the RoboLab simple-tier subset
(~30 min/eval) per arm; **E** parses RoboLab output into wandb for the K=1-vs-K=6 comparison.
Each is its own spec → plan → implementation cycle.
