# Spec B — Video-memory data pipeline (DROID RLDS)

Date: 2026-06-30
Branch: `mem-fidelity-fixes` (off `mem-pi05-rebase`)
Status: design, pending implementation
Depends on: Spec A (committed), Spec A2 (committed)

## Context

The MEM model consumes `video_images` (per-camera `[K,h,w,c]`) and `video_states`
(`[K, state_dim]`) — the K-frame strided observation history the VideoViT encoder
and the K proprioceptive state tokens are built from. But **nothing populates them
from real data**: the DROID RLDS loader emits only single-frame observations, so
`obs.video_images` is always `None` and the model silently falls back to the
single-frame SigLIP branch. (FakeData / `inputs_spec` already provide fake video,
so the `pi0_mem_*_debug` configs train end-to-end; this spec is specifically the
*real DROID training* path.)

This is the third of five sequenced specs (A → A2 → **B** → C → D).

Scope decisions (from brainstorming):
- **Training only.** This is the training data loader. At inference/eval, frames
  arrive decoded from the live camera into a rolling K-frame buffer (a runtime
  concern, Spec D / deployment) — no RLDS, no decode. Episode-start handling must
  match between training and that buffer to avoid distribution shift.
- **Decode: lazy-K** (gather encoded bytes, decode K frames lazily after flatten),
  matching the existing DROID loader's deliberate late-decode-for-low-memory design.
  Plus a **benchmark** to measure decode/runtime cost and decide later whether the
  decode-once optimization is warranted.
- **Episode-start fill: repeat-oldest** (clamp gather index ≥0). In-distribution,
  no encoder change, trivially consistent with the inference buffer.
- **Masks: camera-presence broadcast to K** (base=True, right-wrist=False), mirroring
  the single-frame `image_mask` and the model's frame-0 gating.

## Goal

Populate `video_images` / `video_states` for MEM training on DROID, leaving the
single-frame path and non-MEM RLDS training unchanged.

## Data flow (video-parallel path at each stage)

```
RLDS dataset (droid_rlds_dataset.py)
  gather_history (NEW, before flatten)  → observation.video_{image,wrist_image,joint_position,gripper_position}
  decode_images (EXT)                   → decode the K video frames (lazy, after flatten)
  ↓ repack (config.py, MEM-gated)       → observation/video_* key remaps
DroidInputs (droid_policy.py, NEW video branch)
  → video_image{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb} [K,h,w,c]
  → video_image_mask {True,True,False} broadcast to K
  → video_states = concat(joint[K,7], gripper[K,1]) → [K,8]
  ↓ DeltaActions (unchanged)
ResizeImages (EXT)        → resize each video frame to 224²
PadStatesAndActions (EXT) → video_states [K,8] → [K, action_dim]
  ↓ Observation.from_dict (already consumes video_image/_mask/video_states)
```

## Components

### 1. RLDS gather-history (`src/openpi/training/droid_rlds_dataset.py`)
A `gather_history` `traj_map` (backward mirror of `chunk_actions`), applied
per-trajectory **before `flatten`**, and only when video is requested
(`num_video_frames > 1`). For timestep `t`, `K = num_video_frames`, stride
`S = video_stride_frames`:
- window indices `[t-(K-1)S, …, t-S, t]`, **clamped to ≥0** (repeat-oldest at
  episode start). Current frame is **last** (index K-1), matching VideoViT's
  `e(0)=0`-at-current temporal posemb.
- `tf.gather` the **encoded** `image`/`wrist_image` bytes → `video_image`,
  `video_wrist_image` (length K); gather `joint_position` `[K,7]`,
  `gripper_position` `[K,1]`.
- Use ONE exterior camera, chosen **consistently** across the K frames (don't
  re-randomize per frame).
- `decode_images` extended to decode the K video frames (lazy, after flatten;
  `tf.map_fn`/vectorized decode over K).

Threading: `DroidRldsDataset` gains `num_video_frames` (from `model_config`) and
`video_stride_frames` (from the data config); the data loader passes them.

Decode efficiency: **lazy-K** as above (each frame decoded ~K× per epoch, parallel
via `num_parallel_calls`). The decode-once-per-trajectory alternative (1× decode,
~15× shuffle memory) is **deferred behind the benchmark** (Component 6).

### 2. Stride config (`src/openpi/training/config.py`)
`RLDSDroidDataConfig.video_stride_frames: int = 15` (DROID ≈15 Hz → ~1 s, the
paper's pretrain stride; configurable for longer-horizon post-training).
`num_video_frames` stays on `Pi0MEMConfig`.

### 3. Repack (`RLDSDroidDataConfig.create`)
Add `video_*` mappings to the repack dict **only when the model is PI0_MEM**
(so non-MEM RLDS runs are untouched):
`observation/video_exterior_image_1_left ← observation/video_image`,
`observation/video_wrist_image_left ← observation/video_wrist_image`,
`observation/video_joint_position`, `observation/video_gripper_position`.

### 4. `DroidInputs` video branch (`src/openpi/policies/droid_policy.py`)
In the `PI0_MEM` branch, when video keys are present, build:
- `video_image` = {`base_0_rgb`=video-exterior `[K,h,w,c]`,
  `left_wrist_0_rgb`=video-wrist, `right_wrist_0_rgb`=`zeros_like`},
- `video_image_mask` = {`base_0_rgb`=`[True]*K`, `left_wrist_0_rgb`=`[True]*K`,
  `right_wrist_0_rgb`=`[False]*K`} (camera-presence broadcast),
- `video_states` = `concat(video_joint_position[K,7], video_gripper_position[K,1])`
  → `[K,8]`.
Parse each frame to uint8 `[h,w,c]` (vectorize `_parse_image` over K).
**Micro-opt:** derive the single-frame `image`/`wrist_image` from the current
(last) video frame `video[...,-1,:,:,:]` to avoid a redundant separate decode of
the current frame (the single-frame `images` are still required because the HL
head uses the single-frame encoder).

### 5. Resize / pad (`src/openpi/transforms.py`)
- `ResizeImages`: also resize each `data["video_image"][name]` frame to 224²
  (guarded `if "video_image" in data`; `resize_with_pad` applied per frame —
  reshape `[K,h,w,c]`→`[K·…]` or `map` if it doesn't accept a leading dim).
- `PadStatesAndActions`: also `data["video_states"] = pad_to_dim(..., action_dim,
  axis=-1)` when present.
Both guarded so non-MEM data is untouched.

### 6. Benchmark (decide decode-once later)
A small benchmark (`scripts/benchmark_video_dataloader.py` or a documented
measurement) timing the DROID RLDS MEM loader (K frames, lazy-K) vs the
single-frame loader: report samples/sec, wall-clock, and the decode share of step
time, using the local `pi0_mem_droid_local` subset (or `pi0_mem_droid_stream`).
Outcome gates whether to implement decode-once. Record results in
`docs/benchmark_streaming_vs_local.md` (or a new note).

### Model
No change — `Observation.from_dict` already maps `video_image`/`video_image_mask`/
`video_states`.

## Naming alignment
`DroidInputs` emits top-level `video_image` / `video_image_mask` / `video_states`,
matching the single-frame `image`/`image_mask`/`state` pattern and what
`Observation.from_dict` reads (`data.get("video_image")` → `video_images`, etc.).

## Testing / verification

1. **`gather_history` unit test** on a synthetic tf trajectory (no DROID download):
   len=100, K=6, S=15 → timestep 90 gathers indices `[15,30,45,60,75,90]`;
   timestep 0 gathers all-clamped-to-0; current frame is last (index 5).
2. **`DroidInputs` video test**: extend `make_droid_example` (or a new example)
   with video keys; assert `video_image[name].shape == (K,h,w,c)`, masks
   `{True×K, True×K, False×K}`, `video_states.shape == (K,8)`, and single-frame
   `image == video_image[base][-1]`.
3. **`ResizeImages` video test**: `[K,h',w',c]` → `[K,224,224,c]`.
4. **`PadStatesAndActions` video test**: `video_states [K,8]` → `[K,action_dim]`.
5. **Non-MEM regression**: a non-MEM RLDS data config build path produces no
   `video_*` keys and is unchanged (repack gating).
6. **E2E (manual, needs DROID data):** run `pi0_mem_droid_local` (local subset)
   for a few steps; confirm `video_images`/`video_states` are populated (shapes
   `[b,K,224,224,3]` / `[b,K,action_dim]`) and a step completes. Full FakeData
   smoke can't exercise the RLDS gather.
7. **Benchmark** (Component 6) produces the decode/runtime numbers used to decide
   on decode-once.
