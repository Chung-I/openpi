# Spec B — Video-memory data pipeline (DROID RLDS) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the MEM model's `video_images` (per-camera `[K,h,w,c]`) and `video_states` (`[K,8]`) from the DROID RLDS dataset — a strided K-frame history window — leaving the single-frame path and all non-MEM training unchanged.

**Architecture:** Add a backward `gather_history` step to the RLDS trajectory pipeline (mirror of `chunk_actions`), thread `num_video_frames`/`video_stride_frames` into the loader, remap the new keys (MEM-gated), build the video fields in `DroidInputs`, and extend `ResizeImages`/`PadStatesAndActions` to the video fields. Lazy-K decode (decode K frames late, after flatten); repeat-oldest at episode start; camera-presence masks broadcast to K.

**Tech Stack:** TensorFlow / `tf.data` / `dlimp` (RLDS), NumPy, JAX, pytest, `uv`.

## Global Constraints

- Run via `uv run`; prefix JAX/test commands with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- Branch: `mem-fidelity-fixes` (Specs A + A2 already committed).
- Video gathering is **MEM-gated**: it runs only when `num_video_frames > 1`; every non-MEM RLDS path (pi0/pi0.5/pi0-fast) and the single-frame fields stay byte-for-byte unchanged.
- Window ordering: indices `[t-(K-1)S, …, t-S, t]`, clamped to `>= 0` (repeat-oldest), **current frame last** (index K-1) — matches VideoViT's `e(0)=0`-at-current temporal posemb.
- Masks: camera-presence broadcast to K — `base_0_rgb`=True, `left_wrist_0_rgb`=True, `right_wrist_0_rgb`=False.
- `video_stride_frames` default 15 (DROID ≈15 Hz ≈ 1 s); lives on the data config. `num_video_frames` lives on the model config.
- Decode strategy is **lazy-K**; a benchmark (Task 5) decides whether decode-once is later warranted — do NOT build decode-once now.
- The exterior-camera pick in `restructure` is already per-trajectory; reuse `observation.image` (do not re-randomize per frame).

---

### Task 1: RLDS gather-history + lazy video decode

**Files:**
- Modify: `src/openpi/training/droid_rlds_dataset.py` (add `_video_window_indices` helper; `DroidRldsDataset.__init__` gains `num_video_frames`/`video_stride_frames`; add `gather_history` traj_map + video decode)
- Test: `src/openpi/training/droid_rlds_dataset_test.py` (create if absent)

**Interfaces:**
- Produces: module fn `_video_window_indices(traj_len, num_frames, stride)` returning an int32 tensor `[traj_len, num_frames]` (clamped, current-last). `DroidRldsDataset(..., num_video_frames: int = 1, video_stride_frames: int = 15)`. When `num_video_frames > 1`, each flattened frame gains `observation.video_image`/`video_wrist_image` (`[K,h,w,3]` uint8) and `observation.video_joint_position`/`video_gripper_position` (`[K,7]`/`[K,1]`).

- [ ] **Step 1: Write the failing test for the index helper**

Create `src/openpi/training/droid_rlds_dataset_test.py`:

```python
import numpy as np
import tensorflow as tf

from openpi.training.droid_rlds_dataset import _video_window_indices


def test_video_window_indices_strided_and_current_last():
    idx = _video_window_indices(traj_len=100, num_frames=6, stride=15)
    idx = idx.numpy() if isinstance(idx, tf.Tensor) else np.asarray(idx)
    assert idx.shape == (100, 6)
    # timestep 90: [15, 30, 45, 60, 75, 90], current (90) last
    np.testing.assert_array_equal(idx[90], [15, 30, 45, 60, 75, 90])
    # timestep 0: all clamped to 0 (repeat-oldest)
    np.testing.assert_array_equal(idx[0], [0, 0, 0, 0, 0, 0])
    # current frame is always the last column
    np.testing.assert_array_equal(idx[:, -1], np.arange(100))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/training/droid_rlds_dataset_test.py -v`
Expected: FAIL — `ImportError: cannot import name '_video_window_indices'`.

- [ ] **Step 3: Add the index helper**

In `src/openpi/training/droid_rlds_dataset.py`, at module level (tf is imported inside `__init__`; import it lazily in the helper too):

```python
def _video_window_indices(traj_len, num_frames: int, stride: int):
    """Backward strided window indices, shape [traj_len, num_frames].

    For each timestep t: [t-(num_frames-1)*stride, ..., t-stride, t], clamped to
    >= 0 (repeat-oldest at episode start). The current frame (offset 0) is the
    LAST column, matching the video encoder's e(0)=0-at-current temporal posemb.
    """
    import tensorflow as tf

    offsets = (tf.range(num_frames) - (num_frames - 1)) * stride  # [K], last == 0
    base = tf.range(traj_len)[:, None]  # [T, 1]
    idx = base + offsets[None, :]  # [T, K]
    return tf.maximum(idx, 0)
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/training/droid_rlds_dataset_test.py::test_video_window_indices_strided_and_current_last -v`
Expected: PASS.

- [ ] **Step 5: Add constructor params + gather_history + video decode**

In `DroidRldsDataset.__init__`, add keyword-only params (after `num_parallel_calls`):

```python
        num_video_frames: int = 1,
        video_stride_frames: int = 15,
```

Store them before `prepare_single_dataset` so the closures capture them:

```python
        self.num_video_frames = num_video_frames
        self.video_stride_frames = video_stride_frames
```

Inside `prepare_single_dataset`, after `dataset = dataset.traj_map(restructure, num_parallel_calls)` and before/after `chunk_actions`, add the gather (gated):

```python
            if num_video_frames > 1:
                def gather_history(traj):
                    traj_len = tf.shape(traj["observation"]["image"])[0]
                    idx = _video_window_indices(traj_len, num_video_frames, video_stride_frames)  # [T,K]
                    obs = traj["observation"]
                    obs["video_image"] = tf.gather(obs["image"], idx)  # [T,K] encoded bytes
                    obs["video_wrist_image"] = tf.gather(obs["wrist_image"], idx)
                    obs["video_joint_position"] = tf.gather(obs["joint_position"], idx)  # [T,K,7]
                    obs["video_gripper_position"] = tf.gather(obs["gripper_position"], idx)  # [T,K,1]
                    return traj

                dataset = dataset.traj_map(gather_history, num_parallel_calls)
```

Extend `decode_images` to lazily decode the K video frames when present (leave the single-frame decode as-is so non-MEM is unchanged):

```python
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                if num_video_frames > 1:
                    def _decode_k(frames):  # frames: [K] encoded -> [K,h,w,3]
                        return tf.map_fn(
                            lambda x: tf.io.decode_image(x, expand_animations=False, dtype=tf.uint8),
                            frames,
                            fn_output_signature=tf.uint8,
                        )
                    traj["observation"]["video_image"] = _decode_k(traj["observation"]["video_image"])
                    traj["observation"]["video_wrist_image"] = _decode_k(traj["observation"]["video_wrist_image"])
                return traj
```

(`num_video_frames`/`video_stride_frames` are closure-captured from `__init__`'s parameters.)

- [ ] **Step 6: Run the test file (helper still green; no full-dataset run needed)**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/training/droid_rlds_dataset_test.py -v`
Expected: PASS. (Full-pipeline execution needs DROID data and is covered by Task 5's manual E2E.)

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/droid_rlds_dataset.py src/openpi/training/droid_rlds_dataset_test.py
git commit -m "$(cat <<'EOF'
feat(droid_rlds): strided video-history gather + lazy K-frame decode (Spec B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 2: Thread params + MEM-gated repack

**Files:**
- Modify: `src/openpi/training/config.py` (`DataConfig.video_stride_frames`; `RLDSDroidDataConfig.video_stride_frames` + set it in `create`; MEM-gated `video_*` repack)
- Modify: `src/openpi/training/data_loader.py` (`create_rlds_dataset`/`create_torch_dataloader` pass `num_video_frames` + `video_stride_frames`)
- Test: `src/openpi/training/config_test.py` (or existing config test location)

**Interfaces:**
- Consumes: `DroidRldsDataset(num_video_frames=, video_stride_frames=)` (Task 1).
- Produces: `DataConfig.video_stride_frames: int = 15`; `RLDSDroidDataConfig.video_stride_frames: int = 15`; repack includes `observation/video_*` keys iff `model_config.model_type == PI0_MEM`.

- [ ] **Step 1: Write the failing test**

Append to the config test file:

```python
def test_rlds_droid_repack_has_video_keys_for_mem():
    import pathlib
    from openpi.training import config as _config
    from openpi.models import pi0_mem_config

    factory = _config.RLDSDroidDataConfig(rlds_data_dir="/tmp/x")
    mem_cfg = factory.create(pathlib.Path("/tmp"), pi0_mem_config.Pi0MEMConfig(num_video_frames=6))
    repack = mem_cfg.repack_transforms.inputs[0].structure  # RepackTransform mapping
    assert "observation/video_exterior_image_1_left" in repack
    assert mem_cfg.video_stride_frames == 15
```

(Adapt `.structure` to the actual `RepackTransform` attribute name — check `transforms.RepackTransform`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k test_rlds_droid_repack_has_video_keys_for_mem -v`
Expected: FAIL — no `video_stride_frames` attr / video keys absent.

- [ ] **Step 3: Add `video_stride_frames` to `DataConfig` and `RLDSDroidDataConfig`**

In `src/openpi/training/config.py`, add to the `DataConfig` dataclass:

```python
    # Stride (in dataset frames) between sampled video-memory frames (MEM only).
    video_stride_frames: int = 15
```

Add to `RLDSDroidDataConfig` (the factory) the same field with default 15.

- [ ] **Step 4: MEM-gate the repack + set stride in `RLDSDroidDataConfig.create`**

In `RLDSDroidDataConfig.create`, build the repack mapping conditionally:

```python
        repack_map = {
            "observation/exterior_image_1_left": "observation/image",
            "observation/wrist_image_left": "observation/wrist_image",
            "observation/joint_position": "observation/joint_position",
            "observation/gripper_position": "observation/gripper_position",
            "actions": "actions",
            "prompt": "prompt",
        }
        if model_config.model_type == _model.ModelType.PI0_MEM:
            repack_map.update({
                "observation/video_exterior_image_1_left": "observation/video_image",
                "observation/video_wrist_image_left": "observation/video_wrist_image",
                "observation/video_joint_position": "observation/video_joint_position",
                "observation/video_gripper_position": "observation/video_gripper_position",
            })
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_map)])
```

And include `video_stride_frames=self.video_stride_frames` in the returned `dataclasses.replace(self.create_base_config(...), ...)` call.

- [ ] **Step 5: Thread `num_video_frames` + `video_stride_frames` through the loader**

In `src/openpi/training/data_loader.py`, `create_rlds_dataset` — add params and pass them:

```python
def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_video_frames: int = 1,
) -> Dataset:
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
        num_video_frames=num_video_frames,
        video_stride_frames=data_config.video_stride_frames,
    )
```

In `create_torch_dataloader` (the caller), add a `num_video_frames` param (default 1) and pass it to `create_rlds_dataset`. In `create_data_loader` (where `config.model` is available), pass `num_video_frames=getattr(config.model, "num_video_frames", 1)` down to `create_torch_dataloader`.

- [ ] **Step 6: Run the config test + a non-MEM regression check**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k "rlds_droid_repack or config_registered" -v`
Expected: PASS — MEM repack has video keys; add/confirm an assertion that a non-MEM (`pi0`) `RLDSDroidDataConfig.create` repack has NO `video_*` keys.

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/config.py src/openpi/training/data_loader.py src/openpi/training/config_test.py
git commit -m "$(cat <<'EOF'
feat(config): thread num_video_frames/stride + MEM-gated video repack (Spec B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 3: DroidInputs video branch

**Files:**
- Modify: `src/openpi/policies/droid_policy.py` (`make_droid_example` + `DroidInputs.__call__` video branch)
- Test: `src/openpi/policies/droid_policy_test.py` (create if absent)

**Interfaces:**
- Consumes: repacked keys `observation/video_exterior_image_1_left` (`[K,h,w,3]`), `observation/video_wrist_image_left`, `observation/video_joint_position` (`[K,7]`), `observation/video_gripper_position` (`[K,1]`).
- Produces (top-level, when video keys present and model is PI0_MEM): `video_image` = {`base_0_rgb`,`left_wrist_0_rgb`,`right_wrist_0_rgb`} each `[K,h,w,3]`; `video_image_mask` = {True×K, True×K, False×K}; `video_states` = `[K,8]`. Single-frame `image`/`state` derived from the current (last) video frame.

- [ ] **Step 1: Write the failing test**

Create `src/openpi/policies/droid_policy_test.py`:

```python
import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy


def _mem_example(K=6):
    return {
        "observation/video_exterior_image_1_left": np.random.randint(256, size=(K, 224, 224, 3), dtype=np.uint8),
        "observation/video_wrist_image_left": np.random.randint(256, size=(K, 224, 224, 3), dtype=np.uint8),
        "observation/video_joint_position": np.random.rand(K, 7).astype(np.float32),
        "observation/video_gripper_position": np.random.rand(K, 1).astype(np.float32),
        "prompt": "do something",
    }


def test_droid_inputs_builds_video_fields():
    K = 6
    out = droid_policy.DroidInputs(model_type=_model.ModelType.PI0_MEM)(_mem_example(K))
    assert out["video_image"]["base_0_rgb"].shape == (K, 224, 224, 3)
    assert out["video_image"]["right_wrist_0_rgb"].shape == (K, 224, 224, 3)
    assert out["video_image_mask"]["base_0_rgb"].shape == (K,)
    assert bool(out["video_image_mask"]["base_0_rgb"].all())
    assert not bool(out["video_image_mask"]["right_wrist_0_rgb"].any())
    assert out["video_states"].shape == (K, 8)
    # single-frame image == current (last) video frame
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], out["video_image"]["base_0_rgb"][-1])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/policies/droid_policy_test.py -v`
Expected: FAIL — `KeyError: 'video_image'` (DroidInputs doesn't produce video fields).

- [ ] **Step 3: Add the video branch to `DroidInputs`**

In `src/openpi/policies/droid_policy.py`, at the start of `__call__`, before the existing single-frame logic, detect video and build the fields (only for PI0_MEM with video keys present). Build the K-frame stacks, derive the single-frame inputs from the current (last) video frame, then return both. Concretely, when `"observation/video_exterior_image_1_left" in data` and `self.model_type == _model.ModelType.PI0_MEM`:

```python
        video_keys_present = (
            self.model_type == _model.ModelType.PI0_MEM
            and "observation/video_exterior_image_1_left" in data
        )
        if video_keys_present:
            v_ext = np.stack([_parse_image(f) for f in np.asarray(data["observation/video_exterior_image_1_left"])])
            v_wrist = np.stack([_parse_image(f) for f in np.asarray(data["observation/video_wrist_image_left"])])
            K = v_ext.shape[0]
            video_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            video_images = (v_ext, v_wrist, np.zeros_like(v_ext))
            video_masks = (
                np.ones(K, dtype=bool),
                np.ones(K, dtype=bool),
                np.zeros(K, dtype=bool),
            )
            video_states = np.concatenate(
                [np.asarray(data["observation/video_joint_position"]),
                 np.asarray(data["observation/video_gripper_position"])],
                axis=-1,
            )  # [K, 8]
            # Derive single-frame inputs from the current (last) video frame.
            base_image = v_ext[-1]
            wrist_image = v_wrist[-1]
            state = video_states[-1]
        else:
            # ... existing single-frame parsing (state, base_image, wrist_image) ...
```

Keep the existing single-frame `names/images/image_masks` block to build `image`/`image_mask`. Then, after assembling `inputs`, add the video fields when present:

```python
        if video_keys_present:
            inputs["video_image"] = dict(zip(video_names, video_images, strict=True))
            inputs["video_image_mask"] = dict(zip(video_names, video_masks, strict=True))
            inputs["video_states"] = video_states
```

Extend `make_droid_example` to optionally emit the video keys (so policy tooling has an example) — add a `num_video_frames: int = 0` arg that, when > 0, includes the four `observation/video_*` keys.

- [ ] **Step 4: Run the test to verify it passes**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/policies/droid_policy_test.py -v`
Expected: PASS.

- [ ] **Step 5: Non-MEM regression**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/policies/ -v`
Expected: PASS — a `pi0`/`pi0_fast` example (no video keys) produces no `video_*` fields and the single-frame output is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/openpi/policies/droid_policy.py src/openpi/policies/droid_policy_test.py
git commit -m "$(cat <<'EOF'
feat(droid_policy): build video_image/_mask/video_states for MEM (Spec B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 4: Video-aware ResizeImages + PadStatesAndActions

**Files:**
- Modify: `src/openpi/transforms.py` (`ResizeImages`, `PadStatesAndActions`)
- Test: `src/openpi/transforms_test.py` (or existing transforms test location)

**Interfaces:**
- Consumes: `data["video_image"]` (dict of `[K,h,w,c]`), `data["video_states"]` (`[K,8]`).
- Produces: resized `video_image` frames `[K,height,width,c]`; padded `video_states` `[K,model_action_dim]`.

- [ ] **Step 1: Write the failing test**

Append to the transforms test file:

```python
import numpy as np
from openpi import transforms


def test_resize_images_handles_video():
    data = {
        "image": {"base_0_rgb": np.zeros((180, 320, 3), np.uint8)},
        "video_image": {"base_0_rgb": np.zeros((6, 180, 320, 3), np.uint8)},
    }
    out = transforms.ResizeImages(224, 224)(data)
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert out["video_image"]["base_0_rgb"].shape == (6, 224, 224, 3)


def test_pad_states_and_actions_handles_video_states():
    data = {"state": np.zeros(8, np.float32), "video_states": np.zeros((6, 8), np.float32)}
    out = transforms.PadStatesAndActions(model_action_dim=32)(data)
    assert out["state"].shape == (32,)
    assert out["video_states"].shape == (6, 32)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k "resize_images_handles_video or pad_states_and_actions_handles_video" -v`
Expected: FAIL — `video_image`/`video_states` left unresized/unpadded (KeyError or wrong shape).

- [ ] **Step 3: Extend `ResizeImages`**

In `src/openpi/transforms.py`, `ResizeImages.__call__`:

```python
    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        if "video_image" in data:
            data["video_image"] = {
                k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["video_image"].items()
            }
        return data
```

If `image_tools.resize_with_pad` does not accept a leading frame axis (`[K,h,w,c]`), reshape per-frame: `np.stack([resize_with_pad(f, H, W) for f in v])`. Verify which by reading `image_tools.resize_with_pad`; use the vectorized form if it already supports a batch/leading dim.

- [ ] **Step 4: Extend `PadStatesAndActions`**

```python
    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "video_states" in data:
            data["video_states"] = pad_to_dim(data["video_states"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k "resize_images_handles_video or pad_states_and_actions_handles_video" -v`
Expected: PASS. Then run the full transforms test file to confirm no regression.

- [ ] **Step 6: Commit**

```bash
git add src/openpi/transforms.py src/openpi/transforms_test.py
git commit -m "$(cat <<'EOF'
feat(transforms): video-aware ResizeImages + PadStatesAndActions (Spec B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 5: Dataloader benchmark + E2E verification note

**Files:**
- Create: `scripts/benchmark_video_dataloader.py`
- Modify: `docs/benchmark_streaming_vs_local.md` (record results) — or create it if absent

**Interfaces:**
- Consumes: an RLDS DROID config (`pi0_mem_droid_local` preferred — local subset; falls back to `pi0_mem_droid_stream`).
- Produces: printed metrics (samples/sec, mean step time, and — if measurable — decode share) for the MEM video loader vs the single-frame loader.

- [ ] **Step 1: Write the benchmark script**

Create `scripts/benchmark_video_dataloader.py`:

```python
"""Benchmark DROID RLDS data loading: MEM (K-frame video) vs single-frame.

Usage:
    uv run python scripts/benchmark_video_dataloader.py --config pi0_mem_droid_local --steps 100

Reports samples/sec and mean per-step wall time. Use the result to decide whether
the lazy-K decode is a bottleneck (and thus whether decode-once is worth building).
Requires DROID RLDS data to be available at the config's rlds_data_dir.
"""

import argparse
import time

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()

    cfg = _config.get_config(args.config)
    loader = _data_loader.create_data_loader(cfg, skip_norm_stats=True)
    it = iter(loader)
    # warmup
    for _ in range(5):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(args.steps):
        batch = next(it)
        n += cfg.batch_size
    dt = time.perf_counter() - t0
    print(f"{args.config}: {n / dt:.1f} samples/s, {dt / args.steps * 1000:.1f} ms/step over {args.steps} steps")


if __name__ == "__main__":
    main()
```

(Adapt the `create_data_loader` call to its actual signature — check `data_loader.create_data_loader`.)

- [ ] **Step 2: Verify the script imports and arg-parses (no DROID data needed)**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/benchmark_video_dataloader.py --help`
Expected: prints usage with `--config`/`--steps`, no import errors.

- [ ] **Step 3: (Manual, needs DROID data) Run the benchmark + E2E**

Run (only where DROID data exists):
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/benchmark_video_dataloader.py --config pi0_mem_droid_local --steps 100`
and a few-step training E2E:
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train.py pi0_mem_droid_local`
Expected: loader yields batches with `video_image` of shape `[b,K,224,224,3]` and `video_states` `[b,K,action_dim]`; training step completes. Record samples/sec in `docs/benchmark_streaming_vs_local.md` and note whether decode appears to be the bottleneck (→ decide on decode-once). If DROID data is unavailable in this environment, mark Step 3 as deferred-to-manual in the report and proceed.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_video_dataloader.py docs/benchmark_streaming_vs_local.md
git commit -m "$(cat <<'EOF'
feat(scripts): DROID video dataloader benchmark (Spec B)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- gather_history (lazy-K, repeat-oldest, current-last) → Task 1.
- stride config + num_video_frames threading + MEM-gated repack → Task 2.
- DroidInputs video fields (camera-presence masks, single:=video[-1]) → Task 3.
- ResizeImages + PadStatesAndActions video extension → Task 4.
- Benchmark + E2E (DROID-data) verification → Task 5.
- Model: no change (spec says `from_dict` already consumes the fields) — correctly no task.

**Placeholder scan:** No TBD/TODO. Each code step shows complete code; run steps have exact commands + expected results. Three guarded "adapt to the actual API shape" notes (RepackTransform attribute name in Task 2, `resize_with_pad` leading-dim support in Task 4, `create_data_loader` signature in Task 5) state the binding behavior and how to confirm — guardrails, not placeholders.

**Type consistency:** `num_video_frames`/`video_stride_frames` named identically across `DroidRldsDataset`, `DataConfig`, `RLDSDroidDataConfig`, `create_rlds_dataset`. `_video_window_indices(traj_len, num_frames, stride)` matches its call in `gather_history`. Repacked keys (`observation/video_exterior_image_1_left`, `observation/video_wrist_image_left`, `observation/video_joint_position`, `observation/video_gripper_position`) match between Task 2 (repack) and Task 3 (DroidInputs reads). Top-level `video_image`/`video_image_mask`/`video_states` match what `Observation.from_dict` consumes and what Task 4 resizes/pads.
