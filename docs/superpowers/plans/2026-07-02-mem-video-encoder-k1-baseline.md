# MEM Video-Encoder K=1 Baseline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `num_video_frames=1` a provable, code-backed "video-encoder-off" baseline for the MEM `Pi0MEM` model — routed through the same `VideoViTEncoder` as K=6 and numerically identical to `origin/main`'s pi0.5 image encoder.

**Architecture:** Flip three `num_video_frames > 1` gates to `>= 1` so a PI0_MEM model always takes the video data path (approach A1). Prove encoder equivalence with a committed golden fixture generated from an `origin/main` checkout (T1) plus a siglip-drift guard, and confirm the K=1 path end-to-end with a config/loader unit test (repack + window math) and a model-level forward smoke (T3).

**Tech Stack:** Python, JAX/Flax (linen + nnx), `uv` for env, `pytest`, TensorFlow (RLDS loader), `flax.serialization` for the fixture.

**Spec:** `docs/superpowers/specs/2026-07-02-mem-video-encoder-k1-baseline-design.md`

## Global Constraints

- Run all Python via `uv run` (repo is uv-managed). Use `uv run --group rlds ...` for anything importing `openpi.training.droid_rlds_dataset` end-to-end.
- Prefix JAX processes with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- RLDS DROID configs must use `num_workers=0` (RLDS handles multiprocessing internally).
- Do NOT modify `origin/main` — the fixture is generated in a throwaway worktree checked out at `origin/main`, and only the produced artifact is copied back.
- The K>1 debug configs (`num_video_frames=2`) must keep working; `>= 1` includes them, so no behavior change there.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww
  ```

## File Structure

- Modify `src/openpi/training/config.py:412` — RLDS repack-map gate `> 1` → `>= 1`; also add a new `pi0_mem_k1_debug` config.
- Modify `src/openpi/training/droid_rlds_dataset.py:220` and `:257` — video gather + decode gates `> 1` → `>= 1`.
- Create `scripts/eval/gen_pi05_encoder_fixture.py` — one-shot generator for the origin/main pi0.5-encoder golden fixture.
- Create `tests/fixtures/pi05_encoder_ref.msgpack`, `tests/fixtures/pi05_encoder_ref_image.npy`, `tests/fixtures/pi05_encoder_ref_out.npy`, `tests/fixtures/pi05_encoder_ref_meta.json` — committed golden fixture.
- Modify `src/openpi/models/video_vit_test.py` — add T1 equivalence test + siglip-drift guard.
- Modify `src/openpi/models/pi0_mem_test.py` — add K=1 model-forward smoke.
- Modify `src/openpi/training/` tests — add `tests/training/test_rlds_k1_video_path.py` for the repack + window-math checks.

---

### Task 1: Flip A1 gates (config repack + RLDS gather/decode) for K=1

**Files:**
- Modify: `src/openpi/training/config.py:412`
- Modify: `src/openpi/training/droid_rlds_dataset.py:220`, `:257`
- Test: `tests/training/test_rlds_k1_video_path.py`

**Interfaces:**
- Consumes: `openpi.training.config.RLDSDroidDataConfig`, `openpi.training.droid_rlds_dataset.DroidActionSpace`, `openpi.training.droid_rlds_dataset._video_window_indices`, `openpi.models.pi0_mem_config.Pi0MEMConfig`.
- Produces: after this task, a PI0_MEM model with `num_video_frames=1` yields a `DataConfig.repack_transforms` whose `RepackTransform.structure` includes the `observation/video_*` keys.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_rlds_k1_video_path.py`:

```python
import pathlib

from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.training import config as _config
from openpi.training.droid_rlds_dataset import DroidActionSpace, _video_window_indices


def _k1_data_config():
    return _config.RLDSDroidDataConfig(
        repo_id="droid",
        rlds_data_dir="/tmp/does-not-exist",
        action_space=DroidActionSpace.JOINT_POSITION,
    )


def test_rlds_repack_includes_video_keys_at_k1(tmp_path: pathlib.Path):
    model_cfg = Pi0MEMConfig(
        num_video_frames=1, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    dc = _k1_data_config().create(tmp_path, model_cfg)
    structure = dc.repack_transforms.inputs[0].structure
    assert "observation/video_exterior_image_1_left" in structure
    assert "observation/video_wrist_image_left" in structure


def test_video_window_indices_k1_is_current_frame():
    import numpy as np

    idx = np.asarray(_video_window_indices(5, 1, 15))
    np.testing.assert_array_equal(idx, [[0], [1], [2], [3], [4]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/training/test_rlds_k1_video_path.py -v`
Expected: `test_rlds_repack_includes_video_keys_at_k1` FAILS (video keys absent — the gate is `> 1`). `test_video_window_indices_k1_is_current_frame` PASSES (window math already correct at K=1).

- [ ] **Step 3: Flip the repack-map gate**

In `src/openpi/training/config.py:412`, change:

```python
        if model_config.model_type == _model.ModelType.PI0_MEM and getattr(model_config, "num_video_frames", 1) > 1:
```

to:

```python
        if model_config.model_type == _model.ModelType.PI0_MEM and getattr(model_config, "num_video_frames", 1) >= 1:
```

- [ ] **Step 4: Flip the RLDS gather + decode gates**

In `src/openpi/training/droid_rlds_dataset.py:220`, change `if num_video_frames > 1:` to `if num_video_frames >= 1:`.
In `src/openpi/training/droid_rlds_dataset.py:257`, change `if num_video_frames > 1:` to `if num_video_frames >= 1:`.

(These two run inside the RLDS pipeline and are exercised end-to-end in Task 5; they must be flipped together with Step 3 so the repack's referenced `observation/video_image` key is actually produced by the loader.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/training/test_rlds_k1_video_path.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/config.py src/openpi/training/droid_rlds_dataset.py tests/training/test_rlds_k1_video_path.py
git commit -m "feat(mem): route K=1 PI0_MEM through the video data path (gates >=1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww"
```

---

### Task 2: Add `pi0_mem_k1_debug` config + model-level K=1 forward smoke

**Files:**
- Modify: `src/openpi/training/config.py` (append a new config to the `_CONFIGS` list)
- Test: `src/openpi/models/pi0_mem_test.py`, `tests/training/test_rlds_k1_video_path.py`

**Interfaces:**
- Consumes: `openpi.training.config.get_config`, `Pi0MEMConfig`, `config.fake_obs`, `config.fake_act`, `model.compute_loss`.
- Produces: a registered config name `pi0_mem_k1_debug` with `model.num_video_frames == 1` and `wandb_enabled is False`.

- [ ] **Step 1: Write the failing model-forward test**

Add to `src/openpi/models/pi0_mem_test.py`:

```python
def test_pi0_mem_k1_forward_uses_video_path():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=1,
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)
    # K=1 obs must still carry a video tensor with a single frame.
    assert obs.video_images is not None
    assert np.asarray(obs.video_images["base_0_rgb"]).shape[1] == 1
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_k1_forward_uses_video_path -v`
Expected: PASS (the model already accepts K=1 video via `fake_obs`; this test locks in that the K=1 path stays finite and single-frame). If it FAILS, investigate before proceeding — that is a real model-side gap.

- [ ] **Step 3: Write the failing config test**

Add to `tests/training/test_rlds_k1_video_path.py`:

```python
def test_pi0_mem_k1_debug_config_registered():
    cfg = _config.get_config("pi0_mem_k1_debug")
    assert cfg.model.num_video_frames == 1
    assert cfg.wandb_enabled is False
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/training/test_rlds_k1_video_path.py::test_pi0_mem_k1_debug_config_registered -v`
Expected: FAIL — `ValueError`/`KeyError` (config name not found).

- [ ] **Step 5: Register the config**

Find the existing `pi0_mem_debug` entry in the `_CONFIGS` list in `src/openpi/training/config.py`. Directly after it, add a K=1 twin (mirror every field of `pi0_mem_debug`, changing only `num_video_frames`):

```python
    TrainConfig(
        name="pi0_mem_k1_debug",
        model=pi0_mem_config.Pi0MEMConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            num_video_frames=1,
        ),
        data=FakeDataConfig(),
        batch_size=2,
        wandb_enabled=False,
    ),
```

Note: copy the exact `data=`, `batch_size=`, and any other fields from the real `pi0_mem_debug` entry so the twin differs only in `name` and `num_video_frames`. Verify the module alias used for `Pi0MEMConfig` in `config.py` (it is imported as `pi0_mem_config`) and match it.

- [ ] **Step 6: Run both tests to verify they pass**

Run: `uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_k1_forward_uses_video_path tests/training/test_rlds_k1_video_path.py::test_pi0_mem_k1_debug_config_registered -v`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/config.py src/openpi/models/pi0_mem_test.py tests/training/test_rlds_k1_video_path.py
git commit -m "feat(mem): add pi0_mem_k1_debug config + K=1 forward smoke

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww"
```

---

### Task 3: Generate the origin/main pi0.5-encoder golden fixture

**Files:**
- Create: `scripts/eval/gen_pi05_encoder_fixture.py`
- Create: `tests/fixtures/pi05_encoder_ref.msgpack`, `tests/fixtures/pi05_encoder_ref_image.npy`, `tests/fixtures/pi05_encoder_ref_out.npy`, `tests/fixtures/pi05_encoder_ref_meta.json`

**Interfaces:**
- Consumes: `openpi.models.siglip.Module` (present on `origin/main`).
- Produces: committed fixture artifacts consumed by Task 4. The reference encoder uses siglip kwargs `dict(num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32")` and a fixed `[1, 16, 16, 3]` image; the meta JSON records `siglip_sha256` of the `origin/main` `siglip.py` used.

- [ ] **Step 1: Write the generator script**

Create `scripts/eval/gen_pi05_encoder_fixture.py`:

```python
"""Generate the golden fixture for the K=1 video-encoder equivalence test.

Run this INSIDE an origin/main checkout so the reference output is produced by
origin/main's siglip.py. It builds the pi0.5-style SigLIP image encoder (same
_siglip.Module class pi0.py uses), runs one fixed image, and writes:
  pi05_encoder_ref.msgpack  - flax params
  pi05_encoder_ref_image.npy - the input image [1, 16, 16, 3]
  pi05_encoder_ref_out.npy   - encoder output
  pi05_encoder_ref_meta.json - {"siglip_sha256": ...} of the siglip.py used

Usage (from an origin/main worktree):
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval/gen_pi05_encoder_fixture.py <out_dir>
"""

import hashlib
import json
import pathlib
import sys

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as _siglip

SIGLIP_KWARGS = dict(
    num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32"
)


def main(out_dir: str) -> None:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = np.asarray(
        jax.random.normal(jax.random.key(42), (1, 16, 16, 3)), dtype=np.float32
    )
    module = _siglip.Module(**SIGLIP_KWARGS)
    variables = module.init(jax.random.key(0), jnp.asarray(image), train=False)
    output, _ = module.apply(variables, jnp.asarray(image), train=False)

    (out / "pi05_encoder_ref.msgpack").write_bytes(
        flax.serialization.to_bytes(variables["params"])
    )
    np.save(out / "pi05_encoder_ref_image.npy", image)
    np.save(out / "pi05_encoder_ref_out.npy", np.asarray(output))

    siglip_sha = hashlib.sha256(pathlib.Path(_siglip.__file__).read_bytes()).hexdigest()
    (out / "pi05_encoder_ref_meta.json").write_text(
        json.dumps({"siglip_sha256": siglip_sha, "siglip_kwargs": SIGLIP_KWARGS}, indent=2)
    )
    print(f"wrote fixture to {out} (siglip_sha256={siglip_sha[:12]}...)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
```

- [ ] **Step 2: Commit the generator script (on this branch)**

```bash
git add scripts/eval/gen_pi05_encoder_fixture.py
git commit -m "feat(mem): add origin/main pi0.5-encoder fixture generator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww"
```

- [ ] **Step 3: Create a throwaway origin/main worktree and generate the fixture there**

```bash
git fetch origin
GEN=$(mktemp -d)/openpi-origin-main
git worktree add "$GEN" origin/main
cp scripts/eval/gen_pi05_encoder_fixture.py "$GEN/scripts/eval/gen_pi05_encoder_fixture.py" 2>/dev/null \
  || (mkdir -p "$GEN/scripts/eval" && cp scripts/eval/gen_pi05_encoder_fixture.py "$GEN/scripts/eval/")
( cd "$GEN" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval/gen_pi05_encoder_fixture.py "$GEN/_fixture_out" )
```

Expected: prints `wrote fixture to .../_fixture_out (siglip_sha256=...)`.

- [ ] **Step 4: Copy the fixture into this branch and clean up the worktree**

```bash
mkdir -p tests/fixtures
cp "$GEN"/_fixture_out/pi05_encoder_ref.msgpack tests/fixtures/
cp "$GEN"/_fixture_out/pi05_encoder_ref_image.npy tests/fixtures/
cp "$GEN"/_fixture_out/pi05_encoder_ref_out.npy tests/fixtures/
cp "$GEN"/_fixture_out/pi05_encoder_ref_meta.json tests/fixtures/
git worktree remove --force "$GEN"
ls -la tests/fixtures/
```

Expected: four fixture files present; `pi05_encoder_ref.msgpack` is small (KBs, mu/2 variant).

- [ ] **Step 5: Commit the fixture**

```bash
git add tests/fixtures/pi05_encoder_ref.msgpack tests/fixtures/pi05_encoder_ref_image.npy tests/fixtures/pi05_encoder_ref_out.npy tests/fixtures/pi05_encoder_ref_meta.json
git commit -m "test(mem): commit origin/main pi0.5-encoder golden fixture

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww"
```

---

### Task 4: T1 equivalence test + siglip-drift guard

**Files:**
- Test: `src/openpi/models/video_vit_test.py`

**Interfaces:**
- Consumes: `tests/fixtures/pi05_encoder_ref*` (Task 3), `openpi.models.video_vit.VideoViTConfig`, `openpi.models.video_vit.VideoViTEncoder`, `openpi.models.siglip`.
- Produces: `test_k1_matches_origin_main_pi05_encoder`, `test_siglip_unchanged_since_fixture`.

- [ ] **Step 1: Write the equivalence + guard tests**

Add to `src/openpi/models/video_vit_test.py`:

```python
import hashlib
import json
import pathlib

_FIX = pathlib.Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def _load_fixture_params(template):
    import flax.serialization

    data = (_FIX / "pi05_encoder_ref.msgpack").read_bytes()
    return flax.serialization.from_bytes(template, data)


def test_k1_matches_origin_main_pi05_encoder():
    """K=1 VideoViTEncoder, loaded with origin/main pi0.5-encoder weights, must
    reproduce the origin/main encoder output bit-for-bit (atol 1e-5)."""
    meta = json.loads((_FIX / "pi05_encoder_ref_meta.json").read_text())
    kwargs = meta["siglip_kwargs"]

    image = np.load(_FIX / "pi05_encoder_ref_image.npy")  # [1, 16, 16, 3]
    ref_out = np.load(_FIX / "pi05_encoder_ref_out.npy")
    video = jnp.asarray(image)[:, None, :, :, :]  # [1, 1, 16, 16, 3]

    video_vit = VideoViTEncoder(
        config=VideoViTConfig(num_video_frames=1, temporal_attn_every_n_layers=4),
        siglip_kwargs=flax.core.FrozenDict(kwargs),
    )
    template = video_vit.init(jax.random.key(0), video, train=False)["params"]
    params = _load_fixture_params(template)  # raises if param trees differ
    out, _ = video_vit.apply({"params": params}, video, train=False)

    np.testing.assert_allclose(
        np.array(out), ref_out, atol=1e-5,
        err_msg="K=1 VideoViTEncoder must match origin/main pi0.5 image encoder",
    )


def test_siglip_unchanged_since_fixture():
    """Guard: siglip.py must match the origin/main version the fixture was built
    from. If this fails, regenerate the fixture (Task 3)."""
    meta = json.loads((_FIX / "pi05_encoder_ref_meta.json").read_text())
    current = hashlib.sha256(pathlib.Path(_siglip.__file__).read_bytes()).hexdigest()
    assert current == meta["siglip_sha256"], (
        "siglip.py changed vs origin/main fixture; regenerate the golden fixture."
    )
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest src/openpi/models/video_vit_test.py -v`
Expected: all PASS, including the existing `test_single_frame_matches_siglip`, the new `test_k1_matches_origin_main_pi05_encoder`, and `test_siglip_unchanged_since_fixture`.

If `test_k1_matches_origin_main_pi05_encoder` fails with a structure error from `from_bytes`, the video/siglip param trees diverged — stop and investigate (this is the equivalence the gate exists to catch). If `test_siglip_unchanged_since_fixture` fails, `siglip.py` on this branch differs from `origin/main`; either restore it or, if the change is intentional and equivalent, regenerate the fixture via Task 3.

- [ ] **Step 3: Commit**

```bash
git add src/openpi/models/video_vit_test.py
git commit -m "test(mem): prove K=1 video encoder == origin/main pi0.5 encoder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01NSnUsRsmMx5SmK3bo1wdww"
```

---

### Task 5: End-to-end K=1 verification + regression sweep

**Files:** none (verification only).

**Interfaces:**
- Consumes: everything above; the `pi0_mem_k1_debug` config (Task 2) and a local RLDS subset (or the streaming config) for the loader-gate end-to-end check.

- [ ] **Step 1: Full MEM/video regression sweep**

Run: `uv run pytest src/openpi/models/video_vit_test.py src/openpi/models/pi0_mem_test.py tests/training/test_rlds_k1_video_path.py -q`
Expected: all PASS, 0 failures (definition-of-done #1 and #3).

- [ ] **Step 2: End-to-end K=1 data→model smoke through the real RLDS loader**

This exercises the flipped loader gates (`droid_rlds_dataset.py:220,257`), which the unit tests cannot cover without data. Point at a local RLDS subset (mirror `pi0_mem_droid_local`'s `rlds_data_dir`) or the streaming source, with `num_video_frames=1`.

Run (adjust `rlds_data_dir` / config to your environment; keep `num_workers=0`):
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --group rlds python scripts/train.py \
  pi0_mem_droid_local --model.num_video_frames=1 --exp_name=k1_smoke \
  --num_train_steps=10 --batch_size=2 --wandb_enabled=False --overwrite
```
Expected: 10 steps complete with no shape/mask errors and finite loss; batches carry `video_*` tensors with a single frame. (If `pi0_mem_droid_local` is unavailable locally, use the streaming `pi0_mem_droid_stream` equivalently with the same overrides.)

- [ ] **Step 3: Confirm the gate summary (evidence for the spec's DoD)**

Run: `uv run pytest tests/training/test_rlds_k1_video_path.py -v`
Record output showing `test_rlds_repack_includes_video_keys_at_k1`, `test_video_window_indices_k1_is_current_frame`, and `test_pi0_mem_k1_debug_config_registered` all PASS.

- [ ] **Step 4: No commit needed** (verification only). If Step 2 required a config tweak, commit that separately with a clear message.

---

## Self-Review

**Spec coverage:**
- Change 1 (A1 gate flips, all three locations) → Task 1 (Steps 3–4). ✓
- Change 2 (T1 origin/main golden fixture) → Task 3 (generate) + Task 4 (assert). ✓
- Change 2 (siglip-unchanged guard) → Task 4 Step 1 `test_siglip_unchanged_since_fixture`. ✓
- Change 3 (`pi0_mem_k1_debug` config + K=1 data→model smoke) → Task 2 + Task 5 Step 2. ✓
- DoD #1 (pytest green incl. T1/guard/T3) → Task 5 Step 1. ✓
- DoD #2 (~10-step K=1 RLDS smoke) → Task 5 Step 2. ✓
- DoD #3 (existing MEM tests green; K=2 unaffected) → Task 5 Step 1 (runs full pi0_mem_test.py, which includes K=2 RTC tests). ✓
- Risk "scan layout" → handled: fixture built with `scan=False` (generator `SIGLIP_KWARGS`). ✓
- Risk "fixture size" → tiny `mu/2` variant + `16x16` image. ✓
- Risk "_video_window_indices at K=1" → Task 1 Step 1 `test_video_window_indices_k1_is_current_frame`. ✓
- Risk "serving fallback preserved" → the `img` fallback in `pi0_mem.py` is untouched (no task modifies it); `test_single_frame_matches_siglip` and existing serving paths remain. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands include expected output. ✓

**Type consistency:** `siglip_kwargs` uses `num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32"` consistently in the generator (`SIGLIP_KWARGS`), the fixture meta, and the T1 test (loaded from meta). `VideoViTConfig(num_video_frames=1, temporal_attn_every_n_layers=4)` matches the existing `test_single_frame_matches_siglip`. Repack key names (`observation/video_exterior_image_1_left`, `observation/video_wrist_image_left`) match `config.py:414-415`. ✓

**Note for the implementer:** In Task 2 Step 5, copy the *exact* fields of the real `pi0_mem_debug` entry (its `data=`/`batch_size=` may differ from the illustrative snippet); the twin must differ only in `name` and `num_video_frames`.
