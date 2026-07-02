# MEM Video-Encoder Eval — Sub-project B (Training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two short-run (10k-step) LoRA-finetuned `Pi0MEM` DROID checkpoints — K=6 (video encoder on) and K=1 (encoder off) — from `pi05_droid`, launched in parallel on NCHC with wandb, so C/D can eval whether the video encoder helps.

**Architecture:** Two `TrainConfig` arms derived from the proven `pi0_mem_droid_stream`, differing only in `num_video_frames`. A new weight loader seeds `video_img` (VideoViTEncoder) from the checkpoint's pretrained SigLIP (`img`), un-stacking `scan=True`→`scan=False`, so both arms start from a trained vision encoder. Guarded by unit tests (freeze partition, merge invariant, remap equivalence) plus an on-NCHC weight key-diff and a 10-step smoke before the 10k runs.

**Tech Stack:** JAX/Flax NNX, tyro configs, RLDS/tf.data DROID streaming from GCS, SLURM (NCHC), Weights & Biases.

## Global Constraints

- Both arms are **identical except `num_video_frames`** (K=6 vs K=1). Same seed (42), lr, batch, steps.
- `action_horizon = 16` (DROID-native; not the paper's 50). `action_dim=32`, `pi05=True`.
- `lora=True`; init from `gs://openpi-assets/checkpoints/pi05_droid/params`.
- `missing_regex = ".*(lora|state_proj|video_img).*"` (kept broad; remap seeds `video_img` before merge).
- `num_workers=0` (RLDS requirement — it does its own tf.data parallelism).
- `fsdp_devices=4`, `batch_size=128`, `grad_accum_steps=1`, `num_train_steps=10_000`, `save_interval=2_000`, `keep_period=10_000`.
- wandb: `project_name="video-encoder-eval"`, `wandb_enabled=True`.
- NCHC: `--account=MST114563`, workdir `/work/roboleon1295/openpi`, `uv run --no-sync`, `export HF_HOME=/work/roboleon1295/hf_cache`, logs under `/work/roboleon1295/openpi/logs/`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- Unit tests must stay fast: use `jax.eval_shape` (no weight allocation) for full-model key inspection; use the tiny `mu/2` SigLIP variant for numeric encoder tests. `dummy` gemma variants have **no** LoRA params, so LoRA numeric fwd/bwd is verified in the NCHC smoke, not a CPU unit test.

---

### Task 1: LoRA freeze-partition guard (T-B1)

Verifies the mechanic the whole experiment rests on: under `lora=True`, `video_img`/`state_proj`/`lora` are trainable and the base `.*llm.*` is frozen — at both K=1 and K=6.

**Files:**
- Test: `src/openpi/models/pi0_mem_test.py` (append)

**Interfaces:**
- Consumes: `Pi0MEMConfig(lora=True).get_freeze_filter()` (exists, `pi0_mem_config.py`); `config.create(rng)`.
- Produces: nothing consumed downstream (guard test).

- [ ] **Step 1: Write the test**

Append to `src/openpi/models/pi0_mem_test.py`:

```python
import flax.traverse_util as _tu


def _param_partition(num_video_frames):
    """Return (trainable_keys, frozen_keys) for a LoRA Pi0MEM, via eval_shape (no weights)."""
    cfg = Pi0MEMConfig(lora=True, num_video_frames=num_video_frames)
    freeze = cfg.get_freeze_filter()

    def f(rng):
        model = cfg.create(rng)
        trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(freeze))).to_pure_dict()
        frozen = nnx.state(model, nnx.All(nnx.Param, freeze)).to_pure_dict()
        return trainable, frozen

    trainable, frozen = jax.eval_shape(f, jax.random.key(0))
    tks = set(_tu.flatten_dict(trainable, sep="/"))
    fks = set(_tu.flatten_dict(frozen, sep="/"))
    return tks, fks


import pytest


@pytest.mark.parametrize("k", [1, 6])
def test_lora_freeze_partition(k):
    trainable, frozen = _param_partition(k)
    # video_img (VideoViTEncoder), state_proj, and lora adapters must be trainable.
    assert any("video_img" in key for key in trainable), "video_img must be trainable"
    assert any("state_proj" in key for key in trainable), "state_proj must be trainable"
    assert any("lora" in key for key in trainable), "lora adapters must be trainable"
    # Base LLM weights (non-lora) must be frozen; no lora param may be frozen.
    assert any("llm" in key for key in frozen), "base llm must be frozen"
    assert all("lora" not in key for key in frozen), "no lora param may be frozen"
    # Any trainable llm-path key must be a lora adapter (base attn/ffn stay frozen).
    assert all("lora" in key for key in trainable if "llm" in key)
```

- [ ] **Step 2: Run test to verify behavior**

Run: `uv run pytest src/openpi/models/pi0_mem_test.py::test_lora_freeze_partition -v`
Expected: PASS for `k=1` and `k=6` (this locks existing `get_freeze_filter` behavior; it should pass immediately). If it FAILS, the freeze filter or module paths regressed — investigate before proceeding.

- [ ] **Step 3: Commit**

```bash
git add src/openpi/models/pi0_mem_test.py
git commit -m "test(mem): guard LoRA freeze partition (video_img/state_proj/lora trainable, base llm frozen)"
```

---

### Task 2: `missing_regex` merge invariant (T-B2)

Encodes the silent-random-init footgun as an invariant: `_merge_params` must never silently drop a reference key; only `lora`/`state_proj`/`video_img` fall to fresh init when the checkpoint lacks them.

**Files:**
- Create: `src/openpi/training/weight_loaders_test.py`

**Interfaces:**
- Consumes: `openpi.training.weight_loaders._merge_params(loaded, params, *, missing_regex)` (exists).
- Produces: nothing downstream (guard test).

- [ ] **Step 1: Write the test**

Create `src/openpi/training/weight_loaders_test.py`:

```python
import numpy as np

from openpi.training import weight_loaders as wl

_REGEX = r".*(lora|state_proj|video_img).*"


def test_merge_params_no_silent_drop_and_fresh_init():
    # Reference (target) MEM param tree: base + MEM-new keys.
    ref = {
        "PaliGemma": {
            "llm": {"layer/kernel": np.zeros((4, 4), np.float32)},
            "img": {"embedding/kernel": np.zeros((2, 2), np.float32)},
            "video_img": {"embedding/kernel": np.ones((2, 2), np.float32)},  # MEM-new (fresh)
            "llm_lora": {"lora_a": np.ones((4, 2), np.float32)},             # MEM-new (fresh)
        },
        "state_proj": {"kernel": np.ones((3, 4), np.float32)},               # MEM-new (fresh)
    }
    # "pi0.5-like" loaded tree = reference minus the MEM-new keys, WITHOUT the video_img remap.
    loaded = {
        "PaliGemma": {
            "llm": {"layer/kernel": np.full((4, 4), 7.0, np.float32)},
            "img": {"embedding/kernel": np.full((2, 2), 9.0, np.float32)},
        }
    }

    merged = wl._merge_params(loaded, ref, missing_regex=_REGEX)

    import flax.traverse_util as tu
    flat_ref = tu.flatten_dict(ref, sep="/")
    flat_merged = tu.flatten_dict(merged, sep="/")

    # 1. Nothing silently dropped: every reference key is present.
    assert set(flat_merged) == set(flat_ref)
    # 2. Base keys took the LOADED values.
    np.testing.assert_array_equal(flat_merged["PaliGemma/llm/layer/kernel"], np.full((4, 4), 7.0))
    np.testing.assert_array_equal(flat_merged["PaliGemma/img/embedding/kernel"], np.full((2, 2), 9.0))
    # 3. MEM-new keys took FRESH init (the reference values), since loaded lacked them.
    np.testing.assert_array_equal(flat_merged["PaliGemma/video_img/embedding/kernel"], np.ones((2, 2)))
    np.testing.assert_array_equal(flat_merged["PaliGemma/llm_lora/lora_a"], np.ones((4, 2)))
    np.testing.assert_array_equal(flat_merged["state_proj/kernel"], np.ones((3, 4)))
```

- [ ] **Step 2: Run test**

Run: `uv run pytest src/openpi/training/weight_loaders_test.py::test_merge_params_no_silent_drop_and_fresh_init -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/openpi/training/weight_loaders_test.py
git commit -m "test(mem): missing_regex merge invariant (no silent drop; only MEM-new keys fresh-init)"
```

---

### Task 3: SigLIP → `video_img` scan-unstack remap loader (T-B4)

The one piece of new engineering. Seed `video_img` from the checkpoint's pretrained SigLIP (`img`), converting the `scan=True` stacked encoder blocks to the `scan=False` per-layer layout `VideoViTEncoder` expects.

**Files:**
- Modify: `src/openpi/training/weight_loaders.py`
- Test: `src/openpi/training/weight_loaders_test.py` (append)

**Interfaces:**
- Consumes: `_merge_params`, `_model.restore_params`, `download.maybe_download` (exist in module).
- Produces:
  - `weight_loaders._unstack_scanned_encoderblocks(flat: dict[str, np.ndarray]) -> dict[str, np.ndarray]`
  - `weight_loaders._seed_video_img_from_img(loaded: at.Params) -> at.Params`
  - `weight_loaders.SiglipToVideoImgWeightLoader(params_path: str, missing_regex: str = ".*(lora|state_proj|video_img).*")` — a `WeightLoader`. Consumed by Task 4's configs.

- [ ] **Step 1: Write the failing unit test for the un-stack helper**

Append to `src/openpi/training/weight_loaders_test.py`:

```python
def test_unstack_scanned_encoderblocks():
    depth = 3
    flat = {
        "embedding/kernel": np.zeros((2, 2, 3, 4), np.float32),          # non-block: copied as-is
        "pos_embedding": np.zeros((1, 5, 8), np.float32),                # non-block: copied as-is
        "Transformer/encoderblock/LayerNorm_0/scale": np.arange(depth * 8, dtype=np.float32).reshape(depth, 8),
        "Transformer/encoder_norm/scale": np.zeros((8,), np.float32),    # non-block: copied as-is
    }
    out = wl._unstack_scanned_encoderblocks(flat)

    assert out["embedding/kernel"].shape == (2, 2, 3, 4)
    assert out["pos_embedding"].shape == (1, 5, 8)
    assert out["Transformer/encoder_norm/scale"].shape == (8,)
    # Stacked block split into depth per-layer blocks, leading axis removed.
    for i in range(depth):
        assert out[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"].shape == (8,)
        np.testing.assert_array_equal(
            out[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"],
            np.arange(depth * 8, dtype=np.float32).reshape(depth, 8)[i],
        )
    assert "Transformer/encoderblock/LayerNorm_0/scale" not in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/openpi/training/weight_loaders_test.py::test_unstack_scanned_encoderblocks -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_unstack_scanned_encoderblocks'`.

- [ ] **Step 3: Implement the remap + loader**

In `src/openpi/training/weight_loaders.py`, add after `_merge_params`:

```python
def _unstack_scanned_encoderblocks(flat_img: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert a scan=True SigLIP subtree to the scan=False per-layer layout.

    scan=True stores all transformer blocks as one stacked `encoderblock` param with a
    leading depth axis; scan=False (what VideoViTEncoder uses) expects per-layer
    `encoderblock_{i}` params. Keys are '/'-joined paths relative to the SigLIP root.
    Non-encoderblock keys (embedding, pos_embedding, Transformer/encoder_norm, head) are
    copied unchanged. A subtree already in scan=False layout is returned unchanged.
    """
    out: dict[str, np.ndarray] = {}
    for key, value in flat_img.items():
        parts = key.split("/")
        if "encoderblock" in parts:  # exact segment match (not the "encoderblock_{i}" prefix)
            idx = parts.index("encoderblock")
            depth = value.shape[0]
            for lyr in range(depth):
                new_key = "/".join(parts[:idx] + [f"encoderblock_{lyr}"] + parts[idx + 1 :])
                out[new_key] = value[lyr]
        else:
            out[key] = value
    return out


def _seed_video_img_from_img(loaded: at.Params) -> at.Params:
    """Return a copy of `loaded` with `PaliGemma/video_img` populated from the un-stacked
    `PaliGemma/img` SigLIP params. No-op if `PaliGemma/img` is absent."""
    pg = loaded.get("PaliGemma")
    if not isinstance(pg, dict) or "img" not in pg:
        return loaded
    flat_img = flax.traverse_util.flatten_dict(pg["img"], sep="/")
    flat_video = _unstack_scanned_encoderblocks(flat_img)
    new_pg = dict(pg)
    new_pg["video_img"] = flax.traverse_util.unflatten_dict(flat_video, sep="/")
    return {**loaded, "PaliGemma": new_pg}


@dataclasses.dataclass(frozen=True)
class SiglipToVideoImgWeightLoader(WeightLoader):
    """CheckpointWeightLoader that additionally seeds `PaliGemma/video_img` from the
    checkpoint's pretrained SigLIP (`PaliGemma/img`).

    Without this, `video_img` falls through `missing_regex` to a random init, and a short
    finetune would train the LL/video vision encoder from scratch.
    """

    params_path: str
    missing_regex: str = ".*(lora|state_proj|video_img).*"

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded = _seed_video_img_from_img(loaded)
        return _merge_params(loaded, params, missing_regex=self.missing_regex)
```

- [ ] **Step 4: Run to verify the un-stack test passes**

Run: `uv run pytest src/openpi/training/weight_loaders_test.py::test_unstack_scanned_encoderblocks -v`
Expected: PASS.

- [ ] **Step 5: Write the failing equivalence test (the real de-risk)**

Append to `src/openpi/training/weight_loaders_test.py`:

```python
def test_remapped_video_img_matches_siglip():
    """Un-stacked scan=True SigLIP weights, loaded into a K=1 VideoViTEncoder, must
    reproduce the SigLIP output bit-for-bit (atol 1e-5). This is A's equivalence proof
    extended through the scan un-stack that B introduces."""
    import flax.core
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    import openpi.models.siglip as _siglip
    from openpi.models.video_vit import VideoViTConfig, VideoViTEncoder

    kw = dict(num_classes=32, variant="mu/2", pool_type="none", dtype_mm="float32")
    siglip = _siglip.Module(scan=True, **kw)              # checkpoint layout (stacked blocks)
    video = VideoViTEncoder(
        config=VideoViTConfig(num_video_frames=1, temporal_attn_every_n_layers=4),
        siglip_kwargs=flax.core.FrozenDict(scan=False, **kw),  # per-layer layout
    )

    image = jnp.ones((1, 224, 224, 3))
    clip = image[:, None]  # [1, 1, 224, 224, 3]
    rng = jax.random.key(0)

    siglip_vars = siglip.init(rng, image, train=False)
    siglip_out, _ = siglip.apply(siglip_vars, image, train=False)

    # Un-stack the SigLIP params and load them into the K=1 VideoViTEncoder.
    flat_img = tu.flatten_dict(siglip_vars["params"], sep="/")
    flat_video = wl._unstack_scanned_encoderblocks(flat_img)
    video_params = {"params": tu.unflatten_dict(flat_video, sep="/")}
    video_out, _ = video.apply(video_params, clip, train=False)

    np.testing.assert_allclose(np.array(siglip_out), np.array(video_out), atol=1e-5)
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest src/openpi/training/weight_loaders_test.py::test_remapped_video_img_matches_siglip -v`
Expected: PASS. If it FAILS on a key mismatch (`apply` error) the SigLIP/VideoViT param paths have diverged — reconcile against `video_vit.py` block/head naming before proceeding.

- [ ] **Step 7: Run the full weight-loader + A test suite (no regressions)**

Run: `uv run pytest src/openpi/training/weight_loaders_test.py src/openpi/models/video_vit_test.py -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/openpi/training/weight_loaders.py src/openpi/training/weight_loaders_test.py
git commit -m "feat(mem): SiglipToVideoImgWeightLoader — seed video_img from pretrained SigLIP (scan un-stack)"
```

---

### Task 4: The two verify configs + build smoke

Register `pi0_mem_droid_k1_verify` and `pi0_mem_droid_k6_verify`, differing only in `num_video_frames`, wired to the new loader.

**Files:**
- Modify: `src/openpi/training/config.py` (add two `TrainConfig`s to the registry, next to `pi0_mem_droid_stream`)
- Test: `src/openpi/training/config_test.py` (append)

**Interfaces:**
- Consumes: `weight_loaders.SiglipToVideoImgWeightLoader` (Task 3); `RLDSDroidDataConfig`, `pi0_mem_config.Pi0MEMConfig`, `_optimizer.CosineDecaySchedule`, `AssetsConfig` (exist).
- Produces: registry names `"pi0_mem_droid_k1_verify"`, `"pi0_mem_droid_k6_verify"` for `_config.get_config(...)`.

- [ ] **Step 1: Write the failing config smoke test**

Append to `src/openpi/training/config_test.py`:

```python
import pytest


@pytest.mark.parametrize("name,k", [("pi0_mem_droid_k1_verify", 1), ("pi0_mem_droid_k6_verify", 6)])
def test_video_eval_verify_configs(name, k):
    from openpi.training import config as _config
    from openpi.training import weight_loaders

    cfg = _config.get_config(name)
    assert cfg.model.num_video_frames == k
    assert cfg.model.action_horizon == 16
    assert cfg.model.lora is True
    assert cfg.model.pi05 is True
    assert cfg.num_train_steps == 10_000
    assert cfg.fsdp_devices == 4
    assert cfg.batch_size == 128
    assert cfg.grad_accum_steps == 1
    assert cfg.seed == 42
    assert cfg.num_workers == 0
    assert cfg.project_name == "video-encoder-eval"
    assert cfg.wandb_enabled is True
    # Non-empty freeze filter (LoRA) and the video_img-seeding loader.
    from flax import nnx
    assert not isinstance(cfg.freeze_filter, nnx.filterlib.Nothing.__class__ if False else type(nnx.Nothing))
    assert isinstance(cfg.weight_loader, weight_loaders.SiglipToVideoImgWeightLoader)
    assert cfg.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_droid/params"
```

Note: the freeze-filter assertion above is awkward; replace its two lines with the simpler check that the filter differs from the default `nnx.Nothing()`:

```python
    assert cfg.freeze_filter is not None
    assert repr(cfg.freeze_filter) != repr(nnx.Nothing())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest "src/openpi/training/config_test.py::test_video_eval_verify_configs" -v`
Expected: FAIL — `get_config` raises for the unknown config names.

- [ ] **Step 3: Add the two configs**

In `src/openpi/training/config.py`, immediately after the `pi0_mem_droid_stream` `TrainConfig` (ends ~line 1090), insert:

```python
    TrainConfig(
        # Video-encoder-eval sub-project B: K=6 arm (video encoder ON).
        name="pi0_mem_droid_k6_verify",
        exp_name="pi0_mem_droid_k6_verify",
        project_name="video-encoder-eval",
        model=pi0_mem_config.Pi0MEMConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            num_video_frames=6,
            lora=True,
        ),
        freeze_filter=pi0_mem_config.Pi0MEMConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            num_video_frames=6,
            lora=True,
        ).get_freeze_filter(),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            rlds_data_dir="gs://gresearch/robotics",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="droid",
                    version="1.0.1",
                    weight=1.0,
                    filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                ),
            ),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.SiglipToVideoImgWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        fsdp_devices=4,
        num_train_steps=10_000,
        batch_size=128,
        grad_accum_steps=1,
        seed=42,
        save_interval=2_000,
        keep_period=10_000,
        num_workers=0,
    ),
    TrainConfig(
        # Video-encoder-eval sub-project B: K=1 arm (video encoder OFF, A-validated baseline).
        name="pi0_mem_droid_k1_verify",
        exp_name="pi0_mem_droid_k1_verify",
        project_name="video-encoder-eval",
        model=pi0_mem_config.Pi0MEMConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            num_video_frames=1,
            lora=True,
        ),
        freeze_filter=pi0_mem_config.Pi0MEMConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            num_video_frames=1,
            lora=True,
        ).get_freeze_filter(),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            rlds_data_dir="gs://gresearch/robotics",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            datasets=(
                droid_rlds_dataset.RLDSDataset(
                    name="droid",
                    version="1.0.1",
                    weight=1.0,
                    filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
                ),
            ),
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.SiglipToVideoImgWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        fsdp_devices=4,
        num_train_steps=10_000,
        batch_size=128,
        grad_accum_steps=1,
        seed=42,
        save_interval=2_000,
        keep_period=10_000,
        num_workers=0,
    ),
```

- [ ] **Step 4: Run the smoke test**

Run: `uv run pytest "src/openpi/training/config_test.py::test_video_eval_verify_configs" -v`
Expected: PASS for both names.

- [ ] **Step 5: Confirm no other config broke (registry integrity)**

Run: `uv run pytest src/openpi/training/config_test.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/config.py src/openpi/training/config_test.py
git commit -m "feat(mem): pi0_mem_droid_k1_verify / k6_verify configs (LoRA, pi05_droid, horizon 16)"
```

---

### Task 5: NCHC pre-launch weight key-diff + documented 10-step smoke (T-B3 realization)

On NCHC (GCS + real `pi05_droid` checkpoint), prove that loading leaves **only** `lora`/`state_proj` freshly initialized and that `video_img` is fully **seeded from the remap** (not random). Then a 10-step smoke exercises the real LoRA fwd/bwd (finite decreasing loss) — the numeric T-B3 check that `dummy` variants can't cover on CPU.

**Files:**
- Create: `scripts/nchc/check_mem_droid_weights.py`

**Interfaces:**
- Consumes: `_config.get_config(name)`; `cfg.model.create`, `cfg.weight_loader.load`; `at.check_pytree_equality`.
- Produces: a CLI that exits non-zero on a bad load (used by Task 6's sbatch as a gate).

- [ ] **Step 1: Write the check script**

Create `scripts/nchc/check_mem_droid_weights.py`:

```python
"""Pre-launch check: verify pi05_droid weights load into a MEM verify config with ONLY
the expected MEM-new keys (lora, state_proj) freshly initialized, and that video_img is
fully seeded from the remapped SigLIP (not random init).

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
    scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify
"""

import re
import sys

import flax.nnx as nnx
import flax.traverse_util as tu
import jax

import openpi.shared.array_typing as at
from openpi.training import config as _config


def main(config_name: str) -> None:
    cfg = _config.get_config(config_name)

    def _init(rng):
        model = cfg.model.create(rng)
        return nnx.state(model, nnx.Param).to_pure_dict()

    params_shape = jax.eval_shape(_init, jax.random.key(0))
    ref_keys = set(tu.flatten_dict(params_shape, sep="/"))

    loaded = cfg.weight_loader.load(params_shape)
    loaded_keys = {
        k for k, v in tu.flatten_dict(loaded, sep="/").items() if not isinstance(v, jax.ShapeDtypeStruct)
    }

    missing = ref_keys - loaded_keys  # keys that fell through to fresh init
    regex = re.compile(cfg.weight_loader.missing_regex)
    bad = sorted(k for k in missing if not regex.fullmatch(k))

    video_img_keys = {k for k in ref_keys if "video_img" in k}
    video_img_seeded = video_img_keys & loaded_keys

    print(f"config: {config_name}")
    print(f"ref params: {len(ref_keys)}  loaded: {len(loaded_keys)}  fresh-init: {len(missing)}")
    print("fresh-init sample:", sorted(missing)[:12])
    print(f"video_img: {len(video_img_seeded)}/{len(video_img_keys)} seeded from checkpoint")

    assert not bad, f"Base weights left uninitialized (path mismatch): {bad[:20]}"
    assert video_img_seeded, "video_img NOT seeded from checkpoint — remap failed (would be random init)"
    assert video_img_seeded == video_img_keys, (
        f"Only {len(video_img_seeded)}/{len(video_img_keys)} video_img keys seeded"
    )
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=False)
    print("OK: only lora/state_proj fresh-init; video_img fully seeded via remap.")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run the key-diff on NCHC for both arms**

On NCHC (`/work/roboleon1295/openpi`, token/HF_HOME set):

```bash
export HF_HOME=/work/roboleon1295/hf_cache
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k1_verify
```

Expected (both): prints `OK: only lora/state_proj fresh-init; video_img fully seeded via remap.` and exits 0. The `fresh-init sample` list must contain only paths matching `lora`/`state_proj`.

- [ ] **Step 3: Run the 10-step LoRA smoke on NCHC (T-B3 numeric check)**

For each arm (single GPU, tiny batch, no wandb, throwaway exp):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py \
  pi0_mem_droid_k1_verify --exp-name=smoke_k1 --overwrite --wandb-enabled=False \
  --num-train-steps=10 --batch-size=8 --fsdp-devices=1
```

Expected: completes 10 steps, prints a **finite** loss that is **not increasing** across steps, `video_*` batch keys present, no shape errors. Repeat with `pi0_mem_droid_k6_verify` / `--exp-name=smoke_k6`. (This is the real gemma_2b_lora fwd/bwd that CPU unit tests can't cover — LoRA numeric T-B3.)

- [ ] **Step 4: Commit the script**

```bash
git add scripts/nchc/check_mem_droid_weights.py
git commit -m "feat(mem): NCHC pre-launch weight key-diff (video_img seeded; only lora/state_proj fresh)"
```

---

### Task 6: NCHC parallel-launch sbatch (the two 10k arms)

Run both arms in parallel on one 8-GPU node (4 GPUs each), gated by the pre-launch key-diff.

**Files:**
- Create: `scripts/nchc/train_mem_video_eval.sbatch`

**Interfaces:**
- Consumes: `scripts/nchc/check_mem_droid_weights.py` (Task 5); the two configs (Task 4); `scripts/train.py`.
- Produces: two checkpoints under `/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k{1,6}_verify/.../10000` + two wandb runs in `video-encoder-eval`.

- [ ] **Step 1: Confirm the 8-GPU partition name**

Run on NCHC: `sinfo -o "%P %G %l"` — note the partition that offers `gpu:8`. Use it for `--partition` below (the template uses `gpu` as a placeholder to replace).

- [ ] **Step 2: Write the sbatch script**

Create `scripts/nchc/train_mem_video_eval.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=mem_video_eval
#SBATCH --account=MST114563
#SBATCH --partition=gpu            # REPLACE with the 8-GPU partition from `sinfo -o "%P %G"`
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_video_eval_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache

K6_LOG=logs/mem_video_eval_k6_${SLURM_JOB_ID}.log
K1_LOG=logs/mem_video_eval_k1_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast if the remap/regex is wrong (before any GPU-hours). ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify || { echo "K6 WEIGHT CHECK FAILED"; exit 1; }
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k1_verify || { echo "K1 WEIGHT CHECK FAILED"; exit 1; }

# --- Two arms in parallel: 4 GPUs each, disjoint device sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k6_verify > "$K6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k1_verify > "$K1_LOG" 2>&1 &
P1=$!

wait $P6; echo "K6_EXIT=$?"
wait $P1; echo "K1_EXIT=$?"

echo "=== K6 tail ==="; tail -n 40 "$K6_LOG"
echo "=== K1 tail ==="; tail -n 40 "$K1_LOG"
```

- [ ] **Step 3: Commit the sbatch**

```bash
git add scripts/nchc/train_mem_video_eval.sbatch
git commit -m "feat(mem): NCHC sbatch — parallel K=1/K=6 10k verify arms (gated by weight key-diff)"
```

- [ ] **Step 4: Submit and verify (operator, on NCHC)**

```bash
sbatch scripts/nchc/train_mem_video_eval.sbatch
```

Expected end state:
- Both weight checks print `OK` (job did not exit early).
- Two wandb runs `pi0_mem_droid_k6_verify` / `pi0_mem_droid_k1_verify` appear under project `video-encoder-eval` with decreasing loss curves.
- `K6_EXIT=0` and `K1_EXIT=0`.
- Final checkpoints exist at `checkpoints/pi0_mem_droid_k6_verify/pi0_mem_droid_k6_verify/10000` and the K=1 equivalent, each with an `assets/` dir — ready for sub-project C.

---

## Self-Review

**Spec coverage:**
- Change 1 (two configs) → Task 4. ✓
- Change 2 (SigLIP→video_img scan-unstack remap) → Task 3. ✓
- Change 3 (NCHC parallel launch sbatch) → Task 6. ✓
- F4 `action_horizon=16` → Task 4 configs + Task 4 smoke assertion. ✓
- Gates: A-suite green → Task 3 Step 7 (video_vit_test) + run `pi0_mem_test.py` in Task 1; T-B1 → Task 1; T-B2 → Task 2; T-B3 → Task 5 Step 3 (NCHC smoke; documented realization since `dummy` lacks LoRA); T-B4 → Task 3 Steps 5–6; NCHC key-diff → Task 5. ✓
- DoD final checkpoints + wandb runs → Task 6 Step 4. ✓

**Placeholder scan:** One deliberate operator value — the SLURM `--partition` (Task 6 Step 1 gives the `sinfo` command to resolve it; template marks it `# REPLACE`). This is environment config the planner cannot know; every other value is concrete. No `TBD`/`TODO`/"handle edge cases".

**Type consistency:** `SiglipToVideoImgWeightLoader(params_path, missing_regex)`, `_unstack_scanned_encoderblocks(flat)`, `_seed_video_img_from_img(loaded)` are defined in Task 3 and referenced identically in Tasks 4–5. Config names `pi0_mem_droid_k1_verify` / `pi0_mem_droid_k6_verify` are consistent across Tasks 4–6.

**Note on guard tests (T-B1, T-B2):** these lock *existing* behavior, so they pass on first run rather than following the red→green cycle. Their value is regression protection; a first-run failure signals a pre-existing regression to investigate.
