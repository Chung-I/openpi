# pi0.5 Layer Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut pi0.5's shared 18-layer Gemma stack down to 6 layers for a ~2x inference speedup, then recover task accuracy with a LoRA finetune on DROID joint-position data.

**Architecture:** The 18 transformer blocks are a single `nn.scan` shared by both the PaliGemma backbone and the action expert, so every parameter under `PaliGemma/llm/layers/` carries a leading depth axis. Building the model at `depth=6` and gathering 6 slices out of the checkpoint yields a genuinely 6-layer model — no masking, no wasted compute, both towers truncated at once. Two arms train identically except for which layers they keep, so the depth-18 arm isolates truncation's cost from finetuning's effect and doubles as an identity test.

**Tech Stack:** JAX / Flax NNX, `nn.scan`-based Gemma, LoRA adapters, DROID RLDS via `tf.data`, wandb, Slurm on NCHC H200s, RoboLab (Isaac Sim) for evaluation.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-21-pi05-layer-truncation-design.md`. Read it before starting.
- **Branch:** `worktree-pi05-layer-truncation`, in worktree `.claude/worktrees/pi05-layer-truncation`. Run everything from the worktree; do not `cd` to the main checkout.
- **Layers kept (arm A):** exactly `(0, 3, 7, 11, 14, 17)`. Ascending, unique, 0-based indices into the original 18-layer stack.
- **Base checkpoint:** `gs://openpi-assets-simeval/pi05_droid_jointpos/params`.
- **Norm stats:** `gs://openpi-assets-simeval/pi05_droid_jointpos/assets`, `asset_id="droid"`. **Joint-position stats, never velocity** — velocity stats on position targets distort the flow loss and produce a timid policy.
- **Freeze filter:** always `nnx.Not(nnx_utils.PathRegex(".*lora.*"))` written explicitly. Never call `Pi0Config.get_freeze_filter()` for these configs — it freezes only `.*llm.*`, leaving SigLIP trainable, which previously collapsed this exact policy to 0%.
- **Scope:** the JAX `Pi0` path only. `pi0_fast.py` and the PyTorch path are out of scope and must fail loudly rather than silently ignore truncation.
- **Test convention:** tests live beside their source as `<module>_test.py`. `pyproject.toml` sets `testpaths = ["src", "scripts", "packages"]` and defines a `manual` marker for tests that need a cluster or GCS.
- **Do not modify:** `src/openpi/models/gemma.py`. Truncation is expressed entirely through config depth and weight gathering.

## Prerequisites

The worktree has no virtualenv yet. Create one before Task 1:

```bash
cd /home/chungyili/Codes/openpi/.claude/worktrees/pi05-layer-truncation
uv sync
uv run python -c "import flax.nnx; print('ok')"
```

Expected: `ok`. Every task's tests run on CPU in seconds — no GPU needed until Task 4.

## File Structure

| File | Responsibility |
|---|---|
| `src/openpi/models/pi0_config.py` (modify) | Owns `keep_layers`: validation, and turning it into truncated gemma configs. |
| `src/openpi/models/pi0.py` (modify, line 70-71) | Consumes the truncated configs when building the model. |
| `src/openpi/models_pytorch/pi0_pytorch.py` (modify, line 90) | Guard: refuse `keep_layers` rather than silently build full depth. |
| `src/openpi/models/pi0_test.py` (modify) | Tests for validation, depth override, and the built model's scan axis. |
| `src/openpi/training/weight_loaders.py` (modify) | Owns the gather: slicing a checkpoint's scan axis down to `keep_layers`. |
| `src/openpi/training/weight_loaders_test.py` (create) | Tests for the gather, including the identity guard. |
| `src/openpi/training/config.py` (modify) | The two experiment arms. |
| `src/openpi/training/config_test.py` (create) | Tests that both arms are well-formed and consistent. |
| `scripts/bench_inference.py` (create) | Gate 0: measures inference latency for any config. |
| `scripts/bench_inference_test.py` (create) | Tests the timing harness on a dummy-sized model. |
| `scripts/check_truncation.py` (create) | Gate 1: pre-launch weight-shape and loss check. |
| `scripts/check_truncation_test.py` (create) | Tests the checker's pure logic. |
| `scripts/nchc/train_truncation.sbatch` (create) | Runs both arms concurrently on one 8-GPU allocation. |
| `docs/eval/TRUNCATION.md` (create) | How to reproduce: train, serve, evaluate, and where results go. |

---

### Task 1: `keep_layers` on Pi0Config

Adds the config knob and wires it into model construction. Nothing loads weights yet — this task only changes what shape of model gets built.

**Files:**
- Modify: `src/openpi/models/pi0_config.py`
- Modify: `src/openpi/models/pi0.py:70-71`
- Modify: `src/openpi/models_pytorch/pi0_pytorch.py:90-91`
- Test: `src/openpi/models/pi0_test.py`

**Interfaces:**
- Consumes: nothing — this is the first task.
- Produces:
  - `Pi0Config.keep_layers: tuple[int, ...] | None = None`
  - `Pi0Config.gemma_configs() -> tuple[_gemma.Config, _gemma.Config]` returning `(paligemma_config, action_expert_config)`, depth-truncated when `keep_layers` is set.

- [ ] **Step 1: Write the failing tests**

Append to `src/openpi/models/pi0_test.py`. Add `import pytest` to the existing imports at the top of the file.

```python
def test_keep_layers_defaults_to_none_and_full_depth():
    config = _pi0_config.Pi0Config(pi05=True)
    assert config.keep_layers is None
    paligemma, expert = config.gemma_configs()
    assert paligemma.depth == 18
    assert expert.depth == 18


def test_keep_layers_truncates_both_towers():
    config = _pi0_config.Pi0Config(pi05=True, keep_layers=(0, 3, 7, 11, 14, 17))
    paligemma, expert = config.gemma_configs()
    assert paligemma.depth == 6
    assert expert.depth == 6
    # Only depth changes. Widths and head counts must be untouched.
    assert paligemma.width == 2048
    assert expert.width == 1024
    assert paligemma.num_heads == 8
    assert expert.num_heads == 8


def test_keep_layers_is_coerced_to_a_tuple():
    # Pi0Config is a frozen dataclass; a list field would make it unhashable.
    config = _pi0_config.Pi0Config(pi05=True, keep_layers=[0, 3])
    assert config.keep_layers == (0, 3)


@pytest.mark.parametrize(
    "bad",
    [
        (),            # empty
        (0, 0),        # duplicate
        (3, 1),        # unsorted
        (0, 18),       # index == base depth
        (0, 99),       # index beyond base depth
        (-1, 2),       # negative
    ],
)
def test_keep_layers_validation_rejects(bad):
    with pytest.raises(ValueError):
        _pi0_config.Pi0Config(pi05=True, keep_layers=bad)


def test_keep_layers_shrinks_the_model_scan_axis():
    # The "dummy" variant is depth 4, so this builds abstractly in milliseconds on CPU.
    config = _pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", keep_layers=(0, 2)
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(abstract_model, nnx.Param).flat_state()
    layer_shapes = {path: leaf.value.shape for path, leaf in state.items() if "layers" in path}
    assert layer_shapes, "expected scanned layer params under a 'layers' path element"
    for path, shape in layer_shapes.items():
        assert shape[0] == 2, f"{path} has scan axis {shape[0]}, expected 2"


def test_keep_layers_none_leaves_the_scan_axis_at_full_depth():
    config = _pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(abstract_model, nnx.Param).flat_state()
    layer_shapes = {path: leaf.value.shape for path, leaf in state.items() if "layers" in path}
    assert layer_shapes
    for path, shape in layer_shapes.items():
        assert shape[0] == 4, f"{path} has scan axis {shape[0]}, expected 4"


def test_truncated_model_runs_a_forward_pass():
    """Building at the right shape is not enough -- the model must still run end to end.

    The "dummy" variant is tiny (width 64, depth 4), so this is a real forward pass in
    well under a second on CPU.
    """
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        keep_layers=(0, 2),
    )
    model = config.create(jax.random.key(0))
    actions = model.sample_actions(jax.random.key(1), config.fake_obs(batch_size=2), num_steps=2)
    assert actions.shape == (2, 4, config.action_dim)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest src/openpi/models/pi0_test.py -v -k keep_layers
```

Expected: FAIL — `TypeError: Pi0Config.__init__() got an unexpected keyword argument 'keep_layers'`.

- [ ] **Step 3: Add the field and validation to `Pi0Config`**

In `src/openpi/models/pi0_config.py`, add the field immediately after `pytorch_compile_mode` (line 35):

```python
    # Indices of the original transformer layers to keep, 0-based and ascending, into the
    # base variant's full stack. None keeps every layer, which is the unmodified model.
    # Both towers share one nn.scan (see gemma.py), so this truncates the PaliGemma
    # backbone and the action expert together. Pair with
    # weight_loaders.LayerSubsetWeightLoader using the same indices.
    keep_layers: tuple[int, ...] | None = None
```

Then append to `__post_init__`, after the existing `pytorch_compile_mode` assertion:

```python
        if self.keep_layers is not None:
            # Coerce first: a list field would make this frozen dataclass unhashable.
            object.__setattr__(self, "keep_layers", tuple(self.keep_layers))
            base_depth = _gemma.get_config(self.paligemma_variant).depth
            if not self.keep_layers:
                raise ValueError("keep_layers must not be empty.")
            if list(self.keep_layers) != sorted(set(self.keep_layers)):
                raise ValueError(f"keep_layers must be sorted and unique, got {self.keep_layers}.")
            if not all(0 <= i < base_depth for i in self.keep_layers):
                raise ValueError(
                    f"keep_layers indices must lie in [0, {base_depth}) for variant "
                    f"'{self.paligemma_variant}', got {self.keep_layers}."
                )
```

Add the accessor as a new method on `Pi0Config`, directly above `get_freeze_filter`:

```python
    def gemma_configs(self) -> tuple[_gemma.Config, _gemma.Config]:
        """Returns the (paligemma, action expert) gemma configs.

        Depths are overridden to len(keep_layers) when truncating. Both must match:
        gemma.Module asserts every expert has the same depth, since they share one scan.
        """
        paligemma_config = _gemma.get_config(self.paligemma_variant)
        action_expert_config = _gemma.get_config(self.action_expert_variant)
        if self.keep_layers is not None:
            depth = len(self.keep_layers)
            paligemma_config = dataclasses.replace(paligemma_config, depth=depth)
            action_expert_config = dataclasses.replace(action_expert_config, depth=depth)
        return paligemma_config, action_expert_config
```

- [ ] **Step 4: Wire it into the JAX model**

In `src/openpi/models/pi0.py`, replace lines 70-71:

```python
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
```

with:

```python
        paligemma_config, action_expert_config = config.gemma_configs()
```

- [ ] **Step 5: Guard the PyTorch path**

`src/openpi/models_pytorch/pi0_pytorch.py:90-91` independently calls `_gemma.get_config` from the *same* `Pi0Config`, so without this guard a truncated config would silently build a full-depth PyTorch model. Insert immediately before line 90 (`paligemma_config = _gemma.get_config(...)`):

```python
        if config.keep_layers is not None:
            raise NotImplementedError(
                "keep_layers (layer truncation) is supported on the JAX Pi0 path only. "
                "This PyTorch path would silently build a full-depth model."
            )
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest src/openpi/models/pi0_test.py -v
```

Expected: PASS, including the four pre-existing `test_pi0_*` tests — they must still pass unchanged, proving `keep_layers=None` is a no-op.

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/pi0_config.py src/openpi/models/pi0.py \
        src/openpi/models_pytorch/pi0_pytorch.py src/openpi/models/pi0_test.py
git commit -m "feat(pi0): keep_layers config for gemma depth truncation

Both towers share one nn.scan and gemma.Module asserts equal depth, so a single
len(keep_layers) override truncates the PaliGemma backbone and action expert
together. Defaults to None, leaving the model bit-identical.

The PyTorch path builds from the same Pi0Config but calls get_config itself, so
it raises rather than silently returning a full-depth model."
```

---

### Task 2: `LayerSubsetWeightLoader`

Slices the checkpoint's scan axis so the gathered weights match the truncated model built in Task 1.

**Files:**
- Modify: `src/openpi/training/weight_loaders.py`
- Test: create `src/openpi/training/weight_loaders_test.py`

**Interfaces:**
- Consumes: nothing at runtime from Task 1, but the indices passed here must equal the model's `keep_layers`.
- Produces:
  - `weight_loaders._gather_scanned_layers(loaded_params: at.Params, keep_layers: tuple[int, ...]) -> at.Params`
  - `weight_loaders.LayerSubsetWeightLoader(params_path: str, keep_layers: tuple[int, ...])` — a frozen dataclass implementing the `WeightLoader` protocol's `load(params) -> params`.

- [ ] **Step 1: Write the failing tests**

Create `src/openpi/training/weight_loaders_test.py`:

```python
import flax.traverse_util
import numpy as np
import pytest

from openpi.training import weight_loaders


def _fake_checkpoint_params(depth: int = 18):
    """Mirrors the real pi0.5 layout: scanned layers plus unscanned siblings.

    Verified against a real checkpoint — everything under PaliGemma/llm/layers/ carries a
    leading depth axis; embedder, final_norm, SigLIP and the action projections do not.
    The `_1` suffix marks action-expert parameters, which are scanned in the same stack.
    """
    return {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "q_einsum": {"w": np.arange(depth * 2, dtype=np.float32).reshape(depth, 2)},
                        "q_einsum_1": {"w": np.arange(depth * 2, dtype=np.float32).reshape(depth, 2) * 10},
                    },
                    "mlp": {"linear": np.arange(depth * 3, dtype=np.float32).reshape(depth, 3)},
                    "pre_attention_norm": {"scale": np.arange(depth * 4, dtype=np.float32).reshape(depth, 4)},
                },
                "final_norm": {"scale": np.ones(4, dtype=np.float32)},
                "embedder": {"input_embedding": np.ones((5, 4), dtype=np.float32)},
            },
            "img": {"head": {"kernel": np.ones((2, 2), dtype=np.float32)}},
        },
        "action_out_proj": {"kernel": np.ones((3, 3), dtype=np.float32)},
    }


def test_gather_selects_the_requested_layers():
    params = _fake_checkpoint_params(depth=18)
    keep = (0, 3, 7, 11, 14, 17)
    out = weight_loaders._gather_scanned_layers(params, keep)

    q = out["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum"]["w"]
    assert q.shape == (6, 2)
    np.testing.assert_array_equal(q, params["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum"]["w"][list(keep)])


def test_gather_truncates_the_action_expert_tower_too():
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, (0, 3, 7, 11, 14, 17))
    # `_1`-suffixed params are the action expert; they live in the same scan.
    assert out["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum_1"]["w"].shape == (6, 2)


def test_gather_leaves_unscanned_params_untouched():
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, (0, 3, 7, 11, 14, 17))

    for path in [
        ("PaliGemma", "llm", "final_norm", "scale"),
        ("PaliGemma", "llm", "embedder", "input_embedding"),
        ("PaliGemma", "img", "head", "kernel"),
        ("action_out_proj", "kernel"),
    ]:
        expected = params
        actual = out
        for key in path:
            expected, actual = expected[key], actual[key]
        np.testing.assert_array_equal(actual, expected, err_msg="/".join(path))


def test_gather_over_the_full_range_is_the_identity():
    """The most valuable guard here: catches wrong-axis and wrong-order bugs."""
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, tuple(range(18)))

    flat_in = flax.traverse_util.flatten_dict(params, sep="/")
    flat_out = flax.traverse_util.flatten_dict(out, sep="/")
    assert flat_in.keys() == flat_out.keys()
    for k in flat_in:
        np.testing.assert_array_equal(flat_out[k], flat_in[k], err_msg=k)


def test_gather_preserves_the_requested_order():
    params = _fake_checkpoint_params(depth=18)
    src = params["PaliGemma"]["llm"]["layers"]["mlp"]["linear"]
    out = weight_loaders._gather_scanned_layers(params, (2, 5))

    got = out["PaliGemma"]["llm"]["layers"]["mlp"]["linear"]
    np.testing.assert_array_equal(got[0], src[2])
    np.testing.assert_array_equal(got[1], src[5])


def test_gather_rejects_an_index_beyond_the_checkpoint_depth():
    params = _fake_checkpoint_params(depth=18)
    with pytest.raises(ValueError, match="out of range"):
        weight_loaders._gather_scanned_layers(params, (0, 18))


def test_layer_subset_loader_is_a_weight_loader():
    loader = weight_loaders.LayerSubsetWeightLoader("gs://example/params", keep_layers=(0, 3))
    assert isinstance(loader, weight_loaders.WeightLoader)
    assert loader.keep_layers == (0, 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest src/openpi/training/weight_loaders_test.py -v
```

Expected: FAIL — `AttributeError: module 'openpi.training.weight_loaders' has no attribute '_gather_scanned_layers'`.

- [ ] **Step 3: Implement the gather and the loader**

In `src/openpi/training/weight_loaders.py`, add after the `CheckpointWeightLoader` class:

```python
# Every parameter under this prefix is produced by the shared nn.scan over transformer
# blocks and carries a leading depth axis. Verified against a real checkpoint: the
# embedder, final norms, SigLIP and the action projections sit outside it.
_SCANNED_LAYER_PREFIX = "PaliGemma/llm/layers/"


def _gather_scanned_layers(loaded_params: at.Params, keep_layers: tuple[int, ...]) -> at.Params:
    """Slices the transformer stack's scan axis down to `keep_layers`.

    Both the PaliGemma backbone and the action expert live in one scan, so this single
    gather truncates both towers. Non-scanned parameters pass through unchanged.
    """
    flat = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    index = list(keep_layers)
    result = {}
    for k, v in flat.items():
        if k.startswith(_SCANNED_LAYER_PREFIX):
            depth = v.shape[0]
            if max(index) >= depth:
                raise ValueError(
                    f"keep_layers index {max(index)} is out of range for '{k}', "
                    f"whose scan axis has depth {depth}."
                )
            result[k] = v[index]
        else:
            result[k] = v
    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class LayerSubsetWeightLoader(WeightLoader):
    """Loads a checkpoint keeping only a subset of its transformer layers.

    `keep_layers` must match the `Pi0Config.keep_layers` of the model being loaded into —
    the model is built at `depth=len(keep_layers)`, and the gathered params must match
    those shapes.

    Example:
        LayerSubsetWeightLoader("gs://.../params", keep_layers=(0, 3, 7, 11, 14, 17))
    """

    params_path: str
    keep_layers: tuple[int, ...]

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = _gather_scanned_layers(loaded_params, self.keep_layers)
        # Add all missing LoRA weights. These come from the freshly built model, so they
        # already carry the truncated depth.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")
```

No new imports are needed — `dataclasses`, `flax.traverse_util`, `numpy as np`, `_model`, `download` and `at` are all already imported at the top of the file.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/openpi/training/weight_loaders_test.py -v
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/weight_loaders.py src/openpi/training/weight_loaders_test.py
git commit -m "feat(weights): LayerSubsetWeightLoader gathers the scan axis

Every param under PaliGemma/llm/layers/ carries a leading depth axis from the
shared nn.scan, so one gather truncates both towers; embedder, final norms,
SigLIP and action projections pass through. LoRA params are filled from the
freshly built model and already carry the truncated depth.

Includes an identity guard -- keep_layers=range(18) must reproduce the input
exactly -- which is what catches wrong-axis and wrong-order bugs."
```

---

### Task 3: The two experiment arms

**Files:**
- Modify: `src/openpi/training/config.py`
- Test: create `src/openpi/training/config_test.py`

**Interfaces:**
- Consumes: `Pi0Config.keep_layers` (Task 1), `weight_loaders.LayerSubsetWeightLoader` (Task 2).
- Produces: config names `pi05_droid_jointpos_trunc6` and `pi05_droid_jointpos_trunc18`, both retrievable via `_config.get_config(name)`.

- [ ] **Step 1: Write the failing tests**

Create `src/openpi/training/config_test.py`:

```python
import flax.nnx as nnx
import jax
import pytest

import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.weight_loaders as weight_loaders

TRUNCATION_ARMS = ["pi05_droid_jointpos_trunc6", "pi05_droid_jointpos_trunc18"]


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_truncation_arm_exists(name):
    assert _config.get_config(name).name == name


def test_trunc6_keeps_the_specified_layers():
    config = _config.get_config("pi05_droid_jointpos_trunc6")
    assert config.model.keep_layers == (0, 3, 7, 11, 14, 17)


def test_trunc18_is_the_full_depth_control():
    config = _config.get_config("pi05_droid_jointpos_trunc18")
    assert config.model.keep_layers == tuple(range(18))


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_loader_indices_match_the_model(name):
    """A mismatch here produces a shape error only after the checkpoint downloads."""
    config = _config.get_config(name)
    assert isinstance(config.weight_loader, weight_loaders.LayerSubsetWeightLoader)
    assert config.weight_loader.keep_layers == config.model.keep_layers


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_arms_use_lora_on_both_towers(name):
    config = _config.get_config(name)
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_arms_freeze_everything_except_lora(name):
    """Guards the recorded failure: the stock filter leaves SigLIP trainable, which
    fully trained SigLIP on DROID and collapsed this policy to a timid 0%."""
    config = _config.get_config(name)
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(config.freeze_filter))).flat_state()
    assert trainable, "expected some trainable params"
    # NOTE: paths are tuples of path elements and LoRA params are named "lora_a"/"lora_b",
    # so `"lora" in path` (exact-element membership) is ALWAYS False. Substring-match each
    # element instead. Verified: this yields exactly 20 trainable params, all LoRA.
    for path in trainable:
        assert any("lora" in element for element in path), f"{path} is trainable but is not a LoRA param"
    # SigLIP especially must stay frozen: training it on DROID collapsed this policy to 0%.
    assert not any("img" in path for path in trainable), "SigLIP must be frozen"


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_arms_use_jointpos_action_space_and_assets(name):
    """Velocity norm stats on position targets distort the flow loss."""
    config = _config.get_config(name)
    assert config.data.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION
    assert "jointpos" in config.data.assets.assets_dir
    assert config.data.assets.asset_id == "droid"


def test_arms_share_every_hyperparameter_but_depth():
    """The control only isolates truncation if nothing else differs."""
    a = _config.get_config("pi05_droid_jointpos_trunc6")
    b = _config.get_config("pi05_droid_jointpos_trunc18")
    for field in ["num_train_steps", "batch_size", "fsdp_devices", "seed", "save_interval", "keep_period", "num_workers"]:
        assert getattr(a, field) == getattr(b, field), field
    assert a.data.datasets == b.data.datasets
    assert a.weight_loader.params_path == b.weight_loader.params_path


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_arms_train_for_20k_steps_with_rlds_settings(name):
    config = _config.get_config(name)
    assert config.num_train_steps == 20_000  # 10k was previously diagnosed as too few
    assert config.num_workers == 0  # RLDS parallelizes internally via tf.data
    assert config.batch_size == 128
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest src/openpi/training/config_test.py -v
```

Expected: FAIL — `ValueError: Config 'pi05_droid_jointpos_trunc6' not found.`

- [ ] **Step 3: Add the import and the arm builder**

In `src/openpi/training/config.py`, add to the imports (after `import openpi.shared.normalize as _normalize`):

```python
import openpi.shared.nnx_utils as nnx_utils
```

Then define the builder immediately above the `_CONFIGS = [` list:

```python
def _truncation_arm(keep_layers: tuple[int, ...]) -> TrainConfig:
    """One arm of the layer-truncation experiment.

    Arm A keeps 6 of 18 layers; arm B keeps all 18 and is the control that separates the
    cost of truncation from the effect of finetuning. Everything except `keep_layers` is
    identical between arms by construction -- that is the point of building both here.
    """
    return TrainConfig(
        name=f"pi05_droid_jointpos_trunc{len(keep_layers)}",
        exp_name=f"pi05_droid_jointpos_trunc{len(keep_layers)}",
        project_name="layer-truncation",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            keep_layers=keep_layers,
        ),
        # Train LoRA adapters only. Deliberately NOT Pi0Config.get_freeze_filter(): that
        # freezes only ".*llm.*", leaving SigLIP trainable, which previously fully trained
        # SigLIP on DROID and collapsed this exact policy to a timid 0%.
        freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*")),
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
                # Joint-POSITION norm stats. Velocity stats on position targets distort the
                # flow loss, which was diagnosed as a cause of a timid policy.
                assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.LayerSubsetWeightLoader(
            "gs://openpi-assets-simeval/pi05_droid_jointpos/params",
            keep_layers=keep_layers,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        fsdp_devices=4,
        num_train_steps=20_000,
        batch_size=128,
        seed=42,
        save_interval=2_500,
        # Retains 5k/10k/15k/20k permanently so both arms can be read at matched steps.
        keep_period=5_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    )
```

Register both arms in `_CONFIGS`, immediately after the `pi05_droid_finetune` entry:

```python
    #
    # Layer-truncation configs. See docs/superpowers/specs/2026-07-21-pi05-layer-truncation-design.md
    #
    _truncation_arm((0, 3, 7, 11, 14, 17)),
    _truncation_arm(tuple(range(18))),
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/openpi/training/config_test.py -v
```

Expected: PASS. If `test_arms_freeze_everything_except_lora` fails, the freeze filter is wrong — fix it rather than relaxing the test; this is the guard against a recorded 0% failure.

- [ ] **Step 5: Verify the configs load through the CLI path**

```bash
uv run python -c "
import openpi.training.config as c
for n in ['pi05_droid_jointpos_trunc6', 'pi05_droid_jointpos_trunc18']:
    cfg = c.get_config(n)
    print(n, cfg.model.keep_layers, cfg.batch_size, cfg.num_train_steps)
"
```

Expected:
```
pi05_droid_jointpos_trunc6 (0, 3, 7, 11, 14, 17) 128 20000
pi05_droid_jointpos_trunc18 (0, 1, 2, ..., 17) 128 20000
```

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/config.py src/openpi/training/config_test.py
git commit -m "feat(config): layer-truncation arms trunc6 and trunc18

Both arms built by one function so they differ only in keep_layers -- the
control is only a control if nothing else varies, and a test asserts it.

Freeze filter is written explicitly rather than via get_freeze_filter(), which
would leave SigLIP trainable; a test asserts only LoRA params are trainable.
Assets point at joint-position norm stats, also asserted."
```

---

### Task 4: Inference benchmark (Gate 0)

Confirms the predicted 66.0 -> 33.1 ms on the actual DROID arms. The spec's Gate 0 was already answered by a prior latency profile on `pi05_libero`, so this is a confirmation of transfer, not a stop/go gate.

**Files:**
- Create: `scripts/bench_inference.py`
- Test: create `scripts/bench_inference_test.py`

**Interfaces:**
- Consumes: config names from Task 3.
- Produces: `bench_inference.BenchResult` (dataclass with `p50_ms`, `p95_ms`, `min_ms`, `num_layers`), and `bench_inference.benchmark(args: Args) -> BenchResult`.

- [ ] **Step 1: Write the failing test**

Create `scripts/bench_inference_test.py`:

```python
import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import bench_inference


def test_benchmark_runs_on_a_dummy_model():
    """Uses the depth-4 'dummy' variant so this runs on CPU in seconds."""
    args = bench_inference.Args(config_name="debug_pi05", warmup=1, repeats=2, num_steps=2)
    result = bench_inference.benchmark(args)

    assert result.p50_ms > 0
    assert result.p95_ms >= result.p50_ms
    assert result.min_ms <= result.p50_ms
    assert result.num_layers == 4  # the "dummy" gemma variant is depth 4


def test_num_layers_reflects_keep_layers():
    base = _config.get_config("debug_pi05")
    truncated = dataclasses.replace(base, model=dataclasses.replace(base.model, keep_layers=(0, 2)))
    assert bench_inference.num_layers_of(truncated) == 2
    assert bench_inference.num_layers_of(base) == 4
```

`scripts/` is a package (it has `__init__.py`), so sibling modules are imported
relatively — `from . import bench_inference`, matching `scripts/train_test.py`. That file
also sets `JAX_PLATFORMS=cpu` before the first JAX import, which keeps unit tests off the
GPU; do the same here.

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest scripts/bench_inference_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bench_inference'`.

- [ ] **Step 3: Write the benchmark script**

Create `scripts/bench_inference.py`:

```python
"""Measures pi0.5 single-observation inference latency for a given train config.

Weights are random: timing is faithful, task success is not measured. This mirrors how
the reference latency profile was produced, and is what makes the benchmark runnable
without downloading an 11 GiB checkpoint.

Usage:
    uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc18
    uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc6
"""

import dataclasses
import statistics
import time

import jax
import numpy as np
import tyro

import openpi.models.gemma as _gemma
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config


@dataclasses.dataclass
class Args:
    # Name of a train config, e.g. "pi05_droid_jointpos_trunc6".
    config_name: str
    # Flow-matching denoise steps. 10 is the deployed default (pi0.py sample_actions).
    num_steps: int = 10
    # Discarded calls that pay for JIT compilation.
    warmup: int = 3
    # Timed calls. The reference profile used a median of 8-12.
    repeats: int = 12
    batch_size: int = 1


@dataclasses.dataclass
class BenchResult:
    p50_ms: float
    p95_ms: float
    min_ms: float
    num_layers: int


def num_layers_of(train_config: _config.TrainConfig) -> int:
    """Depth the model will actually be built at."""
    model_config = train_config.model
    if getattr(model_config, "keep_layers", None) is not None:
        return len(model_config.keep_layers)
    return _gemma.get_config(model_config.paligemma_variant).depth


def benchmark(args: Args) -> BenchResult:
    train_config = _config.get_config(args.config_name)
    model_config = train_config.model

    model = model_config.create(jax.random.key(0))
    obs = model_config.fake_obs(batch_size=args.batch_size)
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(0)

    for _ in range(args.warmup):
        jax.block_until_ready(sample_actions(rng, obs, num_steps=args.num_steps))

    times_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(sample_actions(rng, obs, num_steps=args.num_steps))
        times_ms.append((time.perf_counter() - start) * 1e3)

    return BenchResult(
        p50_ms=statistics.median(times_ms),
        p95_ms=float(np.percentile(times_ms, 95)),
        min_ms=min(times_ms),
        num_layers=num_layers_of(train_config),
    )


def main(args: Args) -> None:
    result = benchmark(args)
    print(f"config      : {args.config_name}")
    print(f"layers      : {result.num_layers}")
    print(f"denoise     : {args.num_steps} steps")
    print(f"p50         : {result.p50_ms:.1f} ms")
    print(f"p95         : {result.p95_ms:.1f} ms")
    print(f"min         : {result.min_ms:.1f} ms")


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest scripts/bench_inference_test.py -v
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_inference.py scripts/bench_inference_test.py
git commit -m "feat(bench): single-obs inference latency benchmark

Random weights -- timing is faithful, task success is not measured -- so it runs
without downloading an 11 GiB checkpoint. Defaults to 10 denoise steps, the
deployed value, not the 16 used in some published latency tables."
```

- [ ] **Step 6: Run Gate 0 on a GPU and record the numbers**

This needs a GPU. A 3B bf16 model needs roughly 6 GB for inference, so the local RTX 5090 is sufficient — the recorded OOM on that card applies to 3B *gradients*.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc18
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc6
```

Predicted from the reference profile, at 10 denoise steps and 968 prefix tokens:

| config | predicted p50 |
|---|---|
| `pi05_droid_jointpos_trunc18` | ~66.0 ms |
| `pi05_droid_jointpos_trunc6` | ~33.1 ms |

Record both measured numbers — they go into `docs/eval/TRUNCATION.md` in Task 6. **If the measured ratio is below 1.5x, stop and report before spending training GPU-hours**; that would mean the profile failed to transfer and the spec's premise needs revisiting.

---

### Task 5: Pre-launch weight check (Gate 1)

A shape mismatch between loader and model surfaces only after an 11 GiB download and a Slurm queue wait. This catches it in seconds, and runs as a gate inside the sbatch before either arm starts.

**Files:**
- Create: `scripts/check_truncation.py`
- Test: create `scripts/check_truncation_test.py`

**Interfaces:**
- Consumes: config names from Task 3.
- Produces: `check_truncation.check(config_name: str) -> dict[str, int]` mapping scanned-param path to its depth; raises on mismatch.

- [ ] **Step 1: Write the failing test**

Create `scripts/check_truncation_test.py`:

```python
import os

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from . import check_truncation


def test_scan_depths_reads_the_leading_axis():
    params = {
        "PaliGemma": {
            "llm": {
                "layers": {"attn": {"q_einsum": {"w": np.zeros((6, 2))}}},
                "final_norm": {"scale": np.zeros(4)},
            }
        }
    }
    depths = check_truncation.scan_depths(params)
    assert depths == {"PaliGemma/llm/layers/attn/q_einsum/w": 6}


def test_scan_depths_ignores_unscanned_params():
    params = {"action_out_proj": {"kernel": np.zeros((3, 3))}}
    assert check_truncation.scan_depths(params) == {}


def test_assert_uniform_depth_accepts_a_consistent_stack():
    check_truncation.assert_uniform_depth({"a": 6, "b": 6}, expected=6)


def test_assert_uniform_depth_rejects_a_wrong_depth():
    with pytest.raises(ValueError, match="expected 6"):
        check_truncation.assert_uniform_depth({"a": 18}, expected=6)


def test_assert_uniform_depth_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="no scanned"):
        check_truncation.assert_uniform_depth({}, expected=6)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest scripts/check_truncation_test.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'check_truncation'`.

- [ ] **Step 3: Write the checker**

Create `scripts/check_truncation.py`:

```python
"""Pre-launch gate for the layer-truncation runs.

Builds the model's parameter shapes, runs the config's weight loader against them, and
verifies every scanned transformer parameter came back at the truncated depth. A
mismatch here would otherwise surface only after an 11 GiB checkpoint download and a
Slurm queue wait.

Usage:
    uv run python scripts/check_truncation.py pi05_droid_jointpos_trunc6
"""

import sys

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np

import openpi.models.gemma as _gemma
import openpi.training.config as _config

_SCANNED_LAYER_PREFIX = "PaliGemma/llm/layers/"


def scan_depths(params) -> dict[str, int]:
    """Leading-axis size of every scanned transformer parameter, keyed by path."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return {k: v.shape[0] for k, v in flat.items() if k.startswith(_SCANNED_LAYER_PREFIX)}


def assert_uniform_depth(depths: dict[str, int], expected: int) -> None:
    """Every scanned parameter must sit at exactly `expected` depth."""
    if not depths:
        raise ValueError(f"found no scanned params under '{_SCANNED_LAYER_PREFIX}'")
    wrong = {k: d for k, d in depths.items() if d != expected}
    if wrong:
        sample = list(wrong.items())[:5]
        raise ValueError(f"expected {expected} layers, but {len(wrong)} params disagree: {sample}")


def check(config_name: str) -> dict[str, int]:
    train_config = _config.get_config(config_name)
    model_config = train_config.model

    if model_config.keep_layers is None:
        expected = _gemma.get_config(model_config.paligemma_variant).depth
    else:
        expected = len(model_config.keep_layers)

    # Abstract build: shapes only, no device memory for the 3B model.
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    params_shape = nnx.state(abstract_model, nnx.Param).to_pure_dict()

    model_depths = scan_depths(params_shape)
    assert_uniform_depth(model_depths, expected)
    print(f"model built at depth {expected} ({len(model_depths)} scanned params) OK")

    loaded = train_config.weight_loader.load(params_shape)
    loaded_depths = scan_depths(loaded)
    assert_uniform_depth(loaded_depths, expected)
    print(f"weights loaded at depth {expected} ({len(loaded_depths)} scanned params) OK")

    # Every scanned param the model wants must have been loaded, at a matching shape.
    flat_model = flax.traverse_util.flatten_dict(params_shape, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    missing = sorted(set(flat_model) - set(flat_loaded))
    if missing:
        raise ValueError(f"{len(missing)} params were not loaded, e.g. {missing[:5]}")
    for k, want in flat_model.items():
        got = flat_loaded[k]
        if tuple(want.shape) != tuple(np.shape(got)):
            raise ValueError(f"shape mismatch for '{k}': model wants {want.shape}, loader gave {np.shape(got)}")
    print(f"all {len(flat_model)} params match the model's shapes OK")

    return loaded_depths


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_truncation.py <config_name>", file=sys.stderr)
        return 2
    try:
        check(sys.argv[1])
    except (ValueError, KeyError) as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        return 1
    print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Two corrections applied during execution, kept here so the plan matches the tree:**

1. The open question about `nnx.state(abstract_model, nnx.Param).to_pure_dict()` is **settled: it is correct.** Verified by reconstructing the real `init_train_state` closure from `scripts/train.py:90-116` (including the bf16 freeze-filter cast and `TrainState` wrapping) and diffing its `params.to_pure_dict()` against the gate's tree for `pi05_droid_jointpos_trunc6` — 71 leaves each, identical key sets and shapes. No fallback to `train.py:122` is needed.

2. The hand-rolled missing-key and shape-comparison block above was **replaced with the trainer's own validation**, so the gate is provably equivalent rather than equivalent-given-today's-loaders. It checks both directions of structural mismatch and dtypes:

```python
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    print("all params match the model's shapes and dtypes OK")
```

This needs `import openpi.shared.array_typing as at`, and `main()`'s exception handler must cover whatever `check_pytree_equality` raises so a mismatch still exits non-zero with the friendly `CHECK FAILED` message.

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest scripts/check_truncation_test.py -v
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Run the whole suite, then commit**

```bash
uv run pytest src/openpi/models/pi0_test.py src/openpi/training/weight_loaders_test.py \
              src/openpi/training/config_test.py scripts/check_truncation_test.py -v
git add scripts/check_truncation.py scripts/check_truncation_test.py
git commit -m "feat(check): pre-launch depth gate for truncation runs

Verifies the model's scanned params and the loaded checkpoint agree on depth,
and that every param the model wants came back at a matching shape. Catches
index and axis bugs in seconds instead of after an 11 GiB download and a queue
wait. Runs as a gate inside the training sbatch."
```

---

### Task 6: Cluster launch and reproduction docs

**Files:**
- Create: `scripts/nchc/train_truncation.sbatch`
- Create: `docs/eval/TRUNCATION.md`

**Interfaces:**
- Consumes: config names from Task 3, `scripts/check_truncation.py` from Task 5, `scripts/bench_inference.py` from Task 4.
- Produces: nothing importable — this is the operational surface.

- [ ] **Step 1: Write the sbatch script**

Both arms share one 8-GPU allocation, 4 GPUs each, so the experiment needs a single scheduling slot. This matters: another account has been observed holding 32-56 GPUs on MST114563, making two separate slots a long wait.

Create `scripts/nchc/train_truncation.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=trunc
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/trunc_%j.log
set -x
cd /work/roboleon1295/openpi

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache

A6_LOG=logs/trunc6_${SLURM_JOB_ID}.log
A18_LOG=logs/trunc18_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast before burning any GPU-hours. ---
for cfg in pi05_droid_jointpos_trunc6 pi05_droid_jointpos_trunc18; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
    scripts/check_truncation.py "$cfg" || { echo "WEIGHT CHECK FAILED: $cfg"; exit 1; }
done

# --- Two arms in parallel: 4 GPUs each, disjoint device sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc6 > "$A6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc18 > "$A18_LOG" 2>&1 &
P18=$!

wait $P6;  echo "TRUNC6_EXIT=$?"
wait $P18; echo "TRUNC18_EXIT=$?"

echo "=== trunc6 tail ===";  tail -n 40 "$A6_LOG"
echo "=== trunc18 tail ==="; tail -n 40 "$A18_LOG"
```

- [ ] **Step 2: Verify the sbatch is syntactically valid**

```bash
bash -n scripts/nchc/train_truncation.sbatch && echo "syntax ok"
```

Expected: `syntax ok`.

- [ ] **Step 3: Write the reproduction doc**

Create `docs/eval/TRUNCATION.md`:

````markdown
# pi0.5 Layer Truncation — Reproduction

Cuts the shared 18-layer gemma stack to 6 layers `(0, 3, 7, 11, 14, 17)` for a ~2x
inference speedup, then recovers accuracy with a LoRA finetune on DROID joint-position
data. Design: `docs/superpowers/specs/2026-07-21-pi05-layer-truncation-design.md`.

## Arms

| config | layers | role |
|---|---|---|
| `pi05_droid_jointpos_trunc6` | 6 of 18 | the truncated model under test |
| `pi05_droid_jointpos_trunc18` | all 18 | control; isolates truncation from finetuning |

Everything except `keep_layers` is identical, enforced by
`src/openpi/training/config_test.py::test_arms_share_every_hyperparameter_but_depth`.

## 1. Latency (Gate 0)

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py \
  --config-name pi05_droid_jointpos_trunc18
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/bench_inference.py \
  --config-name pi05_droid_jointpos_trunc6
```

Predicted at 10 denoise steps and 968 prefix tokens: 66.0 ms -> 33.1 ms, 1.996x.
Runs on a single RTX 5090 — inference of a 3B bf16 model needs ~6 GB.

| config | predicted p50 | measured p50 |
|---|---|---|
| `pi05_droid_jointpos_trunc18` | 66.0 ms | _fill in_ |
| `pi05_droid_jointpos_trunc6` | 33.1 ms | _fill in_ |

## 2. Train

```bash
sbatch scripts/nchc/train_truncation.sbatch
```

Both arms, 4 GPUs each, on one 8-GPU allocation. 20k steps, batch 128. The script gates
on `scripts/check_truncation.py` first, so a loader/model depth mismatch fails in seconds
rather than after the checkpoint downloads.

Watch the first 100 steps in wandb (project `layer-truncation`). Arm A starting far above
arm B and failing to descend means LoRA is not bridging the depth cut — see Escalation.

## 3. Serve

Stock serving; no MEM session server is involved. `RLDSDroidDataConfig` with
`JOINT_POSITION` already appends `AbsoluteActions(make_bool_mask(7, -1))`, so the served
policy emits absolute joint targets.

```bash
uv run python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config=pi05_droid_jointpos_trunc6 \
  --policy.dir=checkpoints/pi05_droid_jointpos_trunc6/pi05_droid_jointpos_trunc6/19999
```

Note the argument order: `--port` is top-level and must precede the `policy:checkpoint`
subcommand.

## 4. Evaluate

RoboLab, 5 tasks x 16 episodes, matching `docs/eval/RESULTS.md` so results are comparable.

```bash
python policies/pi0_family/run.py --remote-host <host> --remote-port 8000 \
  --policy pi05 --num-envs 16 --num-episodes-adaptive 16 --video-mode none --headless \
  --task BananaInBowlTask BagelsOnPlateTask BowlInBinTask MarkerInMugTask MustardInRightBinTask
```

## 5. Report

Two deltas, both stated explicitly:

- **A minus B** — truncation alone, finetuning held constant.
- **A minus vanilla (40%)** — the practical question, confounded with finetuning; label it.

Success: arm A retains at least 80% of arm B's overall success rate (relative — if B
scores 45%, A must clear 36%). At 16 episodes per task the per-task confidence intervals
are wide; consistency of direction across the 5 tasks matters more than the aggregate. If
the aggregate is small and directions are mixed, re-run the two tasks with the most
headroom at 50-100 episodes.

## Escalation if arm A underfits

In order, stopping as soon as accuracy recovers:

1. Unfreeze the action in/out projections.
2. Full-rank finetune the 6 kept blocks instead of LoRA.
3. Add teacher distillation from the depth-18 model.

## Deliberately out of scope

The dead third image slot (~6.9 ms, ~10%, no retraining), longer action chunks, SigLIP
truncation, and truncating the MEM K=6 model. Each is independently testable and should
stay that way. See the spec's "Deferred" section.
````

- [ ] **Step 4: Commit**

```bash
git add scripts/nchc/train_truncation.sbatch docs/eval/TRUNCATION.md
git commit -m "docs(trunc): cluster launch script and reproduction guide

Both arms share one 8-GPU allocation so the experiment needs a single scheduling
slot -- contention on MST114563 makes two slots a long wait. Gates on
check_truncation.py before either arm starts.

Doc covers latency, training, serving, eval, how to report both deltas, and the
escalation ladder if LoRA underfits the truncated backbone."
```

---

## Verification checklist

Before declaring the implementation complete:

- [ ] `uv run pytest src/openpi/models/pi0_test.py src/openpi/training/weight_loaders_test.py src/openpi/training/config_test.py scripts/bench_inference_test.py scripts/check_truncation_test.py -v` — all pass.
- [ ] The four pre-existing `test_pi0_*` tests still pass, proving `keep_layers=None` is a no-op.
- [ ] `uv run ruff check src scripts` and `uv run ruff format --check src scripts` are clean.
- [ ] Gate 0 measured on a GPU, both numbers recorded in `docs/eval/TRUNCATION.md`, ratio at or above 1.5x.
- [ ] `scripts/check_truncation.py pi05_droid_jointpos_trunc6` passes against the real checkpoint (needs GCS access; run on the cluster).
- [ ] Neither arm's config calls `Pi0Config.get_freeze_filter()`.
