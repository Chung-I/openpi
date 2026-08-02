"""Config-integrity tests for VLASH-on-DROID Task 3: `pi05_droid_jointpos_vlash`.

The jointpos RLDS recipe (RLDSDroidDataConfig with action_space=JOINT_POSITION,
rlds_data_dir="gs://gresearch/robotics", jointpos norm stats/params) is ported from
`_truncation_arm(...)` on branch `chungyi/pi05-layer-truncation` (src/openpi/training/config.py,
not present on this branch). What's identical: rlds_data_dir, action_space, the DeltaActions
mask (dims 0..6), num_workers=0. What's deliberately different: `assets_dir`/weight_loader
`params_path` point at the local nano4 checkpoint
(`/work/roboleon1295/checkpoints/pi05_droid_jointpos/{assets,params}`) instead of the
`gs://openpi-assets-simeval/pi05_droid_jointpos/...` GCS location the truncation branch uses,
since that checkpoint is already staged locally on nano4 (see Task 0 notes) and this avoids an
extra GCS dependency for norm-stat loading. `pi05_droid_jointpos` itself is not a standalone
TrainConfig on either branch -- on chungi/pi05-layer-truncation it only exists inline inside
`_truncation_arm`; there is no bare "pi05_droid_jointpos" config name to compare against.
"""

import dataclasses
import re

import flax.nnx as nnx
import jax
import pytest

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.transforms_vlash as transforms_vlash

CONFIG_NAME = "pi05_droid_jointpos_vlash"


def test_config_exists():
    config = _config.get_config(CONFIG_NAME)
    assert config.name == CONFIG_NAME


def test_model_is_pi05_with_state_cond_and_action_horizon_15():
    config = _config.get_config(CONFIG_NAME)
    assert config.model.pi05 is True
    assert config.model.state_cond is True
    assert config.model.action_horizon == 15
    assert config.model.discrete_state_input is False


def test_data_is_rlds_jointpos_matching_truncation_branch_recipe():
    """What's ported verbatim from `_truncation_arm` on chungyi/pi05-layer-truncation."""
    config = _config.get_config(CONFIG_NAME)
    assert isinstance(config.data, _config.RLDSDroidDataConfig)
    assert config.data.rlds_data_dir == "gs://gresearch/robotics"
    assert config.data.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION
    assert config.data.datasets[0].filter_dict_path == "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"
    assert config.num_workers == 0  # RLDS parallelizes internally via tf.data
    # Deliberately NOT ported: local nano4 paths instead of the GCS asset/params locations the
    # truncation branch uses (see module docstring).
    assert config.data.assets.assets_dir == "/work/roboleon1295/checkpoints/pi05_droid_jointpos/assets"
    assert config.weight_loader.params_path == "/work/roboleon1295/checkpoints/pi05_droid_jointpos/params"


def test_vlash_delta_max_is_3():
    config = _config.get_config(CONFIG_NAME)
    assert config.data.vlash_delta_max == 3


def test_rlds_window_is_18():
    """action_horizon (15) + delta_max (3) = 18 extra steps of lookahead for the offset roll."""
    config = _config.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.rlds_action_horizon == 18
    assert config.model.action_horizon == 15


def test_shuffle_buffer_size_defaults_to_none():
    """Production default (DroidRldsDataset's own 250_000) unless overridden, e.g. via
    --data.shuffle-buffer-size=<n> for a quick smoke test that doesn't need full shuffling
    entropy and would otherwise risk OOM on a modest single-GPU --mem allocation."""
    config = _config.get_config(CONFIG_NAME)
    assert config.data.shuffle_buffer_size is None
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.shuffle_buffer_size is None


def test_offset_transform_present_after_delta_actions():
    config = _config.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    inputs = data_config.data_transforms.inputs
    kinds = [type(t) for t in inputs]
    assert _transforms.DeltaActions in kinds
    assert transforms_vlash.VlashTemporalOffset in kinds
    assert kinds.index(transforms_vlash.VlashTemporalOffset) > kinds.index(_transforms.DeltaActions)

    offset_transform = inputs[kinds.index(transforms_vlash.VlashTemporalOffset)]
    assert offset_transform.delta_max == 3
    assert offset_transform.action_horizon == 15


def test_tokenizer_uses_pi05_no_state():
    config = _config.get_config(CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    tokenize_transforms = [t for t in data_config.model_transforms.inputs if isinstance(t, _transforms.TokenizePrompt)]
    assert len(tokenize_transforms) == 1
    tok = tokenize_transforms[0]
    assert tok.pi05_no_state is True
    assert tok.discrete_state_input is False


def test_weight_loader_allows_the_three_fresh_state_modules_missing():
    config = _config.get_config(CONFIG_NAME)
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    pattern = re.compile(config.weight_loader.missing_regex)
    for key in ("state_proj/kernel", "state_proj/bias", "state_mlp_in/kernel", "state_mlp_out/bias"):
        assert pattern.fullmatch(key), f"{key} should be allowed missing"
    # Sanity: an unrelated key must NOT be swallowed by the same regex.
    assert not pattern.fullmatch("time_mlp_in/kernel")


def test_upstream_pi05_droid_is_untouched():
    """Guards against Task 3 accidentally changing the pre-existing upstream pi05_droid config."""
    config = _config.get_config("pi05_droid")
    assert config.model.pi05 is True
    assert config.model.action_horizon == 15
    assert config.model.state_cond is False
    assert not isinstance(config.data, _config.RLDSDroidDataConfig)


def test_state_cond_requires_pi05():
    """Task 1 review follow-up: state_cond=True must assert pi05=True in __post_init__."""
    with pytest.raises(ValueError, match="state_cond"):
        pi0_config.Pi0Config(pi05=False, state_cond=True)


# ----------------------------------------------------------------------------------------------
# Task 4: shared-observation variant `pi05_droid_jointpos_vlash_shared`.
# ----------------------------------------------------------------------------------------------

SHARED_CONFIG_NAME = "pi05_droid_jointpos_vlash_shared"


def test_shared_config_flags():
    config = _config.get_config(SHARED_CONFIG_NAME)
    assert config.model.vlash_shared_obs is True
    assert config.model.state_cond is True
    assert config.data.vlash_shared_obs is True
    assert config.data.vlash_delta_max == 3


def test_shared_config_uses_all_offsets_and_branch_split():
    config = _config.get_config(SHARED_CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    data_kinds = [type(t) for t in data_config.data_transforms.inputs]
    assert transforms_vlash.VlashAllOffsets in data_kinds
    assert transforms_vlash.VlashTemporalOffset not in data_kinds
    assert data_kinds.index(transforms_vlash.VlashAllOffsets) > data_kinds.index(_transforms.DeltaActions)
    # SplitVlashBranches must be the LAST model transform (after Normalize, which runs
    # between data_transforms and model_transforms, and after PadStatesAndActions), so every
    # branch state is normalized/padded identically before the stack is split.
    model_kinds = [type(t) for t in data_config.model_transforms.inputs]
    assert model_kinds[-1] is transforms_vlash.SplitVlashBranches
    assert model_kinds.index(transforms_vlash.SplitVlashBranches) > model_kinds.index(_transforms.PadStatesAndActions)
    # The RLDS window still needs the delta_max lookahead.
    assert data_config.rlds_action_horizon == 18


def test_shared_obs_flag_mismatch_raises():
    import dataclasses

    config = _config.get_config(SHARED_CONFIG_NAME)
    plain_model = dataclasses.replace(config.model, vlash_shared_obs=False)
    with pytest.raises(ValueError, match="vlash_shared_obs"):
        config.data.create(config.assets_dirs, plain_model)


def test_unshared_vlash_config_untouched_by_task4():
    """The Task 3 config must keep sampling a single offset (no branch stacking)."""
    config = _config.get_config(CONFIG_NAME)
    assert config.model.vlash_shared_obs is False
    assert config.data.vlash_shared_obs is False
    data_config = config.data.create(config.assets_dirs, config.model)
    data_kinds = [type(t) for t in data_config.data_transforms.inputs]
    assert transforms_vlash.VlashTemporalOffset in data_kinds
    assert transforms_vlash.VlashAllOffsets not in data_kinds
    assert transforms_vlash.SplitVlashBranches not in [type(t) for t in data_config.model_transforms.inputs]


# ----------------------------------------------------------------------------------------------
# Task 5: opt-in LoRA variant `pi05_droid_jointpos_vlash_lora`.
#
# Identical to `pi05_droid_jointpos_vlash` (Task 3) except: (a) the PaliGemma/action-expert
# gemma variants are swapped for their `_lora` forms (openpi's standard LoRA mechanism -- see
# `pi0_libero_low_mem_finetune`/`pi0_fast_libero_low_mem_finetune` above and
# `Pi0Config.get_freeze_filter`), and (b) `freeze_filter` freezes everything EXCEPT the LoRA
# adapters and the three fresh state-conditioning modules. The three state modules are kept
# trainable deliberately: `Pi0Config.get_freeze_filter()` alone would already leave them
# trainable (they aren't nested under ".*llm.*"), but it would ALSO leave SigLIP and the
# action/time projection heads trainable, which is exactly the failure mode the
# chungyi/pi05-layer-truncation branch's LoRA arms documented (leaving SigLIP trainable
# alongside a frozen-except-LoRA LLM caused an imbalance that collapsed the DROID policy to
# 0%). So this variant freezes everything but LoRA + the state tower, matching that branch's
# `nnx.Not(PathRegex(".*lora.*"))` pattern with an explicit carve-out for the state modules.
# ----------------------------------------------------------------------------------------------

LORA_CONFIG_NAME = "pi05_droid_jointpos_vlash_lora"

_STATE_MODULE_NAMES = ("state_proj", "state_mlp_in", "state_mlp_out")


def test_lora_config_exists():
    config = _config.get_config(LORA_CONFIG_NAME)
    assert config.name == LORA_CONFIG_NAME


def test_lora_config_uses_lora_gemma_variants():
    config = _config.get_config(LORA_CONFIG_NAME)
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"


def test_lora_config_matches_vlash_otherwise():
    """Everything but the gemma variants/freeze_filter/weight_loader.missing_regex must match
    `pi05_droid_jointpos_vlash` -- this is an opt-in LoRA *variant*, not a different recipe."""
    base = _config.get_config(CONFIG_NAME)
    lora = _config.get_config(LORA_CONFIG_NAME)
    assert lora.model.pi05 is True
    assert lora.model.state_cond is True
    assert lora.model.action_horizon == base.model.action_horizon == 15
    assert lora.model.discrete_state_input is False
    assert lora.model.action_dim == base.model.action_dim
    assert isinstance(lora.data, _config.RLDSDroidDataConfig)
    assert lora.data.rlds_data_dir == base.data.rlds_data_dir
    assert lora.data.action_space == base.data.action_space
    assert lora.data.vlash_delta_max == base.data.vlash_delta_max == 3
    assert lora.data.assets.assets_dir == base.data.assets.assets_dir
    assert lora.weight_loader.params_path == base.weight_loader.params_path
    assert lora.num_workers == 0


def test_lora_config_window_and_offset_transform():
    config = _config.get_config(LORA_CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.rlds_action_horizon == 18  # 15 + delta_max(3)
    inputs = data_config.data_transforms.inputs
    kinds = [type(t) for t in inputs]
    assert transforms_vlash.VlashTemporalOffset in kinds
    offset_transform = inputs[kinds.index(transforms_vlash.VlashTemporalOffset)]
    assert offset_transform.delta_max == 3
    assert offset_transform.action_horizon == 15


def test_lora_weight_loader_missing_regex_allows_lora_and_state_modules():
    config = _config.get_config(LORA_CONFIG_NAME)
    assert isinstance(config.weight_loader, weight_loaders.CheckpointWeightLoader)
    pattern = re.compile(config.weight_loader.missing_regex)
    for key in ("state_proj/kernel", "state_mlp_in/kernel", "state_mlp_out/bias"):
        assert pattern.fullmatch(key), f"{key} should be allowed missing"
    for key in (
        "PaliGemma/llm/layers/attn/q_einsum/lora_a",
        "PaliGemma/llm/layers/attn/q_einsum_1/lora_b",
        "PaliGemma/llm/layers/mlp/gating_einsum_lora_a",
    ):
        assert pattern.fullmatch(key), f"{key} should be allowed missing"
    # Sanity: a real base weight must NOT be swallowed by the same regex.
    assert not pattern.fullmatch("PaliGemma/llm/layers/attn/attn_vec_einsum/w")
    assert not pattern.fullmatch("time_mlp_in/kernel")


def test_lora_freeze_filter_partitions_params():
    """Shape-only param-partition check (mirrors `pi0_test.py::_get_frozen_state`): base
    attention weights frozen, LoRA adapters trainable, all three state modules trainable."""
    config = _config.get_config(LORA_CONFIG_NAME)
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))

    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, config.freeze_filter)).flat_state()
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(config.freeze_filter))).flat_state()
    frozen_paths = list(frozen)
    trainable_paths = list(trainable)

    # (a) A known base attention kernel is frozen.
    assert ("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "w") in frozen_paths

    # (b) LoRA params are trainable, and no LoRA param is frozen.
    assert any("lora_a" in p or "lora_b" in p for path in trainable_paths for p in path)
    assert not any("lora_a" in p or "lora_b" in p for path in frozen_paths for p in path)

    # (c) All three fresh state modules are trainable (kernel + bias each).
    for module_name in _STATE_MODULE_NAMES:
        module_trainable_paths = [path for path in trainable_paths if path[0] == module_name]
        assert {path[-1] for path in module_trainable_paths} == {"kernel", "bias"}, module_name
        assert not any(path[0] == module_name for path in frozen_paths), module_name


def test_lora_config_ema_disabled():
    """Matches upstream's LoRA convention (`pi0_libero_low_mem_finetune` etc.): EMA is turned
    off for LoRA finetuning."""
    config = _config.get_config(LORA_CONFIG_NAME)
    assert config.ema_decay is None


# ----------------------------------------------------------------------------------------------
# Task 7: per-offset held-out validation hook wiring (`vlash_val_interval`, `vlash_fixed_delta`).
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [CONFIG_NAME, SHARED_CONFIG_NAME, LORA_CONFIG_NAME])
def test_vlash_configs_enable_val_interval(name):
    config = _config.get_config(name)
    assert config.vlash_val_interval == 1000


def test_upstream_configs_leave_val_interval_disabled():
    """Guards against Task 7 accidentally turning the hook on for non-VLASH configs."""
    config = _config.get_config("pi05_droid")
    assert config.vlash_val_interval is None


def test_vlash_fixed_delta_replaces_random_sampling():
    """`vlash_fixed_delta` (used by `openpi.training.vlash_eval.PerOffsetValLoss.build` to
    cache one held-out batch per delta) swaps in `VlashFixedOffset` -- same pipeline position
    as `VlashTemporalOffset`, but forced instead of sampled."""
    config = _config.get_config(CONFIG_NAME)
    fixed_data_cfg = dataclasses.replace(config.data, vlash_fixed_delta=2)
    data_config = fixed_data_cfg.create(config.assets_dirs, config.model)
    kinds = [type(t) for t in data_config.data_transforms.inputs]
    assert transforms_vlash.VlashFixedOffset in kinds
    assert transforms_vlash.VlashTemporalOffset not in kinds
    offset_transform = data_config.data_transforms.inputs[kinds.index(transforms_vlash.VlashFixedOffset)]
    assert offset_transform.delta == 2
    assert offset_transform.action_horizon == 15


def test_vlash_fixed_delta_out_of_range_raises():
    config = _config.get_config(CONFIG_NAME)
    fixed_data_cfg = dataclasses.replace(config.data, vlash_fixed_delta=4)  # > vlash_delta_max=3
    with pytest.raises(ValueError, match="vlash_fixed_delta"):
        fixed_data_cfg.create(config.assets_dirs, config.model)


def test_vlash_fixed_delta_and_shared_obs_mutually_exclusive():
    config = _config.get_config(SHARED_CONFIG_NAME)
    fixed_data_cfg = dataclasses.replace(config.data, vlash_fixed_delta=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fixed_data_cfg.create(config.assets_dirs, config.model)


# ----------------------------------------------------------------------------------------------
# Ablation (SCOPE ADD): `pi05_droid_jointpos_statecond_d0` -- identical to
# pi05_droid_jointpos_vlash except vlash_delta_max=0 (trains only at the delta=0 anchor, no
# temporal-offset augmentation), isolating the AdaRMS state-conditioning architecture factor
# from offset training.
# ----------------------------------------------------------------------------------------------

D0_CONFIG_NAME = "pi05_droid_jointpos_statecond_d0"


def test_statecond_d0_config_window_delta_max_and_state_cond():
    """Window = action_horizon(15) + vlash_delta_max(0) = 15; delta_max=0; state_cond=True."""
    config = _config.get_config(D0_CONFIG_NAME)
    assert config.data.vlash_delta_max == 0
    assert config.model.state_cond is True
    assert config.model.pi05 is True
    assert config.model.vlash_shared_obs is False
    assert config.data.vlash_shared_obs is False
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.rlds_action_horizon == 15
    kinds = [type(t) for t in data_config.data_transforms.inputs]
    assert transforms_vlash.VlashTemporalOffset in kinds
    offset_transform = data_config.data_transforms.inputs[kinds.index(transforms_vlash.VlashTemporalOffset)]
    assert offset_transform.delta_max == 0
    assert offset_transform.action_horizon == 15


def test_statecond_d0_config_matches_vlash_headline_otherwise():
    """Everything but vlash_delta_max/name/project bookkeeping must match
    pi05_droid_jointpos_vlash -- this is a targeted ablation, not a different recipe."""
    base = _config.get_config(CONFIG_NAME)
    d0 = _config.get_config(D0_CONFIG_NAME)
    assert d0.model.pi05 == base.model.pi05 is True
    assert d0.model.state_cond == base.model.state_cond is True
    assert d0.model.action_horizon == base.model.action_horizon == 15
    assert d0.model.discrete_state_input == base.model.discrete_state_input is False
    assert d0.model.action_dim == base.model.action_dim
    assert isinstance(d0.data, _config.RLDSDroidDataConfig)
    assert d0.data.rlds_data_dir == base.data.rlds_data_dir
    assert d0.data.action_space == base.data.action_space
    assert d0.data.assets.assets_dir == base.data.assets.assets_dir
    assert d0.weight_loader.params_path == base.weight_loader.params_path
    assert d0.weight_loader.missing_regex == base.weight_loader.missing_regex
    assert d0.num_workers == base.num_workers == 0
    assert d0.vlash_val_interval == base.vlash_val_interval == 1000
    # The one deliberate difference:
    assert d0.data.vlash_delta_max == 0
    assert base.data.vlash_delta_max == 3


def test_vlash_headline_config_untouched_by_d0_ablation():
    """Guards against this ablation accidentally mutating the shared `pi05_droid_jointpos_vlash`
    headline config (e.g. via an aliased/mutated dataclass field)."""
    config = _config.get_config(CONFIG_NAME)
    assert config.name == CONFIG_NAME
    assert config.model.pi05 is True
    assert config.model.state_cond is True
    assert config.model.action_horizon == 15
    assert config.data.vlash_delta_max == 3
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.rlds_action_horizon == 18
    kinds = [type(t) for t in data_config.data_transforms.inputs]
    offset_transform = data_config.data_transforms.inputs[kinds.index(transforms_vlash.VlashTemporalOffset)]
    assert offset_transform.delta_max == 3
