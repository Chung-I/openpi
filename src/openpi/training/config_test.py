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

import re

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
