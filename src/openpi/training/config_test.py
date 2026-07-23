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
    for field in [
        "num_train_steps",
        "batch_size",
        "fsdp_devices",
        "seed",
        "save_interval",
        "keep_period",
        "num_workers",
    ]:
        assert getattr(a, field) == getattr(b, field), field
    assert a.data.datasets == b.data.datasets
    assert a.weight_loader.params_path == b.weight_loader.params_path


@pytest.mark.parametrize("name", TRUNCATION_ARMS)
def test_arms_train_for_20k_steps_with_rlds_settings(name):
    config = _config.get_config(name)
    assert config.num_train_steps == 20_000  # 10k was previously diagnosed as too few
    assert config.num_workers == 0  # RLDS parallelizes internally via tf.data
    assert config.batch_size == 128


FULLRANK_ARM = "pi05_droid_jointpos_trunc6_fullrank"


def test_fullrank_arm_exists_and_keeps_the_same_layers():
    config = _config.get_config(FULLRANK_ARM)
    assert config.model.keep_layers == (0, 3, 7, 11, 14, 17)
    assert config.weight_loader.keep_layers == config.model.keep_layers


def test_fullrank_arm_uses_non_lora_variants():
    """Full rank trains the base weights, so the LoRA variants must NOT be used."""
    config = _config.get_config(FULLRANK_ARM)
    assert config.model.paligemma_variant == "gemma_2b"
    assert config.model.action_expert_variant == "gemma_300m"


def test_fullrank_arm_trains_the_kept_blocks_but_not_siglip():
    """The whole point: more trainable capacity than the LoRA arm, with SigLIP still frozen.

    SigLIP must stay frozen -- training it on DROID is the recorded cause of a 0% timid-arm
    policy on this exact setup.
    """
    config = _config.get_config(FULLRANK_ARM)
    abstract_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(config.freeze_filter))).flat_state()
    frozen = nnx.state(abstract_model, nnx.All(nnx.Param, config.freeze_filter)).flat_state()

    assert not any("img" in path for path in trainable), "SigLIP must be frozen"
    assert any("img" in path for path in frozen), "expected SigLIP params in the frozen set"
    # The scanned transformer blocks must be trainable -- that is what full rank means.
    assert any("layers" in path for path in trainable), "kept blocks must be trainable"

    lora_arm = _config.get_config("pi05_droid_jointpos_trunc6")
    lora_abstract = nnx.eval_shape(lora_arm.model.create, jax.random.key(0))
    lora_trainable = nnx.state(lora_abstract, nnx.All(nnx.Param, nnx.Not(lora_arm.freeze_filter))).flat_state()
    assert len(trainable) > len(lora_trainable), "full rank must expose more trainable params than LoRA"


def test_fullrank_arm_matches_the_lora_arm_on_everything_else():
    """Only trainable capacity may differ, or the comparison means nothing."""
    a = _config.get_config("pi05_droid_jointpos_trunc6")
    b = _config.get_config(FULLRANK_ARM)
    for field in [
        "num_train_steps",
        "batch_size",
        "fsdp_devices",
        "seed",
        "save_interval",
        "keep_period",
        "num_workers",
    ]:
        assert getattr(a, field) == getattr(b, field), field
    assert a.lr_schedule == b.lr_schedule
    assert a.weight_loader.params_path == b.weight_loader.params_path
    assert a.data.datasets == b.data.datasets
    assert a.model.action_horizon == b.model.action_horizon


def test_100k_arm_trains_everything_including_siglip():
    """Best-effort arm: nothing frozen, matching upstream's own full-DROID recipe."""
    ours = _config.get_config("pi05_droid_jointpos_trunc6_fullrank_100k")
    upstream = _config.get_config("pi05_full_droid_finetune")

    abstract = nnx.eval_shape(ours.model.create, jax.random.key(0))
    trainable = nnx.state(abstract, nnx.All(nnx.Param, nnx.Not(ours.freeze_filter))).flat_state()
    everything = nnx.state(abstract, nnx.Param).flat_state()
    assert len(trainable) == len(everything), "nothing may be frozen on this arm"
    assert any("img" in path for path in trainable), "SigLIP must train here"

    # Same freeze behaviour as upstream, and the same deliberately FLAT schedule.
    assert type(ours.freeze_filter) is type(upstream.freeze_filter)
    assert ours.lr_schedule.peak_lr == ours.lr_schedule.decay_lr == 5e-5
    assert ours.fsdp_devices == upstream.fsdp_devices == 1


def test_100k_arm_matches_pi05_full_droid_finetune_budget():
    """The 100k arm exists to match upstream's full-DROID recipe, not the LoRA arms."""
    ours = _config.get_config("pi05_droid_jointpos_trunc6_fullrank_100k")
    upstream = _config.get_config("pi05_full_droid_finetune")

    assert ours.num_train_steps == upstream.num_train_steps == 100_000
    assert ours.batch_size == upstream.batch_size == 256
    # Same schedule and optimizer as the upstream full-DROID recipe.
    assert ours.lr_schedule == upstream.lr_schedule
    assert ours.optimizer == upstream.optimizer
    assert ours.ema_decay == upstream.ema_decay
    assert ours.model.action_horizon == upstream.model.action_horizon
    # Still truncated, still full rank, still SigLIP-frozen.
    assert ours.model.keep_layers == (0, 3, 7, 11, 14, 17)
    assert ours.model.paligemma_variant == "gemma_2b"


def test_100k_arm_does_not_disturb_the_matched_arms():
    """Adding it must not change the configs that already ran, or the record is lost."""
    lora = _config.get_config("pi05_droid_jointpos_trunc6")
    matched = _config.get_config("pi05_droid_jointpos_trunc6_fullrank")
    for cfg in (lora, matched):
        assert cfg.batch_size == 128
        assert cfg.num_train_steps == 20_000
        assert cfg.fsdp_devices == 4
        assert cfg.save_interval == 2_500
        assert cfg.keep_period == 5_000
    # The matched full-rank arm must still freeze SigLIP -- that is what makes it matched.
    matched_abstract = nnx.eval_shape(matched.model.create, jax.random.key(0))
    matched_trainable = nnx.state(matched_abstract, nnx.All(nnx.Param, nnx.Not(matched.freeze_filter))).flat_state()
    assert not any("img" in path for path in matched_trainable)
