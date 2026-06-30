import jax

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig


def test_config_defaults():
    config = Pi0MEMConfig()
    assert config.num_video_frames == 6
    assert config.temporal_attn_every_n_layers == 4
    assert config.max_memory_tokens == 128
    assert config.max_subtask_tokens == 64
    assert config.model_type == _model.ModelType.PI0_MEM


def test_config_creates_model():
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(0))
    assert model is not None


def test_inputs_spec_has_mem_fields():
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    obs_spec, act_spec = config.inputs_spec()
    assert obs_spec.tokenized_memory is not None
    assert obs_spec.tokenized_memory_mask is not None
    assert obs_spec.video_images is not None
    assert obs_spec.video_states is not None


def test_fake_obs_has_fast_action_fields():
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    obs = config.fake_obs(2)
    assert obs.tokenized_action is not None
    assert obs.tokenized_action.shape == (2, config.max_action_tokens)
    assert obs.tokenized_action_mask.shape == (2, config.max_action_tokens)
    assert obs.tokenized_action_loss_mask.shape == (2, config.max_action_tokens)


def test_get_freeze_filter_partitions_params():
    config = Pi0MEMConfig(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", lora=True)
    freeze = config.get_freeze_filter()
    # frozen: freeze filter returns True for the path
    def frozen(path_keys):
        return freeze(tuple(path_keys), object())
    # trainable: freeze filter returns False for the path (path-based check; nnx.Param type is a
    # production concern handled via nnx.All(nnx.Param, nnx.Not(freeze)) in training/config.py)
    def is_trainable(path_keys):
        return not freeze(tuple(path_keys), object())
    # backbone base weight: frozen
    assert frozen(["PaliGemma", "llm", "layers", "attn", "kernel"]) is True
    # backbone lora adapter: trainable
    assert is_trainable(["PaliGemma", "llm", "layers", "attn", "lora_a"]) is True
    # video encoder: trainable
    assert is_trainable(["PaliGemma", "video_img", "embedding", "kernel"]) is True
