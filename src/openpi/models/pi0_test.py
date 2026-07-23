import flax.nnx as nnx
import jax
import pytest

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


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
        (),  # empty
        (0, 0),  # duplicate
        (3, 1),  # unsorted
        (0, 18),  # index == base depth
        (0, 99),  # index beyond base depth
        (-1, 2),  # negative
    ],
)
def test_keep_layers_validation_rejects(bad):
    with pytest.raises(ValueError):  # noqa: PT011 - all failure modes share the same ValueError type
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
