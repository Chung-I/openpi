import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils


def test_pi0_mem_ll_loss():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)


def test_pi0_mem_sample_actions():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)


def test_pi0_mem_hl_loss():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)

    # Target tokens for HL: subtask + memory text (tokenized)
    target_len = 32
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)

    loss = nnx_utils.module_jit(model.compute_loss_hl)(key, obs, target_tokens, target_mask)
    assert loss.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(loss))


def test_pi0_mem_combined_loss():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # hl_targets: tokenized subtask + memory text, shape [b, t]
    target_len = 32
    hl_targets = jnp.ones((batch_size, target_len), dtype=jnp.int32)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act, hl_targets=hl_targets)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))


def _mem_model_and_obs(batch_size: int = 2):
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = config.fake_obs(batch_size)
    return config, model, obs


def test_pi0_mem_ll_prefix_excludes_memory():
    # The LL prefix length must NOT depend on the language memory: the low-level
    # policy conditions on video + subtask + goal only (MEM paper, Fig. 1).
    _, model, obs = _mem_model_and_obs()
    assert obs.tokenized_memory is not None  # precondition: fake_obs provides memory
    obs_no_mem = dataclasses.replace(obs, tokenized_memory=None, tokenized_memory_mask=None)
    len_with = model.embed_prefix_ll(obs)[0].shape[1]
    len_without = model.embed_prefix_ll(obs_no_mem)[0].shape[1]
    assert len_with == len_without


def test_pi0_mem_hl_prefix_includes_memory():
    # Regression guard: the HL prefix MUST still use the language memory.
    _, model, obs = _mem_model_and_obs()
    obs_no_mem = dataclasses.replace(obs, tokenized_memory=None, tokenized_memory_mask=None)
    len_with = model.embed_prefix_hl(obs)[0].shape[1]
    len_without = model.embed_prefix_hl(obs_no_mem)[0].shape[1]
    assert len_with > len_without


def _fast_action_obs(config, batch_size):
    # fake_obs provides tokenized_action (ones); make the loss mask non-empty.
    obs = config.fake_obs(batch_size)
    return dataclasses.replace(
        obs,
        tokenized_action_loss_mask=jnp.ones_like(obs.tokenized_action_loss_mask),
    )


def test_pi0_mem_fast_loss_shape_finite():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = _fast_action_obs(config, 2)
    prefix = model.embed_prefix_ll(obs)
    loss = nnx_utils.module_jit(model.compute_loss_fast)(obs, *prefix)
    assert loss.shape == (2,)
    assert jnp.all(jnp.isfinite(loss))


def _grad_abs_by_path(model, scalar_loss_fn):
    graphdef, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda p: scalar_loss_fn(nnx.merge(graphdef, p)))(params)
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        key = "/".join(
            str(getattr(p, "key", getattr(p, "idx", p))) for p in path
        )
        out[key] = float(jnp.sum(jnp.abs(leaf)))
    return out


def _is_backbone(path: str) -> bool:
    # gemma expert-0 (backbone): under llm, NOT an action-expert (_1) leaf, NOT lora.
    return "llm" in path and "_1" not in path


def test_flow_loss_does_not_touch_backbone_or_video():
    key = jax.random.key(1)
    # flow-only: fast and hl weights zero.
    config = Pi0MEMConfig(
        paligemma_variant="dummy", action_expert_variant="dummy",
        fast_loss_weight=0.0, hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = config.fake_obs(1), config.fake_act(1)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    for path, g in norms.items():
        if _is_backbone(path) or "video_img" in path:
            assert g == 0.0, f"flow loss leaked into {path}: {g}"
    # action expert / projections must receive gradient
    assert any(g > 0 for p, g in norms.items() if "_1" in p or "proj" in p or "time_mlp" in p)


def test_fast_loss_trains_backbone_and_video():
    key = jax.random.key(2)
    config = Pi0MEMConfig(
        paligemma_variant="dummy", action_expert_variant="dummy",
        ll_loss_weight=0.0, hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = _fast_action_obs(config, 1), config.fake_act(1)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    assert any(g > 0 for p, g in norms.items() if _is_backbone(p))
    assert any(g > 0 for p, g in norms.items() if "video_img" in p)


def test_get_prefix_weights_linear_matches_reference():
    from openpi.models.pi0_mem import get_prefix_weights
    import numpy as np

    w = np.asarray(get_prefix_weights(2, 6, 10, "linear"))
    np.testing.assert_allclose(w, [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], atol=1e-6)


def test_get_prefix_weights_schedules():
    from openpi.models.pi0_mem import get_prefix_weights
    import numpy as np

    # ones: all 1 except positions >= end
    np.testing.assert_allclose(np.asarray(get_prefix_weights(0, 4, 6, "ones")), [1, 1, 1, 1, 0, 0])
    # end == 0 -> entire prefix ignored (all zeros)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(3, 0, 5, "linear")), [0, 0, 0, 0, 0])
    # zeros: 1 below start, else 0 (and 0 at/after end)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(2, 5, 6, "zeros")), [1, 1, 0, 0, 0, 0])
