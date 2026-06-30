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
