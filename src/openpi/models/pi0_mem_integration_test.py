"""Integration tests for Pi0MEM: end-to-end training steps and config registration.

These tests verify that the full MEM pipeline works together:
  - Config registration in the training registry
  - Forward + backward pass (LL policy)
  - HL policy loss
  - Combined HL + LL loss
  - Gradient flow through the LL loss (with finiteness check)
  - Overfitting on a single batch (loss decreases significantly)
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils
from openpi.training.config import get_config


def test_config_registered():
    """Verify the debug config is accessible via get_config."""
    from openpi.training.config import get_config

    config = get_config("pi0_mem_debug")
    assert config.model.model_type == _model.ModelType.PI0_MEM


def test_end_to_end_training_step():
    """Full forward pass with fake data: LL loss, HL loss, and action sampling."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # LL loss — shape [batch, action_horizon], all finite
    ll_loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert ll_loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(ll_loss))

    # HL loss — shape [batch], all finite
    target_len = 16
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)
    hl_loss = nnx_utils.module_jit(model.compute_loss_hl)(key, obs, target_tokens, target_mask)
    assert hl_loss.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(hl_loss))

    # Action sampling — shape [batch, action_horizon, action_dim], all finite
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert jnp.all(jnp.isfinite(actions))


def test_end_to_end_combined_loss():
    """HL + LL combined loss via compute_loss(hl_targets=...) is finite."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    target_len = config.max_subtask_tokens + config.max_memory_tokens
    hl_targets = jnp.ones((batch_size, target_len), dtype=jnp.int32)

    combined_loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act, hl_targets=hl_targets)
    assert combined_loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(combined_loss))


def test_ll_loss_is_differentiable():
    """Verify gradients flow through the LL loss.

    Uses nnx.split / nnx.merge so that jax.grad operates on a plain pytree
    (State dict) rather than the NNX module directly, avoiding issues with the
    ToNNX bridge wrapper inside Pi0MEM.
    """
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 1  # Use batch_size=1 to keep memory pressure low in the full test suite
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # Split the model into a static graph definition + a differentiable State pytree.
    graphdef, params = nnx.split(model)

    @jax.jit
    def loss_fn(params):
        m = nnx.merge(graphdef, params)
        loss = m.compute_loss(key, obs, act)
        return jnp.mean(loss)

    grads = jax.grad(loss_fn)(params)

    # Verify at least some gradients are non-zero and all are finite.
    flat_grads = jax.tree.leaves(grads)
    has_nonzero = any(jnp.any(g != 0) for g in flat_grads if isinstance(g, jax.Array))
    assert has_nonzero, "Expected some non-zero gradients"
    assert all(jnp.all(jnp.isfinite(g)) for g in flat_grads if isinstance(g, jax.Array)), (
        "Expected all gradients to be finite"
    )


def test_pi0_legacy_mode():
    """Pi05=False backward compat: model creates and produces finite LL loss."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        pi05=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 1
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    ll_loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert ll_loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(ll_loss))


def test_overfitting_single_batch():
    """Model should overfit a single fixed batch: final loss < 50% of initial loss."""
    training_config = get_config("pi0_mem_debug")
    model_config: Pi0MEMConfig = training_config.model  # type: ignore[assignment]

    key = jax.random.key(42)
    model = model_config.create(key)

    batch_size = 1
    obs = model_config.fake_obs(batch_size)
    act = model_config.fake_act(batch_size)

    graphdef, params = nnx.split(model)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        def loss_fn(p):
            m = nnx.merge(graphdef, p)
            loss = m.compute_loss(key, obs, act)
            return jnp.mean(loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    initial_loss, params, opt_state = train_step(params, opt_state)

    for _ in range(99):
        final_loss, params, opt_state = train_step(params, opt_state)

    assert float(final_loss) < 0.5 * float(initial_loss), (
        f"Expected final loss ({float(final_loss):.4f}) to be less than 50% of "
        f"initial loss ({float(initial_loss):.4f})"
    )


