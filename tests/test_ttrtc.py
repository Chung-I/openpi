"""TT-RTC port (arXiv 2512.05964): per-token flow time, prefix conditioning, masked loss.

Dummy variants keep this CPU-cheap. Conventions under test: this codebase's flow time is
1=noise -> 0=target, so ground-truth prefix tokens are pinned at 0.0 (the reference,
0=noise -> 1=target, pins at 1.0). A convention slip here is exactly the class of bug the
RTC port hit (sign inversion), hence the reduction-to-plain invariants below.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0, pi0_config


def _tiny(**kw):
    return pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=10, **kw
    )


def test_per_token_time_matches_scalar_when_uniform():
    """embed_suffix with a constant [b, ah] time vector must equal the scalar-time path
    (pins the flatten/reshape in the per-token embedding and rank-3 adaRMS broadcast)."""
    cfg = _tiny()
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    x_t = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    t_scalar = jnp.array([0.7])
    t_tok = jnp.full((1, cfg.action_horizon), 0.7)
    toks_a, _, _, cond_a = model.embed_suffix(obs, x_t, t_scalar)
    toks_b, _, _, cond_b = model.embed_suffix(obs, x_t, t_tok)
    np.testing.assert_allclose(np.asarray(toks_a), np.asarray(toks_b), atol=1e-6)
    # scalar cond is [b, w]; per-token is [b, ah, w] with identical rows
    assert cond_b.ndim == 3
    np.testing.assert_allclose(np.asarray(cond_b[:, 0]), np.asarray(cond_a), atol=1e-6)
    np.testing.assert_allclose(np.asarray(cond_b[:, -1]), np.asarray(cond_a), atol=1e-6)


def test_ttrtc_loss_masks_prefix_and_rescales():
    """With an injected delay, prefix positions contribute exactly zero and the postfix is
    rescaled so mean(loss) equals the reference's masked mean."""
    cfg = _tiny(ttrtc_delay_max=4)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    rng = jax.random.key(0)
    d = jnp.array([3])
    loss = model._compute_loss_ttrtc(rng, obs, actions, delay=d)
    loss = np.asarray(loss)
    assert loss.shape == (1, cfg.action_horizon)
    np.testing.assert_allclose(loss[0, :3], 0.0, atol=0)
    assert np.all(loss[0, 3:] > 0)
    # rescaling: mean over ah == mean over the 7 unmasked positions' raw values
    raw_mean = loss[0, 3:].sum() / cfg.action_horizon  # what mean() sees
    assert np.isclose(loss.mean(), raw_mean)


def test_ttrtc_sampler_delay0_reduces_to_plain():
    """inference_delay=0 -> no prefix pinned; must reproduce sample_actions bit-for-bit
    under the same rng (pins the time-convention port)."""
    cfg = _tiny(ttrtc_delay_max=4)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    rng = jax.random.key(5)
    plain = model.sample_actions(rng, obs, num_steps=4)
    prev = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    ttrtc = model.sample_actions_ttrtc(rng, obs, prev, 0, num_steps=4)
    np.testing.assert_allclose(np.asarray(ttrtc), np.asarray(plain), atol=1e-5)


def test_ttrtc_sampler_pins_prefix_exactly():
    """The first d output actions must equal the committed prefix EXACTLY -- hard
    conditioning, unlike RTC's soft guidance."""
    cfg = _tiny(ttrtc_delay_max=4)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    rng = jax.random.key(5)
    prev = jnp.ones((1, cfg.action_horizon, cfg.action_dim)) * 0.3
    out = model.sample_actions_ttrtc(rng, obs, prev, 3, num_steps=4)
    np.testing.assert_array_equal(np.asarray(out[:, :3]), np.asarray(prev[:, :3]))
    assert not np.allclose(np.asarray(out[:, 3:]), np.asarray(prev[:, 3:]))
