"""RTC port (arXiv 2506.07339): soft-mask weights + guided sampler invariants.

Reference: real-time-chunking-kinetix/src/model.py. Dummy variants keep this
CPU-cheap (the full-size lesson: test_state_cond once OOM-killed a 30G box).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0, pi0_config
from openpi.policies.policy import _rtc_prefix_weights


def _tiny():
    return pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=10
    )


def test_prefix_weights_match_reference_docstring():
    # get_prefix_weights docstring: start=2, end=6, total=10 (linear schedule)
    w = _rtc_prefix_weights(2, 6, 10, "linear")
    np.testing.assert_allclose(w, [1, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0, 0, 0], atol=1e-9)
    # frozen region is exactly ones; beyond end exactly zeros; exp decays monotonically
    e = _rtc_prefix_weights(3, 8, 10, "exp")
    assert np.all(e[:3] == 1.0) and np.all(e[8:] == 0.0)
    assert np.all(np.diff(e[2:9]) <= 1e-12)
    # end takes precedence over start
    assert np.all(_rtc_prefix_weights(5, 0, 10, "exp") == 0.0)


def test_zero_weights_reduce_to_plain_sampler():
    """With all-zero weights the guidance term vanishes, so sample_actions_rtc must
    reproduce sample_actions bit-for-bit under the same rng (same noise draw, same
    integration path). This pins the time-convention port: any sign/constant error
    in the guidance math would still perturb the trajectory here if it leaked into
    the unguided path."""
    cfg = _tiny()
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    rng = jax.random.key(7)
    plain = model.sample_actions(rng, obs, num_steps=4)
    prev = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    rtc = model.sample_actions_rtc(
        rng, obs, prev, jnp.zeros(cfg.action_horizon), num_steps=4
    )
    np.testing.assert_allclose(np.asarray(rtc), np.asarray(plain), atol=1e-5)


def test_guidance_pulls_toward_prev_chunk():
    """With strong weights on the early positions, the guided sample must land closer
    to the previous chunk there than the unguided sample does."""
    cfg = _tiny()
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    rng = jax.random.key(3)
    plain = model.sample_actions(rng, obs, num_steps=4)
    prev = plain + 0.5  # a nearby but distinct target chunk
    w = _rtc_prefix_weights(4, 8, cfg.action_horizon, "exp")
    rtc = model.sample_actions_rtc(rng, obs, prev, jnp.asarray(w), num_steps=4)
    d_guided = float(jnp.abs(rtc[:, :4] - prev[:, :4]).mean())
    d_plain = float(jnp.abs(plain[:, :4] - prev[:, :4]).mean())
    assert np.isfinite(d_guided)
    assert d_guided < d_plain, f"guidance did not pull toward prev: {d_guided=} {d_plain=}"
