"""RTC serving path (policy.py): cache, alignment, and the re-anchor gate.

test_rtc.py pins the sampler math; this pins the layer above it -- the layer the
LIBERO run relies on: rtc/* key stripping, per-env_id prefix cache, and the
critical gate that re-anchoring (a DROID-jointpos-specific correction) stays OFF
for LIBERO-style requests that carry no observation/joint_position.

Dummy variants keep this CPU-cheap (see test_rtc.py's OOM lesson).
"""

import dataclasses

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models import pi0, pi0_config
from openpi.policies.policy import Policy


def _policy():
    cfg = pi0_config.Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=10
    )
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    return cfg, Policy(model, sample_kwargs={"num_steps": 2})


def _libero_style_inputs(cfg):
    """Model-ready inputs WITHOUT observation/joint_position (the LIBERO shape)."""
    obs = cfg.fake_obs(batch_size=1).to_dict()
    return jax.tree.map(lambda x: np.asarray(x)[0], obs)


def _rtc_keys(env_id, d=1, executed=5):
    return {"rtc/mode": "rtc", "rtc/env_id": env_id, "rtc/inference_delay": d, "rtc/executed": executed}


def test_first_request_plain_then_cached_rtc():
    cfg, policy = _policy()
    inputs = _libero_style_inputs(cfg)
    r1 = policy.infer({**inputs, **_rtc_keys(1)})
    assert r1["actions"].shape == (cfg.action_horizon, cfg.action_dim)
    # cache populated after the first (plain-path) request
    assert 1 in policy._rtc_prev
    # second request takes the RTC path (alignment + guided sampler) and must
    # return finite, correctly-shaped actions
    r2 = policy.infer({**inputs, **_rtc_keys(1)})
    assert r2["actions"].shape == (cfg.action_horizon, cfg.action_dim)
    assert np.all(np.isfinite(r2["actions"]))


def test_reanchor_gate_off_for_libero_requests():
    """No observation/joint_position -> raw_joints is None in the cache, so the
    quantile re-anchor (jointpos-specific) can never fire."""
    cfg, policy = _policy()
    inputs = _libero_style_inputs(cfg)
    policy.infer({**inputs, **_rtc_keys(7)})
    _, cached_joints = policy._rtc_prev[7]
    assert cached_joints is None
    # and the rtc path still runs without norm stats or joints
    r2 = policy.infer({**inputs, **_rtc_keys(7)})
    assert np.all(np.isfinite(r2["actions"]))


def test_env_id_cache_isolation():
    """A fresh env_id must NOT inherit another episode's prefix: its first
    request goes down the plain path (bit-identical to a no-rtc request under
    the same rng), even while another env's cache is hot."""
    cfg, policy = _policy()
    inputs = _libero_style_inputs(cfg)
    policy.infer({**inputs, **_rtc_keys(1)})  # warm env 1
    # clone the policy rng state by reading it, then issue the two requests
    # with identical rng by resetting between calls
    rng_before = policy._rng
    a = policy.infer({**inputs, **_rtc_keys(2)})["actions"]  # env 2: cache miss
    policy._rng = rng_before
    b = policy.infer(dict(inputs))["actions"]  # no rtc at all
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert set(policy._rtc_prev) == {1, 2}


def test_rtc_keys_stripped_before_transforms():
    """rtc/* keys must never leak into the input transforms/model dict."""
    cfg, policy = _policy()
    inputs = _libero_style_inputs(cfg)
    # a transform that would crash on any unexpected rtc key
    seen = {}

    def spy(d):
        seen.update({k: True for k in d if str(k).startswith("rtc/")})
        return d

    policy._input_transform = spy
    policy.infer({**inputs, **_rtc_keys(3)})
    assert not seen, f"rtc keys leaked into transforms: {list(seen)}"
