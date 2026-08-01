"""Shared-observation training (VLASH Task 4), KV-broadcast design.

The shared-obs path computes the (images + language) prefix ONCE per batch element, then
broadcasts its KV cache across the `(delta_max + 1)`-replicated suffix batch, where each
branch carries its own rolled state (per-branch adarms_cond) and offset action chunk. The
tests below pin down the three correctness properties from the task brief:

1. branch isolation -- perturbing branch 1's state/actions must not change branch 0's loss;
2. equivalence -- the shared-obs loss equals per-branch plain `compute_loss` with identical
   weights/noise/time (THE key correctness test for the KV-broadcast design);
3. prefix KV is computed once (embed_prefix called once, at batch size B not B*K) and the
   broadcast cache rows are identical across branch replicas.
"""

import unittest.mock as mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.transforms_vlash import SplitVlashBranches
from openpi.transforms_vlash import VlashAllOffsets
from openpi.transforms_vlash import apply_offset

# On Ampere+ GPUs JAX lowers float32 matmuls to TF32 (~1e-3 relative error), which swamps the
# tight tolerances of the equivalence test (one-pass vs two-pass contraction orders differ).
# Force true fp32 matmuls for this module so fp reordering noise stays ~1e-6.
jax.config.update("jax_default_matmul_precision", "highest")

BATCH = 2
DELTA_MAX = 1
NUM_BRANCHES = DELTA_MAX + 1
HORIZON = 2
ACTION_DIM = 4


@pytest.fixture(scope="module")
def model_cfg() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        state_cond=True,
        vlash_shared_obs=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        max_token_len=16,
        # float32 so the equivalence test can use tight tolerances.
        dtype="float32",
    )


@pytest.fixture(scope="module")
def model(model_cfg) -> pi0.Pi0:
    m = pi0.Pi0(model_cfg, rngs=nnx.Rngs(0))
    # gemma's adaRMS modulation Dense layers are zero-initialized (adaLN-zero), so at fresh
    # init the cond pathway is a numerical no-op and no state perturbation could ever move the
    # loss. Perturb every "Dense_0" param (the adaRMS modulation layers inside RMSNorm) so the
    # per-branch state conditioning is actually exercised by the loss-level tests below.
    graphdef, state = nnx.split(m)
    pure = state.to_pure_dict()
    counter = iter(range(1_000_000))

    def perturb(path, x):
        if any(getattr(p, "key", None) == "Dense_0" for p in path):
            key = jax.random.fold_in(jax.random.key(123), next(counter))
            return x + 0.05 * jax.random.normal(key, x.shape, x.dtype)
        return x

    state.replace_by_pure_dict(jax.tree_util.tree_map_with_path(perturb, pure))
    return nnx.merge(graphdef, state)


@pytest.fixture(scope="module")
def batch(model_cfg):
    """(observation with vlash_states, actions, noise, time) -- noise/time injected so the
    equivalence test can run the plain path with bit-identical flow-matching inputs."""
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    vlash_states = jax.random.normal(k1, (BATCH, NUM_BRANCHES, ACTION_DIM))
    obs = model_cfg.fake_obs(batch_size=BATCH)
    obs = obs.replace(state=vlash_states[:, 0], vlash_states=vlash_states)
    actions = jax.random.normal(k2, (BATCH, NUM_BRANCHES, HORIZON, ACTION_DIM))
    noise = jax.random.normal(k3, actions.shape)
    time = jnp.array([[0.3, 0.7], [0.55, 0.9]])
    return obs, actions, noise, time


# ---------------------------------------------------------------------------------------------
# Transform side: all-offsets emission + post-normalize branch split.
# ---------------------------------------------------------------------------------------------


def _sample(H=4, dmax=2, adim=8):  # noqa: N803  (fixture shape mirrors tests/test_vlash_offsets.py)
    actions = np.zeros((H + dmax, adim), dtype=np.float32)
    actions[:, 0] = np.arange(H + dmax) + 1  # deltas 1,2,3,...
    actions[:, 7] = 0.1 * (np.arange(H + dmax) + 1)  # absolute gripper cmds
    state = np.zeros(adim, dtype=np.float32)
    state[0] = 10.0
    state[7] = 0.5
    return {"state": state, "actions": actions}


def test_all_offsets_emits_every_branch():
    s = _sample()
    out = VlashAllOffsets(delta_max=2, action_horizon=4)(dict(s))
    assert out["state"].shape == (3, 8)
    assert out["actions"].shape == (3, 4, 8)
    for delta in range(3):
        expected = apply_offset(dict(s), delta=delta, action_horizon=4)
        np.testing.assert_array_equal(out["state"][delta], expected["state"])
        np.testing.assert_array_equal(out["actions"][delta], expected["actions"])
    # Branch 0 is the exact identity.
    np.testing.assert_array_equal(out["state"][0], s["state"])
    np.testing.assert_array_equal(out["actions"][0], s["actions"][:4])


def test_all_offsets_does_not_mutate_input_arrays():
    s = _sample()
    state_before, actions_before = s["state"].copy(), s["actions"].copy()
    VlashAllOffsets(delta_max=2, action_horizon=4)({"state": s["state"], "actions": s["actions"]})
    np.testing.assert_array_equal(s["state"], state_before)
    np.testing.assert_array_equal(s["actions"], actions_before)


def test_split_vlash_branches():
    stacked = np.arange(3 * 32, dtype=np.float32).reshape(3, 32)
    out = SplitVlashBranches()({"state": stacked.copy(), "actions": np.zeros((3, 4, 32), np.float32)})
    np.testing.assert_array_equal(out["vlash_states"], stacked)
    np.testing.assert_array_equal(out["state"], stacked[0])
    assert out["state"].shape == (32,)
    # Actions stay stacked (all branches go to the model).
    assert out["actions"].shape == (3, 4, 32)


# ---------------------------------------------------------------------------------------------
# Config flag validation.
# ---------------------------------------------------------------------------------------------


def test_vlash_shared_obs_defaults_false():
    cfg = pi0_config.Pi0Config(pi05=True)
    assert cfg.vlash_shared_obs is False


def test_vlash_shared_obs_requires_pi05():
    with pytest.raises(ValueError, match="vlash_shared_obs"):
        pi0_config.Pi0Config(pi05=False, vlash_shared_obs=True)


def test_vlash_shared_obs_requires_state_cond():
    with pytest.raises(ValueError, match="vlash_shared_obs"):
        pi0_config.Pi0Config(pi05=True, state_cond=False, vlash_shared_obs=True)


# ---------------------------------------------------------------------------------------------
# Model side: branch isolation, equivalence, KV broadcast.
# ---------------------------------------------------------------------------------------------


def test_shared_obs_loss_shape(model, batch):
    obs, actions, noise, time = batch
    loss = model.compute_loss_shared_obs(jax.random.key(0), obs, actions, noise=noise, time=time)
    assert loss.shape == (BATCH, NUM_BRANCHES, HORIZON)
    assert bool(jnp.all(jnp.isfinite(loss)))


def test_branch_isolation(model, batch):
    """Changing branch 1's state AND actions must not change branch 0's loss."""
    obs, actions, noise, time = batch
    rng = jax.random.key(0)
    loss_a = model.compute_loss_shared_obs(rng, obs, actions, noise=noise, time=time)

    perturbed_states = obs.vlash_states.at[:, 1].add(1.0)
    obs_b = obs.replace(vlash_states=perturbed_states)
    actions_b = actions.at[:, 1].add(0.5)
    loss_b = model.compute_loss_shared_obs(rng, obs_b, actions_b, noise=noise, time=time)

    np.testing.assert_allclose(np.asarray(loss_a[:, 0]), np.asarray(loss_b[:, 0]), rtol=0, atol=1e-6)
    assert not np.allclose(np.asarray(loss_a[:, 1]), np.asarray(loss_b[:, 1]), atol=1e-6)


def test_branch_state_reaches_adarms_cond(model, batch):
    """Perturbing ONLY branch 1's state (same actions) must change branch 1's loss: the rolled
    state must actually reach the per-branch adarms_cond."""
    obs, actions, noise, time = batch
    rng = jax.random.key(0)
    loss_a = model.compute_loss_shared_obs(rng, obs, actions, noise=noise, time=time)
    obs_b = obs.replace(vlash_states=obs.vlash_states.at[:, 1].add(1.0))
    loss_b = model.compute_loss_shared_obs(rng, obs_b, actions, noise=noise, time=time)
    assert not np.allclose(np.asarray(loss_a[:, 1]), np.asarray(loss_b[:, 1]), atol=1e-6)


def test_equivalence_with_per_branch_plain_forward(model, batch):
    """THE key correctness test: shared-obs loss for a batch must equal per-branch losses
    computed the plain (unshared) way with identical weights, noise, and time."""
    obs, actions, noise, time = batch
    rng = jax.random.key(0)
    shared = model.compute_loss_shared_obs(rng, obs, actions, noise=noise, time=time)
    for i in range(NUM_BRANCHES):
        obs_i = obs.replace(state=obs.vlash_states[:, i], vlash_states=None)
        plain_i = model._compute_loss_single(  # noqa: SLF001
            rng, obs_i, actions[:, i], noise=noise[:, i], time=time[:, i]
        )
        np.testing.assert_allclose(
            np.asarray(shared[:, i]),
            np.asarray(plain_i),
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"shared-obs branch {i} disagrees with the plain forward",
        )


def test_compute_loss_dispatches_to_shared_obs(model, batch):
    """With vlash_shared_obs=True, the stock compute_loss entry point (used by train.py) must
    route to the shared-obs path."""
    obs, actions, noise, time = batch
    rng = jax.random.key(7)
    via_dispatch = model.compute_loss(rng, obs, actions, noise=noise, time=time)
    direct = model.compute_loss_shared_obs(rng, obs, actions, noise=noise, time=time)
    np.testing.assert_allclose(np.asarray(via_dispatch), np.asarray(direct), rtol=0, atol=1e-6)


def test_shared_obs_requires_vlash_states(model, batch):
    obs, actions, noise, time = batch
    obs_no_branches = obs.replace(vlash_states=None)
    with pytest.raises(ValueError, match="vlash_states"):
        model.compute_loss_shared_obs(jax.random.key(0), obs_no_branches, actions, noise=noise, time=time)


def test_broadcast_kv_cache_rows_identical_across_branches():
    """Each batch element's prefix KV must appear identically in every branch replica, laid out
    b-major to match the `(b k)` suffix flattening."""
    k = jax.random.normal(jax.random.key(1), (3, 2, 5, 1, 4))  # [layers, b, t, kv_heads, head_dim]
    v = jax.random.normal(jax.random.key(2), (3, 2, 5, 1, 4))
    bk, bv = pi0._broadcast_kv_cache((k, v), NUM_BRANCHES)  # noqa: SLF001
    assert bk.shape == (3, 2 * NUM_BRANCHES, 5, 1, 4)
    for i in range(2):
        for j in range(NUM_BRANCHES):
            np.testing.assert_array_equal(np.asarray(bk[:, i * NUM_BRANCHES + j]), np.asarray(k[:, i]))
            np.testing.assert_array_equal(np.asarray(bv[:, i * NUM_BRANCHES + j]), np.asarray(v[:, i]))


def test_prefix_computed_once_at_batch_size_b(model, batch):
    """The prefix (images + language) must be embedded exactly once, at batch size B -- not
    B * num_branches. This is the compute-saving claim of the shared-obs design."""
    obs, actions, noise, time = batch
    calls = []
    orig = pi0.Pi0.embed_prefix

    def spy(self, observation):
        out = orig(self, observation)
        calls.append(out[0].shape)
        return out

    with mock.patch.object(pi0.Pi0, "embed_prefix", spy):
        model.compute_loss_shared_obs(jax.random.key(0), obs, actions, noise=noise, time=time)

    assert len(calls) == 1, f"embed_prefix called {len(calls)} times, expected exactly 1"
    assert calls[0][0] == BATCH, f"prefix batch dim is {calls[0][0]}, expected {BATCH} (shared, not replicated)"
