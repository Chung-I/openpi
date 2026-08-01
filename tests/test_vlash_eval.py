"""Unit smoke test for the Task 7 per-offset held-out validation hook
(`openpi.training.vlash_eval.PerOffsetValLoss`).

This is deliberately network/RLDS-free: `PerOffsetValLoss.build()` streams DROID from GCS to
cache one held-out batch per delta, which is exactly the part that must NOT run in a fast local
unit test. Instead, this constructs `PerOffsetValLoss` directly with hand-built fake batches
(mirroring `tests/test_shared_obs.py`'s dummy-model fixture pattern) and exercises `.compute()`
once against a freshly initialized model with random weights, per the Task 7 brief.
"""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from openpi.models import pi0
from openpi.models import pi0_config
import openpi.training.config as _config
from openpi.training.vlash_eval import PerOffsetValLoss
from openpi.training.vlash_eval import _fixed_delta_config

BATCH = 2
HORIZON = 2
ACTION_DIM = 4
DELTAS = (0, 1, 2, 3)


@pytest.fixture(scope="module")
def model_cfg() -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        state_cond=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=ACTION_DIM,
        action_horizon=HORIZON,
        max_token_len=16,
    )


@pytest.fixture(scope="module")
def model(model_cfg) -> pi0.Pi0:
    return pi0.Pi0(model_cfg, rngs=nnx.Rngs(0))


@pytest.fixture(scope="module")
def val_hook(model_cfg) -> PerOffsetValLoss:
    obs = model_cfg.fake_obs(batch_size=BATCH)
    actions = model_cfg.fake_act(batch_size=BATCH)
    # Each delta gets its own (fake) held-out batch, mirroring how `.build()` caches a
    # distinct batch per fixed-delta data loader -- content doesn't matter here, only that
    # `.compute()` runs cleanly and deterministically over whatever is cached.
    batches = {
        delta: (obs.replace(state=obs.state + float(i)), actions + float(i) * 0.1) for i, delta in enumerate(DELTAS)
    }
    return PerOffsetValLoss(batches=batches)


def test_compute_returns_one_finite_loss_per_delta(val_hook, model):
    out = val_hook.compute(model)
    assert set(out.keys()) == {f"val_loss/delta_{d}" for d in DELTAS}
    for k, v in out.items():
        assert isinstance(v, float), k
        assert jnp.isfinite(v), f"{k} is not finite: {v}"


def test_compute_is_deterministic_across_calls(val_hook, model):
    """Fixed eval rng: calling compute() twice on the SAME weights must give identical
    losses (isolates model changes from noise/time resampling across evals)."""
    first = val_hook.compute(model)
    second = val_hook.compute(model)
    assert first == second


# ---------------------------------------------------------------------------------------------
# Regression: `_fixed_delta_config` must not trip the data/model `vlash_shared_obs` agreement
# check for a SHARED-obs training arm. Caught live by the Task 7 equivalence-gate job
# (pi05_droid_jointpos_vlash_shared crashed at startup with "vlash_shared_obs mismatch:
# data-side=False, model-side=True") before the fix below (also override the fixed-delta
# config's MODEL side, not just its data side).
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", ["pi05_droid_jointpos_vlash", "pi05_droid_jointpos_vlash_shared"])
def test_fixed_delta_config_data_model_agree(config_name):
    config = _config.get_config(config_name)
    for delta in range(config.data.vlash_delta_max + 1):
        fixed_config = _fixed_delta_config(config, delta=delta, eval_batch_size=8)
        assert fixed_config.model.vlash_shared_obs is False
        assert fixed_config.data.vlash_shared_obs is False
        # Must not raise: this is exactly where the mismatch ValueError fired.
        data_config = fixed_config.data.create(fixed_config.assets_dirs, fixed_config.model)
        assert data_config.rlds_action_horizon == config.model.action_horizon + config.data.vlash_delta_max
