import flax.nnx as nnx
import jax

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
