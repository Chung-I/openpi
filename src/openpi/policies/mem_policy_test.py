import jax
import jax.numpy as jnp

from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.policies.mem_policy import MEMPolicy


def test_mem_policy_step():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    policy = MEMPolicy(model, config, hl_interval_steps=5)
    policy.reset()

    obs = config.fake_obs(batch_size=1)
    actions = policy.step(key, obs)
    assert actions.shape == (1, config.action_horizon, config.action_dim)


def test_mem_policy_reset_clears_memory():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    policy = MEMPolicy(model, config, hl_interval_steps=2)
    policy.reset()

    assert policy.memory == ""
    assert policy.subtask == ""
    assert policy.step_count == 0


def test_mem_policy_hl_runs_at_correct_interval():
    """HL should run at step 0, 5, 10, ... but not in between."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    policy = MEMPolicy(model, config, hl_interval_steps=5)
    policy.reset()

    obs = config.fake_obs(batch_size=1)

    # Step 0: HL runs (step_count=0 → 0 % 5 == 0)
    policy.step(key, obs)
    assert policy.step_count == 1

    # Steps 1-4: HL does NOT run
    for _ in range(4):
        policy.step(key, obs)
    assert policy.step_count == 5

    # Step 5: HL runs again (5 % 5 == 0)
    policy.step(key, obs)
    assert policy.step_count == 6


def test_mem_policy_step_increments_counter():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    # Use large hl_interval_steps so HL only runs on step 0
    policy = MEMPolicy(model, config, hl_interval_steps=100)
    policy.reset()

    obs = config.fake_obs(batch_size=1)
    policy.step(key, obs)
    assert policy.step_count == 1

    policy.step(key, obs)
    assert policy.step_count == 2

    policy.reset()
    assert policy.step_count == 0
