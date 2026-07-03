import jax
import jax.numpy as jnp
import pytest

from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.policies.mem_policy import MEMPolicy
from openpi.training.robomind_hl import HL_TARGET_TEMPLATE


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


class _FakeModel:
    """Fully fake Pi0MEM substitute for routing tests — no real model, no XLA compilation.

    Exposes exactly the interface MEMPolicy uses: action_horizon, action_dim, and the
    three sampling methods.  Each method records a tag in self.calls so tests can assert
    which sampler MEMPolicy chose.
    """

    def __init__(self, action_horizon: int, action_dim: int):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.calls: list[str] = []

    def predict_subtask_and_memory(self, rng, observation, *, max_new_tokens):
        self.calls.append("hl")
        # Return a small dummy token-id array; MEMPolicy decodes it (fast, no compute).
        return jnp.ones((1, 4), dtype=jnp.int32)

    def sample_actions(self, rng, observation, *, num_steps):
        self.calls.append("plain")
        return jnp.zeros((1, self.action_horizon, self.action_dim), dtype=jnp.float32)

    def sample_actions_rtc(
        self,
        rng,
        observation,
        *,
        prev_action_chunk,
        inference_delay,
        prefix_attention_horizon,
        prefix_attention_schedule,
        max_guidance_weight,
        num_steps,
    ):
        self.calls.append("rtc")
        return jnp.zeros((1, self.action_horizon, self.action_dim), dtype=jnp.float32)


def test_mem_policy_rtc_routing():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    fake = _FakeModel(config.action_horizon, config.action_dim)
    # hl_interval high so HL only runs at step 0; focus on the LL sampler routing.
    policy = MEMPolicy(fake, config, hl_interval_steps=100, num_flow_steps=2, use_rtc=True)
    policy.reset()
    assert policy.prev_action_chunk is None

    obs = config.fake_obs(batch_size=1)
    a0 = policy.step(key, obs)          # first step: no prev chunk -> plain
    assert policy.prev_action_chunk is not None
    assert policy.prev_action_chunk.shape == (1, config.action_horizon, config.action_dim)
    a1 = policy.step(key, obs)          # second step: prev chunk present -> rtc
    assert "plain" in fake.calls
    assert "rtc" in fake.calls
    assert a1.shape == (1, config.action_horizon, config.action_dim)

    policy.reset()
    assert policy.prev_action_chunk is None


def test_mem_policy_no_rtc_never_calls_rtc():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    fake = _FakeModel(config.action_horizon, config.action_dim)
    policy = MEMPolicy(fake, config, hl_interval_steps=100, num_flow_steps=2)  # use_rtc defaults False
    policy.reset()
    obs = config.fake_obs(batch_size=1)
    policy.step(key, obs)
    policy.step(key, obs)
    assert "rtc" not in fake.calls
    assert "plain" in fake.calls


@pytest.mark.parametrize(
    ("subtask", "memory"),
    [
        ("move towards the lid of the trash bin", "(none yet)"),
        ("pick up the red block", "already opened the drawer; grasp failed once"),
        ("place bread on the table", ""),
        ("push the drawer closed.", "cabinet: closed. drawer: open"),
    ],
)
def test_nl_target_round_trip(subtask, memory):
    text = HL_TARGET_TEMPLATE.format(subtask=subtask, memory=memory)
    parsed_subtask, parsed_memory = MEMPolicy._parse_hl_output(text)
    assert parsed_subtask == subtask.strip()
    assert parsed_memory == memory.strip()


def test_target_format_is_natural_language():
    """Pin the serialization to NL (no XML tags) — the reason for this change."""
    text = HL_TARGET_TEMPLATE.format(subtask="pick up cup", memory="drawer open")
    assert "<subtask>" not in text and "<memory>" not in text
    assert text == "Subtask: pick up cup Memory: drawer open"


def test_parse_missing_fields_returns_empty():
    assert MEMPolicy._parse_hl_output("garbage with no markers") == ("", "")
