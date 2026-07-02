import dataclasses

import flax.nnx as nnx
import flax.traverse_util as _tu
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils


def test_from_dict_normalizes_uint8_video_images():
    """Real DROID video frames arrive uint8; from_dict must normalize them to [-1, 1]
    float32 like single-frame images (FakeData's float32 video spec hid this gap)."""
    b, k, h, w = 2, 3, 4, 4
    data = {
        "image": {"base_0_rgb": np.zeros((b, h, w, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((b,), bool)},
        "state": np.zeros((b, 8), np.float32),
        "video_image": {"base_0_rgb": np.full((b, k, h, w, 3), 255, np.uint8)},
        "video_image_mask": {"base_0_rgb": np.ones((b, k), bool)},
        "video_states": np.zeros((b, k, 8), np.float32),
    }
    obs = _model.Observation.from_dict(data)
    video = np.asarray(obs.video_images["base_0_rgb"])
    assert video.dtype == np.float32
    assert np.allclose(video, 1.0)  # 255 -> +1.0
    # single-frame images still normalized (regression guard)
    assert np.asarray(obs.images["base_0_rgb"]).dtype == np.float32


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


def test_pi0_mem_k1_forward_uses_video_path():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=1,
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)
    # K=1 obs must still carry a video tensor with a single frame.
    assert obs.video_images is not None
    assert np.asarray(obs.video_images["base_0_rgb"]).shape[1] == 1
    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))


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


def _fast_action_obs(config, batch_size):
    # fake_obs provides tokenized_action (ones); make the loss mask non-empty.
    obs = config.fake_obs(batch_size)
    return dataclasses.replace(
        obs,
        tokenized_action_loss_mask=jnp.ones_like(obs.tokenized_action_loss_mask),
    )


def test_pi0_mem_fast_loss_shape_finite():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = _fast_action_obs(config, 2)
    prefix = model.embed_prefix_ll(obs)
    loss = nnx_utils.module_jit(model.compute_loss_fast)(obs, *prefix)
    assert loss.shape == (2,)
    assert jnp.all(jnp.isfinite(loss))


def _grad_abs_by_path(model, scalar_loss_fn):
    graphdef, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda p: scalar_loss_fn(nnx.merge(graphdef, p)))(params)
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        key = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        out[key] = float(jnp.sum(jnp.abs(leaf)))
    return out


def _is_backbone(path: str) -> bool:
    # gemma expert-0 (backbone): under llm, NOT an action-expert (_1) leaf, NOT lora.
    return "llm" in path and "_1" not in path


def test_flow_loss_does_not_touch_backbone_or_video():
    key = jax.random.key(1)
    # flow-only: fast and hl weights zero.
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        fast_loss_weight=0.0,
        hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = config.fake_obs(1), config.fake_act(1)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    for path, g in norms.items():
        if _is_backbone(path) or "video_img" in path:
            assert g == 0.0, f"flow loss leaked into {path}: {g}"
    # action expert / projections must receive gradient
    assert any(g > 0 for p, g in norms.items() if "_1" in p or "proj" in p or "time_mlp" in p)


def test_fast_loss_trains_backbone_and_video():
    key = jax.random.key(2)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        ll_loss_weight=0.0,
        hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = _fast_action_obs(config, 1), config.fake_act(1)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    assert any(g > 0 for p, g in norms.items() if _is_backbone(p))
    assert any(g > 0 for p, g in norms.items() if "video_img" in p)


def test_get_prefix_weights_linear_matches_reference():
    import numpy as np

    from openpi.models.pi0_mem import get_prefix_weights

    w = np.asarray(get_prefix_weights(2, 6, 10, "linear"))
    np.testing.assert_allclose(w, [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], atol=1e-6)


def test_get_prefix_weights_schedules():
    import numpy as np

    from openpi.models.pi0_mem import get_prefix_weights

    # ones: all 1 except positions >= end
    np.testing.assert_allclose(np.asarray(get_prefix_weights(0, 4, 6, "ones")), [1, 1, 1, 1, 0, 0])
    # end == 0 -> entire prefix ignored (all zeros)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(3, 0, 5, "linear")), [0, 0, 0, 0, 0])
    # zeros: 1 below start, else 0 (and 0 at/after end)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(2, 5, 6, "zeros")), [1, 1, 0, 0, 0, 0])


def _rtc_model_and_obs(batch_size=1):
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    model = config.create(key)
    obs = config.fake_obs(batch_size)
    return key, config, model, obs


def test_sample_actions_rtc_shape():
    key, config, model, obs = _rtc_model_and_obs()
    prev = jnp.zeros((1, config.action_horizon, config.action_dim))
    out = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        prev_action_chunk=prev,
        inference_delay=1,
        prefix_attention_horizon=config.action_horizon,
        num_steps=4,
    )
    assert out.shape == (1, config.action_horizon, config.action_dim)


def test_sample_actions_rtc_guidance_off_matches_plain():
    # With prefix_attention_horizon=0 the weights are all zero -> no guidance ->
    # the tau-frame integration is numerically identical to plain sample_actions.
    key, config, model, obs = _rtc_model_and_obs()
    noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
    prev = jnp.ones((1, config.action_horizon, config.action_dim))  # irrelevant when weights==0
    plain = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4, noise=noise)
    rtc = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        prev_action_chunk=prev,
        inference_delay=0,
        prefix_attention_horizon=0,
        num_steps=4,
        noise=noise,
    )
    assert jnp.allclose(plain, rtc, atol=1e-2)


def test_sample_actions_rtc_pins_prefix():
    # Strong guidance over the whole horizon pulls the output toward prev_action_chunk
    # more than the unguided sample does.
    key, config, model, obs = _rtc_model_and_obs()
    noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
    prev = jnp.ones((1, config.action_horizon, config.action_dim)) * 0.5
    plain = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=8, noise=noise)
    rtc = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        prev_action_chunk=prev,
        inference_delay=0,
        prefix_attention_horizon=config.action_horizon,
        prefix_attention_schedule="ones",
        max_guidance_weight=10.0,
        num_steps=8,
        noise=noise,
    )
    plain_dist = jnp.mean(jnp.abs(plain - prev))
    rtc_dist = jnp.mean(jnp.abs(rtc - prev))
    assert rtc_dist < plain_dist


def test_get_prefix_weights_exp_monotone():
    import numpy as np

    from openpi.models.pi0_mem import get_prefix_weights

    w = np.asarray(get_prefix_weights(2, 6, 10, "exp"))
    assert w.shape == (10,)
    assert w[0] == 1.0  # below start -> full weight
    assert w[6] == 0.0  # at/after end -> zero
    assert np.all(np.diff(w) <= 1e-6)  # monotone non-increasing
    assert np.all((w >= 0) & (w <= 1))


def test_get_prefix_weights_invalid_schedule_raises():
    import pytest

    from openpi.models.pi0_mem import get_prefix_weights

    with pytest.raises(ValueError):
        get_prefix_weights(0, 4, 6, "bogus")


def _param_partition(num_video_frames):
    """Return (trainable_keys, frozen_keys) for a LoRA Pi0MEM, via eval_shape (no weights)."""
    cfg = Pi0MEMConfig(lora=True, num_video_frames=num_video_frames)
    freeze = cfg.get_freeze_filter()

    def f(rng):
        model = cfg.create(rng)
        trainable = nnx.state(model, nnx.All(nnx.Param, nnx.Not(freeze))).to_pure_dict()
        frozen = nnx.state(model, nnx.All(nnx.Param, freeze)).to_pure_dict()
        return trainable, frozen

    trainable, frozen = jax.eval_shape(f, jax.random.key(0))
    tks = set(_tu.flatten_dict(trainable, sep="/"))
    fks = set(_tu.flatten_dict(frozen, sep="/"))
    return tks, fks


@pytest.mark.parametrize("k", [1, 6])
def test_lora_freeze_partition(k):
    trainable, frozen = _param_partition(k)
    # video_img (VideoViTEncoder), state_proj, and lora adapters must be trainable.
    assert any("video_img" in key for key in trainable), "video_img must be trainable"
    assert any("state_proj" in key for key in trainable), "state_proj must be trainable"
    assert any("lora" in key for key in trainable), "lora adapters must be trainable"
    # Base LLM weights (non-lora) must be frozen; no lora param may be frozen.
    assert any("llm" in key for key in frozen), "base llm must be frozen"
    assert all("lora" not in key for key in frozen), "no lora param may be frozen"
    # Any trainable llm-path key must be a lora adapter (base attn/ffn stay frozen).
    assert all("lora" in key for key in trainable if "llm" in key)
