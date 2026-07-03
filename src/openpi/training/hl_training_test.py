import functools
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils
from openpi.training import hl_training, optimizer, sharding


def _dummy_config():
    model = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", lora=True)
    cfg = types.SimpleNamespace(
        model=model,
        optimizer=optimizer.AdamW(),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-2, decay_steps=200, decay_lr=1e-2),
        ema_decay=None,
        grad_accum_steps=1,
        seed=0,
    )
    # Train only the (tiny) dummy LLM; freeze the full-size SigLIP image encoder so the
    # optimizer state + activations stay small (and mirror the real config, where SigLIP
    # is frozen). The dummy variant has no LoRA params, so we key off the llm path here.
    cfg.freeze_filter = nnx.Not(nnx_utils.PathRegex(".*llm.*"))
    cfg.trainable_filter = nnx.All(nnx.Param, nnx.Not(cfg.freeze_filter))
    return cfg


def _fixed_batch(model_config, batch_size=2, target_len=8):
    obs = model_config.fake_obs(batch_size)
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)
    return obs, target_tokens, target_mask


def test_overfit_single_batch_drops_loss():
    cfg = _dummy_config()
    mesh = sharding.make_mesh(1)
    rng = jax.random.key(0)
    state, _ = hl_training.init_hl_train_state(cfg, rng, mesh, resume=False)

    # jit the step (frees intermediates -> low memory) with config captured statically.
    step = jax.jit(functools.partial(hl_training.hl_train_step, cfg))

    batch = _fixed_batch(cfg.model)
    losses = []
    with sharding.set_mesh(mesh):
        for _ in range(30):
            state, info = step(rng, state, batch)
            losses.append(float(info["loss"]))
    assert losses[-1] < 0.5 * losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_evaluate_hl_returns_metrics():
    cfg = _dummy_config()
    model = cfg.model.create(jax.random.key(0))

    class _Tok:
        def encode(self, text):
            return [ord(c) % 50 + 3 for c in text][:16]

        def decode(self, ids):
            return "".join(chr((i - 3) % 50 + 65) for i in ids)

    class _DS:
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {
                "image": np.zeros((*_model.IMAGE_RESOLUTION, 3), np.uint8),
                "image_mask": np.array(True),
                "state": np.zeros(32, np.float32),
                "tokenized_prompt": np.zeros(48, np.int32),
                "tokenized_prompt_mask": np.zeros(48, bool),
                "tokenized_memory": np.zeros(128, np.int32),
                "tokenized_memory_mask": np.zeros(128, bool),
                "target_tokens": np.ones(32, np.int32),
                "target_mask": np.concatenate([np.ones(5, bool), np.zeros(27, bool)]),
            }

    metrics = hl_training.evaluate_hl(
        model, _DS(), _Tok(), batch_size=2, max_new_tokens=4, gen_examples=2, rng=jax.random.key(0)
    )
    for key in ["ce_loss", "subtask_exact_match", "memory_exact_match", "token_accuracy"]:
        assert key in metrics and np.isfinite(metrics[key])
