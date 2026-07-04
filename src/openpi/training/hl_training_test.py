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

    metrics, samples = hl_training.evaluate_hl(
        model, _DS(), _Tok(), batch_size=2, max_new_tokens=4, gen_examples=2, rng=jax.random.key(0), n_samples=2
    )
    for key in ["ce_loss", "subtask_exact_match", "memory_exact_match", "token_accuracy"]:
        assert key in metrics and np.isfinite(metrics[key])
    # per-update bucketed keys present and finite
    for key in [
        "subtask_exact_match_update", "memory_exact_match_update", "token_accuracy_update",
        "subtask_exact_match_noupdate", "memory_exact_match_noupdate", "token_accuracy_noupdate",
        "n_update", "n_noupdate",
    ]:
        assert key in metrics and np.isfinite(metrics[key])
    assert isinstance(samples, list) and len(samples) <= 2
    if samples:
        assert {"target", "generated", "subtask_match", "memory_match", "update"} <= set(samples[0])


def test_upsample_update_balances_pool():
    import types as _types

    rows = [{"update": (i % 5 == 0)} for i in range(100)]  # 20 update=True, 80 update=False

    class _DS:
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return i  # collate not exercised; we only inspect the sampled index distribution

    # Monkeypatch collate to return the raw index list so we can count update vs no-update draws.
    import openpi.training.robomind_hl as rh

    orig = rh.collate_hl
    rh.collate_hl = lambda batch: list(batch)
    try:
        ds = _DS()
        ds.rows = rows
        it = hl_training.make_hl_batch_iterator(
            ds, batch_size=10, rng=np.random.default_rng(0), upsample_update=True
        )
        drawn = []
        for _ in range(20):  # 20 batches * 10 = 200 draws over the balanced pool
            drawn.extend(next(it))
        frac_update = np.mean([hl_training._row_is_update(rows[i]) for i in drawn])
        assert 0.4 < frac_update < 0.6, f"upsampled update fraction {frac_update:.2f} not ~0.5"
    finally:
        rh.collate_hl = orig


def test_evaluate_hl_buckets_by_update():
    cfg = _dummy_config()
    model = cfg.model.create(jax.random.key(0))

    class _Tok:
        def encode(self, text):
            return [ord(c) % 50 + 3 for c in text][:16]

        def decode(self, ids):
            return "".join(chr((i - 3) % 50 + 65) for i in ids)

    def _ex():
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

    class _DS:
        rows = [{"update": (i % 2 == 0), "goal": "g"} for i in range(10)]  # 5 update / 5 no-update

        def __len__(self):
            return 10

        def __getitem__(self, i):
            return _ex()

    # batch_size=3 makes chunks straddle the update/no-update boundary (upd_idx then noupd_idx).
    metrics, samples = hl_training.evaluate_hl(
        model, _DS(), _Tok(), batch_size=3, max_new_tokens=4, gen_examples=4, rng=jax.random.key(0), n_samples=3
    )
    assert metrics["n_update"] == 4 and metrics["n_noupdate"] == 4  # gen_examples=4 of each bucket
    for m in ["subtask_exact_match", "memory_exact_match", "token_accuracy"]:
        assert np.isfinite(metrics[f"{m}_update"]) and np.isfinite(metrics[f"{m}_noupdate"])
    assert samples and all(s["update"] for s in samples)  # update bucket is evaluated first
