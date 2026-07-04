"""HL-only training: init, batch iterator, and train step.

Mirrors scripts/train.py's nnx init + step pattern, but drives compute_loss_hl over
RobomindHLDataset instead of compute_loss over RLDS. Kept separate from
scripts/train.py (owned by the LL thread) by design.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_hl_train_state(config, init_rng, mesh, *, resume):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps).gradient_transformation()

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    if resume:
        return train_state_shape, state_sharding

    weight_loader = getattr(config, "weight_loader", None)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if weight_loader is None or isinstance(weight_loader, _weight_loaders.NoOpWeightLoader):
        train_state = jax.jit(init, out_shardings=state_sharding)(init_rng)
        return train_state, state_sharding

    import flax.traverse_util as traverse_util

    loaded = weight_loader.load(train_state_shape.params.to_pure_dict())
    at.check_pytree_equality(
        expected=train_state_shape.params.to_pure_dict(), got=loaded, check_shapes=True, check_dtypes=True
    )
    partial = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    train_state = jax.jit(
        init, donate_argnums=(1,), in_shardings=replicated, out_shardings=state_sharding
    )(init_rng, partial)
    return train_state, state_sharding


@at.typecheck
def hl_train_step(
    config,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, at.Int[at.Array, "b t"], at.Bool[at.Array, "b t"]],
):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, target_tokens, target_mask = batch

    def loss_fn(model, rng, obs, tgt, mask):
        return jnp.mean(model.compute_loss_hl(rng, obs, tgt, mask, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, target_tokens, target_mask
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
    return new_state, info


def _row_is_update(row) -> bool:
    """Whether a manifest row is a subtask/memory transition (update=True). Robust to bool/str."""
    return str(row.get("update", False)).lower() == "true"


def make_hl_batch_iterator(dataset, *, batch_size, rng: np.random.Generator, shuffle=True, upsample_update=False):
    """Yield collate_hl outputs (obs_dict, target_tokens, target_mask) forever.

    upsample_update: oversample the (minority) update=True examples so they appear about as
    often as update=False, without modifying the dataset — the index pool repeats update=True
    indices to match the update=False count.
    """
    from openpi.training.robomind_hl import collate_hl

    n = len(dataset)
    order = np.arange(n)
    if upsample_update:
        rows = getattr(dataset, "rows", None)
        upd = [i for i in range(n) if rows is not None and _row_is_update(rows[i])]
        noupd = [i for i in range(n) if not (rows is not None and _row_is_update(rows[i]))]
        if upd and noupd:
            reps = int(np.ceil(len(noupd) / len(upd)))
            order = np.array(noupd + (upd * reps)[: len(noupd)])  # ~balanced pool; dataset untouched

    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            yield collate_hl([dataset[int(i)] for i in idx])


def _decode_ids_to_text(tokenizer, ids) -> str:
    # Truncate at the FIRST EOS (batched greedy decode over-generates past per-example EOS);
    # skip PAD. Keeping post-EOS tokens would corrupt the parsed memory field.
    clean = []
    for x in np.asarray(ids).tolist():
        x = int(x)
        if x == 1:  # EOS -> stop
            break
        if x == 0:  # PAD -> skip
            continue
        clean.append(x)
    if not clean:
        return ""
    try:
        return tokenizer.decode(clean)
    except Exception:  # noqa: BLE001
        return ""


def evaluate_hl(model, dataset, tokenizer, *, batch_size, max_new_tokens, gen_examples, rng, n_samples=0):
    """Held-out eval: teacher-forced CE over the dataset + generation metrics on a subset.

    Returns ``(metrics, samples)`` where ``samples`` is up to ``n_samples`` dicts
    ``{goal, target, generated, subtask_match, memory_match}`` for logging/inspection.
    """
    from openpi.policies.mem_policy import MEMPolicy
    from openpi.training.robomind_hl import collate_hl

    model.eval()
    n = len(dataset)

    # --- CE loss over full batches (jitted forward keeps memory low). ---
    ce_fn = nnx_utils.module_jit(model.compute_loss_hl)
    ce_sum, ce_count = 0.0, 0
    for start in range(0, n - batch_size + 1, batch_size):
        obs_dict, tgt, mask = collate_hl([dataset[i] for i in range(start, start + batch_size)])
        # collate_hl yields numpy; the eager generation path needs jax arrays for jaxtyping.
        obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(obs_dict))
        loss = ce_fn(rng, obs, jnp.asarray(tgt), jnp.asarray(mask))
        ce_sum += float(jnp.sum(loss))
        ce_count += int(loss.shape[0])
    ce_loss = ce_sum / max(ce_count, 1)

    # --- Generation metrics, bucketed by the `update` flag ---
    # update=False targets copy the input memory (a shortcut the model can exploit), so we
    # report update=True and update=False separately over a BALANCED subset: up to
    # `gen_examples` of each bucket (so the ~4:1 imbalance can't hide the transition cases).
    rows = getattr(dataset, "rows", None)

    def _is_update(i):
        return rows is not None and _row_is_update(rows[i])

    if rows is not None:
        upd_idx = [i for i in range(n) if _is_update(i)][:gen_examples]
        noupd_idx = [i for i in range(n) if not _is_update(i)][:gen_examples]
    else:
        upd_idx, noupd_idx = [], list(range(min(gen_examples, n)))
    idxs = upd_idx + noupd_idx

    # per bucket: [subtask_hits, memory_hits, token_hits, token_total, count]
    acc = {"update": [0, 0, 0, 0, 0], "noupdate": [0, 0, 0, 0, 0]}
    samples = []
    for bstart in range(0, len(idxs), batch_size):
        chunk = idxs[bstart : bstart + batch_size]
        obs_dict, tgt, mask = collate_hl([dataset[i] for i in chunk])
        obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(obs_dict))
        gen = np.asarray(model.predict_subtask_and_memory_cached(rng, obs, max_new_tokens=max_new_tokens))
        for j, di in enumerate(chunk):
            gtxt = _decode_ids_to_text(tokenizer, gen[j])
            ttxt = _decode_ids_to_text(tokenizer, np.asarray(tgt[j]))
            gs, gm = MEMPolicy._parse_hl_output(gtxt)
            ts, tm = MEMPolicy._parse_hl_output(ttxt)
            length = min(int(gen[j].shape[0]), int(np.asarray(mask[j]).sum()))
            th = int(np.sum(np.asarray(gen[j])[:length] == np.asarray(tgt[j])[:length]))
            a = acc["update" if _is_update(di) else "noupdate"]
            a[0] += int(gs == ts)
            a[1] += int(gm == tm)
            a[2] += th
            a[3] += length
            a[4] += 1
            if len(samples) < n_samples:
                samples.append({
                    "goal": rows[di].get("goal", "") if rows is not None else "",
                    "target": ttxt,
                    "generated": gtxt,
                    "subtask_match": int(gs == ts),
                    "memory_match": int(gm == tm),
                    "update": _is_update(di),
                })

    def _bucket(name):
        a = acc[name]
        c, tt = max(a[4], 1), max(a[3], 1)
        return {
            f"subtask_exact_match_{name}": a[0] / c,
            f"memory_exact_match_{name}": a[1] / c,
            f"token_accuracy_{name}": a[2] / tt,
            f"n_{name}": a[4],
        }

    tot = max(acc["update"][4] + acc["noupdate"][4], 1)
    tt_tot = max(acc["update"][3] + acc["noupdate"][3], 1)
    metrics = {
        "ce_loss": ce_loss,
        # "overall" here is over the BALANCED subset (≈50/50 update), not the natural distribution.
        "subtask_exact_match": (acc["update"][0] + acc["noupdate"][0]) / tot,
        "memory_exact_match": (acc["update"][1] + acc["noupdate"][1]) / tot,
        "token_accuracy": (acc["update"][2] + acc["noupdate"][2]) / tt_tot,
        **_bucket("update"),
        **_bucket("noupdate"),
    }
    return metrics, samples
