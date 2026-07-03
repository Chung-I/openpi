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


def make_hl_batch_iterator(dataset, *, batch_size, rng: np.random.Generator, shuffle=True):
    """Yield collate_hl outputs (obs_dict, target_tokens, target_mask) forever."""
    from openpi.training.robomind_hl import collate_hl

    n = len(dataset)
    order = np.arange(n)
    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            yield collate_hl([dataset[int(i)] for i in idx])
