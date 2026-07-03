"""HL-only training entrypoint for Pi0MEM.

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train_hl.py pi0_mem_hl_fr3_base
    ... scripts/train_hl.py pi0_mem_hl_fr3_base --overfit-batch   # debug gate
"""

import dataclasses
import functools
import json
import logging
import pathlib
import sys
import types

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
from openpi.training import config_hl, hl_splits, hl_training
from openpi.training.robomind_hl import RobomindHLDataset, load_paligemma_sp


def _datasets(config, tokenizer):
    splits = hl_splits.load_splits(config.splits_path)
    dev_unseen_tasks = set(splits["dev_unseen_tasks"])
    train_eps = set(splits["train_episodes"])
    dev_seen_eps = set(splits["dev_seen_episodes"])

    def make(manifest, frames, ep_ids):
        return RobomindHLDataset(
            manifest,
            frames,
            tokenizer,
            episode_ids=ep_ids,
            max_prompt_tokens=config.max_prompt_tokens,
            max_memory_tokens=config.max_memory_tokens,
            max_target_tokens=config.max_target_tokens,
        )

    train = make(config.manifest_train, config.frames_train, train_eps)
    dev_seen = make(config.manifest_train, config.frames_train, dev_seen_eps)
    # dev_unseen: all episodes whose canonical task is held out.
    all_rows = [json.loads(x) for x in pathlib.Path(config.manifest_train).read_text().splitlines() if x.strip()]
    dev_unseen_eps = {
        r["episode_id"] for r in all_rows if hl_splits.canonical_task(r["episode_id"]) in dev_unseen_tasks
    }
    dev_unseen = make(config.manifest_train, config.frames_train, dev_unseen_eps)
    test = make(config.manifest_test, config.frames_test, None)
    return train, {"dev_seen": dev_seen, "dev_unseen": dev_unseen}, test


def _to_obs_batch(collated):
    obs_dict, tgt, mask = collated
    obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(obs_dict))
    return obs, jnp.asarray(tgt), jnp.asarray(mask)


def main(config: config_hl.HLTrainConfig, *, overfit_batch: bool = False):
    logging.basicConfig(level=logging.INFO, force=True)  # force: openpi imports pre-config the root logger
    # Use a writable, persistent cache dir (compute nodes may have a non-writable /tmp).
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    mesh = sharding.make_mesh(config.fsdp_devices)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    tokenizer = load_paligemma_sp()
    train_ds, dev, test_ds = _datasets(config, tokenizer)
    it = hl_training.make_hl_batch_iterator(
        train_ds, batch_size=config.batch_size, rng=np.random.default_rng(config.seed)
    )

    ckpt_mgr, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    wandb.init(
        mode="online" if config.wandb_enabled else "disabled",
        name=config.exp_name,
        project=config.project_name,
        config=dataclasses.asdict(config),
    )
    # Stub data_loader for checkpoints.save_state: no HL norm stats to save.
    stub_dl = types.SimpleNamespace(data_config=lambda: types.SimpleNamespace(norm_stats=None, asset_id=None))

    state, _ = hl_training.init_hl_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = _checkpoints.restore_state(ckpt_mgr, state, stub_dl)

    ptrain = jax.jit(functools.partial(hl_training.hl_train_step, config), donate_argnums=(1,))
    fixed = _to_obs_batch(next(it)) if overfit_batch else None

    def _eval_and_log(datasets: dict, step: int):
        model = nnx.merge(state.model_def, state.params)
        for name, ds in datasets.items():
            metrics = hl_training.evaluate_hl(
                model,
                ds,
                tokenizer,
                batch_size=config.batch_size,
                max_new_tokens=config.max_new_tokens,
                gen_examples=config.eval_gen_examples,
                rng=train_rng,
            )
            wandb.log({f"{name}/{k}": v for k, v in metrics.items()}, step=step)
            logging.info("step %d %s: %s", step, name, metrics)

    for step in tqdm.tqdm(range(int(state.step), config.num_train_steps)):
        batch = fixed if overfit_batch else _to_obs_batch(next(it))
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        if step % config.log_interval == 0:
            reduced = {k: float(v) for k, v in info.items()}
            wandb.log({f"train/{k}": v for k, v in reduced.items()}, step=step)
            logging.info("step %d train: %s", step, reduced)
        if not overfit_batch and step > 0 and step % config.eval_interval == 0:
            _eval_and_log(dev, step)
        if not overfit_batch and step > 0 and step % config.save_interval == 0:
            _checkpoints.save_state(ckpt_mgr, state, stub_dl, step)

    if not overfit_batch:
        _eval_and_log({"test": test_ds}, config.num_train_steps)
    ckpt_mgr.wait_until_finished()


if __name__ == "__main__":
    overfit = "--overfit-batch" in sys.argv
    if overfit:
        sys.argv.remove("--overfit-batch")
    main(config_hl.cli(), overfit_batch=overfit)
