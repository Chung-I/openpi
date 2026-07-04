"""Per-update bucketed HL metrics from a checkpoint (for the no-upsample = A baseline).

Prints subtask/memory exact-match + token-accuracy split into update=True / update=False
for the requested split(s).

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval_hl_metrics.py \
      pi0_mem_hl_fr3_base --splits dev_seen dev_unseen test --step 15000 --gen-examples 128
"""

import argparse
import json
import logging
import pathlib
import types

import etils.epath as epath
import flax.nnx as nnx
import jax

import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
from openpi.training import config_hl, hl_splits, hl_training
from openpi.training.robomind_hl import RobomindHLDataset, load_paligemma_sp


def _build_dataset(config, tokenizer, split):
    splits = hl_splits.load_splits(config.splits_path)

    def make(manifest, frames, ep_ids):
        return RobomindHLDataset(
            manifest, frames, tokenizer, episode_ids=ep_ids,
            max_prompt_tokens=config.max_prompt_tokens,
            max_memory_tokens=config.max_memory_tokens,
            max_target_tokens=config.max_target_tokens,
        )

    if split == "test":
        return make(config.manifest_test, config.frames_test, None)
    if split == "dev_seen":
        return make(config.manifest_train, config.frames_train, set(splits["dev_seen_episodes"]))
    if split == "dev_unseen":
        tasks = set(splits["dev_unseen_tasks"])
        rows = [json.loads(x) for x in pathlib.Path(config.manifest_train).read_text().splitlines() if x.strip()]
        eps = {r["episode_id"] for r in rows if hl_splits.canonical_task(r["episode_id"]) in tasks}
        return make(config.manifest_train, config.frames_train, eps)
    raise ValueError(split)


def main():
    logging.basicConfig(level=logging.WARNING, force=True)
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--splits", nargs="+", default=["dev_seen", "dev_unseen", "test"])
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--gen-examples", type=int, default=128)
    args = ap.parse_args()

    config = config_hl.get_config(args.config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    tokenizer = load_paligemma_sp()

    ckpt_mgr, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True
    )
    state, _ = hl_training.init_hl_train_state(config, jax.random.key(0), mesh, resume=True)
    stub_dl = types.SimpleNamespace(data_config=lambda: types.SimpleNamespace(norm_stats=None, asset_id=None))
    state = _checkpoints.restore_state(ckpt_mgr, state, stub_dl, step=args.step)
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    print(f"\n==== {args.config} @ step {args.step or 'latest'}  (upsample={config.upsample_update}) ====")
    for split in args.splits:
        ds = _build_dataset(config, tokenizer, split)
        metrics, _ = hl_training.evaluate_hl(
            model, ds, tokenizer, batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens, gen_examples=args.gen_examples, rng=jax.random.key(0),
        )
        print(f"\n[{split}]  n_update={metrics['n_update']} n_noupdate={metrics['n_noupdate']}")
        for m in ["subtask_exact_match", "memory_exact_match", "token_accuracy"]:
            print(f"  {m:22s} update={metrics[m + '_update']:.3f}   noupdate={metrics[m + '_noupdate']:.3f}")


if __name__ == "__main__":
    main()
