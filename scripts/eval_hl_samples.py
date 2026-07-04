"""Print generated-vs-target HL text for a few dev/test examples from a checkpoint.

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/eval_hl_samples.py \
      pi0_mem_hl_fr3_base --split dev_unseen --n 10 --step 15000
"""

import argparse
import json
import logging
import pathlib
import types

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
from openpi.policies.mem_policy import MEMPolicy
from openpi.training import config_hl, hl_splits, hl_training
from openpi.training.robomind_hl import RobomindHLDataset, collate_hl, load_paligemma_sp


def _build_dataset(config, tokenizer, split, question_prompt=False):
    splits = hl_splits.load_splits(config.splits_path)

    def make(manifest, frames, ep_ids):
        return RobomindHLDataset(
            manifest, frames, tokenizer, episode_ids=ep_ids,
            max_prompt_tokens=config.max_prompt_tokens,
            max_memory_tokens=config.max_memory_tokens,
            max_target_tokens=config.max_target_tokens,
            question_prompt=question_prompt or config.question_prompt,
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
    raise ValueError(f"unknown split: {split}")


def main():
    logging.basicConfig(level=logging.WARNING, force=True)
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="dev_unseen", choices=["dev_seen", "dev_unseen", "test"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--zero-shot", action="store_true", help="use the pretrained weights directly (no HL training / no checkpoint)")
    ap.add_argument("--question-prompt", action="store_true", help="phrase the goal as a pi0.5-style question")
    args = ap.parse_args()

    config = config_hl.get_config(args.config)
    mesh = sharding.make_mesh(config.fsdp_devices)
    tokenizer = load_paligemma_sp()
    ds = _build_dataset(config, tokenizer, args.split, question_prompt=args.question_prompt)

    if args.zero_shot:
        # Pretrained checkpoint (pi05) with fresh zero-init LoRA -> the untrained VLM.
        state, _ = hl_training.init_hl_train_state(config, jax.random.key(0), mesh, resume=False)
    else:
        ckpt_mgr, _ = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True
        )
        state, _ = hl_training.init_hl_train_state(config, jax.random.key(0), mesh, resume=True)
        stub_dl = types.SimpleNamespace(data_config=lambda: types.SimpleNamespace(norm_stats=None, asset_id=None))
        state = _checkpoints.restore_state(ckpt_mgr, state, stub_dl, step=args.step)
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    n = min(args.n, len(ds))
    obs_dict, tgt, _ = collate_hl([ds[i] for i in range(n)])
    obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(obs_dict))
    gen = np.asarray(model.predict_subtask_and_memory_cached(jax.random.key(0), obs, max_new_tokens=config.max_new_tokens))

    print(f"\n==== {args.config} / split={args.split} / n={n} / step={args.step or 'latest'} ====\n")
    for i in range(n):
        gtxt = hl_training._decode_ids_to_text(tokenizer, gen[i])
        ttxt = hl_training._decode_ids_to_text(tokenizer, np.asarray(tgt[i]))
        gs, gm = MEMPolicy._parse_hl_output(gtxt)
        ts, tm = MEMPolicy._parse_hl_output(ttxt)
        print(f"--- example {i}  (goal: {ds.rows[i].get('goal', '')!r}) ---")
        print(f"  TARGET   : {ttxt!r}")
        print(f"  GENERATED: {gtxt!r}")
        print(f"  target  (subtask|memory): {ts!r} | {tm!r}")
        print(f"  gen     (subtask|memory): {gs!r} | {gm!r}\n")


if __name__ == "__main__":
    main()
