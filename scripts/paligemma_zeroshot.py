"""Zero-shot PaliGemma diagnostic on RoboMIND frames, using the CORRECT PaliGemma format.

Loads pi05_base's PaliGemma (no HL training; LoRA is zero-effect at init) and generates
with the paper/HF format:  [image tokens] + BOS + prompt + "\\n"  ->  answer + EOS,
predicting the first answer token from the LAST prefix position (no BOS injected to start
the answer). Contrast with the current HL decode, which injects a BOS -> junk.

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/paligemma_zeroshot.py \
      --split test --n 4 --prompt "caption en"
"""

import argparse
import logging

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0_mem import make_attn_mask
import openpi.training.sharding as sharding
from openpi.training import config_hl, hl_training
from openpi.training.robomind_hl import RobomindHLDataset, collate_hl, load_paligemma_sp

_BOS, _EOS, _PAD = 2, 1, 0


def _paligemma_generate(model, image, tokenizer, prompt, *, max_new_tokens, sep_id):
    """Correct PaliGemma-format greedy generate for one batch of images. image: [b,H,W,3] in [-1,1]."""
    img_tokens, _ = model.PaliGemma.img(image, train=False)  # [b, n_img, d]
    b = img_tokens.shape[0]
    prompt_ids = tokenizer.encode(prompt)
    text_ids = jnp.asarray([[_BOS, *prompt_ids, sep_id]] * b, dtype=jnp.int32)  # bos + prompt + \n
    text_emb = model.PaliGemma.llm(text_ids, method="embed")
    prefix = jnp.concatenate([img_tokens, text_emb], axis=1)  # [b, plen, d]
    plen = prefix.shape[1]

    prefix_mask = jnp.ones((b, plen), dtype=jnp.bool_)
    prefix_ar = jnp.zeros(plen, dtype=jnp.bool_)  # prefix-LM: full/bidirectional attention
    attn = make_attn_mask(prefix_mask, prefix_ar)
    positions = jnp.broadcast_to(jnp.arange(plen), (b, plen))
    (out, _), kv = model.PaliGemma.llm([prefix, None], mask=attn, positions=positions)

    # First answer token comes from the LAST prefix position (the \n) — no BOS injected.
    logits = model.PaliGemma.llm(out[:, -1:, :], method="decode_logits")
    cur = jnp.argmax(logits[:, 0, :], axis=-1, keepdims=True).astype(jnp.int32)
    generated = cur
    for i in range(max_new_tokens - 1):
        emb = model.PaliGemma.llm(cur, method="embed")
        pos = jnp.full((b, 1), plen + i)
        step_mask = jnp.concatenate([prefix_mask, jnp.ones((b, i + 1), jnp.bool_)], axis=1)[:, None, :]
        (o, _), kv = model.PaliGemma.llm([emb, None], positions=pos, mask=step_mask, kv_cache=kv)
        logits = model.PaliGemma.llm(o, method="decode_logits")
        cur = jnp.argmax(logits[:, 0, :], axis=-1, keepdims=True).astype(jnp.int32)
        generated = jnp.concatenate([generated, cur], axis=1)
        if jnp.all(cur == _EOS):
            break
    return np.asarray(generated)


def main():
    logging.basicConfig(level=logging.WARNING, force=True)
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev_unseen"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--prompt", default="caption en")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    config = config_hl.get_config("pi0_mem_hl_fr3_base")
    mesh = sharding.make_mesh(config.fsdp_devices)
    tokenizer = load_paligemma_sp()
    sep_id = tokenizer.encode("\n")[-1]  # newline SEP token id

    manifest = config.manifest_test if args.split == "test" else config.manifest_train
    frames = config.frames_test if args.split == "test" else config.frames_train
    ds = RobomindHLDataset(manifest, frames, tokenizer)

    # Load pi05_base weights (no HL training); LoRA zero-init -> pristine base VLM.
    state, _ = hl_training.init_hl_train_state(config, jax.random.key(0), mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    n = min(args.n, len(ds))
    obs_dict, _, _ = collate_hl([ds[i] for i in range(n)])
    obs = jax.tree.map(jnp.asarray, _model.Observation.from_dict(obs_dict))
    obs = _model.preprocess_observation(None, obs, train=False, image_keys=("base_0_rgb",))
    image = obs.images["base_0_rgb"]

    gen = _paligemma_generate(model, image, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, sep_id=sep_id)

    print(f"\n==== pi05_base PaliGemma zero-shot / prompt={args.prompt!r} / split={args.split} ====\n")
    for i in range(n):
        ids = [int(x) for x in gen[i].tolist() if int(x) not in (_PAD, _EOS)]
        text = tokenizer.decode(ids) if ids else ""
        print(f"--- frame {i}  (goal: {ds.rows[i].get('goal', '')!r}) ---")
        print(f"  GENERATED: {text!r}\n")


if __name__ == "__main__":
    main()
