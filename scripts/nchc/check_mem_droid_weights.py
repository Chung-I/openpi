"""Pre-launch check: verify pi05_droid weights load into a MEM verify config with ONLY
the expected MEM-new keys (lora, state_proj) freshly initialized, and that video_img is
fully seeded from the remapped SigLIP (not random init).

Usage:
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
    scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify
"""

import re
import sys

import flax.nnx as nnx
import flax.traverse_util as tu
import jax

import openpi.shared.array_typing as at
from openpi.training import config as _config


def main(config_name: str) -> None:
    cfg = _config.get_config(config_name)

    def _init(rng):
        model = cfg.model.create(rng)
        return nnx.state(model, nnx.Param).to_pure_dict()

    params_shape = jax.eval_shape(_init, jax.random.key(0))
    ref_keys = set(tu.flatten_dict(params_shape, sep="/"))

    loaded = cfg.weight_loader.load(params_shape)
    loaded_keys = {
        k for k, v in tu.flatten_dict(loaded, sep="/").items() if not isinstance(v, jax.ShapeDtypeStruct)
    }

    missing = ref_keys - loaded_keys  # keys that fell through to fresh init
    regex = re.compile(cfg.weight_loader.missing_regex)
    bad = sorted(k for k in missing if not regex.fullmatch(k))

    video_img_keys = {k for k in ref_keys if "video_img" in k}
    video_img_seeded = video_img_keys & loaded_keys

    print(f"config: {config_name}")
    print(f"ref params: {len(ref_keys)}  loaded: {len(loaded_keys)}  fresh-init: {len(missing)}")
    print("fresh-init sample:", sorted(missing)[:12])
    print(f"video_img: {len(video_img_seeded)}/{len(video_img_keys)} seeded from checkpoint")

    assert not bad, f"Base weights left uninitialized (path mismatch): {bad[:20]}"
    assert video_img_seeded, "video_img NOT seeded from checkpoint — remap failed (would be random init)"
    assert video_img_seeded == video_img_keys, (
        f"Only {len(video_img_seeded)}/{len(video_img_keys)} video_img keys seeded"
    )
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=False)
    print("OK: only lora/state_proj fresh-init; video_img fully seeded via remap.")


if __name__ == "__main__":
    main(sys.argv[1])
