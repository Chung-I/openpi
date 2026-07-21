"""Pre-launch gate for the layer-truncation runs.

Builds the model's parameter shapes, runs the config's weight loader against them, and
verifies every scanned transformer parameter came back at the truncated depth. A
mismatch here would otherwise surface only after an 11 GiB checkpoint download and a
Slurm queue wait.

Usage:
    uv run python scripts/check_truncation.py pi05_droid_jointpos_trunc6
"""

import sys

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np

import openpi.models.gemma as _gemma
import openpi.training.config as _config

_SCANNED_LAYER_PREFIX = "PaliGemma/llm/layers/"


def scan_depths(params) -> dict[str, int]:
    """Leading-axis size of every scanned transformer parameter, keyed by path."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return {k: v.shape[0] for k, v in flat.items() if k.startswith(_SCANNED_LAYER_PREFIX)}


def assert_uniform_depth(depths: dict[str, int], expected: int) -> None:
    """Every scanned parameter must sit at exactly `expected` depth."""
    if not depths:
        raise ValueError(f"found no scanned params under '{_SCANNED_LAYER_PREFIX}'")
    wrong = {k: d for k, d in depths.items() if d != expected}
    if wrong:
        sample = list(wrong.items())[:5]
        raise ValueError(f"expected {expected} layers, but {len(wrong)} params disagree: {sample}")


def check(config_name: str) -> dict[str, int]:
    train_config = _config.get_config(config_name)
    model_config = train_config.model

    if model_config.keep_layers is None:
        expected = _gemma.get_config(model_config.paligemma_variant).depth
    else:
        expected = len(model_config.keep_layers)

    # Abstract build: shapes only, no device memory for the 3B model.
    abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    params_shape = nnx.state(abstract_model, nnx.Param).to_pure_dict()

    model_depths = scan_depths(params_shape)
    assert_uniform_depth(model_depths, expected)
    print(f"model built at depth {expected} ({len(model_depths)} scanned params) OK")

    loaded = train_config.weight_loader.load(params_shape)
    loaded_depths = scan_depths(loaded)
    assert_uniform_depth(loaded_depths, expected)
    print(f"weights loaded at depth {expected} ({len(loaded_depths)} scanned params) OK")

    # Every scanned param the model wants must have been loaded, at a matching shape.
    flat_model = flax.traverse_util.flatten_dict(params_shape, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    missing = sorted(set(flat_model) - set(flat_loaded))
    if missing:
        raise ValueError(f"{len(missing)} params were not loaded, e.g. {missing[:5]}")
    for k, want in flat_model.items():
        got = flat_loaded[k]
        if tuple(want.shape) != tuple(np.shape(got)):
            raise ValueError(f"shape mismatch for '{k}': model wants {want.shape}, loader gave {np.shape(got)}")
    print(f"all {len(flat_model)} params match the model's shapes OK")

    return loaded_depths


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_truncation.py <config_name>", file=sys.stderr)
        return 2
    try:
        check(sys.argv[1])
    except (ValueError, KeyError) as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        return 1
    print("CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
