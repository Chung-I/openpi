"""Measures pi0.5 single-observation inference latency for a given train config.

Weights are random: timing is faithful, task success is not measured. This mirrors how
the reference latency profile was produced, and is what makes the benchmark runnable
without downloading an 11 GiB checkpoint.

Usage:
    uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc18
    uv run python scripts/bench_inference.py --config-name pi05_droid_jointpos_trunc6
"""

import dataclasses
import statistics
import time

import flax.nnx as nnx
import jax
import numpy as np
import tyro

import openpi.models.gemma as _gemma
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config


@dataclasses.dataclass
class Args:
    # Name of a train config, e.g. "pi05_droid_jointpos_trunc6".
    config_name: str
    # Flow-matching denoise steps. 10 is the deployed default (pi0.py sample_actions).
    num_steps: int = 10
    # Discarded calls that pay for JIT compilation.
    warmup: int = 3
    # Timed calls. The reference profile used a median of 8-12.
    repeats: int = 12
    batch_size: int = 1


@dataclasses.dataclass
class BenchResult:
    p50_ms: float
    p95_ms: float
    min_ms: float
    num_layers: int
    # SigLIP-only median latency: the "untouchable floor" component 4 of the spec calls out.
    # Layer truncation cuts the Gemma stack, not the image encoder, so this number should be
    # ~constant across configs while p50_ms drops with num_layers.
    siglip_p50_ms: float


def num_layers_of(train_config: _config.TrainConfig) -> int:
    """Depth the model will actually be built at."""
    model_config = train_config.model
    if getattr(model_config, "keep_layers", None) is not None:
        return len(model_config.keep_layers)
    return _gemma.get_config(model_config.paligemma_variant).depth


def _siglip_only_timings_ms(model, obs, args: Args) -> list[float]:
    """Times just the SigLIP image-encoder forward pass, one call per image slot.

    Mirrors `Pi0.embed_prefix` (`src/openpi/models/pi0.py`): it calls `self.PaliGemma.img(...,
    train=False)` once per entry in `obs.images`, and each call returns a tuple. This is the
    fixed floor -- SigLIP encode, entirely inside `PaliGemma.img` -- that layer truncation of
    the Gemma stack (`PaliGemma.llm`) cannot touch.

    All image-slot calls are compiled into a single `jax.jit`-ed program (like `module_jit`,
    but batching the per-image calls inside one trace rather than jitting each separately).
    That mirrors `sample_actions`, which is also one fused dispatch: a Python-level loop of N
    separately-jitted calls would multiply per-call dispatch overhead by N and read as slower
    than it is, potentially inverting the "siglip is a subset of end-to-end" relationship this
    is meant to establish.

    `train` is omitted rather than passed as `train=False`: it gates a plain Python `not
    train` inside `siglip.Module`, so passing it as a jitted argument turns it into a traced
    value and raises `TracerBoolConversionError`. Its default is already `False`, so omitting
    it is behaviorally identical to `train=False`.
    """
    img_module = model.PaliGemma.img
    graphdef, state = nnx.split(img_module)

    def _encode_all(state, images):
        module = nnx.merge(graphdef, state)
        return [module(image) for image in images]

    encode_all_jit = jax.jit(_encode_all)
    images = list(obs.images.values())

    for _ in range(args.warmup):
        jax.block_until_ready(encode_all_jit(state, images))

    times_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(encode_all_jit(state, images))
        times_ms.append((time.perf_counter() - start) * 1e3)
    return times_ms


def benchmark(args: Args) -> BenchResult:
    train_config = _config.get_config(args.config_name)
    model_config = train_config.model

    model = model_config.create(jax.random.key(0))
    obs = model_config.fake_obs(batch_size=args.batch_size)
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(0)

    for _ in range(args.warmup):
        jax.block_until_ready(sample_actions(rng, obs, num_steps=args.num_steps))

    times_ms = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        jax.block_until_ready(sample_actions(rng, obs, num_steps=args.num_steps))
        times_ms.append((time.perf_counter() - start) * 1e3)

    siglip_times_ms = _siglip_only_timings_ms(model, obs, args)

    return BenchResult(
        p50_ms=statistics.median(times_ms),
        p95_ms=float(np.percentile(times_ms, 95)),
        min_ms=min(times_ms),
        num_layers=num_layers_of(train_config),
        siglip_p50_ms=statistics.median(siglip_times_ms),
    )


def main(args: Args) -> None:
    result = benchmark(args)
    print(f"config      : {args.config_name}")
    print(f"layers      : {result.num_layers}")
    print(f"denoise     : {args.num_steps} steps")
    print(f"p50         : {result.p50_ms:.1f} ms")
    print(f"p95         : {result.p95_ms:.1f} ms")
    print(f"min         : {result.min_ms:.1f} ms")
    print(f"siglip p50  : {result.siglip_p50_ms:.1f} ms  (untouchable floor; not reduced by truncation)")


if __name__ == "__main__":
    main(tyro.cli(Args))
