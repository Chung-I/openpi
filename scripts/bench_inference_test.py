import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import bench_inference


def test_benchmark_runs_on_a_dummy_model():
    """Uses the depth-4 'dummy' variant so this runs on CPU in seconds.

    `num_steps=200` (rather than the module default of 10) is deliberate: SigLIP
    ("So400m/14") is hardcoded regardless of the "dummy" gemma variant (see `Pi0.__init__`),
    so on the depth-4 dummy Gemma stack SigLIP dominates total cost and the two numbers sit
    close together -- close enough that smaller step counts were observed to flip the
    ordering depending on process warmth (e.g. whether other tests already primed JAX/XLA in
    the same process). Enough denoise steps give the Gemma stack -- and hence the end-to-end
    path -- a comfortable, reproducible margin over the SigLIP-only path in both a cold,
    standalone run and a warm run alongside the rest of the suite.
    """
    args = bench_inference.Args(config_name="debug_pi05", warmup=1, repeats=2, num_steps=200)
    result = bench_inference.benchmark(args)

    assert result.p50_ms > 0
    assert result.p95_ms >= result.p50_ms
    assert result.min_ms <= result.p50_ms
    assert result.num_layers == 4  # the "dummy" gemma variant is depth 4
    assert result.siglip_p50_ms > 0
    # SigLIP alone is a strict subset of the end-to-end forward pass (which also runs the
    # full Gemma/action-expert stack over the denoise loop), so it must be faster.
    assert result.siglip_p50_ms < result.p50_ms


def test_num_layers_reflects_keep_layers():
    base = _config.get_config("debug_pi05")
    truncated = dataclasses.replace(base, model=dataclasses.replace(base.model, keep_layers=(0, 2)))
    assert bench_inference.num_layers_of(truncated) == 2
    assert bench_inference.num_layers_of(base) == 4
