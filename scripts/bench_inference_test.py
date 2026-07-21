import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import bench_inference


def test_benchmark_runs_on_a_dummy_model():
    """Uses the depth-4 'dummy' variant so this runs on CPU in seconds."""
    args = bench_inference.Args(config_name="debug_pi05", warmup=1, repeats=2, num_steps=2)
    result = bench_inference.benchmark(args)

    assert result.p50_ms > 0
    assert result.p95_ms >= result.p50_ms
    assert result.min_ms <= result.p50_ms
    assert result.num_layers == 4  # the "dummy" gemma variant is depth 4


def test_num_layers_reflects_keep_layers():
    base = _config.get_config("debug_pi05")
    truncated = dataclasses.replace(base, model=dataclasses.replace(base.model, keep_layers=(0, 2)))
    assert bench_inference.num_layers_of(truncated) == 2
    assert bench_inference.num_layers_of(base) == 4
