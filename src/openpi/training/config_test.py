"""Tests for training config (Task 2: MEM-gated repack + video param threading)."""

import pathlib

import pytest


def test_rlds_droid_repack_has_video_keys_for_mem():
    from openpi.training import config as _config
    from openpi.models import pi0_mem_config

    factory = _config.RLDSDroidDataConfig(rlds_data_dir="/tmp/x")
    mem_cfg = factory.create(pathlib.Path("/tmp"), pi0_mem_config.Pi0MEMConfig(num_video_frames=6))
    repack = mem_cfg.repack_transforms.inputs[0].structure  # RepackTransform.structure attribute
    assert "observation/video_exterior_image_1_left" in repack, (
        f"Expected 'observation/video_exterior_image_1_left' in repack keys, got: {list(repack.keys())}"
    )
    assert "observation/video_wrist_image_left" in repack
    assert mem_cfg.video_stride_frames == 15


def test_rlds_droid_repack_no_video_keys_for_non_mem():
    from openpi.training import config as _config
    from openpi.models import pi0_config

    factory = _config.RLDSDroidDataConfig(rlds_data_dir="/tmp/x")
    pi0_cfg = factory.create(pathlib.Path("/tmp"), pi0_config.Pi0Config())
    repack = pi0_cfg.repack_transforms.inputs[0].structure
    video_keys = [k for k in repack if k.startswith("observation/video_")]
    assert video_keys == [], f"Expected no video_* keys for pi0 model, got: {video_keys}"
