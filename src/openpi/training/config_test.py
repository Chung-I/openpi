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


@pytest.mark.parametrize(("name", "k"), [("pi0_mem_droid_k1_verify", 1), ("pi0_mem_droid_k6_verify", 6)])
def test_video_eval_verify_configs(name, k):
    from flax import nnx

    from openpi.training import config as _config
    from openpi.training import weight_loaders

    cfg = _config.get_config(name)
    assert cfg.model.num_video_frames == k
    assert cfg.model.action_horizon == 16
    assert cfg.model.lora is True
    assert cfg.model.pi05 is True
    assert cfg.num_train_steps == 10_000
    assert cfg.fsdp_devices == 4
    assert cfg.batch_size == 128
    assert cfg.grad_accum_steps == 1
    assert cfg.seed == 42
    assert cfg.num_workers == 0
    assert cfg.project_name == "video-encoder-eval"
    assert cfg.wandb_enabled is True
    # Non-empty freeze filter (LoRA) and the video_img-seeding loader.
    assert cfg.freeze_filter is not None
    assert repr(cfg.freeze_filter) != repr(nnx.Nothing())
    assert isinstance(cfg.weight_loader, weight_loaders.SiglipToVideoImgWeightLoader)
    assert cfg.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_droid/params"
