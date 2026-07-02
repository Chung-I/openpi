import pathlib

from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.training import config as _config
from openpi.training.droid_rlds_dataset import DroidActionSpace, _video_window_indices


def _k1_data_config():
    return _config.RLDSDroidDataConfig(
        repo_id="droid",
        rlds_data_dir="/tmp/does-not-exist",
        action_space=DroidActionSpace.JOINT_POSITION,
    )


def test_rlds_repack_includes_video_keys_at_k1(tmp_path: pathlib.Path):
    model_cfg = Pi0MEMConfig(
        num_video_frames=1, paligemma_variant="dummy", action_expert_variant="dummy"
    )
    dc = _k1_data_config().create(tmp_path, model_cfg)
    structure = dc.repack_transforms.inputs[0].structure
    assert "observation/video_exterior_image_1_left" in structure
    assert "observation/video_wrist_image_left" in structure


def test_video_window_indices_k1_is_current_frame():
    import numpy as np

    idx = np.asarray(_video_window_indices(5, 1, 15))
    np.testing.assert_array_equal(idx, [[0], [1], [2], [3], [4]])


def test_pi0_mem_k1_debug_config_registered():
    cfg = _config.get_config("pi0_mem_k1_debug")
    assert cfg.model.num_video_frames == 1
    assert cfg.wandb_enabled is False


def test_num_video_frames_for_is_zero_for_non_mem():
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.data_loader import _num_video_frames_for

    assert _num_video_frames_for(Pi0Config()) == 0


def test_num_video_frames_for_matches_mem_k():
    from openpi.training.data_loader import _num_video_frames_for

    assert (
        _num_video_frames_for(
            Pi0MEMConfig(num_video_frames=1, paligemma_variant="dummy", action_expert_variant="dummy")
        )
        == 1
    )
    assert (
        _num_video_frames_for(
            Pi0MEMConfig(num_video_frames=6, paligemma_variant="dummy", action_expert_variant="dummy")
        )
        == 6
    )
