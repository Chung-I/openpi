import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy


def _mem_example(K=6):
    return {
        "observation/video_exterior_image_1_left": np.random.randint(256, size=(K, 224, 224, 3), dtype=np.uint8),
        "observation/video_wrist_image_left": np.random.randint(256, size=(K, 224, 224, 3), dtype=np.uint8),
        "observation/video_joint_position": np.random.rand(K, 7).astype(np.float32),
        "observation/video_gripper_position": np.random.rand(K, 1).astype(np.float32),
        "prompt": "do something",
    }


def test_droid_inputs_builds_video_fields():
    K = 6
    out = droid_policy.DroidInputs(model_type=_model.ModelType.PI0_MEM)(_mem_example(K))
    assert out["video_image"]["base_0_rgb"].shape == (K, 224, 224, 3)
    assert out["video_image"]["right_wrist_0_rgb"].shape == (K, 224, 224, 3)
    assert out["video_image_mask"]["base_0_rgb"].shape == (K,)
    assert bool(out["video_image_mask"]["base_0_rgb"].all())
    assert not bool(out["video_image_mask"]["right_wrist_0_rgb"].any())
    assert out["video_states"].shape == (K, 8)
    # single-frame image == current (last) video frame
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], out["video_image"]["base_0_rgb"][-1])
