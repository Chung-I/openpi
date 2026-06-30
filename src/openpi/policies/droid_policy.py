import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example(num_video_frames: int = 0) -> dict:
    """Creates a random input example for the Droid policy.

    Args:
        num_video_frames: When > 0, include the four ``observation/video_*``
            keys with ``num_video_frames`` frames (for PI0_MEM).
    """
    example = {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }
    if num_video_frames > 0:
        K = num_video_frames
        example["observation/video_exterior_image_1_left"] = np.random.randint(
            256, size=(K, 224, 224, 3), dtype=np.uint8
        )
        example["observation/video_wrist_image_left"] = np.random.randint(
            256, size=(K, 224, 224, 3), dtype=np.uint8
        )
        example["observation/video_joint_position"] = np.random.rand(K, 7).astype(np.float32)
        example["observation/video_gripper_position"] = np.random.rand(K, 1).astype(np.float32)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        video_keys_present = (
            self.model_type == _model.ModelType.PI0_MEM
            and "observation/video_exterior_image_1_left" in data
        )
        if video_keys_present:
            v_ext = np.stack([_parse_image(f) for f in np.asarray(data["observation/video_exterior_image_1_left"])])
            v_wrist = np.stack([_parse_image(f) for f in np.asarray(data["observation/video_wrist_image_left"])])
            K = v_ext.shape[0]
            video_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            video_images = (v_ext, v_wrist, np.zeros_like(v_ext))
            video_masks = (
                np.ones(K, dtype=bool),
                np.ones(K, dtype=bool),
                np.zeros(K, dtype=bool),
            )
            video_states = np.concatenate(
                [np.asarray(data["observation/video_joint_position"]),
                 np.asarray(data["observation/video_gripper_position"])],
                axis=-1,
            )  # [K, 8]
            # Derive single-frame inputs from the current (last) video frame.
            base_image = v_ext[-1]
            wrist_image = v_wrist[-1]
            state = video_states[-1]
        else:
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([data["observation/joint_position"], gripper_pos])

            # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
            # stores as float32 (C,H,W), gets skipped for policy inference
            base_image = _parse_image(data["observation/exterior_image_1_left"])
            wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05 | _model.ModelType.PI0_MEM:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if video_keys_present:
            inputs["video_image"] = dict(zip(video_names, video_images, strict=True))
            inputs["video_image_mask"] = dict(zip(video_names, video_masks, strict=True))
            inputs["video_states"] = video_states

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}
