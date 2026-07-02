import flax.traverse_util as tu
import numpy as np

from openpi.training import weight_loaders as wl

_REGEX = r".*(lora|state_proj|video_img).*"


def test_merge_params_no_silent_drop_and_fresh_init():
    # Reference (target) MEM param tree: base + MEM-new keys.
    ref = {
        "PaliGemma": {
            "llm": {"layer/kernel": np.zeros((4, 4), np.float32)},
            "img": {"embedding/kernel": np.zeros((2, 2), np.float32)},
            "video_img": {"embedding/kernel": np.ones((2, 2), np.float32)},  # MEM-new (fresh)
            "llm_lora": {"lora_a": np.ones((4, 2), np.float32)},  # MEM-new (fresh)
        },
        "state_proj": {"kernel": np.ones((3, 4), np.float32)},  # MEM-new (fresh)
    }
    # "pi0.5-like" loaded tree = reference minus the MEM-new keys, WITHOUT the video_img remap.
    loaded = {
        "PaliGemma": {
            "llm": {"layer/kernel": np.full((4, 4), 7.0, np.float32)},
            "img": {"embedding/kernel": np.full((2, 2), 9.0, np.float32)},
        }
    }

    merged = wl._merge_params(loaded, ref, missing_regex=_REGEX)  # noqa: SLF001

    flat_ref = tu.flatten_dict(ref, sep="/")
    flat_merged = tu.flatten_dict(merged, sep="/")

    # 1. Nothing silently dropped: every reference key is present.
    assert set(flat_merged) == set(flat_ref)
    # 2. Base keys took the LOADED values.
    np.testing.assert_array_equal(flat_merged["PaliGemma/llm/layer/kernel"], np.full((4, 4), 7.0))
    np.testing.assert_array_equal(flat_merged["PaliGemma/img/embedding/kernel"], np.full((2, 2), 9.0))
    # 3. MEM-new keys took FRESH init (the reference values), since loaded lacked them.
    np.testing.assert_array_equal(flat_merged["PaliGemma/video_img/embedding/kernel"], np.ones((2, 2)))
    np.testing.assert_array_equal(flat_merged["PaliGemma/llm_lora/lora_a"], np.ones((4, 2)))
    np.testing.assert_array_equal(flat_merged["state_proj/kernel"], np.ones((3, 4)))
