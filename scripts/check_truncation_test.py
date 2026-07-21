import os

import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from . import check_truncation


def test_scan_depths_reads_the_leading_axis():
    params = {
        "PaliGemma": {
            "llm": {
                "layers": {"attn": {"q_einsum": {"w": np.zeros((6, 2))}}},
                "final_norm": {"scale": np.zeros(4)},
            }
        }
    }
    depths = check_truncation.scan_depths(params)
    assert depths == {"PaliGemma/llm/layers/attn/q_einsum/w": 6}


def test_scan_depths_ignores_unscanned_params():
    params = {"action_out_proj": {"kernel": np.zeros((3, 3))}}
    assert check_truncation.scan_depths(params) == {}


def test_assert_uniform_depth_accepts_a_consistent_stack():
    check_truncation.assert_uniform_depth({"a": 6, "b": 6}, expected=6)


def test_assert_uniform_depth_rejects_a_wrong_depth():
    with pytest.raises(ValueError, match="expected 6"):
        check_truncation.assert_uniform_depth({"a": 18}, expected=6)


def test_assert_uniform_depth_rejects_an_empty_stack():
    with pytest.raises(ValueError, match="no scanned"):
        check_truncation.assert_uniform_depth({}, expected=6)


def test_assert_layers_agree_accepts_matching_indices():
    check_truncation.assert_layers_agree((0, 3, 7, 11, 14, 17), (0, 3, 7, 11, 14, 17))


def test_assert_layers_agree_accepts_no_loader_opinion():
    """A loader without a `keep_layers` attribute (e.g. CheckpointWeightLoader) is exempt."""
    check_truncation.assert_layers_agree((0, 3, 7, 11, 14, 17), None)
    check_truncation.assert_layers_agree(None, None)


def test_assert_layers_agree_rejects_same_length_different_indices():
    """The failure this fix exists for: a depth-only check would miss this."""
    with pytest.raises(ValueError, match=r"weight loader keeps layers \(1, 3\)"):
        check_truncation.assert_layers_agree((0, 2), (1, 3))


def test_assert_layers_agree_rejects_loader_layers_when_model_is_untruncated():
    with pytest.raises(ValueError, match="model was built for \\(\\)"):
        check_truncation.assert_layers_agree(None, (0, 1))
