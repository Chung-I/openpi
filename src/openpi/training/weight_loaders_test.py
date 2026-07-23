import flax.traverse_util
import numpy as np
import pytest

from openpi.training import weight_loaders


def _fake_checkpoint_params(depth: int = 18):
    """Mirrors the real pi0.5 layout: scanned layers plus unscanned siblings.

    Verified against a real checkpoint — everything under PaliGemma/llm/layers/ carries a
    leading depth axis; embedder, final_norm, SigLIP and the action projections do not.
    The `_1` suffix marks action-expert parameters, which are scanned in the same stack.
    """
    return {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "attn": {
                        "q_einsum": {"w": np.arange(depth * 2, dtype=np.float32).reshape(depth, 2)},
                        "q_einsum_1": {"w": np.arange(depth * 2, dtype=np.float32).reshape(depth, 2) * 10},
                    },
                    "mlp": {"linear": np.arange(depth * 3, dtype=np.float32).reshape(depth, 3)},
                    "pre_attention_norm": {"scale": np.arange(depth * 4, dtype=np.float32).reshape(depth, 4)},
                },
                "final_norm": {"scale": np.ones(4, dtype=np.float32)},
                "embedder": {"input_embedding": np.ones((5, 4), dtype=np.float32)},
            },
            "img": {
                "head": {"kernel": np.ones((2, 2), dtype=np.float32)},
                # NOT scanned (outside `PaliGemma/llm/layers/`), but its leading dim coincidentally
                # equals `depth`. A shape-based gather heuristic would wrongly slice this; the
                # gather must select by key prefix only. See
                # test_gather_leaves_unscanned_params_untouched.
                "embedding": np.arange(depth * 4, dtype=np.float32).reshape(depth, 4),
            },
        },
        "action_out_proj": {"kernel": np.ones((3, 3), dtype=np.float32)},
    }


def test_gather_selects_the_requested_layers():
    params = _fake_checkpoint_params(depth=18)
    keep = (0, 3, 7, 11, 14, 17)
    out = weight_loaders._gather_scanned_layers(params, keep)  # noqa: SLF001

    q = out["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum"]["w"]
    assert q.shape == (6, 2)
    np.testing.assert_array_equal(q, params["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum"]["w"][list(keep)])


def test_gather_truncates_the_action_expert_tower_too():
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, (0, 3, 7, 11, 14, 17))  # noqa: SLF001
    # `_1`-suffixed params are the action expert; they live in the same scan.
    assert out["PaliGemma"]["llm"]["layers"]["attn"]["q_einsum_1"]["w"].shape == (6, 2)


def test_gather_leaves_unscanned_params_untouched():
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, (0, 3, 7, 11, 14, 17))  # noqa: SLF001

    for path in [
        ("PaliGemma", "llm", "final_norm", "scale"),
        ("PaliGemma", "llm", "embedder", "input_embedding"),
        ("PaliGemma", "img", "head", "kernel"),
        # Leading dim == depth (18), but it lives outside `PaliGemma/llm/layers/`. A gather
        # that keyed off shape instead of key prefix would wrongly slice this.
        ("PaliGemma", "img", "embedding"),
        ("action_out_proj", "kernel"),
    ]:
        expected = params
        actual = out
        for key in path:
            expected, actual = expected[key], actual[key]
        np.testing.assert_array_equal(actual, expected, err_msg="/".join(path))


def test_gather_over_the_full_range_is_the_identity():
    """The most valuable guard here: catches wrong-axis and wrong-order bugs."""
    params = _fake_checkpoint_params(depth=18)
    out = weight_loaders._gather_scanned_layers(params, tuple(range(18)))  # noqa: SLF001

    flat_in = flax.traverse_util.flatten_dict(params, sep="/")
    flat_out = flax.traverse_util.flatten_dict(out, sep="/")
    assert flat_in.keys() == flat_out.keys()
    for k in flat_in:
        np.testing.assert_array_equal(flat_out[k], flat_in[k], err_msg=k)


def test_gather_preserves_the_requested_order():
    params = _fake_checkpoint_params(depth=18)
    src = params["PaliGemma"]["llm"]["layers"]["mlp"]["linear"]
    out = weight_loaders._gather_scanned_layers(params, (2, 5))  # noqa: SLF001

    got = out["PaliGemma"]["llm"]["layers"]["mlp"]["linear"]
    np.testing.assert_array_equal(got[0], src[2])
    np.testing.assert_array_equal(got[1], src[5])


def test_gather_rejects_an_index_beyond_the_checkpoint_depth():
    params = _fake_checkpoint_params(depth=18)
    with pytest.raises(ValueError, match="out of range"):
        weight_loaders._gather_scanned_layers(params, (0, 18))  # noqa: SLF001


def test_gather_rejects_an_empty_keep_layers():
    params = _fake_checkpoint_params(depth=18)
    with pytest.raises(ValueError, match="non-empty"):
        weight_loaders._gather_scanned_layers(params, ())  # noqa: SLF001


def test_gather_rejects_a_negative_index_instead_of_silently_wrapping():
    """(-1, 3) would otherwise silently wrap to [layer 17, layer 3] via numpy's negative indexing."""
    params = _fake_checkpoint_params(depth=18)
    with pytest.raises(ValueError, match="out of range"):
        weight_loaders._gather_scanned_layers(params, (-1, 3))  # noqa: SLF001


def test_gather_rejects_an_out_of_range_index_even_alongside_valid_ones():
    """(5, 2, 2): index 5 is out of range for depth 4 and must be caught, even though 2 (repeated) is valid.

    A max-only check on this exact tuple would also catch it (max=5), but per-index checking
    is what correctly localizes the complaint to the offending index rather than the tuple.
    """
    params = _fake_checkpoint_params(depth=4)
    with pytest.raises(ValueError, match="out of range"):
        weight_loaders._gather_scanned_layers(params, (5, 2, 2))  # noqa: SLF001


def test_layer_subset_loader_is_a_weight_loader():
    loader = weight_loaders.LayerSubsetWeightLoader("gs://example/params", keep_layers=(0, 3))
    assert isinstance(loader, weight_loaders.WeightLoader)
    assert loader.keep_layers == (0, 3)
