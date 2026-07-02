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


def test_unstack_scanned_encoderblocks():
    depth = 3
    flat = {
        "embedding/kernel": np.zeros((2, 2, 3, 4), np.float32),          # non-block: copied as-is
        "pos_embedding": np.zeros((1, 5, 8), np.float32),                # non-block: copied as-is
        "Transformer/encoderblock/LayerNorm_0/scale": np.arange(depth * 8, dtype=np.float32).reshape(depth, 8),
        "Transformer/encoder_norm/scale": np.zeros((8,), np.float32),    # non-block: copied as-is
    }
    out = wl._unstack_scanned_encoderblocks(flat)  # noqa: SLF001

    assert out["embedding/kernel"].shape == (2, 2, 3, 4)
    assert out["pos_embedding"].shape == (1, 5, 8)
    assert out["Transformer/encoder_norm/scale"].shape == (8,)
    # Stacked block split into depth per-layer blocks, leading axis removed.
    for i in range(depth):
        assert out[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"].shape == (8,)
        np.testing.assert_array_equal(
            out[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"],
            np.arange(depth * 8, dtype=np.float32).reshape(depth, 8)[i],
        )
    assert "Transformer/encoderblock/LayerNorm_0/scale" not in out


def test_remapped_video_img_matches_siglip():
    """Un-stacked scan=True SigLIP weights, loaded into a K=1 VideoViTEncoder, must
    reproduce the SigLIP output to ~1e-6 under highest matmul precision (asserted at
    atol 1e-5). This is A's equivalence proof extended through the scan un-stack that B
    introduces."""
    import flax.core
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    import openpi.models.siglip as _siglip
    from openpi.models.video_vit import VideoViTConfig
    from openpi.models.video_vit import VideoViTEncoder

    kw = {"variant": "mu/2", "pool_type": "none", "dtype_mm": "float32"}
    siglip = _siglip.Module(scan=True, **kw)              # checkpoint layout (stacked blocks)
    video = VideoViTEncoder(
        config=VideoViTConfig(num_video_frames=1, temporal_attn_every_n_layers=4),
        siglip_kwargs=flax.core.FrozenDict(scan=False, **kw),  # per-layer layout
    )

    image = jnp.ones((1, 224, 224, 3))
    clip = image[:, None]  # [1, 1, 224, 224, 3]
    rng = jax.random.key(0)

    siglip_vars = siglip.init(rng, image, train=False)

    # The un-stack crosses scan=True (stacked encoderblocks) -> scan=False (unrolled per-layer)
    # execution paths, whose default-precision matmul reduction order differs (~5e-4 drift).
    # "highest" precision makes both paths use full-precision matmuls, proving the remap is
    # algebraically exact (the two outputs then agree to ~1e-6).
    with jax.default_matmul_precision("highest"):
        siglip_out, _ = siglip.apply(siglip_vars, image, train=False)

        # Un-stack the SigLIP params and load them into the K=1 VideoViTEncoder.
        flat_img = tu.flatten_dict(siglip_vars["params"], sep="/")
        flat_video = wl._unstack_scanned_encoderblocks(flat_img)  # noqa: SLF001
        video_params = {"params": tu.unflatten_dict(flat_video, sep="/")}
        video_out, _ = video.apply(video_params, clip, train=False)

    assert not np.allclose(np.array(siglip_out), 0.0), (
        "SigLIP output is degenerate (all zeros) — comparison would be vacuous"
    )
    np.testing.assert_allclose(np.array(siglip_out), np.array(video_out), atol=1e-5)
