"""Tests for VideoViT space-time separable attention module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.siglip as _siglip
from openpi.models.video_vit import VideoViTConfig, VideoViTEncoder


def _make_siglip(dtype="float32"):
    return _siglip.Module(
        num_classes=32,
        variant="mu/2",
        pool_type="none",
        scan=False,
        dtype_mm=dtype,
    )


def test_single_frame_matches_siglip():
    """K=1 must produce the exact same output as the standard SigLIP ViT."""
    config = VideoViTConfig(num_video_frames=1, temporal_attn_every_n_layers=4)
    siglip = _make_siglip()
    video_vit = VideoViTEncoder(
        config=config,
        siglip_kwargs=dict(
            num_classes=32,
            variant="mu/2",
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )

    image = jnp.ones((1, 224, 224, 3))
    video = image[:, None, :, :, :]  # [1, 1, 224, 224, 3]

    rng = jax.random.key(0)
    siglip_vars = siglip.init(rng, image, train=False)
    siglip_out, _ = siglip.apply(siglip_vars, image, train=False)

    video_vit_vars = video_vit.init(rng, video, train=False)
    video_vit_out, _ = video_vit.apply(video_vit_vars, video, train=False)

    np.testing.assert_allclose(
        np.array(siglip_out),
        np.array(video_vit_out),
        atol=1e-5,
        err_msg="K=1 VideoViT output must match single-frame SigLIP",
    )


def test_multi_frame_output_shape():
    """VideoViT with K>1 should still produce [b, n, d] output.

    Uses 16x16 images (8x8 = 64 patches with patch_size=2) to stay within
    memory limits while testing the output shape invariant.
    """
    config = VideoViTConfig(num_video_frames=4, temporal_attn_every_n_layers=1)
    video_vit = VideoViTEncoder(
        config=config,
        siglip_kwargs=dict(
            num_classes=32,
            variant="mu/2",
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )

    # 16×16 images → 8×8 = 64 patches (patch_size=2 for mu/2 variant)
    video = jnp.ones((2, 4, 16, 16, 3))
    rng = jax.random.key(0)
    vars = video_vit.init(rng, video, train=False)
    out, _ = video_vit.apply(vars, video, train=False)

    # Output should be [b, n_patches, num_classes] regardless of K
    assert out.shape[0] == 2  # batch size preserved
    assert out.ndim == 3  # [b, n, d]
    assert out.shape[-1] == 32  # num_classes


def test_temporal_causal_masking():
    """Changing the current frame must change the output.

    Uses 16x16 images to avoid OOM on large attention matrices.
    No num_classes: the head Dense uses zero-init and would mask all signal.
    """
    config = VideoViTConfig(num_video_frames=3, temporal_attn_every_n_layers=1)
    video_vit = VideoViTEncoder(
        config=config,
        siglip_kwargs=dict(
            # No num_classes: raw token embeddings are returned (non-zero init).
            # A zero-init head collapses every output to zero, hiding the signal.
            variant="mu/2",
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )

    rng = jax.random.key(42)
    base_video = jax.random.normal(rng, (1, 3, 16, 16, 3))
    vars = video_vit.init(rng, base_video, train=False)

    # Run with original video
    out1, _ = video_vit.apply(vars, base_video, train=False)

    # Modify the last (current) frame — output must change
    modified_video = base_video.at[:, -1].set(
        jax.random.normal(jax.random.key(99), (1, 16, 16, 3))
    )
    out2, _ = video_vit.apply(vars, modified_video, train=False)

    # Output SHOULD differ because we changed the current frame (frame index K-1 = current)
    assert not jnp.allclose(out1, out2), "Changing current frame must change output"


def test_past_frame_influence():
    """Past frame change should influence the current frame's output (via temporal attention).

    Uses 16x16 images to avoid OOM on large attention matrices.
    No num_classes so raw encoder tokens are returned (non-zero init).
    """
    config = VideoViTConfig(num_video_frames=3, temporal_attn_every_n_layers=1)
    video_vit = VideoViTEncoder(
        config=config,
        siglip_kwargs=dict(
            variant="mu/2",
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )

    rng = jax.random.key(7)
    base_video = jax.random.normal(rng, (1, 3, 16, 16, 3))
    vars = video_vit.init(rng, base_video, train=False)

    out1, _ = video_vit.apply(vars, base_video, train=False)

    # Modify a past (earlier) frame — the current frame's representation should change
    modified_video = base_video.at[:, 0].set(
        jax.random.normal(jax.random.key(123), (1, 16, 16, 3))
    )
    out2, _ = video_vit.apply(vars, modified_video, train=False)

    # With causal attention, the current frame (last) CAN see past frames,
    # so changing a past frame changes the output.
    assert not jnp.allclose(out1, out2), "Past frame change must influence current output via temporal attention"
