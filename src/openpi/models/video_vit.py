"""VideoViT: Space-time separable attention for multi-frame visual encoding.

Extends the SigLIP ViT to process K video frames with divided space-time
attention (temporal attention every N layers), then drops past-frame tokens.
Output shape is [b, n_patches, d] — identical to single-frame SigLIP.

Critical invariant: K=1 output is numerically identical to standard SigLIP
when initialised with the same RNG (same parameter paths, no temporal ops).
"""

import dataclasses

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as _siglip


@dataclasses.dataclass(frozen=True)
class VideoViTConfig:
    num_video_frames: int = 6
    temporal_attn_every_n_layers: int = 4


def temporal_posemb_sincos(
    timesteps: jnp.ndarray,
    width: int,
    temperature: float = 10_000.0,
) -> jnp.ndarray:
    """Sinusoidal temporal position embedding with e(0) = 0.

    Uses [sin(t*w), cos(t*w) - 1] so the zero-th position maps to
    the all-zeros vector. This means the current frame (t=0) receives
    no additive temporal bias — only past frames are shifted.

    Args:
        timesteps: int array [K] where 0 = current frame; past frames
                   use negative indices (e.g. -5, -4, ..., -1, 0).
        width: embedding dimension (must be even).
        temperature: frequency base.

    Returns:
        [K, width] temporal embeddings with emb[K-1] == 0 (current frame).
    """
    assert width % 2 == 0
    half = width // 2
    omega = 1.0 / (temperature ** (jnp.arange(half) / max(half - 1, 1)))
    angles = jnp.outer(timesteps.astype(jnp.float32), omega)  # [K, half]
    # sin(0)=0, cos(0)-1=0 → e(0) = zero vector
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles) - 1.0], axis=-1)


class TemporalAttention(nn.Module):
    """Causal temporal self-attention across K frames for each spatial patch.

    Receives tokens from all K frames flattened on the batch axis ([b*K, n, d]),
    transposes to attend over the temporal axis per patch, then restores shape.
    Only called when K > 1.
    """

    num_heads: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x: jnp.ndarray, num_frames: int) -> jnp.ndarray:
        """
        Args:
            x: [b*K, n, d] spatial tokens, all frames stacked on batch dim.
            num_frames: K.

        Returns:
            [b*K, n, d] after causal temporal attention.
        """
        bk, n, d = x.shape
        b = bk // num_frames
        K = num_frames

        # [b*K, n, d] → [b, n, K, d]
        x_4d = x.reshape(b, K, n, d)
        x_patch = jnp.transpose(x_4d, (0, 2, 1, 3))  # [b, n, K, d]
        x_flat = x_patch.reshape(b * n, K, d)  # [b*n, K, d]

        # Causal mask: each frame can attend to itself and earlier frames only
        causal_mask = jnp.tril(jnp.ones((K, K), dtype=bool))  # [K, K]

        y = nn.LayerNorm(dtype=self.dtype_mm)(x_flat)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y, mask=causal_mask)
        x_flat = x_flat + y

        # Restore [b*K, n, d]
        x_patch = x_flat.reshape(b, n, K, d)
        x_4d = jnp.transpose(x_patch, (0, 2, 1, 3))  # [b, K, n, d]
        return x_4d.reshape(b * K, n, d)


class VideoEncoder1DBlock(nn.Module):
    """Transformer encoder block with optional temporal attention.

    Parameter structure for spatial components matches SigLIP Encoder1DBlock
    exactly when use_temporal_attn=False (same Flax auto-numbered names).
    When use_temporal_attn=True, a named 'temporal_sa' sub-module is prepended
    but the spatial component numbering is unaffected.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_temporal_attn: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        *,
        num_frames: int = 1,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        out = {}

        # Temporal attention: explicit name 'temporal_sa' → doesn't shift
        # the auto-numbered spatial layer names (LayerNorm_0, etc.)
        if self.use_temporal_attn and num_frames > 1:
            x = TemporalAttention(
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
                name="temporal_sa",
            )(x, num_frames)

        # Spatial self-attention  (same names as SigLIP Encoder1DBlock)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        # MLP block
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = _siglip.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class _VideoTransformerEncoder(nn.Module):
    """Transformer encoder with per-layer temporal attention and token dropping.

    Intended to be instantiated as name='Transformer' inside VideoViTEncoder
    so that its parameter paths match SigLIP's Encoder(name='Transformer').
    """

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    # 0-based layer indices that receive temporal attention
    temporal_attn_layers: tuple = ()

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        K: int,
        b: int,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Args:
            x: [b*K, n, d] after patch embedding + positional embedding.
            K: number of video frames (runtime, not compile-time).
            b: original batch size.
            deterministic: disable dropout when True.

        Returns:
            (x, out) where x is [b, n, d] (current frame only).
        """
        out = {}
        temporal_set = set(self.temporal_attn_layers)
        last_temporal = max(temporal_set) if temporal_set else -1
        K_cur = K  # tracks effective K as we process layers

        for lyr in range(self.depth):
            use_temporal = lyr in temporal_set
            block = VideoEncoder1DBlock(
                name=f"encoderblock_{lyr}",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_temporal_attn=use_temporal,
            )
            x, block_out = block(x, num_frames=K_cur, deterministic=deterministic)
            out[f"block{lyr:02d}"] = block_out

            # After the last temporal attention layer, drop past-frame tokens.
            # Only the current frame (last along K axis) is kept.
            if use_temporal and K_cur > 1 and lyr == last_temporal:
                n_patches = x.shape[1]
                x = x.reshape(b, K_cur, n_patches, x.shape[-1])[:, -1, :, :]
                K_cur = 1  # remaining layers process a single frame

        out["pre_ln"] = x
        x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)
        return x, out


class VideoViTEncoder(nn.Module):
    """SigLIP ViT extended with space-time separable attention for video input.

    Processes K frames with divided space-time attention (temporal attention
    every N layers), drops past-frame tokens after the last temporal layer,
    and returns tokens with the same shape as single-frame SigLIP.

    For K=1 (single frame), temporal ops are skipped entirely and the output
    is numerically identical to standard SigLIP given the same RNG seed,
    because the parameter paths are structurally identical.
    """

    config: VideoViTConfig
    # kwargs forwarded to decode_variant + used as SigLIP config
    siglip_kwargs: dict

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        *,
        train: bool = False,
    ) -> tuple[jnp.ndarray, dict]:
        """
        Args:
            images: [b, K, h, w, 3] video frames.

        Returns:
            (tokens, out) where tokens has shape [b, n_patches, width]
            (or [b, n_patches, num_classes] if num_classes is given).
        """
        out = {}
        K = self.config.num_video_frames
        b = images.shape[0]
        assert images.shape[1] == K, f"Expected {K} frames, got {images.shape[1]}"

        # ------------------------------------------------------------------
        # Unpack SigLIP variant config
        # ------------------------------------------------------------------
        variant_params = _siglip.decode_variant(self.siglip_kwargs.get("variant"))
        width = self.siglip_kwargs.get("width", variant_params.get("width", 768))
        depth = self.siglip_kwargs.get("depth", variant_params.get("depth", 12))
        mlp_dim = self.siglip_kwargs.get("mlp_dim", variant_params.get("mlp_dim"))
        num_heads = self.siglip_kwargs.get("num_heads", variant_params.get("num_heads", 12))
        patch_size_raw = self.siglip_kwargs.get("patch_size", variant_params.get("patch_size", (16, 16)))
        patch_size = patch_size_raw if isinstance(patch_size_raw, tuple) else (patch_size_raw, patch_size_raw)
        posemb = self.siglip_kwargs.get("posemb", "learn")
        pool_type = self.siglip_kwargs.get("pool_type", "none")
        dtype_mm = self.siglip_kwargs.get("dtype_mm", "float32")
        dropout = self.siglip_kwargs.get("dropout", 0.0)
        num_classes = self.siglip_kwargs.get("num_classes")

        # ------------------------------------------------------------------
        # Patch embedding — name="embedding" matches SigLIP
        # ------------------------------------------------------------------
        images_flat = images.reshape(b * K, *images.shape[2:])
        images_flat = jnp.asarray(images_flat, jnp.float32)

        x = nn.Conv(
            width,
            patch_size,
            strides=patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(images_flat)  # [b*K, h_p, w_p, width]

        _, h_p, w_p, c = x.shape
        n_patches = h_p * w_p
        x = x.reshape(b * K, n_patches, c)

        # ------------------------------------------------------------------
        # Spatial positional embedding — name="pos_embedding" matches SigLIP
        # ------------------------------------------------------------------
        x = x + _siglip.get_posemb(
            self, posemb, (h_p, w_p), c, "pos_embedding", jnp.float32
        )

        # ------------------------------------------------------------------
        # Temporal positional embedding — only added for K > 1.
        # For K=1 this block is skipped entirely, preserving numerical
        # equivalence with standard SigLIP.
        # e(0) = 0 (zero vector) by construction, but we skip rather than
        # add zeros to be safe and explicit.
        # ------------------------------------------------------------------
        if K > 1:
            # timesteps: current frame = 0, past frames = -(K-1), ..., -1
            timesteps = jnp.arange(K) - (K - 1)
            temporal_emb = temporal_posemb_sincos(timesteps, c)  # [K, c]
            # broadcast over patches and batch
            temporal_emb_bk = jnp.tile(temporal_emb, (b, 1)).reshape(b * K, 1, c)
            x = x + temporal_emb_bk

        if pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [b * K, 1, 1]), x], axis=1)

        x = nn.Dropout(rate=dropout)(x, not train)
        x = x.astype(dtype_mm)

        # ------------------------------------------------------------------
        # Determine which layers receive temporal attention
        # ------------------------------------------------------------------
        every_n = self.config.temporal_attn_every_n_layers
        # 0-based layer indices: (1, 2, ..., depth) mapped to (0, 1, ..., depth-1)
        # "every N layers" = layers 1*N-1, 2*N-1, ... (0-based)
        temporal_attn_layers = tuple(
            lyr for lyr in range(depth)
            if (lyr + 1) % every_n == 0
        )

        # ------------------------------------------------------------------
        # Transformer encoder — name="Transformer" matches SigLIP's Encoder
        # ------------------------------------------------------------------
        x, enc_out = _VideoTransformerEncoder(
            name="Transformer",
            depth=depth,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype_mm=dtype_mm,
            temporal_attn_layers=temporal_attn_layers,
        )(x, K=K, b=b, deterministic=not train)
        out["encoder"] = enc_out
        encoded = out["encoded"] = x

        # ------------------------------------------------------------------
        # Pooling — match SigLIP behaviour for pool_type
        # ------------------------------------------------------------------
        if pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif pool_type in ("0", "tok"):
            x = out["head_input"] = x[:, 0]
            if pool_type == "tok":
                encoded = encoded[:, 1:]
        elif pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{pool_type}'")

        # ------------------------------------------------------------------
        # Head projection — name="head" matches SigLIP
        # ------------------------------------------------------------------
        if num_classes:
            head = nn.Dense(
                num_classes,
                dtype=dtype_mm,
                name="head",
                kernel_init=nn.initializers.zeros,
            )
            x = out["logits"] = head(x)

        return x, out
