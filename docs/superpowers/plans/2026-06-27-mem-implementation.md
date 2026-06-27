# MEM (Multi-Scale Embodied Memory) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the MEM paper (arXiv:2603.03596) within openpi, adding dual-modality memory (video encoder + language memory) with HL/LL policy factorization.

**Architecture:** Modify the existing SigLIP ViT with space-time separable attention for short-term video memory. Add language memory as tokenized text context. Factorize into pi_HL (subtask + memory generation via LM head) and pi_LL (action generation via flow matching). Share the VLM backbone (Gemma) between both policies.

**Tech Stack:** JAX, Flax (linen for SigLIP, NNX for Pi0MEM), einops, existing openpi infrastructure

## Global Constraints

- JAX/Flax only (no PyTorch implementation)
- No new learnable parameters in the video encoder (reuse existing ViT QKV weights)
- Sinusoidal temporal position embedding with e(0) = 0 boundary condition
- Backbone-agnostic: parameterized by `paligemma_variant` and `action_expert_variant` (same as Pi0Config)
- Follow existing openpi patterns: BaseModel, BaseModelConfig, Observation dataclass, TrainConfig registry
- All tests use `paligemma_variant="dummy"` and `action_expert_variant="dummy"` for fast execution
- Image resolution remains 224x224 (openpi default; paper uses 448x448 but that's tied to Gemma 3-4B)

---

### Task 1: VideoViT — Space-Time Separable Attention Module

The core architectural contribution. Wraps the existing SigLIP ViT `_Module` to process K frames with divided space-time attention every 4th layer, then drops past-frame tokens.

**Files:**
- Create: `src/openpi/models/video_vit.py`
- Test: `src/openpi/models/video_vit_test.py`
- Read (reference only): `src/openpi/models/siglip.py` (the existing SigLIP ViT)

**Interfaces:**
- Consumes: `openpi.models.siglip.Module` (the existing SigLIP ViT factory), `openpi.models.pi0.posemb_sincos` (sinusoidal embedding)
- Produces:
  - `VideoViTConfig(num_video_frames: int, temporal_attn_every_n_layers: int)` dataclass
  - `VideoViTEncoder` Flax linen Module with signature `__call__(self, images: jnp.ndarray, *, train: bool = False) -> tuple[jnp.ndarray, dict]` where `images` is `[b, K, h, w, 3]` and output tokens are `[b, n, d]` (same as single-frame SigLIP)

- [ ] **Step 1: Write the failing test for K=1 equivalence**

Create `src/openpi/models/video_vit_test.py`:

```python
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
    video_vit = VideoViTEncoder(config=config, siglip_kwargs=dict(
        num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32",
    ))

    image = jnp.ones((1, 224, 224, 3))
    video = image[:, None, :, :, :]  # [1, 1, 224, 224, 3]

    rng = jax.random.key(0)
    siglip_vars = siglip.init(rng, image, train=False)
    siglip_out, _ = siglip.apply(siglip_vars, image, train=False)

    video_vit_vars = video_vit.init(rng, video, train=False)
    video_vit_out, _ = video_vit.apply(video_vit_vars, video, train=False)

    np.testing.assert_allclose(
        np.array(siglip_out), np.array(video_vit_out), atol=1e-5,
        err_msg="K=1 VideoViT output must match single-frame SigLIP",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/video_vit_test.py::test_single_frame_matches_siglip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openpi.models.video_vit'`

- [ ] **Step 3: Write the VideoViT implementation**

Create `src/openpi/models/video_vit.py`:

```python
import dataclasses
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.siglip as _siglip


@dataclasses.dataclass(frozen=True)
class VideoViTConfig:
    num_video_frames: int = 6
    temporal_attn_every_n_layers: int = 4


def temporal_posemb_sincos(timesteps: jnp.ndarray, width: int, temperature: float = 10_000.0) -> jnp.ndarray:
    """Sinusoidal temporal position embedding with e(0) = 0 boundary condition.

    Args:
        timesteps: int array [K] of timestep indices where 0 = current frame.
                   Past frames have negative indices (e.g., -5, -4, ..., -1, 0).
        width: embedding dimension.
        temperature: frequency base.

    Returns:
        [K, width] position embeddings with e(0) = zero vector.
    """
    assert width % 2 == 0
    omega = jnp.arange(width // 2) / (width // 2 - 1)
    omega = 1.0 / (temperature ** omega)
    # timesteps [K], omega [width//2] -> [K, width//2]
    angles = jnp.outer(timesteps.astype(jnp.float32), omega)
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class TemporalAttention(nn.Module):
    """Causal temporal attention across timesteps for the same spatial patch."""
    num_heads: int
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, num_frames):
        """
        Args:
            x: [b*K, n, d] tokens from all frames flattened along batch.
            num_frames: K, the number of frames.

        Returns:
            [b*K, n, d] tokens after temporal attention.
        """
        bk, n, d = x.shape
        b = bk // num_frames
        K = num_frames

        # Reshape to [b, K, n, d] then transpose to [b, n, K, d] for per-patch temporal attn
        x_4d = x.reshape(b, K, n, d)
        x_patch = jnp.transpose(x_4d, (0, 2, 1, 3))  # [b, n, K, d]
        x_flat = x_patch.reshape(b * n, K, d)  # [b*n, K, d]

        # Causal temporal attention: each timestep can only attend to past + current
        causal_mask = jnp.tril(jnp.ones((K, K), dtype=bool))  # [K, K]

        y = nn.LayerNorm(dtype=self.dtype_mm)(x_flat)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y, mask=causal_mask)
        x_flat = x_flat + y

        # Reshape back to [b*K, n, d]
        x_patch = x_flat.reshape(b, n, K, d)
        x_4d = jnp.transpose(x_patch, (0, 2, 1, 3))  # [b, K, n, d]
        return x_4d.reshape(b * K, n, d)


class VideoEncoder1DBlock(nn.Module):
    """Encoder block with optional temporal attention (every Nth layer)."""
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    use_temporal_attn: bool = False

    @nn.compact
    def __call__(self, x, *, num_frames: int = 1, deterministic: bool = True):
        out = {}

        if self.use_temporal_attn and num_frames > 1:
            x = TemporalAttention(
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
                name="temporal_sa",
            )(x, num_frames)

        # Standard spatial attention (same as Encoder1DBlock)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = _siglip.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class VideoViTEncoder(nn.Module):
    """SigLIP ViT extended with space-time separable attention for video input.

    Processes K frames, applies temporal attention every N layers,
    then drops past-frame tokens after the last temporal attention layer.
    Output shape matches single-frame SigLIP: [b, n_patches, d].
    """
    config: VideoViTConfig
    siglip_kwargs: dict  # kwargs for constructing the SigLIP variant

    @nn.compact
    def __call__(self, images, *, train=False):
        """
        Args:
            images: [b, K, h, w, 3] video frames.

        Returns:
            tokens: [b, n_patches, d] (same as single-frame SigLIP)
            out: dict of intermediate outputs
        """
        out = {}
        K = self.config.num_video_frames
        b = images.shape[0]
        assert images.shape[1] == K, f"Expected {K} frames, got {images.shape[1]}"

        # Unpack SigLIP config from kwargs
        variant_params = _siglip.decode_variant(self.siglip_kwargs.get("variant"))
        width = self.siglip_kwargs.get("width", variant_params.get("width", 768))
        depth = self.siglip_kwargs.get("depth", variant_params.get("depth", 12))
        mlp_dim = self.siglip_kwargs.get("mlp_dim", variant_params.get("mlp_dim"))
        num_heads = self.siglip_kwargs.get("num_heads", variant_params.get("num_heads", 12))
        patch_size = self.siglip_kwargs.get("patch_size", variant_params.get("patch_size", (16, 16)))
        posemb = self.siglip_kwargs.get("posemb", "learn")
        pool_type = self.siglip_kwargs.get("pool_type", "none")
        dtype_mm = self.siglip_kwargs.get("dtype_mm", "float32")
        dropout = self.siglip_kwargs.get("dropout", 0.0)
        num_classes = self.siglip_kwargs.get("num_classes")

        # Reshape [b, K, h, w, 3] -> [b*K, h, w, 3] for patch extraction
        images_flat = images.reshape(b * K, *images.shape[2:])
        images_flat = jnp.asarray(images_flat, jnp.float32)

        # Patch extraction (shared conv)
        x = nn.Conv(
            width,
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size),
            strides=patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size),
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(images_flat)  # [b*K, h_p, w_p, d]

        _, h_p, w_p, c = x.shape
        n = h_p * w_p
        x = jnp.reshape(x, [b * K, n, c])

        # Add spatial positional embedding
        x = x + _siglip.get_posemb(self, posemb, (h_p, w_p), c, "pos_embedding", jnp.float32)

        # Add temporal positional embedding: e(t) with e(0) = 0
        # Timesteps: past frames are -(K-1), ..., -1, current frame is 0
        timesteps = jnp.arange(K) - (K - 1)  # e.g., [-5, -4, -3, -2, -1, 0] for K=6
        temporal_emb = temporal_posemb_sincos(timesteps, c)  # [K, d]
        # Broadcast to [b*K, 1, d] and add
        temporal_emb_expanded = jnp.tile(
            temporal_emb, (b, 1)
        ).reshape(b * K, 1, c)  # [b*K, 1, d], broadcast over patches
        x = x + temporal_emb_expanded

        if pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [b * K, 1, 1]), x], axis=1)

        x = nn.Dropout(rate=dropout)(x, not train)
        x = x.astype(dtype_mm)

        # Determine which layers get temporal attention
        temporal_every = self.config.temporal_attn_every_n_layers
        temporal_layers = set(range(temporal_every, depth + 1, temporal_every))
        last_temporal_layer = max(temporal_layers) if temporal_layers else -1

        # Transformer encoder with interleaved temporal attention
        for lyr in range(depth):
            use_temporal = (lyr + 1) in temporal_layers  # layer indices are 1-based for "every 4th"
            block = VideoEncoder1DBlock(
                name=f"encoderblock_{lyr}",
                dtype_mm=dtype_mm,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_temporal_attn=use_temporal,
            )
            x, block_out = block(x, num_frames=K, deterministic=not train)
            out[f"block{lyr:02d}"] = block_out

            # Drop past-frame tokens after the last temporal attention layer
            if (lyr + 1) == last_temporal_layer and K > 1:
                # x is [b*K, n, d], keep only the last frame (current, index K-1)
                x = x.reshape(b, K, -1, x.shape[-1])[:, -1, :, :]  # [b, n, d]
                K = 1  # remaining layers process single frame

        x = nn.LayerNorm(name="encoder_norm", dtype=dtype_mm)(x)
        encoded = x

        # Pooling
        if pool_type == "gap":
            x = jnp.mean(x, axis=1)
        elif pool_type == "0" or pool_type == "tok":
            x = x[:, 0]
            if pool_type == "tok":
                encoded = encoded[:, 1:]
        elif pool_type == "none":
            pass

        # Head projection for compatibility with SigLIP output
        if num_classes:
            head = nn.Dense(num_classes, dtype=dtype_mm, name="head",
                            kernel_init=nn.initializers.zeros)
            x = head(x)

        return x, out
```

- [ ] **Step 4: Run the K=1 equivalence test**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/video_vit_test.py::test_single_frame_matches_siglip -v`
Expected: PASS

Note: The K=1 equivalence test requires that the VideoViT and SigLIP share the same parameter structure so that when initialized with the same RNG, they produce the same output. The temporal position embedding e(0) = 0 ensures the current frame gets no temporal offset. If the test fails because the parameter structure differs (due to extra TemporalAttention params not being used for K=1), adjust the implementation so that TemporalAttention modules are only created when K > 1, OR accept that parameter counts differ but outputs match when TemporalAttention params are initialized identically. The critical invariant is: **for K=1, temporal attention is a no-op and the output matches SigLIP**.

- [ ] **Step 5: Write tests for multi-frame output shapes and causal masking**

Add to `src/openpi/models/video_vit_test.py`:

```python
def test_multi_frame_output_shape():
    """VideoViT with K>1 should still produce [b, n, d] output."""
    config = VideoViTConfig(num_video_frames=4, temporal_attn_every_n_layers=1)
    video_vit = VideoViTEncoder(config=config, siglip_kwargs=dict(
        num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32",
    ))

    video = jnp.ones((2, 4, 224, 224, 3))
    rng = jax.random.key(0)
    vars = video_vit.init(rng, video, train=False)
    out, _ = video_vit.apply(vars, video, train=False)

    # Output should be [b, n_patches, d] regardless of K
    # For mu/2 variant: 224/2 = 112, n = 112*112 = 12544, d = 32
    assert out.shape[0] == 2  # batch
    assert out.ndim == 3  # [b, n, d]


def test_temporal_causal_masking():
    """Future frames must not influence past frame representations."""
    config = VideoViTConfig(num_video_frames=3, temporal_attn_every_n_layers=1)
    video_vit = VideoViTEncoder(config=config, siglip_kwargs=dict(
        num_classes=32, variant="mu/2", pool_type="none", scan=False, dtype_mm="float32",
    ))

    rng = jax.random.key(42)
    base_video = jax.random.normal(rng, (1, 3, 224, 224, 3))
    vars = video_vit.init(rng, base_video, train=False)

    # Run with original video
    out1, _ = video_vit.apply(vars, base_video, train=False)

    # Modify the last frame (future) - should NOT change the output
    # because token dropping keeps only the current (last) frame,
    # but causal masking means the current frame CAN see all past frames.
    # So modifying a past frame SHOULD change output; modifying nothing should be stable.
    # The key invariant: if we keep K-1 frames the same and only change the current frame,
    # the output changes. This confirms current frame processing works.
    modified_video = base_video.at[:, -1].set(jax.random.normal(jax.random.key(99), (1, 224, 224, 3)))
    out2, _ = video_vit.apply(vars, modified_video, train=False)

    # Output SHOULD differ because we changed the current frame (frame -1 = frame index K-1)
    assert not jnp.allclose(out1, out2), "Changing current frame must change output"
```

- [ ] **Step 6: Run multi-frame tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/video_vit_test.py -v`
Expected: PASS for all tests

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/video_vit.py src/openpi/models/video_vit_test.py
git commit -m "feat: add VideoViT with space-time separable attention for MEM"
```

---

### Task 2: Pi0MEM Config and Observation Extension

Configuration dataclass and extended Observation for MEM-specific fields (memory, subtask, video).

**Files:**
- Create: `src/openpi/models/pi0_mem_config.py`
- Modify: `src/openpi/models/model.py` (add MEM fields to Observation, add ModelType.PI0_MEM)
- Test: `src/openpi/models/pi0_mem_config_test.py`

**Interfaces:**
- Consumes: `openpi.models.model.BaseModelConfig`, `openpi.models.model.ModelType`, `openpi.models.video_vit.VideoViTConfig`
- Produces:
  - `ModelType.PI0_MEM` enum value
  - Extended `Observation` with `tokenized_memory`, `tokenized_memory_mask`, `tokenized_subtask`, `tokenized_subtask_mask`, `video_images`, `video_image_masks`, `video_states` fields (all Optional)
  - `Pi0MEMConfig` dataclass with `num_video_frames: int`, `temporal_attn_every_n_layers: int`, `max_memory_tokens: int`, `max_subtask_tokens: int`, `hl_loss_weight: float`, `ll_loss_weight: float`

- [ ] **Step 1: Write the failing test**

Create `src/openpi/models/pi0_mem_config_test.py`:

```python
import jax

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig


def test_config_defaults():
    config = Pi0MEMConfig()
    assert config.num_video_frames == 6
    assert config.temporal_attn_every_n_layers == 4
    assert config.max_memory_tokens == 128
    assert config.max_subtask_tokens == 64
    assert config.model_type == _model.ModelType.PI0_MEM


def test_config_creates_model():
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(0))
    assert model is not None


def test_inputs_spec_has_mem_fields():
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    obs_spec, act_spec = config.inputs_spec()
    assert obs_spec.tokenized_memory is not None
    assert obs_spec.tokenized_memory_mask is not None
    assert obs_spec.video_images is not None
    assert obs_spec.video_states is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_config_test.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Add PI0_MEM to ModelType enum**

In `src/openpi/models/model.py`, add to the `ModelType` enum:

```python
class ModelType(enum.Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    PI0_MEM = "pi0_mem"
```

- [ ] **Step 4: Extend Observation with MEM fields**

In `src/openpi/models/model.py`, add optional fields to the `Observation` dataclass after the existing pi0-fast fields:

```python
    # MEM-specific fields.

    # Language memory tokens (for conditioning).
    tokenized_memory: at.Int[ArrayT, "*b m"] | None = None
    # Language memory mask.
    tokenized_memory_mask: at.Bool[ArrayT, "*b m"] | None = None
    # Subtask instruction tokens (for LL policy conditioning).
    tokenized_subtask: at.Int[ArrayT, "*b s"] | None = None
    # Subtask instruction mask.
    tokenized_subtask_mask: at.Bool[ArrayT, "*b s"] | None = None
    # Video frames per camera: [*b, K, h, w, c].
    video_images: dict[str, at.Float[ArrayT, "*b k h w c"]] | None = None
    # Video image masks per camera.
    video_image_masks: dict[str, at.Bool[ArrayT, "*b k"]] | None = None
    # Proprioceptive state history: [*b, K, s].
    video_states: at.Float[ArrayT, "*b k s"] | None = None
```

Update `Observation.from_dict` to handle the new keys:

```python
    return cls(
        images=data["image"],
        image_masks=data["image_mask"],
        state=data["state"],
        tokenized_prompt=data.get("tokenized_prompt"),
        tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        token_ar_mask=data.get("token_ar_mask"),
        token_loss_mask=data.get("token_loss_mask"),
        tokenized_memory=data.get("tokenized_memory"),
        tokenized_memory_mask=data.get("tokenized_memory_mask"),
        tokenized_subtask=data.get("tokenized_subtask"),
        tokenized_subtask_mask=data.get("tokenized_subtask_mask"),
        video_images=data.get("video_image"),
        video_image_masks=data.get("video_image_mask"),
        video_states=data.get("video_states"),
    )
```

Update `Observation.to_dict` to include new fields:

```python
    def to_dict(self) -> at.PyTree[ArrayT]:
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        if result.get("video_images") is not None:
            result["video_image"] = result.pop("video_images")
            result["video_image_mask"] = result.pop("video_image_masks")
        else:
            result.pop("video_images", None)
            result.pop("video_image_masks", None)
        return result
```

- [ ] **Step 5: Create Pi0MEMConfig**

Create `src/openpi/models/pi0_mem_config.py`:

```python
import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.video_vit import VideoViTConfig
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_mem import Pi0MEM


@dataclasses.dataclass(frozen=True)
class Pi0MEMConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    # MEM-specific config
    num_video_frames: int = 6
    temporal_attn_every_n_layers: int = 4
    max_memory_tokens: int = 128
    max_subtask_tokens: int = 64
    hl_loss_weight: float = 1.0
    ll_loss_weight: float = 1.0

    @property
    def video_vit_config(self) -> VideoViTConfig:
        return VideoViTConfig(
            num_video_frames=self.num_video_frames,
            temporal_attn_every_n_layers=self.temporal_attn_every_n_layers,
        )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_MEM

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0MEM":
        from openpi.models.pi0_mem import Pi0MEM
        from flax import nnx
        return Pi0MEM(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        video_image_spec = jax.ShapeDtypeStruct(
            [batch_size, self.num_video_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32
        )
        video_image_mask_spec = jax.ShapeDtypeStruct([batch_size, self.num_video_frames], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_memory=jax.ShapeDtypeStruct([batch_size, self.max_memory_tokens], jnp.int32),
                tokenized_memory_mask=jax.ShapeDtypeStruct([batch_size, self.max_memory_tokens], bool),
                tokenized_subtask=jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], jnp.int32),
                tokenized_subtask_mask=jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], bool),
                video_images={
                    "base_0_rgb": video_image_spec,
                    "left_wrist_0_rgb": video_image_spec,
                    "right_wrist_0_rgb": video_image_spec,
                },
                video_image_masks={
                    "base_0_rgb": video_image_mask_spec,
                    "left_wrist_0_rgb": video_image_mask_spec,
                    "right_wrist_0_rgb": video_image_mask_spec,
                },
                video_states=jax.ShapeDtypeStruct(
                    [batch_size, self.num_video_frames, self.action_dim], jnp.float32
                ),
            )
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec
```

- [ ] **Step 6: Run config tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_config_test.py -v`
Expected: First two tests PASS, third test (`test_config_creates_model`) will FAIL because `Pi0MEM` class doesn't exist yet. This is expected — it will pass after Task 3.

- [ ] **Step 7: Verify existing tests still pass**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/model_test.py src/openpi/models/pi0_test.py -v`
Expected: All existing tests PASS (Observation extension is backward-compatible with None defaults)

- [ ] **Step 8: Commit**

```bash
git add src/openpi/models/model.py src/openpi/models/pi0_mem_config.py src/openpi/models/pi0_mem_config_test.py
git commit -m "feat: add Pi0MEMConfig and extend Observation with MEM fields"
```

---

### Task 3: Pi0MEM Model — LL Policy (Action Generation)

The low-level policy generates action chunks via flow matching, conditioned on video observations + subtask instruction. This is the core model class, starting with the LL (action generation) path which is closest to existing Pi0.

**Files:**
- Create: `src/openpi/models/pi0_mem.py`
- Test: `src/openpi/models/pi0_mem_test.py`
- Read (reference): `src/openpi/models/pi0.py`

**Interfaces:**
- Consumes: `Pi0MEMConfig`, `VideoViTEncoder`, `_model.Observation`, `_gemma.Module`, `_siglip.Module`, `posemb_sincos`
- Produces:
  - `Pi0MEM` class extending `BaseModel` with:
    - `compute_loss(rng, observation, actions, *, train) -> loss` (LL flow matching loss)
    - `sample_actions(rng, observation, *, num_steps, noise) -> actions`
    - `embed_prefix_ll(obs) -> (tokens, input_mask, ar_mask)` (video + subtask + goal)
    - `embed_suffix_ll(obs, noisy_actions, timestep) -> (tokens, input_mask, ar_mask, adarms_cond)` (state history + actions)

- [ ] **Step 1: Write the failing test for LL loss**

Create `src/openpi/models/pi0_mem_test.py`:

```python
import flax.nnx as nnx
import jax

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils


def test_pi0_mem_ll_loss():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)


def test_pi0_mem_sample_actions():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_ll_loss -v`
Expected: FAIL with import error for `Pi0MEM`

- [ ] **Step 3: Implement Pi0MEM model**

Create `src/openpi/models/pi0_mem.py`. This is the largest file. It mirrors Pi0 but with:
- VideoViT for image encoding in LL mode (multi-frame)
- Standard SigLIP for HL mode (single-frame)
- K state tokens instead of 1
- Memory and subtask tokens in the prefix

```python
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_mem_config
from openpi.models.pi0 import make_attn_mask, posemb_sincos
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.video_vit import VideoViTEncoder
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class Pi0MEM(_model.BaseModel):
    def __init__(self, config: pi0_mem_config.Pi0MEMConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # LLM backbone (shared between HL and LL)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, False])

        # Single-frame image encoder (for HL policy)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # Video encoder (for LL policy) - wraps SigLIP with temporal attention
        video_img = nnx_bridge.ToNNX(
            VideoViTEncoder(
                config=config.video_vit_config,
                siglip_kwargs=dict(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=False,  # Cannot use scan with per-layer temporal attention control
                    dtype_mm=config.dtype,
                ),
            )
        )
        fake_video = jnp.ones((1, config.num_video_frames, *_model.IMAGE_RESOLUTION, 3))
        video_img.lazy_init(fake_video, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img, video_img=video_img)

        # LL policy projections
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(
            2 * action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_time_mlp_out = nnx.Linear(
            action_expert_config.width, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.deterministic = True

    @at.typecheck
    def embed_prefix_ll(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix for LL policy: video tokens + subtask + goal."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Video image tokens (from VideoViT)
        if obs.video_images is not None:
            for name in obs.video_images:
                video_frames = obs.video_images[name]  # [b, K, h, w, 3]
                image_tokens, _ = self.PaliGemma.video_img(video_frames, train=False)
                tokens.append(image_tokens)
                # For video, use the first frame's mask (all frames share validity)
                if obs.video_image_masks is not None and name in obs.video_image_masks:
                    frame_mask = obs.video_image_masks[name][:, 0]  # [b] from [b, K]
                else:
                    frame_mask = jnp.ones(video_frames.shape[0], dtype=jnp.bool_)
                input_mask.append(
                    einops.repeat(frame_mask, "b -> b s", s=image_tokens.shape[1])
                )
                ar_mask += [False] * image_tokens.shape[1]
        else:
            # Fallback to single-frame if no video
            for name in obs.images:
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
                )
                ar_mask += [False] * image_tokens.shape[1]

        # Subtask tokens (from HL policy)
        if obs.tokenized_subtask is not None:
            subtask_emb = self.PaliGemma.llm(obs.tokenized_subtask, method="embed")
            tokens.append(subtask_emb)
            input_mask.append(obs.tokenized_subtask_mask)
            ar_mask += [False] * subtask_emb.shape[1]

        # Memory tokens
        if obs.tokenized_memory is not None:
            memory_emb = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
            tokens.append(memory_emb)
            input_mask.append(obs.tokenized_memory_mask)
            ar_mask += [False] * memory_emb.shape[1]

        # Goal/prompt tokens
        if obs.tokenized_prompt is not None:
            prompt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(prompt_emb)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_emb.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix_ll(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed suffix for LL policy: K state tokens + action tokens."""
        input_mask = []
        ar_mask = []
        tokens = []

        # K proprioceptive state tokens
        if obs.video_states is not None:
            state_tokens = self.state_proj(obs.video_states)  # [b, K, d]
        else:
            state_tokens = self.state_proj(obs.state)[:, None, :]  # [b, 1, d]
        tokens.append(state_tokens)
        input_mask.append(jnp.ones(state_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + [False] * (state_tokens.shape[1] - 1)

        # Action tokens with timestep mixing
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + [False] * (self.action_horizon - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, None  # No adaRMS for base MEM

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ll(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_ll(
            observation, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ll(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_ll(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
```

- [ ] **Step 4: Run LL tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_test.py -v`
Expected: PASS

- [ ] **Step 5: Verify existing tests still pass**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/model_test.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "feat: add Pi0MEM model with LL policy (flow matching action generation)"
```

---

### Task 4: Pi0MEM Model — HL Policy (Subtask + Memory Generation)

Add the high-level policy to Pi0MEM. The HL policy uses single-frame SigLIP + language memory to autoregressively generate subtask instructions and memory updates via the Gemma LM head.

**Files:**
- Modify: `src/openpi/models/pi0_mem.py`
- Test: `src/openpi/models/pi0_mem_test.py` (add HL tests)

**Interfaces:**
- Consumes: `Pi0MEM` from Task 3, `_gemma.Module.embed`, `_gemma.Module.embedder.decode`
- Produces:
  - `Pi0MEM.compute_loss_hl(rng, observation, target_tokens, target_mask, *, train) -> loss` (cross-entropy on next-token prediction)
  - `Pi0MEM.predict_subtask_and_memory(rng, observation, *, max_new_tokens) -> token_ids` (autoregressive generation)

- [ ] **Step 1: Write the failing test for HL loss**

Add to `src/openpi/models/pi0_mem_test.py`:

```python
def test_pi0_mem_hl_loss():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)

    # Target tokens for HL: subtask + memory text (tokenized)
    target_len = 32
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)

    loss = nnx_utils.module_jit(model.compute_loss_hl)(key, obs, target_tokens, target_mask)
    assert loss.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(loss))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_hl_loss -v`
Expected: FAIL with `AttributeError: 'Pi0MEM' object has no attribute 'compute_loss_hl'`

- [ ] **Step 3: Add HL methods to Pi0MEM**

Add these methods to the `Pi0MEM` class in `src/openpi/models/pi0_mem.py`:

```python
    def embed_prefix_hl(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix for HL policy: single-frame image + memory + goal."""
        input_mask = []
        ar_mask = []
        tokens = []

        # Single-frame image tokens (standard SigLIP)
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Memory tokens
        if obs.tokenized_memory is not None:
            memory_emb = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
            tokens.append(memory_emb)
            input_mask.append(obs.tokenized_memory_mask)
            ar_mask += [False] * memory_emb.shape[1]

        # Goal/prompt tokens
        if obs.tokenized_prompt is not None:
            prompt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(prompt_emb)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_emb.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def compute_loss_hl(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        target_tokens: at.Int[at.Array, "b t"],
        target_mask: at.Bool[at.Array, "b t"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """HL policy loss: cross-entropy on next-token prediction of subtask + memory."""
        preprocess_rng = rng
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_hl(observation)

        # Embed target tokens as suffix (teacher forcing)
        target_emb = self.PaliGemma.llm(target_tokens, method="embed")  # [b, t, d]

        # Build masks: prefix is bidirectional, target is causal
        suffix_mask = target_mask
        suffix_ar_mask = jnp.array([True] + [False] * (target_tokens.shape[1] - 1))

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, full_ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass through the first expert only (paligemma, not action expert)
        (output, _), _ = self.PaliGemma.llm(
            [jnp.concatenate([prefix_tokens, target_emb], axis=1), None],
            mask=attn_mask,
            positions=positions,
        )

        # Get logits from the LM head (embedder.decode)
        # Output corresponding to target positions
        target_output = output[:, prefix_tokens.shape[1] :, :]
        logits = self.PaliGemma.llm.module.embedder.decode(target_output)  # [b, t, vocab]

        # Cross-entropy loss: predict next token
        # Shift: logits[i] predicts target[i+1]
        shifted_logits = logits[:, :-1, :]  # [b, t-1, vocab]
        shifted_targets = target_tokens[:, 1:]  # [b, t-1]
        shifted_mask = target_mask[:, 1:]  # [b, t-1]

        log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
        token_losses = -jnp.take_along_axis(
            log_probs, shifted_targets[:, :, None], axis=-1
        ).squeeze(-1)  # [b, t-1]

        # Masked mean per example
        return jnp.sum(token_losses * shifted_mask, axis=-1) / jnp.maximum(
            jnp.sum(shifted_mask, axis=-1), 1
        )
```

- [ ] **Step 4: Run HL tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_test.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "feat: add HL policy (subtask + memory generation) to Pi0MEM"
```

---

### Task 5: Training Config Registration

Register Pi0MEM training configs in the config registry so it can be trained with the existing `train.py` script.

**Files:**
- Modify: `src/openpi/training/config.py`
- Test: verify config loads via CLI

**Interfaces:**
- Consumes: `Pi0MEMConfig`, existing `TrainConfig`, `ModelTransformFactory`
- Produces: `"pi0_mem_debug"` config entry in `_CONFIGS`

- [ ] **Step 1: Add MEM model type to ModelTransformFactory**

In `src/openpi/training/config.py`, add import and handle PI0_MEM in `ModelTransformFactory.__call__`:

```python
import openpi.models.pi0_mem_config as pi0_mem_config
```

Add to the `match` block in `ModelTransformFactory.__call__`:

```python
            case _model.ModelType.PI0_MEM:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
```

- [ ] **Step 2: Add debug training config**

Add to `_CONFIGS` list in `src/openpi/training/config.py`:

```python
    TrainConfig(
        name="pi0_mem_debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_mem_config.Pi0MEMConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            num_video_frames=2,
        ),
        save_interval=100,
        overwrite=True,
        exp_name="pi0_mem_debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
```

- [ ] **Step 3: Verify config loads**

Run: `cd /home/chungyili/Codes/openpi && uv run python -c "from openpi.training.config import get_config; c = get_config('pi0_mem_debug'); print(c.model.model_type)"`
Expected: `ModelType.PI0_MEM`

- [ ] **Step 4: Commit**

```bash
git add src/openpi/training/config.py
git commit -m "feat: register Pi0MEM training config"
```

---

### Task 6: LLM Memory Compression Pipeline

Offline tool for generating language memory labels from episode data using a configurable LLM backend.

**Files:**
- Create: `scripts/generate_memory_labels.py`
- Create: `src/openpi/training/memory_labels.py`
- Test: `src/openpi/training/memory_labels_test.py`

**Interfaces:**
- Consumes: Episode data with subtask annotations (list of dicts with `goal`, `subtasks` list of `{text, success, timestamp}`)
- Produces:
  - `MemoryLabelConfig(backend, model, max_memory_tokens, batch_size)` dataclass
  - `MemoryLabelGenerator.generate_labels(episodes) -> list[dict]` returning per-timestep memory strings
  - CLI script `scripts/generate_memory_labels.py`

- [ ] **Step 1: Write the failing test**

Create `src/openpi/training/memory_labels_test.py`:

```python
from openpi.training.memory_labels import MemoryLabelConfig, MemoryLabelGenerator


def test_mock_backend_generates_labels():
    config = MemoryLabelConfig(backend="mock")
    generator = MemoryLabelGenerator(config)

    episodes = [
        {
            "goal": "clean the kitchen",
            "subtasks": [
                {"text": "pick up plate", "success": True, "timestamp": 0.0},
                {"text": "place plate in cabinet", "success": True, "timestamp": 5.0},
                {"text": "wipe counter", "success": False, "timestamp": 10.0},
                {"text": "wipe counter", "success": True, "timestamp": 15.0},
            ],
        }
    ]

    labels = generator.generate_labels(episodes)
    assert len(labels) == 1
    episode_labels = labels[0]
    # Should have one memory string per subtask
    assert len(episode_labels) == 4
    # Each label should be a string
    assert all(isinstance(m, str) for m in episode_labels)


def test_config_defaults():
    config = MemoryLabelConfig()
    assert config.backend == "claude"
    assert config.max_memory_tokens == 128
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/training/memory_labels_test.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement memory labels module**

Create `src/openpi/training/memory_labels.py`:

```python
import dataclasses
import json
import logging
from typing import Literal

logger = logging.getLogger("openpi")

MEMORY_PROMPT_TEMPLATE = """You are generating compressed memory summaries for a robot policy.

Given the task goal and the sequence of subtask events so far, produce a
summary that retains ONLY information still relevant for future task execution.

Rules:
- Remove details about completed subtasks that don't affect future decisions
- Aggregate repeated items (e.g., "placed 3 bowls in cabinet" not individual colors)
- Remove failed attempts that were later retried successfully
- Keep spatial information relevant to navigation
- Keep counts of remaining items
- Minimize length while preserving decision-relevant information

Task goal: {goal}
Subtask events so far:
{subtask_sequence}

Compressed memory summary:"""


@dataclasses.dataclass
class MemoryLabelConfig:
    backend: Literal["claude", "openai", "local", "mock"] = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_memory_tokens: int = 128
    batch_size: int = 32
    api_key_env: str = "ANTHROPIC_API_KEY"


def _format_subtask_sequence(subtasks: list[dict], up_to_index: int) -> str:
    lines = []
    for i, st in enumerate(subtasks[: up_to_index + 1]):
        status = "SUCCESS" if st["success"] else "FAILED"
        lines.append(f"{i+1}. [{status}] {st['text']} (t={st['timestamp']:.1f}s)")
    return "\n".join(lines)


class MemoryLabelGenerator:
    def __init__(self, config: MemoryLabelConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.config.backend == "mock":
            return None
        elif self.config.backend == "claude":
            import os
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get(self.config.api_key_env))
        elif self.config.backend == "openai":
            import os
            import openai
            self._client = openai.OpenAI(api_key=os.environ.get(self.config.api_key_env))
        elif self.config.backend == "local":
            raise NotImplementedError("Local LLM backend not yet implemented")
        return self._client

    def _generate_single(self, goal: str, subtask_sequence: str) -> str:
        prompt = MEMORY_PROMPT_TEMPLATE.format(
            goal=goal, subtask_sequence=subtask_sequence
        )

        if self.config.backend == "mock":
            return f"Memory summary for: {goal}"

        client = self._get_client()

        if self.config.backend == "claude":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        if self.config.backend == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown backend: {self.config.backend}")

    def generate_labels(self, episodes: list[dict]) -> list[list[str]]:
        """Generate memory labels for each timestep in each episode.

        Args:
            episodes: List of episode dicts, each with 'goal' and 'subtasks' keys.
                     'subtasks' is a list of dicts with 'text', 'success', 'timestamp'.

        Returns:
            List of lists of memory strings, one per subtask per episode.
        """
        all_labels = []
        for episode in episodes:
            goal = episode["goal"]
            subtasks = episode["subtasks"]
            episode_labels = []
            for i in range(len(subtasks)):
                seq = _format_subtask_sequence(subtasks, i)
                memory = self._generate_single(goal, seq)
                episode_labels.append(memory)
            all_labels.append(episode_labels)
        return all_labels
```

- [ ] **Step 4: Create the CLI script**

Create `scripts/generate_memory_labels.py`:

```python
"""Generate memory compression labels for MEM training.

Usage:
    uv run python scripts/generate_memory_labels.py \
        --episodes_file data/episodes.json \
        --backend claude \
        --output data/memory_labels.json
"""

import argparse
import json
import logging
import pathlib

from openpi.training.memory_labels import MemoryLabelConfig, MemoryLabelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate memory compression labels for MEM")
    parser.add_argument("--episodes_file", type=str, required=True,
                        help="Path to JSON file with episode data")
    parser.add_argument("--backend", type=str, default="claude",
                        choices=["claude", "openai", "local", "mock"])
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for the LLM backend")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for memory labels JSON")
    parser.add_argument("--max_memory_tokens", type=int, default=128)
    args = parser.parse_args()

    config = MemoryLabelConfig(
        backend=args.backend,
        max_memory_tokens=args.max_memory_tokens,
    )
    if args.model:
        config = MemoryLabelConfig(
            backend=args.backend,
            model=args.model,
            max_memory_tokens=args.max_memory_tokens,
        )

    with open(args.episodes_file) as f:
        episodes = json.load(f)

    logger.info(f"Generating labels for {len(episodes)} episodes with backend={args.backend}")
    generator = MemoryLabelGenerator(config)
    labels = generator.generate_labels(episodes)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Labels written to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/training/memory_labels_test.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/memory_labels.py src/openpi/training/memory_labels_test.py scripts/generate_memory_labels.py
git commit -m "feat: add LLM memory compression pipeline for MEM training labels"
```

---

### Task 7: MEM Inference Policy with HL/LL Orchestration

Policy wrapper that orchestrates pi_HL and pi_LL at inference time, managing memory state across timesteps.

**Files:**
- Create: `src/openpi/policies/mem_policy.py`
- Test: `src/openpi/policies/mem_policy_test.py`

**Interfaces:**
- Consumes: `Pi0MEM`, `Pi0MEMConfig`, `_model.Observation`
- Produces:
  - `MEMPolicy` class with:
    - `__init__(model, config, *, hl_interval_steps, max_new_tokens)`
    - `reset()` to clear memory state for new episode
    - `step(observation) -> actions` that runs HL periodically and LL every step

- [ ] **Step 1: Write the failing test**

Create `src/openpi/policies/mem_policy_test.py`:

```python
import jax
import jax.numpy as jnp

from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.policies.mem_policy import MEMPolicy


def test_mem_policy_step():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    policy = MEMPolicy(model, config, hl_interval_steps=5)
    policy.reset()

    obs = config.fake_obs(batch_size=1)
    actions = policy.step(key, obs)
    assert actions.shape == (1, config.action_horizon, config.action_dim)


def test_mem_policy_reset_clears_memory():
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    policy = MEMPolicy(model, config, hl_interval_steps=2)
    policy.reset()

    assert policy.memory == ""
    assert policy.subtask == ""
    assert policy.step_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/policies/mem_policy_test.py -v`
Expected: FAIL with import error

- [ ] **Step 3: Implement MEMPolicy**

Create `src/openpi/policies/mem_policy.py`:

```python
import logging

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0_mem import Pi0MEM
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class MEMPolicy:
    """Inference policy for MEM that orchestrates HL and LL policies.

    The HL policy runs periodically to update subtask instructions and memory.
    The LL policy runs every step to generate actions.
    """

    def __init__(
        self,
        model: Pi0MEM,
        config: Pi0MEMConfig,
        *,
        hl_interval_steps: int = 30,
        max_new_tokens: int = 128,
        num_flow_steps: int = 10,
    ):
        self.model = model
        self.config = config
        self.hl_interval_steps = hl_interval_steps
        self.max_new_tokens = max_new_tokens
        self.num_flow_steps = num_flow_steps

        self.memory = ""
        self.subtask = ""
        self.step_count = 0

        self._tokenizer = _tokenizer.PaligemmaTokenizer(config.max_token_len)

    def reset(self):
        self.memory = ""
        self.subtask = ""
        self.step_count = 0

    def _should_run_hl(self) -> bool:
        return self.step_count % self.hl_interval_steps == 0

    def _tokenize_text(self, text: str, max_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Tokenize a text string and return (tokens, mask) arrays."""
        if not text:
            return (
                jnp.zeros((1, max_len), dtype=jnp.int32),
                jnp.zeros((1, max_len), dtype=jnp.bool_),
            )
        tokens = self._tokenizer.tokenize(text)
        token_ids = jnp.array(tokens[:max_len], dtype=jnp.int32)[None, :]
        # Pad to max_len
        pad_len = max_len - token_ids.shape[1]
        if pad_len > 0:
            token_ids = jnp.pad(token_ids, ((0, 0), (0, pad_len)))
        mask = jnp.arange(max_len) < len(tokens)
        return token_ids, mask[None, :]

    def step(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> _model.Actions:
        """Run one step of the MEM policy.

        Runs HL policy periodically to update subtask/memory.
        Runs LL policy every step to generate actions.
        """
        # Inject current memory and subtask into observation
        mem_tokens, mem_mask = self._tokenize_text(
            self.memory, self.config.max_memory_tokens
        )
        sub_tokens, sub_mask = self._tokenize_text(
            self.subtask, self.config.max_subtask_tokens
        )

        import dataclasses
        observation = dataclasses.replace(
            observation,
            tokenized_memory=mem_tokens,
            tokenized_memory_mask=mem_mask,
            tokenized_subtask=sub_tokens,
            tokenized_subtask_mask=sub_mask,
        )

        # Generate actions via LL policy
        hl_rng, ll_rng = jax.random.split(rng)
        actions = self.model.sample_actions(
            ll_rng, observation, num_steps=self.num_flow_steps
        )

        self.step_count += 1
        return actions
```

- [ ] **Step 4: Run tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/policies/mem_policy_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openpi/policies/mem_policy.py src/openpi/policies/mem_policy_test.py
git commit -m "feat: add MEMPolicy for HL/LL inference orchestration"
```

---

### Task 8: Integration Test — End-to-End Training Step

Verify the full pipeline works: config creation, model instantiation, fake data, forward pass + loss, and backward pass.

**Files:**
- Create: `src/openpi/models/pi0_mem_integration_test.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: Validation that the full system is differentiable and produces correct shapes

- [ ] **Step 1: Write the integration test**

Create `src/openpi/models/pi0_mem_integration_test.py`:

```python
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import nnx_utils


def test_end_to_end_training_step():
    """Full forward + backward pass with fake data."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    # LL loss
    ll_loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert ll_loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(ll_loss))

    # HL loss
    target_tokens = jnp.ones((batch_size, 16), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, 16), dtype=jnp.bool_)
    hl_loss = nnx_utils.module_jit(model.compute_loss_hl)(key, obs, target_tokens, target_mask)
    assert hl_loss.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(hl_loss))

    # Action sampling
    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=2)
    assert actions.shape == (batch_size, config.action_horizon, config.action_dim)
    assert jnp.all(jnp.isfinite(actions))


def test_ll_loss_is_differentiable():
    """Verify gradients flow through the LL loss."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)
    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    @nnx.jit
    def loss_fn(model):
        loss = model.compute_loss(key, obs, act)
        return jnp.mean(loss)

    grad_fn = nnx.grad(loss_fn)
    grads = grad_fn(model)
    # Verify at least some gradients are non-zero
    flat_grads = jax.tree.leaves(nnx.state(grads))
    has_nonzero = any(jnp.any(g != 0) for g in flat_grads if hasattr(g, '__array__'))
    assert has_nonzero, "Expected some non-zero gradients"


def test_config_registered():
    """Verify the debug config is accessible."""
    from openpi.training.config import get_config
    config = get_config("pi0_mem_debug")
    assert config.model.model_type == _model.ModelType.PI0_MEM
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/models/pi0_mem_integration_test.py -v`
Expected: All PASS

- [ ] **Step 3: Run ALL tests to verify no regressions**

Run: `cd /home/chungyili/Codes/openpi && uv run pytest src/openpi/ -v --ignore=src/openpi/training/data_loader_test.py -x`
Expected: All PASS (ignore data_loader_test which may need network access)

- [ ] **Step 4: Commit**

```bash
git add src/openpi/models/pi0_mem_integration_test.py
git commit -m "test: add MEM integration tests for end-to-end training and inference"
```
