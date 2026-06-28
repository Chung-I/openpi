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

    # Set model-specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore

    # Pi0.5 mode: adaRMSNorm timestep injection in action expert.
    pi05: bool = True
    discrete_state_input: bool = False

    # MEM-specific config.
    num_video_frames: int = 6
    temporal_attn_every_n_layers: int = 4
    max_memory_tokens: int = 128
    max_subtask_tokens: int = 64
    hl_loss_weight: float = 1.0
    ll_loss_weight: float = 1.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

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
        from flax import nnx
        from openpi.models.pi0_mem import Pi0MEM

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
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                tokenized_memory=jax.ShapeDtypeStruct([batch_size, self.max_memory_tokens], jnp.int32),
                tokenized_memory_mask=jax.ShapeDtypeStruct([batch_size, self.max_memory_tokens], jnp.bool_),
                tokenized_subtask=jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], jnp.int32),
                tokenized_subtask_mask=jax.ShapeDtypeStruct([batch_size, self.max_subtask_tokens], jnp.bool_),
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
