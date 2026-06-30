"""Pi0MEM: Memory-augmented Pi0/Pi0.5 model with multi-frame video observation.

The low-level (LL) policy generates action chunks via flow matching, conditioned
on video observations + subtask instruction + episodic memory.

Architecture differences from Pi0/Pi0.5:
- VideoViTEncoder instead of standard SigLIP for image encoding (multi-frame)
- K proprioceptive state tokens (one per video frame) instead of 1
- Subtask and memory tokens in the prefix (from HL policy)
- AR boundary is at the first state token (K tokens, not 1)

When pi05=True (default), uses adaRMSNorm timestep injection matching pi0.5.
When pi05=False, uses the legacy MLP-concat path matching pi0.
"""

import logging

import einops
import flax.core
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

# PaliGemma tokenizer special token IDs (SentencePiece / Gemma vocab).
_PALIGEMMA_BOS_TOKEN_ID: int = 2
_PALIGEMMA_EOS_TOKEN_ID: int = 1


class Pi0MEM(_model.BaseModel):
    def __init__(self, config: pi0_mem_config.Pi0MEMConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05

        pg_variant = config.paligemma_variant
        ax_variant = config.action_expert_variant
        if config.lora:
            pg_variant = pg_variant if "lora" in pg_variant or pg_variant == "dummy" else pg_variant + "_lora"
            ax_variant = ax_variant if "lora" in ax_variant or ax_variant == "dummy" else ax_variant + "_lora"
        paligemma_config = _gemma.get_config(pg_variant)
        action_expert_config = _gemma.get_config(ax_variant)

        # LLM backbone (two-expert: PaliGemma + action expert)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )

        # Single-frame image encoder (fallback when video_images is absent)
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

        # Video encoder (LL policy) — space-time separable attention over K frames
        video_img = nnx_bridge.ToNNX(
            VideoViTEncoder(
                config=config.video_vit_config,
                siglip_kwargs=flax.core.FrozenDict(
                    num_classes=paligemma_config.width,
                    variant="So400m/14",
                    pool_type="none",
                    scan=False,  # per-layer temporal attention control uses a Python loop
                    dtype_mm=config.dtype,
                ),
            )
        )
        fake_video = jnp.ones((1, config.num_video_frames, *_model.IMAGE_RESOLUTION, 3))
        video_img.lazy_init(fake_video, train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img, video_img=video_img)

        # LL policy projections (action expert width)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width, action_expert_config.width, rngs=rngs
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Set by model.train() / model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix_ll(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix for LL policy: video tokens + subtask + goal.

        All prefix tokens use bidirectional attention (ar_mask=False).
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # --- Video image tokens (VideoViT) or fallback to single-frame SigLIP ---
        if obs.video_images is not None:
            for name in obs.video_images:
                video_frames = obs.video_images[name]  # [b, K, h, w, 3]
                image_tokens, _ = self.PaliGemma.video_img(video_frames, train=False)
                tokens.append(image_tokens)
                # Use the first-frame mask to gate the whole camera stream
                if obs.video_image_masks is not None and name in obs.video_image_masks:
                    frame_mask = obs.video_image_masks[name][:, 0]  # [b]
                else:
                    frame_mask = jnp.ones(video_frames.shape[0], dtype=jnp.bool_)
                input_mask.append(
                    einops.repeat(frame_mask, "b -> b s", s=image_tokens.shape[1])
                )
                ar_mask += [False] * image_tokens.shape[1]
        else:
            # Fallback: standard single-frame SigLIP
            for name in obs.images:
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
                )
                ar_mask += [False] * image_tokens.shape[1]

        # --- Subtask tokens (from HL policy, e.g. "pick up the cup") ---
        if obs.tokenized_subtask is not None:
            subtask_emb = self.PaliGemma.llm(obs.tokenized_subtask, method="embed")
            tokens.append(subtask_emb)
            input_mask.append(obs.tokenized_subtask_mask)
            ar_mask += [False] * subtask_emb.shape[1]

        # --- Goal / free-text prompt tokens ---
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
        """Embed suffix for LL policy: K state tokens + action tokens.

        AR boundary: first state token starts a new AR block (prefix cannot attend
        to suffix). Remaining state tokens share that block. Action tokens form
        their own block (attend to state + each other).

        Returns adarms_cond when pi05=True (adaRMSNorm path), None otherwise.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # --- K proprioceptive state tokens (one per video frame) ---
        if obs.video_states is not None:
            state_tokens = self.state_proj(obs.video_states)  # [b, K, d_expert]
        else:
            state_tokens = self.state_proj(obs.state)[:, None, :]  # [b, 1, d_expert]
        tokens.append(state_tokens)
        input_mask.append(jnp.ones(state_tokens.shape[:2], dtype=jnp.bool_))
        # First state token is the AR boundary; remaining share the same block.
        ar_mask += [True] + [False] * (state_tokens.shape[1] - 1)

        # --- Action tokens with flow-matching timestep ---
        action_tokens = self.action_in_proj(noisy_actions)  # [b, H, d_expert]
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )  # [b, d_expert]

        if self.pi05:
            # adaRMSNorm path: time MLP produces conditioning signal
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # MLP-concat path (pi0 legacy)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # Action tokens form a new AR block; they can attend to state tokens.
        ar_mask += [True] + [False] * (self.action_horizon - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # -------------------------------------------------------------------------
    # HL (high-level) policy: single-frame SigLIP + memory → subtask + memory text
    # -------------------------------------------------------------------------

    @at.typecheck
    def embed_prefix_hl(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed prefix for HL policy: single-frame image + memory + goal.

        All prefix tokens use bidirectional attention (ar_mask=False). Uses the
        standard single-frame SigLIP encoder, NOT the video encoder.
        """
        input_mask = []
        ar_mask: list[bool] = []
        tokens = []

        # Single-frame image tokens (standard SigLIP, not VideoViT)
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Memory tokens (episodic memory from prior HL steps)
        if obs.tokenized_memory is not None:
            memory_emb = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
            tokens.append(memory_emb)
            input_mask.append(obs.tokenized_memory_mask)
            ar_mask += [False] * memory_emb.shape[1]

        # Goal / free-text prompt tokens
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
    def compute_loss_hl(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        target_tokens: at.Int[at.Array, "b t"],
        target_mask: at.Bool[at.Array, "b t"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, " b"]:
        """HL policy loss: cross-entropy on next-token prediction of subtask + memory text.

        Uses teacher forcing: the target sequence is fed as input and the model
        is trained to predict the next token at each position.
        """
        observation = _model.preprocess_observation(rng, observation, train=train)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_hl(observation)

        # Embed target tokens as suffix (teacher forcing)
        target_emb = self.PaliGemma.llm(target_tokens, method="embed")  # [b, t, d]

        # Build masks: prefix is bidirectional, target forms a single causal AR block
        # (target tokens can attend to each other and all prefix tokens)
        suffix_ar_mask = jnp.ones(target_tokens.shape[1], dtype=jnp.bool_)
        input_mask = jnp.concatenate([prefix_mask, target_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, full_ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Forward pass through the first expert only (PaliGemma, not action expert)
        (hl_output, _), _ = self.PaliGemma.llm(
            [jnp.concatenate([prefix_tokens, target_emb], axis=1), None],
            mask=attn_mask,
            positions=positions,
        )

        # Slice output positions corresponding to target tokens and project to logits
        prefix_len = prefix_tokens.shape[1]
        target_output = hl_output[:, prefix_len:, :]  # [b, t, d]
        logits = self.PaliGemma.llm(target_output, method="decode_logits")  # [b, t, vocab]

        # Next-token prediction: logits[i] predicts target[i+1]
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

    def predict_subtask_and_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_new_tokens: int = 64,
    ) -> at.Int[at.Array, "b t"]:
        """Autoregressively generate subtask + memory text using the HL policy.

        Uses greedy decoding. This method runs a Python-level loop (no KV cache),
        so it should be called outside of jit when max_new_tokens is variable.

        Returns:
            Generated token IDs of shape [b, max_new_tokens], NOT including the
            leading BOS token.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_hl(observation)
        prefix_len = prefix_tokens.shape[1]

        # Seed the generated sequence with BOS
        generated = jnp.full((batch_size, 1), _PALIGEMMA_BOS_TOKEN_ID, dtype=jnp.int32)

        for _ in range(max_new_tokens):
            gen_len = generated.shape[1]

            # Embed all generated tokens so far
            gen_emb = self.PaliGemma.llm(generated, method="embed")  # [b, gen_len, d]

            # Build attention mask: prefix is bidirectional, each generated token is
            # its own AR block (strictly causal within generated sequence)
            gen_mask = jnp.ones((batch_size, gen_len), dtype=jnp.bool_)
            gen_ar_mask = jnp.ones(gen_len, dtype=jnp.bool_)  # one block per token
            input_mask = jnp.concatenate([prefix_mask, gen_mask], axis=1)
            full_ar_mask = jnp.concatenate([prefix_ar_mask, gen_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, full_ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (hl_output, _), _ = self.PaliGemma.llm(
                [jnp.concatenate([prefix_tokens, gen_emb], axis=1), None],
                mask=attn_mask,
                positions=positions,
            )

            # Get logits at the last generated position and sample greedily
            last_hidden = hl_output[:, prefix_len + gen_len - 1 : prefix_len + gen_len, :]
            logits = self.PaliGemma.llm(last_hidden, method="decode_logits")  # [b, 1, vocab]
            next_token = jnp.argmax(logits[:, 0, :], axis=-1, keepdims=True)  # [b, 1]
            generated = jnp.concatenate([generated, next_token], axis=1)

            if jnp.all(next_token == _PALIGEMMA_EOS_TOKEN_ID):
                break

        # Return generated tokens, stripping the leading BOS token
        return generated[:, 1:]

    @at.typecheck
    def compute_loss_fast(
        self,
        obs: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_ar_mask: at.Bool[at.Array, " s"],
    ) -> at.Float[at.Array, " b"]:
        """FAST discrete-action cross-entropy through the VLM backbone.

        The action postfix is causal; loss is next-token CE on the FAST tokens.
        Trains the backbone (+ LoRA) and, via prefix_tokens, the video encoder.
        """
        action_tokens = obs.tokenized_action
        action_emb = self.PaliGemma.llm(action_tokens, method="embed")
        suffix_ar_mask = jnp.ones(action_tokens.shape[1], dtype=jnp.bool_)
        input_mask = jnp.concatenate([prefix_mask, obs.tokenized_action_mask], axis=1)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, full_ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (out, _), _ = self.PaliGemma.llm(
            [jnp.concatenate([prefix_tokens, action_emb], axis=1), None],
            mask=attn_mask,
            positions=positions,
        )
        prefix_len = prefix_tokens.shape[1]
        action_out = out[:, prefix_len:, :]
        logits = self.PaliGemma.llm(action_out, method="decode_logits")
        shifted_logits = logits[:, :-1, :]
        shifted_targets = action_tokens[:, 1:]
        shifted_mask = obs.tokenized_action_loss_mask[:, 1:]
        log_probs = jax.nn.log_softmax(shifted_logits, axis=-1)
        token_losses = -jnp.take_along_axis(log_probs, shifted_targets[:, :, None], axis=-1).squeeze(-1)
        return jnp.sum(token_losses * shifted_mask, axis=-1) / jnp.maximum(jnp.sum(shifted_mask, axis=-1), 1)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        hl_targets: at.Int[at.Array, "b t"] | None = None,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng, hl_rng = jax.random.split(rng, 4)
        observation_ll = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Encode the LL prefix once (video encoded here, with gradient for the FAST pass).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ll(observation_ll)

        # --- Flow pass (insulated): flow expert attends to a stop-grad'd prefix KV. ---
        sg_prefix = jax.lax.stop_gradient(prefix_tokens)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([sg_prefix, None], mask=prefix_attn_mask, positions=prefix_positions)
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_ll(observation_ll, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_for_suffix, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        ll_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [b, ah]

        total_loss = self.config.ll_loss_weight * ll_loss

        # --- FAST CE pass (trains backbone + video encoder). ---
        if observation_ll.tokenized_action is not None and self.config.fast_loss_weight > 0:
            fast_loss = self.compute_loss_fast(observation_ll, prefix_tokens, prefix_mask, prefix_ar_mask)  # [b]
            total_loss = total_loss + self.config.fast_loss_weight * fast_loss[:, None]

        # --- HL CE (unchanged; uses the raw observation / single-frame prefix). ---
        if hl_targets is not None and self.config.hl_loss_weight > 0:
            hl_targets_mask = jnp.ones_like(hl_targets, dtype=jnp.bool_)
            hl_loss = self.compute_loss_hl(hl_rng, observation, hl_targets, hl_targets_mask, train=train)  # [b]
            total_loss = total_loss + self.config.hl_loss_weight * hl_loss[:, None]

        return total_loss

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
        # Convention: t=1 is noise, t=0 is the target distribution.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with a single prefix forward pass
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ll(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_ll(
                observation, x_t, jnp.broadcast_to(time, (batch_size,))
            )
            # How suffix tokens attend to each other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # How suffix tokens attend to the cached prefix
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            # Full mask: [b, suffix_len, prefix_len + suffix_len]
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
