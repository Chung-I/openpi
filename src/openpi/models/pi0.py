import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _broadcast_kv_cache(kv_cache: _gemma.KVCache, num_branches: int) -> _gemma.KVCache:
    """Repeats each batch element's prefix KV `num_branches` times along the batch axis.

    KVCache layout is [layers, batch, seq, kv_heads, head_dim]; `jnp.repeat` on axis 1 lays
    the replicas out b-major (element i lands at rows i*num_branches ... i*num_branches +
    num_branches - 1), matching the `(b k) -> b k` einops flattening used by the shared-obs
    suffix batch. This is how the ONE prefix forward pass is shared across all temporal-offset
    branches: the broadcast is differentiable, so prefix (PaliGemma) weights still receive
    gradients from every branch through the cached keys/values.
    """
    cache_k, cache_v = kv_cache
    return jnp.repeat(cache_k, num_branches, axis=1), jnp.repeat(cache_v, num_branches, axis=1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.state_cond = config.state_cond
        self.vlash_shared_obs = config.vlash_shared_obs
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            if config.state_cond:
                self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                self.state_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                self.state_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        return self._embed_suffix(obs.state, noisy_actions, timestep)

    @at.typecheck
    def _embed_suffix(
        self,
        state: at.Float[at.Array, "b st"],
        noisy_actions: at.Float[at.Array, "b ah ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """State-array variant of `embed_suffix` so the shared-obs path can feed per-branch
        rolled states (flattened to the batch axis) without constructing a fake Observation."""
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
            if self.state_cond:
                state_emb = self.state_proj(state)
                state_emb = nnx.swish(self.state_mlp_in(state_emb))
                state_emb = nnx.swish(self.state_mlp_out(state_emb))
                adarms_cond = adarms_cond + state_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        noise: at.Float[at.Array, "*b ah ad"] | None = None,
        time: at.Float[at.Array, "*b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        """Flow-matching training loss.

        When `config.vlash_shared_obs` is set, dispatches to `compute_loss_shared_obs`:
        `actions` are then [b, k, ah, ad] (one action chunk per temporal-offset branch) and
        `observation.vlash_states` must hold the per-branch rolled states [b, k, s]; the
        returned loss is [b, k, ah], so the caller's `jnp.mean` averages over branches.

        `noise`/`time` default to being sampled from `rng` and are injectable for tests that
        need bit-identical flow-matching inputs across separate calls.
        """
        if self.vlash_shared_obs:
            return self.compute_loss_shared_obs(rng, observation, actions, train=train, noise=noise, time=time)
        return self._compute_loss_single(rng, observation, actions, train=train, noise=noise, time=time)

    def _compute_loss_single(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        noise: at.Float[at.Array, "*b ah ad"] | None = None,
        time: at.Float[at.Array, "*b"] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        """Stock (single-branch) loss: one observation, one action chunk per batch element."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        if time is None:
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def compute_loss_shared_obs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: at.Float[at.Array, "b k ah ad"],
        *,
        train: bool = False,
        noise: at.Float[at.Array, "b k ah ad"] | None = None,
        time: at.Float[at.Array, "b k"] | None = None,
    ) -> at.Float[at.Array, "b k ah"]:
        """Shared-observation loss over all `k = delta_max + 1` temporal-offset branches.

        KV-broadcast design: the (images + language) prefix is embedded and run through the
        transformer ONCE at batch size `b` -- exactly as in `sample_actions`'s prefill -- and
        its KV cache is broadcast across the `(b k)`-flattened suffix batch. Each branch's
        suffix carries its own noisy action chunk and its own adarms_cond computed from that
        branch's rolled state (`observation.vlash_states[:, k]`), which is exactly the plain
        per-sequence cond path in gemma.py -- no per-token-group cond surgery needed. Every
        suffix query attends to its own batch row only (prefix replica + itself), so branches
        are isolated by construction; the two-pass prefix/suffix split is mathematically
        identical to the stock single-pass forward because prefix tokens never attend to the
        suffix. Loss is per-branch [b, k, ah]; the training caller's `jnp.mean` averages over
        branches.
        """
        if observation.vlash_states is None:
            raise ValueError(
                "compute_loss_shared_obs requires observation.vlash_states (per-branch rolled states "
                "[b, k, s]); is transforms_vlash.SplitVlashBranches in the model transforms?"
            )
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_size, num_branches = actions.shape[:2]
        if noise is None:
            noise = jax.random.normal(noise_rng, actions.shape)
        if time is None:
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size, num_branches)) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 1) shared prefix: ONE forward pass at batch size b, filling the KV cache (same
        # structure as sample_actions' prefill). Gradients flow into the prefix weights
        # through the cached keys/values.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

        # 2) per-branch suffix at batch size b*k; adarms_cond is per-branch via the batch dim.
        flat_states = einops.rearrange(observation.vlash_states, "b k s -> (b k) s")
        flat_x_t = einops.rearrange(x_t, "b k ah ad -> (b k) ah ad")
        flat_time = einops.rearrange(time, "b k -> (b k)")
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_suffix(flat_states, flat_x_t, flat_time)

        # 3) broadcast the prefix KV/mask across branches (b-major, matching the flattening).
        kv_cache = _broadcast_kv_cache(kv_cache, num_branches)
        prefix_mask_rep = jnp.repeat(prefix_mask, num_branches, axis=0)

        # `suffix_attn_mask` is (b*k, suffix_len, suffix_len): how suffix tokens attend to
        # each other within their own branch row.
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is (b*k, suffix_len, prefix_len): suffix tokens see their own
        # batch element's (shared) prefix.
        prefix_attn_mask = einops.repeat(prefix_mask_rep, "bk p -> bk s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask_rep, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        v_t = einops.rearrange(v_t, "(b k) ah ad -> b k ah ad", k=num_branches)

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
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
