"""MEMPolicy: inference-time orchestration of HL (high-level) and LL (low-level) policies.

The HL policy (predict_subtask_and_memory) runs every `hl_interval_steps` steps to update
the current subtask instruction and episodic memory.  The LL policy (sample_actions) runs
every step to generate action chunks conditioned on the current subtask and memory.

State maintained across steps:
    memory      -- free-text episodic memory string from the most recent HL call
    subtask     -- free-text subtask instruction string from the most recent HL call
    step_count  -- monotonically increasing counter, reset at episode boundaries
"""

import dataclasses
import logging
import re

import jax
import jax.numpy as jnp
import sentencepiece

from openpi.models import model as _model
from openpi.models.pi0_mem import Pi0MEM
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.shared import array_typing as at
import openpi.shared.download as download

logger = logging.getLogger("openpi")

# PaliGemma tokenizer special token IDs (SentencePiece / Gemma vocab).
_PALIGEMMA_EOS_TOKEN_ID: int = 1
_PADDING_TOKEN_ID: int = 0


class MEMPolicy:
    """Inference policy for MEM that orchestrates HL and LL policies.

    The HL policy runs periodically to update subtask instructions and episodic memory.
    The LL policy runs every step to generate actions.

    Usage::

        policy = MEMPolicy(model, config, hl_interval_steps=30)
        policy.reset()                        # start of episode
        for obs in episode:
            actions = policy.step(rng, obs)   # actions: [1, action_horizon, action_dim]
    """

    def __init__(
        self,
        model: Pi0MEM,
        config: Pi0MEMConfig,
        *,
        hl_interval_steps: int = 30,
        max_new_tokens: int = 64,
        num_flow_steps: int = 10,
        use_rtc: bool = False,
        inference_delay: int = 1,
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
    ):
        """
        Args:
            model: Initialised Pi0MEM model.
            config: Pi0MEMConfig used to create the model.
            hl_interval_steps: How often to run the HL policy (steps between HL calls).
            max_new_tokens: Maximum tokens the HL policy generates per call.
            num_flow_steps: Number of flow-matching steps for LL action sampling.
            use_rtc: If True, use RTC (receding-time control) guided sampling after the
                first action chunk is available. Default False → identical to prior behavior.
            inference_delay: Number of steps of delay between observation and action chunk
                (passed to sample_actions_rtc).
            prefix_attention_horizon: Number of frames to attend to from the previous chunk.
                Defaults to model.action_horizon when None.
            prefix_attention_schedule: Schedule for prefix attention weights ("exp", "linear",
                etc.) passed to sample_actions_rtc.
            max_guidance_weight: Maximum classifier-free guidance weight for RTC sampling.
        """
        self.model = model
        self.config = config
        self.hl_interval_steps = hl_interval_steps
        self.max_new_tokens = max_new_tokens
        self.num_flow_steps = num_flow_steps
        self.use_rtc = use_rtc
        self.inference_delay = inference_delay
        self.prefix_attention_horizon = (
            prefix_attention_horizon if prefix_attention_horizon is not None else model.action_horizon
        )
        self.prefix_attention_schedule = prefix_attention_schedule
        self.max_guidance_weight = max_guidance_weight
        self.prev_action_chunk = None

        # Episode state — cleared by reset().
        self.memory: str = ""
        self.subtask: str = ""
        self.step_count: int = 0

        # Sentencepiece tokenizer for encoding text → token IDs and decoding HL output.
        # The model file is cached at ~/.cache/openpi/ after the first download.
        path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            self._sp: sentencepiece.SentencePieceProcessor = (
                sentencepiece.SentencePieceProcessor(model_proto=f.read())
            )

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear memory state for a new episode."""
        self.memory = ""
        self.subtask = ""
        self.step_count = 0
        self.prev_action_chunk = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_run_hl(self) -> bool:
        return self.step_count % self.hl_interval_steps == 0

    def _encode_text(self, text: str, max_len: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode *text* into token IDs and a boolean mask, both shaped [1, max_len].

        Empty text returns all-zero tokens with an all-False mask (the model ignores
        these tokens).
        """
        if not text:
            return (
                jnp.zeros((1, max_len), dtype=jnp.int32),
                jnp.zeros((1, max_len), dtype=jnp.bool_),
            )
        ids: list[int] = self._sp.encode(text)
        n = min(len(ids), max_len)
        padded = ids[:n] + [_PADDING_TOKEN_ID] * (max_len - n)
        token_ids = jnp.array(padded, dtype=jnp.int32)[None, :]   # [1, max_len]
        mask = jnp.array(
            [True] * n + [False] * (max_len - n), dtype=jnp.bool_
        )[None, :]  # [1, max_len]
        return token_ids, mask

    def _decode_hl_tokens(self, token_ids: jnp.ndarray) -> str:
        """Decode HL output token IDs to a text string.

        Args:
            token_ids: Integer array of shape [b, t].  Only the first batch
                element is decoded.  EOS (1) and padding (0) tokens are filtered.

        Returns:
            Decoded text string, or "" on failure.
        """
        ids = [
            int(x)
            for x in token_ids[0].tolist()
            if int(x) not in (_PADDING_TOKEN_ID, _PALIGEMMA_EOS_TOKEN_ID)
        ]
        if not ids:
            return ""
        try:
            return self._sp.decode(ids)
        except Exception as exc:
            logger.warning("MEMPolicy: failed to decode HL tokens: %s", exc)
            return ""

    @staticmethod
    def _parse_hl_output(text: str) -> tuple[str, str]:
        """Extract subtask and memory from HL output text.

        Expected format (XML-like tags)::

            <subtask>pick up the cup</subtask><memory>already opened the drawer</memory>

        Returns:
            (subtask, memory) — both default to "" if the tag is absent.
        """
        subtask_match = re.search(r"<subtask>(.*?)</subtask>", text, re.DOTALL)
        memory_match = re.search(r"<memory>(.*?)</memory>", text, re.DOTALL)
        subtask = subtask_match.group(1).strip() if subtask_match else ""
        memory = memory_match.group(1).strip() if memory_match else ""
        return subtask, memory

    def _build_observation(
        self,
        observation: _model.Observation,
        memory: str,
        subtask: str,
    ) -> _model.Observation:
        """Return a copy of *observation* with memory and subtask token fields updated."""
        mem_tokens, mem_mask = self._encode_text(memory, self.config.max_memory_tokens)
        sub_tokens, sub_mask = self._encode_text(subtask, self.config.max_subtask_tokens)
        return dataclasses.replace(
            observation,
            tokenized_memory=mem_tokens,
            tokenized_memory_mask=mem_mask,
            tokenized_subtask=sub_tokens,
            tokenized_subtask_mask=sub_mask,
        )

    # ------------------------------------------------------------------
    # Inference step
    # ------------------------------------------------------------------

    def step(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ) -> _model.Actions:
        """Run one step of the MEM policy.

        1. Injects the current (previous-step) memory and subtask into the observation.
        2. If it is time (every ``hl_interval_steps`` steps), runs the HL policy to
           update ``self.subtask`` and ``self.memory``, then refreshes the observation.
        3. Runs the LL policy to sample an action chunk.

        Args:
            rng: JAX PRNG key.
            observation: Current robot observation (images, state, …).

        Returns:
            Actions array of shape [batch_size, action_horizon, action_dim].
        """
        # Inject previous subtask/memory tokens so both HL and LL can see them.
        obs_with_ctx = self._build_observation(observation, self.memory, self.subtask)

        if self._should_run_hl():
            hl_rng, ll_rng = jax.random.split(rng)

            hl_token_ids = self.model.predict_subtask_and_memory(
                hl_rng, obs_with_ctx, max_new_tokens=self.max_new_tokens
            )
            hl_text = self._decode_hl_tokens(hl_token_ids)
            new_subtask, new_memory = self._parse_hl_output(hl_text)

            self.subtask = new_subtask
            self.memory = new_memory
            logger.debug(
                "MEMPolicy step %d: HL → subtask=%r memory=%r",
                self.step_count,
                self.subtask,
                self.memory,
            )

            # Refresh observation with the updated subtask/memory.
            obs_with_ctx = self._build_observation(obs_with_ctx, self.memory, self.subtask)
        else:
            ll_rng = rng

        if self.use_rtc and self.prev_action_chunk is not None:
            actions = self.model.sample_actions_rtc(
                ll_rng,
                obs_with_ctx,
                prev_action_chunk=self.prev_action_chunk,
                inference_delay=self.inference_delay,
                prefix_attention_horizon=self.prefix_attention_horizon,
                prefix_attention_schedule=self.prefix_attention_schedule,
                max_guidance_weight=self.max_guidance_weight,
                num_steps=self.num_flow_steps,
            )
        else:
            actions = self.model.sample_actions(
                ll_rng, obs_with_ctx, num_steps=self.num_flow_steps
            )
        self.prev_action_chunk = actions
        self.step_count += 1
        return actions
