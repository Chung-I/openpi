# Spec A2 — Knowledge Insulation (FAST head + flow insulation + LoRA) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the MEM low-level policy a discrete FAST action-token head (cross-entropy through the VLM backbone) that trains the backbone-LoRA adapters and the VideoViT encoder, while insulating the continuous flow-matching expert so its gradient never reaches the backbone or video encoder; train with LoRA on the gemma experts and full fine-tuning on the vision encoders.

**Architecture:** Two-pass `compute_loss`: (1) a FAST CE pass — backbone over `[LL-prefix · FAST-action-postfix]`, mirroring the existing `compute_loss_hl`; (2) a flow pass — prefix run into a `kv_cache` that is wrapped in `jax.lax.stop_gradient`, with the flow suffix attending to it, mirroring `sample_actions`. The video encoder is encoded once (with grad for the FAST pass; stop-grad'd for the flow pass). LoRA via `freeze_filter`; insulation via in-forward `stop_gradient`.

**Tech Stack:** JAX, Flax NNX, `physical-intelligence/fast` FAST tokenizer, pytest, `uv`.

## Global Constraints

- Run everything via `uv run`; prefix JAX commands with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- Branch: `mem-fidelity-fixes` (off `mem-pi05-rebase`; Spec A already committed at `689dce2`).
- `sample_actions` MUST remain unchanged (inference uses the flow expert; FAST is training-only).
- Do NOT edit `gemma.py` (two-pass design avoids gemma internals).
- Insulation requirement (verbatim from spec): flow-matching loss gradient must reach only the action expert (`*_1`) + projections (`state_proj`/`action_in_proj`/`action_out_proj`/`time_mlp_*`); it must NOT reach gemma backbone (expert-0, non-`_1`) params, the text-embed table, or `video_img`. The FAST CE loss is what trains backbone + `video_img`.
- LoRA freeze: gemma base frozen, `*lora*` trainable, vision encoders (`img`/`video_img`) + projections trainable (full-FT).
- Loss combination keeps the existing return shape `[b, action_horizon]`; per-example scalar losses (FAST CE, HL CE) are added as `loss[:, None]`.

---

### Task 1: FAST action-token fields on Observation

**Files:**
- Modify: `src/openpi/models/model.py` (Observation dataclass `:107` area, `from_dict`, `to_dict`, `preprocess_observation`)
- Modify: `src/openpi/models/pi0_mem_config.py` (`inputs_spec`)
- Test: `src/openpi/models/pi0_mem_config_test.py`

**Interfaces:**
- Produces: `Observation.tokenized_action: Int[*b, fa] | None`, `Observation.tokenized_action_mask: Bool[*b, fa] | None`, `Observation.tokenized_action_loss_mask: Bool[*b, fa] | None`. `Pi0MEMConfig.max_action_tokens: int = 256` (added here; used by inputs_spec). `fake_obs(b)` returns these as `[b, max_action_tokens]`.

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/models/pi0_mem_config_test.py`:

```python
def test_fake_obs_has_fast_action_fields():
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    obs = config.fake_obs(2)
    assert obs.tokenized_action is not None
    assert obs.tokenized_action.shape == (2, config.max_action_tokens)
    assert obs.tokenized_action_mask.shape == (2, config.max_action_tokens)
    assert obs.tokenized_action_loss_mask.shape == (2, config.max_action_tokens)
```

(Ensure the file imports `Pi0MEMConfig` — it already does in the existing tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_config_test.py::test_fake_obs_has_fast_action_fields -v`
Expected: FAIL — `AttributeError: 'Pi0MEMConfig' object has no attribute 'max_action_tokens'` (or `tokenized_action` is None).

- [ ] **Step 3: Add the Observation fields**

In `src/openpi/models/model.py`, in the `Observation` dataclass MEM-fields block (after `video_states`), add:

```python
    # FAST discrete action tokens (for the knowledge-insulation FAST head).
    tokenized_action: at.Int[ArrayT, "*b fa"] | None = None
    tokenized_action_mask: at.Bool[ArrayT, "*b fa"] | None = None
    tokenized_action_loss_mask: at.Bool[ArrayT, "*b fa"] | None = None
```

In `from_dict`, add (alongside the other MEM `data.get(...)` lines):

```python
            tokenized_action=data.get("tokenized_action"),
            tokenized_action_mask=data.get("tokenized_action_mask"),
            tokenized_action_loss_mask=data.get("tokenized_action_loss_mask"),
```

In `preprocess_observation`, add to the reconstructed `Observation(...)` (alongside the other MEM fields):

```python
        tokenized_action=observation.tokenized_action,
        tokenized_action_mask=observation.tokenized_action_mask,
        tokenized_action_loss_mask=observation.tokenized_action_loss_mask,
```

`to_dict` uses `dataclasses.asdict` and only special-cases `images`/`video_images`; the new fields pass through unchanged — no edit needed there.

- [ ] **Step 4: Add `max_action_tokens` + inputs_spec entries**

In `src/openpi/models/pi0_mem_config.py`, add to the dataclass fields (near `max_subtask_tokens`):

```python
    max_action_tokens: int = 256
```

In `inputs_spec`, after the `video_states` spec and inside the `Observation(...)` block, add:

```python
                tokenized_action=jax.ShapeDtypeStruct([batch_size, self.max_action_tokens], jnp.int32),
                tokenized_action_mask=jax.ShapeDtypeStruct([batch_size, self.max_action_tokens], jnp.bool_),
                tokenized_action_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_action_tokens], jnp.bool_),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_config_test.py -v`
Expected: PASS (new test + existing config tests).

- [ ] **Step 6: Run the existing MEM model tests for regressions**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -v`
Expected: PASS (the new optional fields default to None and are not yet consumed by the model).

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/model.py src/openpi/models/pi0_mem_config.py src/openpi/models/pi0_mem_config_test.py
git commit -m "$(cat <<'EOF'
feat(pi0_mem): add FAST action-token fields to Observation (Spec A2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 2: FAST action tokenization (tokenizer method + transform)

**Files:**
- Modify: `src/openpi/models/tokenizer.py` (add `FASTTokenizer.tokenize_actions`)
- Modify: `src/openpi/transforms.py` (add `TokenizeFASTActions` transform)
- Test: `src/openpi/models/tokenizer_test.py`

**Interfaces:**
- Produces: `FASTTokenizer.tokenize_actions(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(tokens, mask, loss_mask)` each shape `[max_len]`. `transforms.TokenizeFASTActions(tokenizer, max_len)` — a `DataTransformFn` that reads `data["actions"]` and writes `data["tokenized_action"]`, `data["tokenized_action_mask"]`, `data["tokenized_action_loss_mask"]`.

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/models/tokenizer_test.py`:

```python
import numpy as np

from openpi.models.tokenizer import FASTTokenizer


def test_fast_tokenize_actions_shapes_and_postfix():
    tok = FASTTokenizer(max_len=64)
    actions = np.zeros((10, 7), dtype=np.float32)
    tokens, mask, loss_mask = tok.tokenize_actions(actions)
    assert tokens.shape == (64,)
    assert mask.shape == (64,)
    assert loss_mask.shape == (64,)
    # Loss is only on real postfix tokens, which are exactly the masked-in tokens.
    assert bool(loss_mask.any())
    assert not bool(loss_mask[~mask].any())  # no loss on padding
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/tokenizer_test.py::test_fast_tokenize_actions_shapes_and_postfix -v`
Expected: FAIL — `AttributeError: 'FASTTokenizer' object has no attribute 'tokenize_actions'`.

- [ ] **Step 3: Add `tokenize_actions` to FASTTokenizer**

In `src/openpi/models/tokenizer.py`, add this method to `FASTTokenizer` (it reuses the existing `self._fast_tokenizer` and `self._act_tokens_to_paligemma_tokens`, and the `"Action: " + FAST + "|"` postfix convention already used in `tokenize`):

```python
    def tokenize_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize an action chunk into the FAST postfix used by the MEM FAST head.

        Returns (tokens, mask, loss_mask), each of length self._max_len. The
        postfix is causal and is the only region with loss. There is no prompt/
        state prefix here — the MEM model supplies its own (video/subtask/goal)
        prefix; this produces only the discrete action postfix.
        """
        action_tokens = self._fast_tokenizer(actions[None])[0]
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
        postfix = (
            self._paligemma_tokenizer.encode("Action: ")
            + action_tokens_in_pg.tolist()
            + self._paligemma_tokenizer.encode("|", add_eos=True)
        )
        tokens = postfix[: self._max_len]
        mask = [True] * len(tokens)
        loss_mask = [True] * len(tokens)
        pad = self._max_len - len(tokens)
        if pad > 0:
            tokens = tokens + [0] * pad
            mask = mask + [False] * pad
            loss_mask = loss_mask + [False] * pad
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool), np.asarray(loss_mask, dtype=bool)
```

- [ ] **Step 4: Run the tokenizer test to verify it passes**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/tokenizer_test.py::test_fast_tokenize_actions_shapes_and_postfix -v`
Expected: PASS.

- [ ] **Step 5: Add the `TokenizeFASTActions` transform**

In `src/openpi/transforms.py`, add (follow the existing `DataTransformFn` dataclass style; `TokenizePrompt` in the same file is the reference):

```python
@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        tokens, mask, loss_mask = self.tokenizer.tokenize_actions(np.asarray(data["actions"]))
        return {
            **data,
            "tokenized_action": tokens,
            "tokenized_action_mask": mask,
            "tokenized_action_loss_mask": loss_mask,
        }
```

If `transforms.py` does not already import the tokenizer module, add `import openpi.models.tokenizer as _tokenizer` near the other imports. (Check the top of the file first; `TokenizePrompt` already references a tokenizer, so the import likely exists.)

- [ ] **Step 6: Write + run a transform test**

Append to `src/openpi/models/tokenizer_test.py`:

```python
def test_tokenize_fast_actions_transform_writes_fields():
    from openpi import transforms

    tok = FASTTokenizer(max_len=64)
    tf = transforms.TokenizeFASTActions(tok)
    out = tf({"actions": np.zeros((10, 7), dtype=np.float32)})
    assert out["tokenized_action"].shape == (64,)
    assert out["tokenized_action_mask"].shape == (64,)
    assert out["tokenized_action_loss_mask"].shape == (64,)
```

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/tokenizer_test.py -v`
Expected: PASS (both new tests + existing).

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/tokenizer.py src/openpi/transforms.py src/openpi/models/tokenizer_test.py
git commit -m "$(cat <<'EOF'
feat(tokenizer): FAST action-postfix tokenization + transform (Spec A2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 3: FAST CE head + insulated two-pass compute_loss

**Files:**
- Modify: `src/openpi/models/pi0_mem.py` (add `compute_loss_fast`; restructure `compute_loss`)
- Test: `src/openpi/models/pi0_mem_test.py`

**Interfaces:**
- Consumes: `embed_prefix_ll(obs) -> (prefix_tokens, prefix_mask, prefix_ar_mask)`, `embed_suffix_ll(obs, x_t, time) -> (suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond)`, `make_attn_mask`, `self.PaliGemma.llm` (methods `embed`, `decode_logits`; call forms `[e0, e1]` with `kv_cache=`), `Observation.tokenized_action*` (Task 1).
- Produces: `Pi0MEM.compute_loss_fast(obs, prefix_tokens, prefix_mask, prefix_ar_mask) -> Float[b]`. `compute_loss(rng, obs, actions, *, hl_targets=None, train=False) -> Float[b, action_horizon]` with FAST term gated on `obs.tokenized_action is not None and self.config.fast_loss_weight > 0`, and the flow term computed against a stop-grad'd prefix KV.

- [ ] **Step 1: Write the failing tests**

Append to `src/openpi/models/pi0_mem_test.py`:

```python
import flax.nnx as nnx


def _fast_action_obs(config, batch_size):
    # fake_obs provides tokenized_action (ones); make the loss mask non-empty.
    obs = config.fake_obs(batch_size)
    import dataclasses
    return dataclasses.replace(
        obs,
        tokenized_action_loss_mask=jnp.ones_like(obs.tokenized_action_loss_mask),
    )


def test_pi0_mem_fast_loss_shape_finite():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = _fast_action_obs(config, 2)
    prefix = model.embed_prefix_ll(obs)
    loss = nnx_utils.module_jit(model.compute_loss_fast)(obs, *prefix)
    assert loss.shape == (2,)
    assert jnp.all(jnp.isfinite(loss))


def _grad_abs_by_path(model, scalar_loss_fn):
    graphdef, params = nnx.split(model, nnx.Param)
    grads = jax.grad(lambda p: scalar_loss_fn(nnx.merge(graphdef, p)))(params)
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        key = "/".join(
            str(getattr(p, "key", getattr(p, "idx", p))) for p in path
        )
        out[key] = float(jnp.sum(jnp.abs(leaf)))
    return out


def _is_backbone(path: str) -> bool:
    # gemma expert-0 (backbone): under llm, NOT an action-expert (_1) leaf, NOT lora.
    return "llm" in path and "_1" not in path


def test_flow_loss_does_not_touch_backbone_or_video(self_key=1):
    key = jax.random.key(self_key)
    # flow-only: fast and hl weights zero.
    config = Pi0MEMConfig(
        paligemma_variant="dummy", action_expert_variant="dummy",
        fast_loss_weight=0.0, hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    for path, g in norms.items():
        if _is_backbone(path) or "video_img" in path:
            assert g == 0.0, f"flow loss leaked into {path}: {g}"
    # action expert / projections must receive gradient
    assert any(g > 0 for p, g in norms.items() if "_1" in p or "proj" in p or "time_mlp" in p)


def test_fast_loss_trains_backbone_and_video():
    key = jax.random.key(2)
    config = Pi0MEMConfig(
        paligemma_variant="dummy", action_expert_variant="dummy",
        ll_loss_weight=0.0, hl_loss_weight=0.0,
    )
    model = config.create(key)
    obs, act = _fast_action_obs(config, 2), config.fake_act(2)
    norms = _grad_abs_by_path(model, lambda m: m.compute_loss(key, obs, act).mean())
    assert any(g > 0 for p, g in norms.items() if _is_backbone(p))
    assert any(g > 0 for p, g in norms.items() if "video_img" in p)
```

(Remove the unused `self_key` default if your linter objects — it is only there to vary the RNG; a plain `key = jax.random.key(1)` is equivalent.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_fast_loss_shape_finite -v`
Expected: FAIL — `AttributeError: ... has no attribute 'compute_loss_fast'`. (The insulation tests will also fail/error until Step 3/4.)

- [ ] **Step 3: Add `compute_loss_fast`**

In `src/openpi/models/pi0_mem.py`, add this method (mirrors `compute_loss_hl`, but consumes a pre-embedded LL prefix so video is encoded once):

```python
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
```

- [ ] **Step 4: Restructure `compute_loss` into the insulated two-pass form**

Replace the body of `compute_loss` in `src/openpi/models/pi0_mem.py` with:

```python
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
```

Note: the flow pass no longer runs the backbone with gradient, so `prefix_out` is gone; the only place `prefix_tokens` carries gradient is the FAST pass. This is what enforces insulation.

- [ ] **Step 5: Run the Task-3 tests to verify they pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -k "fast_loss or flow_loss or fast_loss_trains" -v`
Expected: PASS — `test_pi0_mem_fast_loss_shape_finite`, `test_flow_loss_does_not_touch_backbone_or_video`, `test_fast_loss_trains_backbone_and_video`.

- [ ] **Step 6: Run the full MEM suite for regressions**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py src/openpi/models/pi0_mem_integration_test.py -v`
Expected: PASS. In particular `test_pi0_mem_ll_loss` (no `tokenized_action` consumed? note: `fake_obs` now includes `tokenized_action`, so the FAST term activates — the test asserts only shape `[b, ah]`, which still holds; loss is finite).

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "$(cat <<'EOF'
feat(pi0_mem): FAST CE head + insulated two-pass compute_loss (Spec A2)

Flow-matching loss now attends to a stop_gradient'd prefix KV cache, so its
gradient cannot reach the backbone or video encoder; the new FAST cross-entropy
head trains them instead. Implements the knowledge-insulation recipe.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 4: LoRA config — flag, freeze filter, loss weight

**Files:**
- Modify: `src/openpi/models/pi0_mem_config.py` (`lora` flag, variant mapping in `create`, `get_freeze_filter`, `fast_loss_weight`)
- Test: `src/openpi/models/pi0_mem_config_test.py`

**Interfaces:**
- Produces: `Pi0MEMConfig.lora: bool = False`, `Pi0MEMConfig.fast_loss_weight: float = 1.0`, `Pi0MEMConfig.get_freeze_filter() -> nnx.filterlib.Filter`. When `lora=True`, `create` uses `gemma_2b_lora`/`gemma_300m_lora` in place of `gemma_2b`/`gemma_300m`.

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/models/pi0_mem_config_test.py`:

```python
import flax.nnx as nnx

from openpi.shared import nnx_utils


def test_get_freeze_filter_partitions_params():
    config = Pi0MEMConfig(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", lora=True)
    freeze = config.get_freeze_filter()
    trainable = nnx.All(nnx.Param, nnx.Not(freeze))
    # Representative paths (as tuples of string keys, matching nnx path nodes).
    def frozen(path_keys):
        return freeze(tuple(path_keys), object())
    def is_trainable(path_keys):
        return trainable(tuple(path_keys), object())
    # backbone base weight: frozen
    assert frozen(["PaliGemma", "llm", "layers", "attn", "kernel"]) is True
    # backbone lora adapter: trainable
    assert is_trainable(["PaliGemma", "llm", "layers", "attn", "lora_a"]) is True
    # video encoder: trainable
    assert is_trainable(["PaliGemma", "video_img", "embedding", "kernel"]) is True
```

(If `nnx_utils.PathRegex` filters expect a different call signature, adapt the path representation in this test to match how `pi0_config`'s freeze filter is exercised; the binding behavior to assert is unchanged: backbone base frozen, `*lora*` and `video_img` trainable.)

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_config_test.py::test_get_freeze_filter_partitions_params -v`
Expected: FAIL — `AttributeError: ... no attribute 'lora'` or `'get_freeze_filter'`.

- [ ] **Step 3: Add the fields, variant mapping, and freeze filter**

In `src/openpi/models/pi0_mem_config.py`:

Add imports near the top (if absent):

```python
import flax.nnx as nnx

from openpi.shared import nnx_utils
```

Add dataclass fields (near `hl_loss_weight`/`ll_loss_weight`):

```python
    lora: bool = False
    fast_loss_weight: float = 1.0
```

Leave `create` unchanged. Map the variants inside `Pi0MEM.__init__`
(`src/openpi/models/pi0_mem.py`) — replace the two existing `get_config` calls
so they honor the flag:

```python
        pg_variant = config.paligemma_variant
        ax_variant = config.action_expert_variant
        if config.lora:
            pg_variant = pg_variant if "lora" in pg_variant or pg_variant == "dummy" else pg_variant + "_lora"
            ax_variant = ax_variant if "lora" in ax_variant or ax_variant == "dummy" else ax_variant + "_lora"
        paligemma_config = _gemma.get_config(pg_variant)
        action_expert_config = _gemma.get_config(ax_variant)
```

Add the freeze filter method to `Pi0MEMConfig` (mirrors `pi0_config.get_freeze_filter`, resolving the same `_lora` suffixing so it matches what `create` builds):

```python
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        pg = self.paligemma_variant + ("_lora" if self.lora and "lora" not in self.paligemma_variant and self.paligemma_variant != "dummy" else "")
        ax = self.action_expert_variant + ("_lora" if self.lora and "lora" not in self.action_expert_variant and self.action_expert_variant != "dummy" else "")
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in pg:
            filters.append(gemma_params_filter)
            if "lora" not in ax:
                filters.append(nnx.Not(action_expert_params_filter))
            has_lora = True
        elif "lora" in ax:
            filters.append(action_expert_params_filter)
            has_lora = True
        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_config_test.py -v`
Expected: PASS.

- [ ] **Step 5: Run the MEM model tests for regressions**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -v`
Expected: PASS (tests use `dummy`, `lora=False`; variant mapping is a no-op for `dummy`).

- [ ] **Step 6: Commit**

```bash
git add src/openpi/models/pi0_mem_config.py src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_config_test.py
git commit -m "$(cat <<'EOF'
feat(pi0_mem): LoRA flag, freeze filter, fast_loss_weight (Spec A2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 5: Wire FAST transform + LoRA training config + smoke test

**Files:**
- Modify: `src/openpi/training/config.py` (`PI0_MEM` case in `ModelTransformFactory`; add `pi0_mem_lora_debug` config)
- Test: `src/openpi/training/config_test.py` (or the existing config-registry test location)

**Interfaces:**
- Consumes: `transforms.TokenizeFASTActions` (Task 2), `_tokenizer.FASTTokenizer`, `Pi0MEMConfig(lora=True).get_freeze_filter()` (Task 4).
- Produces: registry entry `pi0_mem_lora_debug`.

- [ ] **Step 1: Write the failing test**

Append to the config test file (use the same file/pattern existing config tests use; if `_CONFIGS_DICT`/`get_config` exists, assert on it):

```python
def test_pi0_mem_lora_debug_registered():
    from openpi.training import config as _config
    cfg = _config.get_config("pi0_mem_lora_debug")
    assert cfg.model.lora is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k pi0_mem_lora_debug -v`
Expected: FAIL — config name not found.

- [ ] **Step 3: Add FAST tokenization to the PI0_MEM transform group**

In `src/openpi/training/config.py`, in `ModelTransformFactory`'s `PI0_MEM` case, append a `TokenizeFASTActions` to the `inputs` list (after `PadStatesAndActions`):

```python
                        _transforms.TokenizeFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_action_tokens),
                        ),
```

- [ ] **Step 4: Add the `pi0_mem_lora_debug` config**

In the `_CONFIGS` list in `src/openpi/training/config.py`, add (mirrors the existing `pi0_mem_debug`, plus `lora=True` and the freeze filter):

```python
    TrainConfig(
        name="pi0_mem_lora_debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_mem_config.Pi0MEMConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            num_video_frames=2,
            lora=True,
        ),
        freeze_filter=pi0_mem_config.Pi0MEMConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            num_video_frames=2,
            lora=True,
        ).get_freeze_filter(),
        save_interval=100,
        overwrite=True,
        exp_name="pi0_mem_lora_debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
```

Note: with `dummy` variants the freeze filter resolves to `nnx.Nothing` (no LoRA params exist for `dummy`), so this debug config exercises the transform + training plumbing rather than real LoRA freezing; real LoRA freezing is covered by Task 4's filter test. (A real LoRA run uses `gemma_2b`/`gemma_300m` + `lora=True`.)

- [ ] **Step 5: Run the config test to verify it passes**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest -k pi0_mem_lora_debug -v`
Expected: PASS.

- [ ] **Step 6: Smoke-train end-to-end**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train.py pi0_mem_lora_debug`
Expected: completes 10 steps; loss finite (FAST + flow terms both active because `FakeDataConfig` + the transform supply `tokenized_action`); no trace/shape errors.

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/config.py src/openpi/training/config_test.py
git commit -m "$(cat <<'EOF'
feat(config): wire FAST tokenization + pi0_mem_lora_debug (Spec A2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- FAST head (spec A2.2 step 2, A2.5 FAST loss) → Task 3 `compute_loss_fast`.
- Flow insulation (A2.1, A2.2 step 3, Global Constraints) → Task 3 two-pass `compute_loss` + insulation gradient tests.
- Observation FAST fields + inputs_spec (A2.3) → Task 1.
- FAST tokenization transform (A2.3) → Task 2.
- LoRA flag + `get_freeze_filter` + `fast_loss_weight` + `max_action_tokens` (A2.4) → Tasks 1 (`max_action_tokens`) + 4.
- FAST transform wired + `pi0_mem_lora_debug` (A2.4, A2.5 smoke) → Task 5.
- Verification: insulation grad tests (Task 3), FAST loss shape/finite (Task 3), freeze-filter logic (Task 4), smoke train (Task 5), regression (Tasks 1/3/4). All covered.

**Placeholder scan:** No TBD/TODO; each code step shows complete code; each run step has an exact command + expected result. Two notes intentionally flag environment-dependent adaptation (nnx path-key representation in Tasks 3/4 grad/filter tests, and config-test location in Task 5) with the binding behavior stated — these are not placeholders but guardrails for codebase-specific API shapes the implementer must match.

**Type consistency:** `tokenized_action`/`tokenized_action_mask`/`tokenized_action_loss_mask` named identically across model.py, config inputs_spec, tokenizer transform, and `compute_loss_fast`. `compute_loss_fast(obs, prefix_tokens, prefix_mask, prefix_ar_mask) -> [b]` matches its call in `compute_loss`. `lora`/`fast_loss_weight`/`max_action_tokens`/`get_freeze_filter` names match across Tasks 1/4/5. `tokenize_actions -> (tokens, mask, loss_mask)` matches `TokenizeFASTActions`.
