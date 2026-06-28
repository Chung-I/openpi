# MEM Pi0.5 Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase the MEM model from pi0 to pi0.5 by adding adaRMSNorm timestep injection, pi0.5's time MLP, and matching token length.

**Architecture:** Add a `pi05` flag to `Pi0MEMConfig` that switches the action expert's timestep conditioning from MLP-concat to adaRMSNorm — the same pattern used by `Pi0` in `pi0.py:67-100`. The VideoViT encoder, HL policy, and continuous state projection are unchanged. The `pi0_mem_debug` training config defaults to `pi05=True`.

**Tech Stack:** JAX, Flax NNX, openpi model framework

## Global Constraints

- `Pi0MEMConfig(pi05=False)` must produce identical behavior to the current (pre-rebase) implementation.
- `Pi0MEMConfig(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")` must pass all 5 integration tests.
- The `dummy` Gemma variant already supports `adarms=True` — no changes to `gemma.py` needed.
- `state_proj` stays in both pi0 and pi05 modes (continuous state, not discrete).

---

### Task 1: Config — add pi05 flag and update defaults

**Files:**
- Modify: `src/openpi/models/pi0_mem_config.py`
- Modify: `src/openpi/training/config.py:165-175` (ModelTransformFactory PI0_MEM case)
- Modify: `src/openpi/training/config.py:981-994` (pi0_mem_debug registration)

**Interfaces:**
- Produces: `Pi0MEMConfig.pi05: bool` (default `True`), `Pi0MEMConfig.discrete_state_input: bool` (default `False`), `Pi0MEMConfig.max_token_len` (default `200`)

- [ ] **Step 1: Update `Pi0MEMConfig` in `pi0_mem_config.py`**

Add the `pi05` and `discrete_state_input` fields. Change `max_token_len` to use `None` with `__post_init__` defaulting (matching `Pi0Config`'s pattern at `pi0_config.py:27-41`).

Replace lines 17-33 with:

```python
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
            object.__setattr__(self, "max_token_len", 200)
```

- [ ] **Step 2: Update `ModelTransformFactory` PI0_MEM case in `training/config.py`**

At line 165, update the PI0_MEM case to pass `discrete_state_input` (matching the PI05 case at line 127):

```python
            case _model.ModelType.PI0_MEM:
                assert isinstance(model_config, pi0_mem_config.Pi0MEMConfig)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
```

- [ ] **Step 3: Update `pi0_mem_debug` config registration**

At line 981, the config already works as-is since `pi05=True` is the new default. No change needed unless you want to be explicit. Verify the config still loads:

Run: `uv run python -c "from openpi.training.config import get_config; c = get_config('pi0_mem_debug'); print(f'pi05={c.model.pi05}, max_token_len={c.model.max_token_len}')"`
Expected: `pi05=True, max_token_len=200`

- [ ] **Step 4: Commit**

```bash
git add src/openpi/models/pi0_mem_config.py src/openpi/training/config.py
git commit -m "feat(mem): add pi05 flag and update defaults to match pi0.5"
```

---

### Task 2: Model — switch to adaRMSNorm timestep injection

**Files:**
- Modify: `src/openpi/models/pi0_mem.py`

**Interfaces:**
- Consumes: `Pi0MEMConfig.pi05` from Task 1
- Produces: `Pi0MEM.__init__` creates `time_mlp_in/out` when `pi05=True`; `embed_suffix_ll` returns non-None `adarms_cond` when `pi05=True`

- [ ] **Step 1: Update `__init__` to branch on `pi05`**

Replace lines 38-98 (the `__init__` method) with:

```python
class Pi0MEM(_model.BaseModel):
    def __init__(self, config: pi0_mem_config.Pi0MEMConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

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
                    scan=False,
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
```

- [ ] **Step 2: Update `embed_suffix_ll` to branch on `pi05`**

Replace lines 163-213 (the `embed_suffix_ll` method) with:

```python
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
        ar_mask += [True] + [False] * (self.action_horizon - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond
```

- [ ] **Step 3: Update docstring**

Replace lines 1-11 (module docstring) with:

```python
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
```

- [ ] **Step 4: Verify the config quick-check still works**

Run: `uv run python -c "from openpi.models.pi0_mem_config import Pi0MEMConfig; c = Pi0MEMConfig(paligemma_variant='dummy', action_expert_variant='dummy', num_video_frames=2); import jax; m = c.create(jax.random.key(0)); print(f'pi05={c.pi05}, has_time_mlp={hasattr(m, \"time_mlp_in\")}')"`
Expected: `pi05=True, has_time_mlp=True`

- [ ] **Step 5: Commit**

```bash
git add src/openpi/models/pi0_mem.py
git commit -m "feat(mem): switch to adaRMSNorm timestep injection for pi0.5"
```

---

### Task 3: Tests — verify all 5 integration tests pass with pi05=True

**Files:**
- Modify: `src/openpi/models/pi0_mem_integration_test.py`

**Interfaces:**
- Consumes: `Pi0MEMConfig(pi05=True)` from Tasks 1-2

- [ ] **Step 1: Run the existing tests without any test changes**

The `Pi0MEMConfig` now defaults to `pi05=True`. The 4 tests that construct `Pi0MEMConfig(paligemma_variant="dummy", ...)` directly will automatically pick up the new default. The `test_overfitting_single_batch` uses `get_config("pi0_mem_debug")` which also defaults to `pi05=True`.

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_integration_test.py -xvs 2>&1 | tail -20`
Expected: All 5 tests PASS.

- [ ] **Step 2: If tests fail, debug and fix**

The most likely failure: the `dummy` Gemma variant might not support `adarms=True`. Check by running the quick-check from Task 2 Step 4 first.

If the issue is in `gemma.py`, the `Module.__init__` needs `adarms` kwarg. Check `pi0.py:77` which already passes `adarms=config.pi05` — the `dummy` variant handles it. If it still fails, read the error and fix accordingly.

- [ ] **Step 3: Add a backward-compat test for pi05=False**

Add this test to `pi0_mem_integration_test.py` after `test_config_registered`:

```python
def test_pi0_legacy_mode():
    """Pi05=False backward compat: model creates and produces finite LL loss."""
    key = jax.random.key(0)
    config = Pi0MEMConfig(
        pi05=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        num_video_frames=2,
    )
    model = config.create(key)

    batch_size = 1
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    ll_loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert ll_loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(ll_loss))
```

Run: `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_integration_test.py -xvs 2>&1 | tail -20`
Expected: All 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/openpi/models/pi0_mem_integration_test.py
git commit -m "test(mem): verify pi0.5 rebase and add pi0 backward-compat test"
```
