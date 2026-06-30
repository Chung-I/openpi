# Spec D — Inference-time Real-Time Chunking (RTC) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inference-time real-time chunking to Pi0MEM — a guided flow sampler (`sample_actions_rtc`) that makes a new action chunk continue the previous one under inference latency — and wire it into `MEMPolicy`, with no retraining and no change to `sample_actions`/`compute_loss`.

**Architecture:** Port the kinetix `realtime_action` soft-guidance loop. Because pi0_mem's flow uses `t=1`=noise while kinetix uses `τ=1`=clean, run the guidance in the kinetix τ-frame via an adapter (`τ=1−t`, `v_τ=−v_t`); the one-step denoiser and all guidance constants then port verbatim. `MEMPolicy` keeps the previous chunk and drives the guided sampler each step.

**Tech Stack:** JAX (`jax.vjp`, `jax.lax.scan`), Flax NNX, pytest, `uv`.

## Global Constraints

- Run via `uv run`; prefix JAX/test commands with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- Branch: `mem-fidelity-fixes` (Specs A/A2/B/C committed).
- Inference-time soft-guided RTC ONLY. Training-time RTC + hard-mask are deferred (D2). Do NOT add per-position-time conditioning or a `simulated_delay` loss here.
- `sample_actions` and `compute_loss` MUST stay unchanged. No `Pi0MEMConfig` change.
- Time-convention adapter (verbatim): pi0_mem `x_t = t·noise + (1−t)·actions`, model predicts `v_t = noise − actions`. Map to kinetix frame with `τ = 1 − t` and `v_τ = −v_t`. Clean prediction (denoiser) `x_clean = x + v_τ·(1−τ)`. Guidance constants in `τ`: `inv_r2 = (τ² + (1−τ)²)/((1−τ)²)`, `c = nan_to_num((1−τ)/τ, posinf=max_guidance_weight)`, `guidance_weight = min(c·inv_r2, max_guidance_weight)`. Integrate `τ`: 0→1, `dτ = 1/num_steps`.
- `get_prefix_weights(start=inference_delay, end=prefix_attention_horizon, total=action_horizon, schedule)` — ported verbatim from `~/Codes/real-time-chunking-kinetix/src/model.py`.

---

### Task 1: `get_prefix_weights` helper

**Files:**
- Modify: `src/openpi/models/pi0_mem.py` (add module-level `get_prefix_weights`)
- Test: `src/openpi/models/pi0_mem_test.py`

**Interfaces:**
- Produces: `get_prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array` of shape `[total]`. Schedules: `"ones"`, `"zeros"`, `"linear"`, `"exp"`.

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/models/pi0_mem_test.py`:

```python
def test_get_prefix_weights_linear_matches_reference():
    from openpi.models.pi0_mem import get_prefix_weights
    import numpy as np

    w = np.asarray(get_prefix_weights(2, 6, 10, "linear"))
    np.testing.assert_allclose(w, [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0], atol=1e-6)


def test_get_prefix_weights_schedules():
    from openpi.models.pi0_mem import get_prefix_weights
    import numpy as np

    # ones: all 1 except positions >= end
    np.testing.assert_allclose(np.asarray(get_prefix_weights(0, 4, 6, "ones")), [1, 1, 1, 1, 0, 0])
    # end == 0 -> entire prefix ignored (all zeros)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(3, 0, 5, "linear")), [0, 0, 0, 0, 0])
    # zeros: 1 below start, else 0 (and 0 at/after end)
    np.testing.assert_allclose(np.asarray(get_prefix_weights(2, 5, 6, "zeros")), [1, 1, 0, 0, 0, 0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -k get_prefix_weights -v`
Expected: FAIL — `ImportError: cannot import name 'get_prefix_weights'`.

- [ ] **Step 3: Add the helper**

In `src/openpi/models/pi0_mem.py`, add at module level (near the top, after imports):

```python
def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array:
    """Prefix-attention weights for RTC guidance (ported from real-time-chunking-kinetix).

    With start=2, end=6, total=10 (schedule="linear"): [1,1,0.8,0.6,0.4,0.2,0,0,0,0].
    `start` (inclusive) is where the chunk may start changing; `end` (exclusive) is
    where it stops attending to the prefix. `end` takes precedence: if end < start,
    start is pushed down to end; if end == 0 the whole prefix is ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)
```

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -k get_prefix_weights -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "$(cat <<'EOF'
feat(pi0_mem): get_prefix_weights helper for RTC (Spec D)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 2: `Pi0MEM.sample_actions_rtc` (guided sampler)

**Files:**
- Modify: `src/openpi/models/pi0_mem.py` (add `sample_actions_rtc` method)
- Test: `src/openpi/models/pi0_mem_test.py`

**Interfaces:**
- Consumes: `get_prefix_weights` (Task 1), `embed_prefix_ll`, `embed_suffix_ll`, `make_attn_mask`, `action_out_proj`, `self.PaliGemma.llm` (KV-cache call form) — all already in the file.
- Produces: `Pi0MEM.sample_actions_rtc(self, rng, observation, *, prev_action_chunk, inference_delay, prefix_attention_horizon, prefix_attention_schedule="exp", max_guidance_weight=5.0, num_steps=10, noise=None) -> Actions` of shape `[b, action_horizon, action_dim]`.

- [ ] **Step 1: Write the failing tests**

Append to `src/openpi/models/pi0_mem_test.py`:

```python
def _rtc_model_and_obs(batch_size=1):
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    model = config.create(key)
    obs = config.fake_obs(batch_size)
    return key, config, model, obs


def test_sample_actions_rtc_shape():
    key, config, model, obs = _rtc_model_and_obs()
    prev = jnp.zeros((1, config.action_horizon, config.action_dim))
    out = nnx_utils.module_jit(model.sample_actions_rtc)(
        key, obs, prev_action_chunk=prev, inference_delay=1,
        prefix_attention_horizon=config.action_horizon, num_steps=4,
    )
    assert out.shape == (1, config.action_horizon, config.action_dim)


def test_sample_actions_rtc_guidance_off_matches_plain():
    # With prefix_attention_horizon=0 the weights are all zero -> no guidance ->
    # the tau-frame integration is numerically identical to plain sample_actions.
    key, config, model, obs = _rtc_model_and_obs()
    noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
    prev = jnp.ones((1, config.action_horizon, config.action_dim))  # irrelevant when weights==0
    plain = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=4, noise=noise)
    rtc = nnx_utils.module_jit(model.sample_actions_rtc)(
        key, obs, prev_action_chunk=prev, inference_delay=0,
        prefix_attention_horizon=0, num_steps=4, noise=noise,
    )
    assert jnp.allclose(plain, rtc, atol=1e-2)


def test_sample_actions_rtc_pins_prefix():
    # Strong guidance over the whole horizon pulls the output toward prev_action_chunk
    # more than the unguided sample does.
    key, config, model, obs = _rtc_model_and_obs()
    noise = jax.random.normal(key, (1, config.action_horizon, config.action_dim))
    prev = jnp.ones((1, config.action_horizon, config.action_dim)) * 0.5
    plain = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=8, noise=noise)
    rtc = nnx_utils.module_jit(model.sample_actions_rtc)(
        key, obs, prev_action_chunk=prev, inference_delay=0,
        prefix_attention_horizon=config.action_horizon, prefix_attention_schedule="ones",
        max_guidance_weight=10.0, num_steps=8, noise=noise,
    )
    plain_dist = jnp.mean(jnp.abs(plain - prev))
    rtc_dist = jnp.mean(jnp.abs(rtc - prev))
    assert rtc_dist < plain_dist
```

- [ ] **Step 2: Run to verify they fail**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -k sample_actions_rtc -v`
Expected: FAIL — `AttributeError: ... has no attribute 'sample_actions_rtc'`.

- [ ] **Step 3: Implement `sample_actions_rtc`**

Add this method to `Pi0MEM` in `src/openpi/models/pi0_mem.py` (it mirrors `sample_actions`'s prefix-KV setup and per-step suffix forward, wrapped in the τ-frame guidance):

```python
    @override
    def sample_actions_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prev_action_chunk: at.Float[at.Array, "b ah ad"],
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Inference-time RTC: guided flow sampling toward prev_action_chunk.

        Ported from real-time-chunking-kinetix realtime_action (soft-guidance branch),
        run in the kinetix tau-frame (tau = 1 - pi0_mem_time, v_tau = -v_t).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Encode the LL prefix once -> KV cache (same as sample_actions).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ll(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def pi0_velocity(x, t_scalar):
            # pi0_mem velocity v_t at pi0_mem-time t (== body of sample_actions's step).
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_ll(
                observation, x, jnp.broadcast_to(t_scalar, (batch_size,))
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms_cond],
            )
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        weights = get_prefix_weights(
            inference_delay, prefix_attention_horizon, self.action_horizon, prefix_attention_schedule
        )
        dtau = 1.0 / num_steps

        def step(carry, _):
            x, tau = carry

            def denoiser(x_in):
                v_tau = -pi0_velocity(x_in, 1.0 - tau)          # kinetix-frame velocity
                x_clean = x_in + v_tau * (1.0 - tau)            # one-step denoise (clean pred)
                return x_clean, v_tau

            x_clean, vjp_fun, v_tau = jax.vjp(denoiser, x, has_aux=True)
            error = (prev_action_chunk - x_clean) * weights[:, None]
            pinv_correction = vjp_fun(error)[0]
            inv_r2 = (tau**2 + (1 - tau) ** 2) / ((1 - tau) ** 2)
            c = jnp.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
            v_guided = v_tau + guidance_weight * pinv_correction
            return (x + dtau * v_guided, tau + dtau), None

        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        return x_1
```

If `@override` causes a typecheck error (no base-class `sample_actions_rtc`), drop the `@override` decorator — it is not a base method. Keep `@at.typecheck` off this method if the `str` schedule arg trips the typechecker; match the decoration style of `sample_actions` in the file.

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -k sample_actions_rtc -v`
Expected: PASS — shape, guidance-off-matches-plain, pins-prefix.

- [ ] **Step 5: Run the full model suite for regressions**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py -v`
Expected: PASS (incl. unchanged `sample_actions`/`compute_loss` tests).

- [ ] **Step 6: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "$(cat <<'EOF'
feat(pi0_mem): sample_actions_rtc guided inference-time RTC sampler (Spec D)

Ported from real-time-chunking-kinetix realtime_action, run in the kinetix
tau-frame (tau=1-t, v_tau=-v_t) so pi0_mem's t=1=noise flow matches the guidance
formulas. sample_actions and compute_loss are unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 3: MEMPolicy RTC wiring

**Files:**
- Modify: `src/openpi/policies/mem_policy.py` (ctor params, `prev_action_chunk` state, `reset`, `step`)
- Test: `src/openpi/policies/mem_policy_test.py`

**Interfaces:**
- Consumes: `model.sample_actions_rtc(...)` (Task 2), existing `model.sample_actions(...)`.
- Produces: `MEMPolicy(..., use_rtc: bool = False, inference_delay: int = 1, prefix_attention_horizon: int | None = None, prefix_attention_schedule: str = "exp", max_guidance_weight: float = 5.0)`; `self.prev_action_chunk` state (None until first chunk; cleared by `reset()`).

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/policies/mem_policy_test.py`:

```python
class _RecordingModel:
    """Wraps a real Pi0MEM, recording which sampling method MEMPolicy calls."""

    def __init__(self, real):
        self._real = real
        self.action_horizon = real.action_horizon
        self.action_dim = real.action_dim
        self.calls = []

    def predict_subtask_and_memory(self, *a, **k):
        self.calls.append("hl")
        return self._real.predict_subtask_and_memory(*a, **k)

    def sample_actions(self, *a, **k):
        self.calls.append("plain")
        return self._real.sample_actions(*a, **k)

    def sample_actions_rtc(self, *a, **k):
        self.calls.append("rtc")
        return self._real.sample_actions_rtc(*a, **k)


def test_mem_policy_rtc_routing():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    model = config.create(key)
    rec = _RecordingModel(model)
    # hl_interval high so HL only runs at step 0; focus on the LL sampler routing.
    policy = MEMPolicy(rec, config, hl_interval_steps=100, num_flow_steps=2, use_rtc=True)
    policy.reset()
    assert policy.prev_action_chunk is None

    obs = config.fake_obs(batch_size=1)
    a0 = policy.step(key, obs)          # first step: no prev chunk -> plain
    assert policy.prev_action_chunk is not None
    assert policy.prev_action_chunk.shape == (1, config.action_horizon, config.action_dim)
    a1 = policy.step(key, obs)          # second step: prev chunk present -> rtc
    assert "plain" in rec.calls
    assert "rtc" in rec.calls
    assert a1.shape == (1, config.action_horizon, config.action_dim)

    policy.reset()
    assert policy.prev_action_chunk is None


def test_mem_policy_no_rtc_never_calls_rtc():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", num_video_frames=2)
    rec = _RecordingModel(config.create(key))
    policy = MEMPolicy(rec, config, hl_interval_steps=100, num_flow_steps=2)  # use_rtc defaults False
    policy.reset()
    obs = config.fake_obs(batch_size=1)
    policy.step(key, obs)
    policy.step(key, obs)
    assert "rtc" not in rec.calls
    assert "plain" in rec.calls
```

- [ ] **Step 2: Run to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/policies/mem_policy_test.py -k rtc -v`
Expected: FAIL — `MEMPolicy.__init__` has no `use_rtc` (TypeError) / `prev_action_chunk` attribute missing.

- [ ] **Step 3: Add RTC params + state to `MEMPolicy.__init__` and `reset`**

In `src/openpi/policies/mem_policy.py`, extend the `__init__` signature (add after `num_flow_steps`):

```python
        use_rtc: bool = False,
        inference_delay: int = 1,
        prefix_attention_horizon: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
```

In the `__init__` body (with the other assignments):

```python
        self.use_rtc = use_rtc
        self.inference_delay = inference_delay
        self.prefix_attention_horizon = (
            prefix_attention_horizon if prefix_attention_horizon is not None else model.action_horizon
        )
        self.prefix_attention_schedule = prefix_attention_schedule
        self.max_guidance_weight = max_guidance_weight
        self.prev_action_chunk = None
```

In `reset()`, add:

```python
        self.prev_action_chunk = None
```

- [ ] **Step 4: Route the sampler in `step()`**

In `src/openpi/policies/mem_policy.py`, replace the existing LL sampling call:

```python
        actions = self.model.sample_actions(
            ll_rng, obs_with_ctx, num_steps=self.num_flow_steps
        )
```

with:

```python
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
```

(Leave the `self.step_count += 1` line that follows in place.)

- [ ] **Step 5: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/policies/mem_policy_test.py -v`
Expected: PASS — RTC routing + no-RTC tests, and the existing MEMPolicy tests still pass (default `use_rtc=False` → unchanged behavior).

- [ ] **Step 6: Commit**

```bash
git add src/openpi/policies/mem_policy.py src/openpi/policies/mem_policy_test.py
git commit -m "$(cat <<'EOF'
feat(mem_policy): wire inference-time RTC into MEMPolicy.step (Spec D)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- `get_prefix_weights` (D.1) → Task 1.
- `sample_actions_rtc` guided sampler + τ-frame adapter + correctness crux (D.2, time-convention section) → Task 2 (with the guidance-off≈plain test pinning the adapter and the pins-prefix behavioral test).
- MEMPolicy wiring + prev-chunk state + reset (D.3) → Task 3.
- No config change (D.4) → honored (no `Pi0MEMConfig` edits in any task).
- `sample_actions`/`compute_loss` unchanged → honored (Task 2 only adds a method; Task 2 Step 5 regression-checks).
- Testing items (D.5): get_prefix_weights example → Task 1; shape/guidance-off/pins-prefix → Task 2; MEMPolicy routing/reset/no-rtc → Task 3.

**Placeholder scan:** No TBD/TODO; every code step has complete code; run steps have exact commands + expected output. The `@override`/`@at.typecheck` note in Task 2 Step 3 is a concrete "match the file's decoration style; drop @override since there's no base method" instruction, not a placeholder.

**Type consistency:** `get_prefix_weights(start, end, total, schedule)` signature matches its call in `sample_actions_rtc`. `sample_actions_rtc(..., prev_action_chunk, inference_delay, prefix_attention_horizon, prefix_attention_schedule, max_guidance_weight, num_steps, noise)` matches the call from `MEMPolicy.step` (Task 3) exactly. `MEMPolicy` new ctor params (`use_rtc`/`inference_delay`/`prefix_attention_horizon`/`prefix_attention_schedule`/`max_guidance_weight`) and `prev_action_chunk` are consistent between Task 3's `__init__`, `reset`, `step`, and its tests. The τ-frame velocity (`v_tau = -pi0_velocity(x, 1-tau)`) and denoiser (`x + v_tau*(1-tau)`) match the spec's adapter section.
