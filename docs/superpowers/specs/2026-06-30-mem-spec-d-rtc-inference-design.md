# Spec D — Inference-time Real-Time Chunking (RTC) for Pi0MEM

Date: 2026-06-30
Branch: `mem-fidelity-fixes` (off `mem-pi05-rebase`)
Status: design, pending implementation
Depends on: Specs A, A2, B, C (committed). Reference: `~/Codes/real-time-chunking-kinetix/src/model.py`.

## Context

The MEM paper (§III-D) runs on-robot with real-time chunking (RTC) for asynchronous
inference — "inference-time RTC [6] or training-time RTC [7]". RTC removes the
discontinuity between consecutive action chunks under inference latency by making a
fresh chunk agree with the still-executing tail of the previous chunk.

This spec implements **inference-time RTC** (Black et al., arXiv:2506.07339): a
*guided* flow-matching sampler that needs **no retraining**. Each denoise step nudges
the sample toward the previous action chunk in the "prefix" region (positions already
committed during the inference delay), weighted by a prefix schedule, via a
pinv-corrected velocity (a Jacobian-vector product of the one-step denoiser).

**Training-time RTC is deferred** (a follow-up, "D2"): the kinetix training-time
variant sets a *per-position* timestep so the prefix is clean, but pi0_mem's action
expert conditions on a *single scalar* timestep (pi0.5 adaRMSNorm), so it needs a
per-position-time architectural change. The hard-mask inference path is also deferred
with it (hard-masking a clean prefix is only well-founded once the model is trained to
inpaint one). This spec is soft-guided inference RTC only.

This is the fifth and final spec (A → A2 → B → C → **D**).

Decisions (from brainstorming):
- Inference-time soft-guided RTC only; training-time RTC + hard-mask → D2.
- New `sample_actions_rtc` method; `sample_actions` and `compute_loss` unchanged.
- Inference RTC params are sampler args + `MEMPolicy` fields (deployment-time); no
  `Pi0MEMConfig` change.
- Wire RTC into `MEMPolicy.step` (prev-chunk state) so it's usable end-to-end.

## Goal

Pi0MEM can sample an action chunk that smoothly continues the previous chunk under a
given inference delay, via guided flow sampling, and MEMPolicy uses it across steps.

## Time-convention adapter (the correctness crux)

pi0_mem's flow uses `t=1`=noise: `x_t = t·noise + (1−t)·actions`, and the model
predicts `v_t = noise − actions` (= `dx_t/dt`). The kinetix RTC code is written in the
opposite convention (`τ=1`=clean: `x_τ = (1−τ)·noise + τ·actions`, model predicts
`actions − noise = dx_τ/dτ`).

These map exactly with `τ = 1 − t` and `v_τ = −v_t`. So we run the kinetix guidance
**verbatim in the τ-frame** by wrapping pi0_mem's suffix forward in an adapter:

```
kinetix_velocity(x, τ):  t = 1 − τ;  v_t = <pi0_mem suffix forward>(x, t);  return −v_t
```

Then the ported `realtime_action` integrates `x_τ` from τ=0 (noise) to τ=1 (clean) with
`dτ = 1/num_steps`, and the one-step denoiser `x_clean = x_τ + v_τ·(1−τ)` and all
guidance constants (`inv_r2`, `c`, `guidance_weight`) are used unchanged. This avoids
re-deriving the formulas and is validated by behavioral tests (below).

## Components

### 1. `get_prefix_weights(start, end, total, schedule)` (in `pi0_mem.py`)
Ported verbatim from kinetix. `start`=`inference_delay` (inclusive: where the chunk may
start changing), `end`=`prefix_attention_horizon` (exclusive), `total`=`action_horizon`.
Schedules `"ones"`/`"zeros"`/`"linear"`/`"exp"`. Pure function.

### 2. `Pi0MEM.sample_actions_rtc(...)`
Signature:
```python
def sample_actions_rtc(
    self, rng, observation, *,
    prev_action_chunk: Float[b, ah, ad],
    inference_delay: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: str = "exp",
    max_guidance_weight: float = 5.0,
    num_steps: int = 10,
    noise: Float[b, ah, ad] | None = None,
) -> Actions
```
(`noise` mirrors `sample_actions` so a test can drive both samplers with identical
noise; when None it is drawn as `jax.random.normal(rng, [b, ah, ad])`, exactly as
`sample_actions` does.)
- Same prefix KV-cache setup as `sample_actions` (encode prefix once → `kv_cache`).
- Define a suffix-forward closure that, given `x` and a pi0_mem-time `t`, runs
  `embed_suffix_ll(observation, x, t)` → action expert against `kv_cache` →
  `action_out_proj` → `v_t` (exactly the body of `sample_actions`'s `step`).
- `kinetix_velocity(x, τ)` = `−suffix_forward(x, 1−τ)`.
- Port kinetix `realtime_action`'s soft-guidance loop (the `pinv_corrected_velocity`
  branch — NOT the `simulated_delay`/hard-mask branch): per step, `jax.vjp` of
  `denoiser(x) = x + kinetix_velocity(x,τ)·(1−τ)` (returns `x_clean`, aux `v_τ`) w.r.t.
  `x`; `error = (prev_action_chunk − x_clean) · get_prefix_weights(inference_delay,
  prefix_attention_horizon, action_horizon, schedule)[:,None]`; `pinv = vjp(error)`;
  `guidance_weight = min(c·inv_r2, max_guidance_weight)`; step
  `x += dτ·(v_τ + guidance_weight·pinv)`. Integrate τ: 0→1 via `jax.lax.scan`/`while_loop`.
- `sample_actions` stays unchanged.

### 3. `MEMPolicy` RTC wiring (`mem_policy.py`)
- New ctor params: `use_rtc: bool = False`, `inference_delay: int = 1`,
  `prefix_attention_horizon: int | None = None` (default = `action_horizon`),
  `prefix_attention_schedule: str = "exp"`, `max_guidance_weight: float = 5.0`.
- New state `self.prev_action_chunk` (cleared in `reset()`).
- In `step()`: if `use_rtc` and `self.prev_action_chunk is not None`, call
  `model.sample_actions_rtc(ll_rng, obs_with_ctx, prev_action_chunk=self.prev_action_chunk,
  inference_delay=..., prefix_attention_horizon=..., ...)`; else (first step / RTC off)
  call the existing `model.sample_actions`. Store the returned chunk as
  `self.prev_action_chunk`. Behavior with `use_rtc=False` is identical to today.

### 4. Config
No `Pi0MEMConfig` change (training-RTC param is D2). Inference RTC params live on
`sample_actions_rtc` args + `MEMPolicy` fields.

## Files

- `src/openpi/models/pi0_mem.py` — `get_prefix_weights` + `sample_actions_rtc` (+ the
  kinetix-frame adapter). `sample_actions`/`compute_loss` untouched.
- `src/openpi/policies/mem_policy.py` — RTC ctor params, `prev_action_chunk` state,
  `step()`/`reset()` wiring.
- Tests: `src/openpi/models/pi0_mem_test.py`, `src/openpi/policies/mem_policy_test.py`.

## Testing / verification

1. **`get_prefix_weights`** (pure): the docstring example
   `get_prefix_weights(2, 6, 10, "zeros"|"linear")` and the canonical
   `start=2,end=6,total=10` shape `[1,1,4/5,3/5,2/5,1/5,0,0,0,0]` (linear); `end=0` →
   all zeros; `schedule="ones"` → all ones except `>= end`.
2. **`sample_actions_rtc` shape** (dummy model): returns `[b, action_horizon, action_dim]`.
3. **Guidance-off ≈ plain** (the key correctness test): pass the SAME explicit `noise`
   to both; with `prefix_attention_horizon=0` (weights all zero → no guidance),
   `sample_actions_rtc` output equals `sample_actions` output within tight tolerance.
   This holds because the τ-frame increment `dτ·v_τ = (1/n)·(−v_t)` equals
   `sample_actions`'s `dt·v_t = (−1/n)·v_t` step-for-step, so the two integrations are
   numerically identical when guidance is off — confirming the adapter is correct.
4. **Prefix pinned** (behavioral): with a strong schedule (`"ones"`),
   `inference_delay=0`, `prefix_attention_horizon=action_horizon`, large
   `max_guidance_weight`, and a fixed `prev_action_chunk`, the first few output
   positions are closer to `prev_action_chunk` than the unguided sample is (guidance
   pulls the prefix toward the previous chunk).
5. **`MEMPolicy`** (mock model): with `use_rtc=True`, the first `step()` uses plain
   sampling (no prev chunk) and subsequent steps pass the stored `prev_action_chunk`
   into `sample_actions_rtc`; `reset()` clears it; with `use_rtc=False` behavior is
   unchanged (still calls `sample_actions`).

Note: tests run on the `dummy` variant; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
