# MEM on Pi0.5: Architecture Rebase Design

## Context

The MEM implementation currently builds on pi0 (Gemma 2B + 300M action expert, MLP timestep concat). The MEM paper (arXiv:2603.03596) integrates into pi0.6 (Gemma3 4B + 860M, adaRMSNorm), but pi0.6 is not open-sourced. Pi0.5 (arXiv:2504.16054) is open-sourced and already supported in the openpi codebase via `Pi0Config(pi05=True)`. This design rebases MEM onto pi0.5.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State input | Continuous (both HL and LL) | MEM needs per-frame state tokens for temporal alignment; simpler than hybrid |
| Timestep injection | adaRMSNorm | Matches pi0.5/pi0.6; key architectural improvement |
| Weight init | Load from pi05_base checkpoint | Backbone weights transfer directly; MEM-specific layers init randomly |
| Token length | 200 | Matches pi0.5's default; room for subtask + memory tokens |
| Approach | pi05 flag in Pi0MEMConfig | Mirrors existing pi0/pi0.5 pattern; supports both modes |

## Architecture

Pi0.5's key differences from pi0 that affect MEM:

1. **adaRMSNorm**: The action expert uses adaptive RMSNorm to inject flow matching timestep, replacing the MLP concat of timestep + action tokens.
2. **time_mlp**: A 2-layer MLP (`time_mlp_in → swish → time_mlp_out → swish`) produces the adaRMS conditioning signal, replacing `action_time_mlp_in/out`.
3. **No state_proj removal**: Unlike standard pi0.5 which uses discrete state input, MEM keeps the continuous `state_proj` for K proprioceptive state tokens (per-frame temporal alignment).
4. **max_token_len=200**: Matches pi0.5's prompt length to accommodate subtask + memory tokens.

## Changes by File

### 1. `src/openpi/models/pi0_mem_config.py`

- Add `pi05: bool = True` (default True — MEM should be based on pi0.5)
- Add `discrete_state_input: bool = False` (needed for transform factory)
- Change `max_token_len` default to 200
- Add `__post_init__` to handle defaults (matching Pi0Config's pattern)

### 2. `src/openpi/models/pi0_mem.py`

**`__init__`** — when `pi05=True`:
- Pass `adarms=True` to `_gemma.Module`
- Pass `use_adarms=[False, True]` to `llm.lazy_init`
- Create `time_mlp_in` + `time_mlp_out` (Linear layers, action_expert width)
- When `pi05=False` (backward compat): keep `action_time_mlp_in/out`
- `state_proj` stays in both modes (continuous state)

**`embed_suffix_ll`** — when `pi05=True`:
- State tokens: unchanged (continuous, K per frame)
- Action tokens: `action_in_proj(noisy_actions)` only (no time concat)
- Time: `time_mlp_in(time_emb) → swish → time_mlp_out → swish` → returned as `adarms_cond`
- When `pi05=False`: existing MLP concat path unchanged

**HL policy methods**: No changes — they use only the first expert (PaliGemma), not the action expert. adaRMSNorm only affects the action expert.

**`compute_loss` / `sample_actions`**: Already pass `adarms_cond=[None, adarms_cond]` — the only change is that `adarms_cond` is now non-None when pi05=True.

### 3. `src/openpi/training/config.py`

- Update `pi0_mem_debug` config to pass `pi05=True`
- Update `ModelTransformFactory` PI0_MEM case to pass `discrete_state_input`
- Register `pi05_mem_base` config with weight loader pointing to `gs://openpi-assets/checkpoints/pi05_base/params`

### 4. `src/openpi/models/pi0_mem_integration_test.py`

- Update test configs to use `pi05=True` (matching the new default)
- All 5 existing tests should pass — the `dummy` Gemma variant supports adaRMSNorm

## What Does NOT Change

- `video_vit.py` — VideoViT encoder is backbone-agnostic
- `model.py` — Observation dataclass, ModelType.PI0_MEM
- `mem_policy.py` — MEMPolicy inference orchestration
- `memory_labels.py` — LLM compression pipeline
- `generate_memory_labels.py` — CLI script
- HL policy logic (prefix, loss, generation) — uses only first expert

## Invariants

1. `Pi0MEMConfig(pi05=False)` must produce identical behavior to the current implementation
2. `Pi0MEMConfig(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")` must pass all integration tests
3. When loading pi05_base weights, only MEM-specific layers (VideoViT temporal embedding, state_proj for multi-frame) should be randomly initialized; all backbone layers should transfer
