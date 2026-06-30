# Spec A — LL Drops Language-Memory Conditioning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the low-level (LL) policy stop attending to the language memory, so its prefix is `[video · subtask · goal]` — goal `g` and subtask `l_{t+1}` per the MEM paper §III-A factorization `πLL(a | o_{t-K:t}, l_{t+1}, g)`; language memory `m_t` feeds the high-level (HL) policy only (Fig. 1 shows memory entering HL, not LL).

**Architecture:** Single behavioral edit to `embed_prefix_ll` in `src/openpi/models/pi0_mem.py` (delete the memory block). Pinned by two new tests asserting the LL prefix length is independent of memory while the HL prefix length still depends on it.

**Tech Stack:** JAX, Flax NNX, pytest, `uv` runner.

## Global Constraints

- Run everything via `uv run` (project uses uv).
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` for any JAX process (project convention).
- Branch: `mem-fidelity-fixes` (already checked out, off `mem-pi05-rebase`).
- Do not change `embed_prefix_hl`, `MEMPolicy`, configs, transforms, or the data pipeline — those are later specs (A2/B/C/D).

---

### Task 1: Remove language memory from the LL prefix

**Files:**
- Modify: `src/openpi/models/pi0_mem.py:116` (docstring) and `:156-161` (memory block)
- Test: `src/openpi/models/pi0_mem_test.py` (append two tests)

**Interfaces:**
- Consumes: `Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")`, `config.create(key) -> Pi0MEM`, `config.fake_obs(b) -> Observation`, `model.embed_prefix_ll(obs) -> (tokens, input_mask, ar_mask)`, `model.embed_prefix_hl(obs) -> (tokens, input_mask, ar_mask)`. `fake_obs` fills every spec field with ones, so `obs.tokenized_memory` is a non-None `int32[b, max_memory_tokens]`.
- Produces: an `embed_prefix_ll` whose output sequence length does not depend on `obs.tokenized_memory`.

- [ ] **Step 1: Write the failing tests**

Append to `src/openpi/models/pi0_mem_test.py` (add `import dataclasses` at the top of the file, after the existing imports):

```python
def _mem_model_and_obs(batch_size: int = 2):
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = config.fake_obs(batch_size)
    return config, model, obs


def test_pi0_mem_ll_prefix_excludes_memory():
    # The LL prefix length must NOT depend on the language memory: the low-level
    # policy conditions on video + subtask + goal only (MEM paper, Fig. 1).
    _, model, obs = _mem_model_and_obs()
    assert obs.tokenized_memory is not None  # precondition: fake_obs provides memory
    obs_no_mem = dataclasses.replace(obs, tokenized_memory=None, tokenized_memory_mask=None)
    len_with = model.embed_prefix_ll(obs)[0].shape[1]
    len_without = model.embed_prefix_ll(obs_no_mem)[0].shape[1]
    assert len_with == len_without


def test_pi0_mem_hl_prefix_includes_memory():
    # Regression guard: the HL prefix MUST still use the language memory.
    _, model, obs = _mem_model_and_obs()
    obs_no_mem = dataclasses.replace(obs, tokenized_memory=None, tokenized_memory_mask=None)
    len_with = model.embed_prefix_hl(obs)[0].shape[1]
    len_without = model.embed_prefix_hl(obs_no_mem)[0].shape[1]
    assert len_with > len_without
```

- [ ] **Step 2: Run the tests to verify the LL test fails (and the HL test passes)**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_ll_prefix_excludes_memory src/openpi/models/pi0_mem_test.py::test_pi0_mem_hl_prefix_includes_memory -v`

Expected: `test_pi0_mem_ll_prefix_excludes_memory` FAILS on `assert len_with == len_without` (currently `len_with` is larger by `max_memory_tokens`); `test_pi0_mem_hl_prefix_includes_memory` PASSES.

- [ ] **Step 3: Remove the memory block from `embed_prefix_ll`**

In `src/openpi/models/pi0_mem.py`, delete these lines (currently `:156-161`):

```python
        # --- Memory tokens (episodic memory from prior HL steps) ---
        if obs.tokenized_memory is not None:
            memory_emb = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
            tokens.append(memory_emb)
            input_mask.append(obs.tokenized_memory_mask)
            ar_mask += [False] * memory_emb.shape[1]
```

Then update the `embed_prefix_ll` docstring (currently `:116`) from:

```python
        """Embed prefix for LL policy: video tokens + subtask + memory + goal.
```

to:

```python
        """Embed prefix for LL policy: video tokens + subtask + goal.
```

Leave the subtask block (`:149-154`) and the goal/prompt block (`:163-168`) unchanged, so the LL prefix is now video → subtask → prompt.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py::test_pi0_mem_ll_prefix_excludes_memory src/openpi/models/pi0_mem_test.py::test_pi0_mem_hl_prefix_includes_memory -v`

Expected: both PASS.

- [ ] **Step 5: Run the full pi0_mem test suite for regressions**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run pytest src/openpi/models/pi0_mem_test.py src/openpi/models/pi0_mem_integration_test.py -v`

Expected: all PASS (the four pre-existing tests assert only loss/action shapes, which are unchanged by removing prefix tokens).

- [ ] **Step 6: Smoke-train the debug config end-to-end**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train.py pi0_mem_debug`

Expected: completes 10 steps with no trace/shape errors; logged loss values are finite. (This config uses `FakeDataConfig` and `wandb_enabled=False`, so no data download or wandb login is required.)

- [ ] **Step 7: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py docs/superpowers/specs/2026-06-30-mem-spec-a-ll-drop-memory-design.md docs/superpowers/plans/2026-06-30-mem-spec-a-ll-drop-memory.md
git commit -m "$(cat <<'EOF'
fix(pi0_mem): drop language-memory conditioning from LL prefix (Spec A)

The low-level policy now conditions on video + subtask + goal only, matching
the MEM paper factorization pi_LL(a | o_{t-K:t}, l_{t+1}, g) and Fig. 1, where
language memory feeds the high-level policy only. embed_prefix_hl is unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

## Self-Review

**Spec coverage:** The spec's single change (remove the memory block from `embed_prefix_ll`, keep HL/`MEMPolicy` untouched) is Task 1, Step 3. The spec's verification items map to Steps 5 (pytest) and 6 (`pi0_mem_debug` smoke run); the spec's "confirm HL still includes memory" sanity check is `test_pi0_mem_hl_prefix_includes_memory` (Steps 1/4). No gaps.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" placeholders; every step has concrete code or an exact command with expected output.

**Type consistency:** `embed_prefix_ll`/`embed_prefix_hl` return a 3-tuple whose first element is the token array; tests index `[0].shape[1]` consistently. `Pi0MEMConfig`, `config.create`, `config.fake_obs`, and `dataclasses.replace(obs, tokenized_memory=None, tokenized_memory_mask=None)` match the real signatures in `pi0_mem_config.py` / `model.py`.
