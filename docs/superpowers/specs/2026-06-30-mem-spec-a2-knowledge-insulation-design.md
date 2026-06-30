# Spec A2 — Knowledge Insulation: FAST head + flow-expert insulation + LoRA

Date: 2026-06-30
Branch: `mem-fidelity-fixes` (off `mem-pi05-rebase`)
Status: design, pending implementation
Depends on: Spec A (LL drops memory) — already committed (`689dce2`)

## Context

The MEM paper (arXiv:2603.03596 §III-D) trains the low-level policy with the
Knowledge-Insulation recipe (Driess et al., arXiv:2505.23705): the model has
**two** action heads on one VLM — a discrete **FAST** action-token head trained
by next-token cross-entropy through the VLM backbone, and a continuous
**flow-matching** action expert. Crucially, *"gradients don't flow from the
[flow] action expert into the VLM backbone."*

The current `pi0_mem` implementation has only the flow head + an HL text head.
Two consequences:

1. The flow loss currently backprops into the VLM backbone and video encoder
   (no insulation) — contrary to the paper.
2. There is no FAST head. This matters because once the flow expert is
   insulated, **FAST cross-entropy becomes the only action-derived signal that
   trains the VideoViT encoder** (the HL head uses the separate single-frame
   `img` encoder, not `video_img`). So items 2 (FAST) and 3 (insulation) must
   ship together — they are one unit.

Additionally, we deviate from the paper's full fine-tuning and use **LoRA** on
the gemma backbone + action expert for VRAM, full-fine-tuning the (paramater-
light) vision/video encoders.

This is the second of five sequenced specs (A → **A2** → B → C → D); see
`docs/superpowers/specs/2026-06-30-mem-spec-a-ll-drop-memory-design.md`.

## Goal

1. Add a FAST discrete action-token head: backbone CE that trains the
   backbone-LoRA adapters + VideoViT encoder + text embeddings.
2. Insulate the flow expert: its gradient must not reach the backbone or the
   video encoder.
3. Use LoRA on gemma (backbone + action expert), full-FT the vision encoders.

Non-paper deviation recorded: LoRA instead of full FT (VRAM).

## Architecture & gradient routing

```
FAST CE   → backbone-LoRA adapters [train] · VideoViT (video_img) [train] · text-embed [train]
flow MSE  → action-expert-LoRA adapters [train] · state_proj/action_in/out_proj/time_mlp [train]
            (backbone + VideoViT insulated via stop-grad'd prefix KV)
HL CE     → backbone-LoRA + single-frame img encoder [train]   (unchanged; only when hl_targets given)
gemma base weights → FROZEN (LoRA freeze_filter)
```

`sample_actions` is **unchanged**: inference uses the flow expert; FAST is a
training-time co-objective only.

## Forward structure (two-pass, no gemma.py edits)

In `compute_loss`:

1. Encode video once: `video_tokens = video_img(frames)` (with gradient).
2. **FAST CE pass** (mirrors `compute_loss_hl`): backbone (expert 0) processes
   `[prefix(video+subtask+goal) · FAST-action-postfix]`; prefix bidirectional
   (`ar=False`), FAST postfix causal (`ar=True` per token). `decode_logits` on
   the postfix positions → next-token CE, masked by the action loss mask.
   Trains backbone-LoRA + video encoder + text-embed.
3. **Flow pass** (mirrors `sample_actions` KV-cache path): build the prefix KV
   from `stop_gradient(video_tokens)` through the backbone, then wrap the
   resulting **KV cache in `jax.lax.stop_gradient`**; the flow suffix (K state
   tokens + noisy actions, expert 1) attends to the stopped KV → flow MSE.
   Gradient reaches only the action expert + projections.
4. **Combine:** `total = ll_w·flow[b,ah] + fast_w·fast_ce[b,None] + hl_w·hl_ce[b,None]`
   (same broadcast the code already uses for HL). FAST and HL terms are gated:
   FAST only when `tokenized_action is not None and fast_loss_weight > 0`; HL
   only when `hl_targets is not None and hl_loss_weight > 0`.

Why this insulates correctly: the flow loss reads only the stop-grad'd KV, so no
gradient path exists from flow MSE to the backbone weights, the text-embed, or
the video encoder. The video encoder still trains because the FAST pass uses the
non-stopped `video_tokens`.

## Data / Observation changes

New `Observation` fields (`src/openpi/models/model.py`):
- `tokenized_action: Int[*b, fa] | None` — FAST action postfix token ids
  (`"Action:" + FAST + "|"`), FAST ids mapped into the PaliGemma vocab.
- `tokenized_action_mask: Bool[*b, fa] | None` — padding mask.
- `tokenized_action_loss_mask: Bool[*b, fa] | None` — positions that contribute
  to CE (the FAST action tokens).

Wire through `from_dict`, `to_dict`, and `preprocess_observation` exactly like
the existing MEM fields.

New input transform (reuse `tokenizer.FASTTokenizer`): tokenizes the action
chunk to the FAST postfix and fills the three fields. Added to the `PI0_MEM`
case in `config.ModelTransformFactory`. `inputs_spec` (in `pi0_mem_config.py`)
gains the three fields so `FakeDataConfig` / `fake_obs` provide them and the
debug config trains end-to-end with fake FAST tokens.

## Config / LoRA

`Pi0MEMConfig` (`src/openpi/models/pi0_mem_config.py`):
- `lora: bool = False` flag. When true, `create` maps each configured base
  variant to its `_lora` counterpart (`gemma_2b`→`gemma_2b_lora`,
  `gemma_300m`→`gemma_300m_lora`); `dummy` has no LoRA counterpart and is left
  as-is (unit tests use `dummy` + `lora=False` and never invoke
  `freeze_filter`, which is applied by the training loop, not the model).
- `get_freeze_filter() -> nnx.filterlib.Filter` mirroring `pi0_config.py:88-113`:
  freeze gemma base params, keep `*lora*` trainable. Vision encoders (`img`,
  `video_img`), `state_proj`, action projections are outside `.*llm.*` and stay
  trainable (full-FT) — this is the Q2 "full-FT video encoder" choice.
- `fast_loss_weight: float = 1.0`, `max_action_tokens: int = 256`.

Training configs (`src/openpi/training/config.py`): LoRA MEM configs set
`freeze_filter=Pi0MEMConfig(..., lora=True).get_freeze_filter()`. Add a
`pi0_mem_lora_debug` config (FakeData, tiny) for the smoke test.

## Files

- `src/openpi/models/model.py` — three `Observation` fields + wiring.
- `src/openpi/models/pi0_mem.py` — `embed_action_postfix` (FAST embed/mask),
  `compute_loss_fast` (CE), `compute_loss` restructured into the two-pass form
  with stop-grad'd flow KV + combined weighted loss.
- `src/openpi/models/pi0_mem_config.py` — `lora` flag, `get_freeze_filter`,
  `fast_loss_weight`, `max_action_tokens`, `inputs_spec` additions.
- `src/openpi/training/config.py` — FAST tokenization transform in the
  `PI0_MEM` case + `pi0_mem_lora_debug` config + `freeze_filter` on LoRA configs.
- New FAST-action transform (in `transforms.py` or a mem-specific module,
  following existing transform patterns).
- Tests: `src/openpi/models/pi0_mem_test.py`, `pi0_mem_config_test.py`.

## Verification

1. **Insulation gradient tests (the crux).** Run on the `dummy` variant (small,
   fast); insulation is orthogonal to LoRA, so test it on the full
   (non-LoRA) backbone params:
   - `jax.grad` of a flow-only loss (`fast_loss_weight=0`, no `hl_targets`):
     assert every `video_img` leaf grad and every backbone (gemma expert-0,
     i.e. non-`_1`) leaf grad is exactly 0; assert action-expert (expert-1,
     `*_1`) + projection (`state_proj`/`action_*_proj`/`time_mlp`) grads are
     non-zero.
   - `jax.grad` of a FAST-only loss: assert `video_img` and backbone expert-0
     grads are non-zero.
2. FAST CE loss shape `[b]`, finite; combined `compute_loss` returns
   `[b, action_horizon]`, finite.
3. LoRA freeze-filter test (filter logic, no 2B model needed). Assert
   `get_freeze_filter()` (a `PathRegex`-based filter) classifies representative
   param paths correctly: a backbone base path (e.g. `.../llm/.../attn/...`)
   is frozen; a backbone LoRA path (e.g. `.../llm/.../attn/lora_a`) is
   trainable; a `.../video_img/...` path is trainable. Check by applying the
   filter to a small constructed path→leaf tree (mirror however
   `pi0_config.get_freeze_filter` is exercised, if it is).
4. Smoke train: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train.py pi0_mem_lora_debug`
   completes with finite, decreasing loss.
5. Regression: existing `pi0_mem_test.py` (dummy, non-LoRA) still passes; the
   no-FAST `compute_loss` path (`tokenized_action=None`) still returns the flow
   loss shape `[b, action_horizon]`.
