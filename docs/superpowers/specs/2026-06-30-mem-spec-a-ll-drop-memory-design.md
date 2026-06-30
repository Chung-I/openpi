# Spec A — LL policy drops language-memory conditioning

Date: 2026-06-30
Branch: `mem-fidelity-fixes` (off `mem-pi05-rebase`)
Status: design, pending implementation

## Context

A gap audit of the `mem-pi05-rebase` branch against the MEM paper
(arXiv:2603.03596, "Multi-Scale Embodied Memory") found that the low-level (LL)
policy is **over-conditioned**. The paper's factorization is

```
πLL(a_{t:t+H} | o_{t-K:t}, l_{t+1}, g)
```

i.e. the LL policy conditions only on the short observation window, the subtask
instruction `l_{t+1}`, and the goal `g` — **not** on the language memory `m_t`.
Figure 1 of the paper confirms this: the language memory flows only into the
high-level (HL) box; the low-level box receives the subtask + video frames +
noise. Keeping the LL memory-free is the whole point of the HL/LL split — the LL
stays cheap and the HL carries the long-horizon state.

The current implementation (`embed_prefix_ll`) instead injects
`tokenized_memory` into the LL prefix, contradicting the paper and Figure 1.

This spec removes that. It is the first of five sequenced specs:

- **A (this spec)** — LL drops memory conditioning (item 1).
- **A2** — Knowledge insulation: add a FAST discrete action-token head + insulate
  the flow-matching action expert from the VLM backbone (items 2 + 3, which the
  paper designs as one unit — the FAST loss is the *only* action supervision that
  trains the video encoder once the flow expert is insulated). Also where **LoRA
  fine-tuning** lands (deviation from the paper's full FT, for VRAM): use
  `gemma_2b_lora`/`gemma_300m_lora` variants; the insulation stop-gradient then
  targets the backbone LoRA adapters rather than full backbone weights.
- **B** — video-memory data pipeline: populate `video_images` / `video_states`
  (item 7).
- **C** — HL training-data generation via an open-weights Qwen served on NCHC,
  using a pre-annotated subtask dataset (e.g. ForeAct/AgiBot-World) (item 4).
- **D** — inference-time + training-time real-time chunking (RTC) (item 9).

## Goal

The LL policy must not attend to the language memory. The HL policy keeps it.

## Change

Single edit in `src/openpi/models/pi0_mem.py`, function `embed_prefix_ll`:
delete the memory block (the `if obs.tokenized_memory is not None:` stanza that
appends `memory_emb` to the LL prefix). The LL prefix becomes

```
[ video tokens · subtask tokens · prompt(goal) tokens ]
```

Everything else is unchanged:

- `embed_prefix_hl` keeps its memory block (HL needs `m_t`).
- `MEMPolicy` (`src/openpi/policies/mem_policy.py`) is unchanged — it still
  carries memory in the observation it builds; the LL simply ignores it. No
  call-site breaks because the field stays on `Observation`.
- `compute_loss`, `sample_actions` need no logic change (they call
  `embed_prefix_ll`, which now returns a shorter prefix).

## Non-goals

- No FAST head, no gradient insulation (Spec A2).
- No data-pipeline, config, transform, or RTC changes.
- No change to how memory is produced or carried at inference.

## Files

- `src/openpi/models/pi0_mem.py` — remove memory block in `embed_prefix_ll`.
- `src/openpi/models/pi0_mem_test.py` — update any assertion on the LL prefix
  length / token count that assumed memory tokens were present.

## Verification

1. `uv run pytest src/openpi/models/pi0_mem_test.py
   src/openpi/models/pi0_mem_integration_test.py` — green after adjusting
   prefix-length expectations.
2. Smoke train: `uv run python scripts/train.py pi0_mem_debug` (10 steps,
   FakeData) — completes with no trace/shape errors, loss is finite and trends
   down. (Per project convention, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.)
3. Sanity: confirm `embed_prefix_hl` still includes memory (token count
   unchanged for the HL path) so only the LL path shrank.
