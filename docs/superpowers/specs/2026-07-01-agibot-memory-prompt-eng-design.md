# AgiBot World Memory-Label Prompt Engineering — Design

**Goal:** Generate MEM high-level memory labels on a real long-horizon dataset (AgiBot World), and run a manual prompt bake-off to find the prompt that maximizes memory-label **faithfulness, conciseness, temporal coherence, determinism, structural consistency, and decision-relevance**. The winning prompt becomes the default `MEMORY_PROMPT_TEMPLATE`.

**Why AgiBot World:** unlike ForeAct's repackaged per-subtask clips, AgiBot World preserves continuous long-horizon rollouts with *ordered* sub-step annotations (and ~1% failure-recovery labels) — so we get genuine subtask sequences and real success/failure flags, not the synthetic sequences the ForeAct v1 loader had to fabricate.

## Context

- Builds on the existing Spec C pipeline: `src/openpi/training/memory_labels.py` (`MemoryLabelGenerator`, `MemoryLabelConfig`, `Episode`), `src/openpi/training/memory_validation.py` (metrics + judge prompts), `scripts/validate_memory_labels.py`, and the NCHC Qwen-vLLM serving used for the ForeAct run.
- Runs on NCHC (project `MST114563`), serving **Qwen3.6-27B** FP8 via vLLM with `disable_thinking=True` (the toggle added in the follow-up PR). Heuristics-primary scoring; Qwen self-judge is a secondary signal. No external API.
- AgiBot World access (Alpha + Beta) is granted on Hugging Face.

## Non-goals (scoped out)

- Training MEM on the generated labels (this is data generation + prompt selection only).
- Frame/pixel grounding — generation is text-only; we download annotation JSON only.
- Automated prompt search (APE/OPRO). We do a **manual bake-off + refine**.
- A stronger external judge — Qwen self-judge only.

## Generation modes (decided)

MEM's memory is **recursive** at inference: `m_{t+1} = f(m_t, new event)`, with the HL head conditioned on its own previous memory. Training teacher-forces on the previous memory *label*, so the learned mapping is (label `m_t`) → (label `m_{t+1}`).

- **Recursive (primary):** label `m_{t+1}` is generated as `f(label m_t, new subtask event)` — a clean incremental update, matching the operator the model runs at inference. Sequential **within** an episode; parallel **across** episodes.
- **Stateless (baseline):** each `m_t` is regenerated independently from the full ground-truth history up to `t`. This is the current pipeline; it makes the target not a function of the memory input, widening the train-inference gap. We run it **only on the winning prompt** to quantify the coherence/gap difference as a reported number.

## Architecture & components

1. **`scripts/load_agibot_episodes.py`** — new loader (mirrors `load_foreact_episodes.py`).
   - Downloads only `task_info/*.json` from `agibot-world/AgiBotWorld-Alpha` via `huggingface_hub` `allow_patterns` (few MB).
   - Pure parser `episode_from_task_info(obj) -> Episode | None`: extracts the **ordered sub-step instructions** (confirmed field at implementation time — expected `label_info.action_config[*].action_text` with frame ranges) into `subtasks`, derives `goal` (from `task_name`/`init_scene_text`), and maps any failure/recovery annotation to `success_flags` (so some entries are `False`).
   - Filters to long-horizon episodes (**≥3 distinct sub-steps**); samples up to `--max-episodes` (cap 1000). Writes `agibot_episodes.json`.

2. **`src/openpi/training/memory_labels.py`** — extend (backward-compatible):
   - `MemoryLabelConfig` gains `prompt_template: str | None = None` (default → module `MEMORY_PROMPT_TEMPLATE`), `generation_mode: Literal["stateless","recursive"] = "stateless"`, `temperature: float = 0.0`.
   - Recursive generation: per episode, maintain running memory; for step `i`, prompt = `template.format(goal, previous_memory=m_prev, new_event=<subtask_i + status>)`; `m_0` uses `previous_memory="(none yet)"`. Sequential per episode, episodes run concurrently (async path). Stateless path unchanged.
   - `temperature` is passed to the create calls (needed for the determinism probe).

3. **`src/openpi/training/memory_validation.py`** — extend with new metrics (below).

4. **`prompts/` directory** of candidate templates (recursive form, `{goal}`/`{previous_memory}`/`{new_event}` placeholders): `v1_baseline`, `v2_tight_rules`, `v3_fewshot`, `v4_carry_forward`, `v5_structured_natural`, `v6_terse`. The winning prompt also gets a stateless re-expression for the baseline run.

5. **`src/openpi/training/prompt_bakeoff.py`** — harness library: load candidate templates → for each, generate labels over the frozen eval set → score all metrics → aggregate → ranked table. Pure scoring/aggregation logic unit-tested.

6. **`scripts/run_prompt_bakeoff.py`** + NCHC sbatch wrapper — CLI entry; calibrates throughput, runs the bake-off, writes `bakeoff_report.{json,md}`.

## Data flow

1. **Frozen eval set:** `load_agibot_episodes.py` → `agibot_episodes.json` (≤1000 long-horizon episodes). Frozen across all candidates.
2. **Calibrate:** time ~200 generations against the live Qwen server → estimate per-call latency → set episode count so `#prompts × #episodes × avg_subtasks` (+ fixed judge/determinism samples, + recursive sequentiality) fits `--time-budget-min` (default 60). Cap 1000.
3. **Per candidate (recursive):** generate one memory per (episode, timestep). Durable shards keyed by `(prompt_id, mode, episode)` for resume.
4. **Determinism probe:** resample ~30 steps ×5 at `temperature≈0.8`.
5. **Score → rank → report.** Then run the **winning** prompt once in **stateless** mode and report the recursive-vs-stateless delta.
6. **Promote** the winning prompt to default `MEMORY_PROMPT_TEMPLATE` in a small follow-up PR.

## Metrics & composite score

| Property | Measure | Over |
|---|---|---|
| Faithfulness | content-word overlap with history (heuristic) + judge 1–5 | all / sample |
| Conciseness | char compression ratio (heuristic) + judge 1–5 | all / sample |
| Decision-relevance | judge 1–5 ("keeps info needed for next step") | sample |
| Temporal coherence *(new)* | judge on consecutive pairs (m_t, m_{t+1}): preserves still-relevant facts, no contradiction, 1–5 | sampled pairs |
| Determinism *(new)* | mean pairwise content-word Jaccard over ×5 resamples | subset |
| Structural consistency *(new)* | fraction of labels matching a clean shape (no preamble/markdown/JSON/thinking; within a length band) | all |

**Composite (transparent, gated):** normalize each metric to [0,1]. A candidate must clear minimum **faithfulness**, **temporal-coherence**, and **decision-relevance** thresholds (the gate — stops "short but wrong" winning on conciseness). Among gated candidates, rank by a weighted sum emphasizing faithfulness + temporal coherence + decision-relevance, then determinism + structural, with **conciseness as a bounded bonus** (rewarded only above the gate). Weights are a config dict with defaults; the report prints all per-metric columns so the ranking can be re-derived under different weights.

## Error handling

- HF token via `HF_HOME` (custom-cache token note); access granted.
- **`task_info` schema is the main unknown:** first implementation step inspects a real Alpha `task_info.json`, then the pure parser is fixed against a captured fixture — a schema surprise fails in a unit test, not mid-run.
- Durable per-`(prompt_id, mode, episode)` shards → crash/resume safe.
- Judge JSON-parse failures tolerated (existing behavior).
- Calibration prevents budget overrun; `--max-episodes`/`--time-budget-min` configurable.

## Testing (TDD, offline with mocks/fixtures)

- Loader: fixture `task_info.json` → expected `Episode`s, including a **failed sub-step → `success_flags` containing `False`** case; long-horizon filter (≥3) drops short episodes.
- Recursive generation: with a mock client, step `i`'s prompt contains the previous step's returned memory (proves the chain); `m_0` uses the empty sentinel.
- New metrics: `temporal_coherence` (mock judge), `determinism` (identical resamples → 1.0, divergent → low), `structural_consistency` (clean vs preamble/markdown/thinking examples).
- Composite: **gate test** — a short-but-unfaithful candidate is rejected despite top conciseness.
- Harness: mock generator + metrics → deterministic ranking; resume skips existing shards.

## Deliverables

- `load_agibot_episodes.py` (+ tests), recursive generation in `memory_labels.py` (+ tests), new metrics in `memory_validation.py` (+ tests), `prompt_bakeoff.py` (+ tests), `run_prompt_bakeoff.py` + sbatch, `prompts/` candidate set.
- `bakeoff_report.{json,md}`: ranked per-metric table, winning prompt text, recursive-vs-stateless delta, example labels, flagged failures.
- Follow-up PR promoting the winning prompt to the default `MEMORY_PROMPT_TEMPLATE`.
