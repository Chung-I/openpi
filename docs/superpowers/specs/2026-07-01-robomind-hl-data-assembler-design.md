# MEM HL Training-Data Assembler (RoboMIND Franka) — Design

**Goal:** Build the pipeline that turns RoboMIND Franka annotations + HDF5 frames + generated language-memory labels into **training-ready high-level (HL) policy samples** — each pairing an observation `o_t` with `(goal g, previous memory m_t)` and targets `(next subtask l_{t+1}, updated memory m_{t+1})` — for a small first trial. Output is a manifest + extracted frames; wiring into pi0_mem's HL loss and actually training are a separate follow-up spec.

**Why RoboMIND Franka:** it resolves the two gaps against MEM's HL setting. (1) `o_t`: RoboMIND provides **per-subtask frame timestamps** (`start_frame`/`end_frame`), so each memory label aligns to a real observation. (2) Embodiment/camera: the `h5_franka_1rgb` subset is **single-arm Franka with one top RGB camera**, matching pi0.5's DROID/Franka embodiment — HL only trains the VLM, so only the camera view/resolution transfers, and that reduces to "resize `camera_top` to 224".

## Context

- Builds on the merged MEM work and PR #4: reuses `MemoryLabelGenerator` (recursive, train-inference-aligned; default `RECURSIVE_MEMORY_PROMPT_TEMPLATE` = the paper-faithful `v7_paper` prompt) and `Episode`/`MemoryLabels` from `openpi.training.memory_labels`.
- RoboMIND (`x-humanoid-robomind/RoboMIND`) is NOT a HF-`datasets`-native dataset: Franka data is stored as **per-task multi-part archives** (`benchmark1_0_compressed/h5_franka_1rgb/<task>.tar.gz.part-*`) that reassemble into `.../success_episodes/train/<ts>/data/trajectory.hdf5`. Subtask annotations live in `static/language_description_annotation_json/h5_franka_1rgb.json` (161 episodes). HDF5 layout: `rgb_images/camera_top` (+ depth, state, action).
- Annotation record shape (verified):
  ```json
  {"id": "h5_franka_1rgb/bread_in_basket/success_episodes/train/1016_161244/data",
   "response": {"task_summary": "placing bread into a basket",
     "steps": [{"step_description": "move towards the bread",
                "start_frame": "camera_top_0000.jpg", "end_frame": "camera_top_0023.jpg"}, ...]}}
  ```

## Non-goals (scoped out)

- Wiring the manifest into pi0_mem's HL loss / DataConfig, and running HL training (separate follow-up spec).
- RoboMIND `failure_data/` episodes: **deferred**. The assembler stays failure-aware (the failed-subtask no-update rule is implemented and unit-tested with synthetic `success_flags`), but the first trial uses the 161 `h5_franka_1rgb` **success** episodes only.
- Depth, multi-camera (`h5_franka_3rgb`), other embodiments, video-memory `o_{t-K:t}` windows (our HL conditions on a single `o_t`).

## The π_HL dataset construction (core)

Index spaces kept separate: subtasks `i = 1..N` with frame ranges `[s_i, e_i]`, instructions `l_i`, success flags `f_i`; goal `g`; memory states `m_0 = "(none yet)"` and `m_i` = memory after subtask `i`'s event (from the recursive generator: on success `m_i` extends `m_{i-1}`, on failure `m_i = m_{i-1}`); observations `o_τ` sampled at **1 Hz** (`frame ≈ round(τ·fps)`).

The input memory held *throughout* subtask `i` is `m_{i-1}` — it becomes `m_i` only at the boundary. Per sampled observation in subtask `i`:

| where the observation is | input `(o, m, g)` | target subtask | target memory | update |
|---|---|---|---|---|
| **inside** subtask `i` (`s_i ≤ frame < e_i`) | `(o_τ, m_{i-1}, g)` | `l_i` (keep going) | `m_{i-1}` (unchanged) | no |
| **boundary** of `i` (frame ≈ `e_i`) | `(o_τ, m_{i-1}, g)` | `l_{i+1}` (advance) | `m_i` (update) | yes |

- The transition `m_{i-1} → m_i` and `l_i → l_{i+1}` fires exactly once per subtask, at its vision-detected completion (`e_i`). Recursion is consistent: during subtask `i+1` the input memory is `m_i`.
- **Last subtask** boundary: target `("done", m_N)`.
- **Failed** subtask (`f_i = False`): its boundary is a **no-update** sample — target `(l_i, m_{i-1})` — i.e. the memory does not move until the subtask succeeds (v7_paper's rule, now grounded in vision). Deferred as data, present as logic.

## Architecture & components

1. **`scripts/load_robomind_episodes.py`** — parse `h5_franka_1rgb.json` → records `{id, goal, steps=[{subtask, start_idx, end_idx}], success_flags}` (frame idx parsed from `camera_top_0023.jpg` → 23). Provides an `Episode(goal, subtasks, success_flags)` view for label generation. Pure parser (fixture-tested) + the small JSON download.

2. **Memory-label generation** — *reuse the existing pipeline unchanged*: `MemoryLabelGenerator` (recursive, v7_paper) over the RoboMIND `Episode`s on NCHC-Qwen → `robomind_labels.json` (`m_1…m_N` per episode). Not new code; a run.

3. **`scripts/robomind_frames.py`** — selective frame access. Given a record `id` + frame indices: fetch that episode's **task-tar** (all `.part-*`), extract its `trajectory.hdf5`, read `rgb_images/camera_top[idx]`, resize→224; **cache per task** (episodes share a tar → fetch once) and **delete the tar after the task's episodes are done** (bounded disk). `fps` from HDF5 attrs, else `--fps` default. Optional `--stream` (fsspec) path is future.

4. **`scripts/assemble_hl_data.py`** — CPU orchestrator. Selects a subset, and per episode runs the pure `build_samples(record, memories, fps, sample_hz) → list[dict]` (the construction table), extracts+saves each `o_τ` frame, and writes manifest rows.

5. **Output** → `data/robomind_hl/manifest.jsonl` + `data/robomind_hl/frames/`.

## Manifest schema (one JSON row per HL sample)

```json
{"episode_id": "...1016_161244", "subtask_index": 2, "frame": 96, "update": true,
 "image": "frames/<episode>_96.jpg", "goal": "placing bread into a basket",
 "input_memory": "I moved to the bread.", "target_subtask": "move bread towards the basket",
 "target_memory": "I moved to the bread and grabbed it.", "success": true}
```
Self-describing (text targets + image path + provenance), so the later training spec adds only a thin manifest→`Observation` transform.

## Data flow

- **Stage A (once, NCHC-Qwen):** `load_robomind_episodes` → `Episode`s → `MemoryLabelGenerator` (recursive v7) → `robomind_labels.json`.
- **Stage B (CPU + HF fetch):** `assemble_hl_data`: load records + labels → select subset (`--max-episodes`) → per episode: get fps + frames via `robomind_frames` (selective fetch → extract → read → discard) → `build_samples` (1 Hz) → resize+save frames → write `manifest.jsonl`.

## Error handling

- `id` → task-tar mapping resolved from the annotation path; download `.part-*` in order, `cat`, extract only the target episode's HDF5; delete the tar when the task's sampled episodes are done.
- `fps` missing from HDF5 attrs → `--fps` default with a warning.
- Frame index out of range → clamp; missing tar/HDF5 or extract failure → skip the episode with a logged warning (don't abort the run).
- Every manifest row carries `episode_id`/`subtask_index`/`frame` for debugging.

## Testing (TDD, offline with fixtures/mocks)

- `load_robomind_episodes`: fixture annotation record → parsed record + `Episode`; `camera_top_0023.jpg` → frame index 23.
- **`build_samples` (the crux, pure, no I/O):** synthetic subtask ranges + `m_i` + fps → assert: constant within-subtask targets `(l_i, m_{i-1})`; boundary at `e_i` → `(l_{i+1}, m_i)`; last-subtask boundary → `("done", m_N)`; a synthetic **FAILED** subtask → boundary is no-update `(l_i, m_{i-1})`; 1 Hz stride honored.
- `robomind_frames`: mock HDF5 `camera_top` array → 224×224 output; frame-index parsing.
- Manifest round-trips (write → read → same fields).

## Deliverables

- `load_robomind_episodes.py` (+ tests), `robomind_frames.py` (+ tests), `assemble_hl_data.py` with `build_samples` (+ tests), and the two Stage-A/Stage-B run configs.
- `data/robomind_hl/manifest.jsonl` + `frames/` from the 161-episode success trial.
- Follow-up specs (not here): ingest `failure_data/`; manifest→`Observation` transform + pi0_mem HL training.
