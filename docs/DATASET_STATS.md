# MEM HL Dataset Statistics

High-level (HL) "memory" training data for Pi0MEM: `goal → ordered subtasks → recursive first-person memory`, conditioned on head-camera frames. Aggregated across source datasets.

RoboMIND and Galaxea numbers are both **final** (Galaxea full run completed 227/227 on 2026-07-09).

_Last updated: 2026-07-09 (Galaxea 227/227 COMPLETE)._

## Notions

| Notion | Definition |
|---|---|
| **Task** | A dataset *grouping* of demonstrations for one activity. RoboMIND: a task folder (`bread_in_basket`). Galaxea: an archive (`Fold_Clothes_20250617_001`); date-variants count as separate instances, so *unique families* is also given. |
| **Goal** | The natural-language *instruction string* the HL model reads (the `goal` field), e.g. `"fold clothes"`. Distinct-goal count = instruction-text diversity. Free-text annotation inflates this well above the true activity count (see caveat). |
| **Episode** | One robot demonstration (trajectory) of a goal. |
| **Subtask instance** | An ordered *step segment* within an episode (e.g. `"pick up the shirt"`), summed over all episodes. These are what the HL policy predicts and what the memory accumulates. |
| **Subtask vocab** | Count of *distinct* subtask strings — the phrasing diversity of the steps. |
| **HL sample (frame)** | One training row = 1 image + `(input_memory, target_subtask, target_memory)`. ~1 Hz sampling within subtasks. |
| **Transition %** | Share of samples that are subtask *boundaries* (`update=true`, one per subtask instance) vs within-subtask (`update=false`). The class balance the train-time sampler / `--max-samples-per-subtask` cap address. |

## Consolidated statistics

| Source | Tasks (instances / unique) | Distinct goals | Episodes | Subtask instances | Subtask vocab | HL samples | Transition % |
|---|---|---|---|---|---|---|---|
| `robomind_hl` (Franka) | 67 ᵃ | 67 | 161 | 702 | 204 | 3,460 | 20% |
| `robomind_hl_fr3` (Franka FR3) | 82 | 755 | 3,073 | 14,035 | 2,651 | 69,181 | 20% |
| **Galaxea** — FINAL (227/227) | 227 / 175 | 175 | 21,755 | 119,780 | 18,643 | 2,031,019 | 6% |
| **TOTAL** | **376 / 324** | **997** | **24,989** | **134,517** | **21,498** ᵇ | **2,103,660** | ~6% ᶜ |

ᵃ `robomind_hl`'s path-based task count is a quirk (2 dirs); its 67 distinct goals used as the task-type proxy.
ᵇ Vocab total is an upper bound (sums per-source distinct strings; cross-source overlap not deduped).
ᶜ Weighted by sample count (Galaxea's ~2.1 M samples at 6% dominate the blend).

## Caveat: "distinct goals" ≫ "tasks" is annotation phrasing, not activities

In `robomind_hl_fr3`, 82 tasks map to 755 goals (median **9 goals/task**, max 34) — mostly **paraphrase variation**: the same activity labeled many ways (e.g. `close_cap_trash_can` → *"closing the lid of a trash can"* / *"close the cap of a trash can"* / *"Put down the lid of the trash can"* …). A minority of high-goal tasks also bundle genuine object/composition diversity (e.g. `pull_across_pull_in_basket`, 34 goals, mixes heterogeneous activities). So **tasks (~82) is the honest activity count; distinct-goals overstates it ~9×**. The phrasing diversity is good for HL language robustness, but 755 goals ≠ 755 activities (90 of the 755 also appear in >1 task).

Galaxea is the opposite regime — near 1 goal per archive (clean templated annotations), so its goals ≈ its task families.

## Provenance

| Source | Repo | License | Embodiment |
|---|---|---|---|
| RoboMIND | existing HL sets on nano4 (`.../openpi/data/robomind_hl{,_fr3}`) | — | Franka / Franka FR3 |
| Galaxea Open-World | `OpenGalaxea/Galaxea-Open-World-Dataset` | CC-BY-NC-SA | R1 mobile bimanual |
| AgiBot Alpha / Beta | `agibot-world/AgiBotWorld-{Alpha,Beta}` | CC-BY-NC-SA | pipeline built; **held** (9.3 TB / 46 TB, ~90% discarded depth) |

Recipe: recursive first-person memory (`--generation_mode recursive`, `--disable_thinking`) via Qwen3.6-27B served on vLLM; subtasks are dataset-native (never LLM-generated).

Galaxea final output: **18 GB** on disk at `/work/roboleon1295/openpi-galaxea/data/galaxea_hl_full/<archive>/hl/` (per-archive `manifest.jsonl` + `frames/`). All 227 subtasks are `success=True` (Galaxea ships no failure labels).

_Regenerate by aggregating each source's `records.json` + `DONE` markers (Galaxea) or `manifest.jsonl` (RoboMIND)._
