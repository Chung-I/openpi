# HL-only Training Harness for Pi0MEM — Design

- **Date:** 2026-07-03
- **Branch:** `mem-hl-training` (worktree, off `robomind-hl-assembler`)
- **Status:** Approved design; ready for implementation plan
- **Related:** `2026-07-01-robomind-hl-data-assembler-design.md` (produces the data),
  MEM paper (arXiv 2603.03596v1) §training

## 1. Context & motivation

The RoboMIND HL data assembler and the `RobomindHLDataset` → `Observation` transform
(`src/openpi/training/robomind_hl.py`) exist and are tested offline, but nothing wires
them into a training loop. `Pi0MEM.compute_loss_hl` (next-token cross-entropy on the serialized subtask+memory text)
is implemented and unit-tested, but has never been driven by an optimizer over real data.
This project also refines two details of that existing implementation — the target
serialization format (§5.1) and the generation-eval decode path (§5.2) — motivated below.

This project builds a **testing/debugging harness for HL-only training**: a small,
iterable loop that finetunes the high-level policy on RoboMIND Franka data and reports
whether it learns and generalizes. It is explicitly a debugging vehicle, not a
production run.

**Division of labor.** A separate session owns **LL-only** training testing/debugging
(`scripts/train.py`, the action/flow-matching objective, LL configs). This project owns
**HL-only**. The two must not share a mutable workspace — see §8.

### Relationship to the MEM paper

The paper co-trains HL and LL on **one shared VLM backbone** with the action-expert
gradients stopped from flowing into the backbone
(*"Gradients don't flow from the action expert into the VLM backbone."*). Our
`compute_loss_hl` naturally realizes the HL half of that recipe: it runs only the
PaliGemma expert, the single-frame SigLIP encoder (**not** the video encoder), and never
touches the flow-matching action expert. **Joint HL+LL co-training is deferred**; this
harness trains the HL objective in isolation. That is a deliberate simplification of the
paper's joint recipe, documented here so the divergence is explicit.

## 2. Goals / non-goals

**Goals**
- A dedicated `scripts/train_hl.py` that optimizes `compute_loss_hl` over
  `RobomindHLDataset`, reusing openpi's mesh/FSDP/checkpoint/wandb scaffolding.
- LoRA finetune of the PaliGemma VLM; action expert frozen.
- Switch the HL target serialization from XML-style tags to a natural-language framing
  (§5.1), and give the HL policy a faithful KV-cache greedy decode for generation eval
  (§5.2).
- Episode-grouped, task-level train/dev split with two dev slices; franka_1rgb as test.
- Two comparison arms: init from `pi05_base` vs `pi05_droid`.
- First-class debugging signals: overfit-a-batch gate, dev CE curves, generation-quality
  metrics, and a memorization-vs-generalization readout.
- Full isolation from the LL / video-encoder-eval threads.

**Non-goals**
- Joint HL+LL co-training (deferred; future work).
- The video encoder / K-frame path (HL uses single-frame SigLIP).
- Full-backbone (non-LoRA) finetune (may follow later; not now).
- Production-scale training or hyperparameter sweeps beyond the two init arms.
- Any change to `scripts/train.py`, LL configs, or `DataConfig`/RLDS machinery.
- Serving the HL policy under vLLM. vLLM would require exporting the LoRA-adapted
  PaliGemma to HF format and reconstructing the MEM-specific HL prefix (image + memory
  tokens + goal) in vLLM's PaliGemma path — high effort and a fidelity risk that can
  invalidate eval. Out of scope; see §5.2 for why in-graph decode is used instead.

## 3. Background: dataset semantics

Confirmed by inspecting the assembled manifests on nano4:

- **task** — a manipulation skill; encoded in `episode_id` as `<date>_<task>_<run>`
  (e.g. `241021_close_trash_bin_1`). fr3 spans 82 raw task-segments, which collapse to
  ~60 canonical tasks after stripping the `<date>`/`<run>` wrapper (see §4).
- **episode** — one demonstration of a task instance. Each episode carries a
  natural-language `goal` that is a *paraphrase* (fr3 has 755 distinct goal strings).
- **subtask** — one step within an episode (`subtask_index`); the prediction unit. Each
  manifest row asks the HL policy: given `(frame o_t, goal g, input_memory m_t)`, predict
  `(target_subtask l_{t+1}, target_memory m_{t+1})`.

Assembled data on nano4 (`/work/roboleon1295/openpi/data/`):

| Dir | Config | Samples | Episodes | Tasks |
|---|---|---|---|---|
| `robomind_hl_fr3` | franka_3rgb | 69,181 | 3,073 | ~82 |
| `robomind_hl`     | franka_1rgb | 3,460  | 161     | 2 (`bread_in_basket`, `bread_on_table`) |

The test set (franka_1rgb) is a **single bread-manipulation family on a different camera
config** — i.e. a held-out task/domain, not held-out episodes of the training tasks.

## 4. Data splits

All grouping is by `episode_id` (episode-based; no subtask-row leakage), with a fixed
seed, stratified across task families. **Rationale:** the test set measures *new-task /
new-domain* generalization, so the dev set must measure the same axis to be a faithful
proxy. An episode-level split of the same tasks would leak per-task subtask "scripts" and
be optimistically biased.

- **Train** — episodes of the remaining *trained* tasks (fr3's ~60 canonical tasks minus
  the ~8–10 held out for `dev_unseen`), minus `dev_seen`.
- **`dev_seen`** — held-out *episodes* of the trained tasks. Fast learning-sanity signal;
  should overfit quickly. Size target: ~40 episodes.
- **`dev_unseen`** — ~8–10 *whole held-out tasks* from fr3 (all their episodes), spanning
  diverse families (drawer / bread / blocks / markers / …). The generalization proxy that
  matches the test set. Size target: ~300 episodes.
- **Test** — all of `robomind_hl` (franka_1rgb). Final held-out eval only; never trained on.

**Task canonicalization.** Group episodes into tasks by stripping the leading `<date>`
token and trailing `_<run>` integer from the `episode_id` task segment. This heuristic is
**verified during implementation** by eyeballing the resulting groups (e.g. confirming
`close_drawer` and `close_drawer_1` merge as intended) before the split is frozen.

**Persistence.** The split is computed once and written to `splits/fr3_splits.json`
(schema: `{"seed": int, "dev_unseen_tasks": [...], "dev_seen_episodes": [...],
"train_episodes": [...]}`), committed, and consumed read-only by both arms so the two runs
are trained/evaluated on identical splits.

## 5. Harness: `scripts/train_hl.py`

Mirrors `scripts/train.py`'s structure (mesh, `fsdp_sharding`, `init_train_state`,
checkpoint manager, wandb, tqdm) but swaps the data path and loss:

- **Data.** A map-style loader over `RobomindHLDataset` filtered to the `train_episodes`
  set; batches assembled with `collate_hl` → `Observation.from_dict`, sharded along the
  data axis. Shuffling with the config seed. (Frames are read from disk by the dataset;
  worker count / prefetch is a tuning knob, defaulted conservatively.)
- **Loss.** `train_step` calls `model.compute_loss_hl(rng, obs, target_tokens,
  target_mask, train=True)` and means over the batch. Batch type is
  `(Observation, target_tokens, target_mask)` — distinct from `train.py`'s
  `(Observation, Actions)`.
- **Freeze filter.** `Pi0MEMConfig(lora=True).get_freeze_filter()` — trains only LoRA
  adapter params; base PaliGemma and the action expert are frozen (cast to bf16). The
  action expert is additionally never in the `compute_loss_hl` graph, so it receives no
  gradient regardless (matching the paper's stop-gradient).
- **Checkpoints / wandb.** Own checkpoint dir per arm; wandb `project_name="mem-hl-training"`.
- **Evaluation** (periodic, every `eval_interval` steps):
  - **CE loss** on `dev_seen` and `dev_unseen` (teacher-forced `compute_loss_hl`, no grad).
  - **Generation metrics** on a fixed dev subset via the KV-cache greedy decode (§5.2):
    subtask exact-match, memory exact-match, and token-level accuracy. Parsed with the
    same NL separators the training targets use (§5.1).
  - Logged to wandb as `dev_seen/*` and `dev_unseen/*`; the seen−unseen gap is the
    memorization-vs-generalization readout.
  - **Test** eval (same metrics on franka_1rgb) runs once at the end of each arm.

### 5.1 HL target serialization (natural-language framing)

The current format wraps the target in XML-style tags
(`<subtask>..</subtask><memory>..</memory>`). Under the PaliGemma SentencePiece tokenizer
these tags are **not** special tokens — `piece_to_id("<subtask>")` returns the `unk` id —
so each tag is shredded into subword pieces (`<`, `sub`, `task`, `>`), producing
off-distribution merges (`'><'`, `')</'`) and ~35% delimiter-token overhead (a short
target: 23 tokens tagged vs 17 in NL). For a small LoRA finetune, staying near the VLM's
language prior matters, so we **switch to a natural-language framing**, e.g.:

```
Subtask: <target_subtask>. Memory: <target_memory>.<eos>
```

This is applied at load time in `robomind_hl.py` (`HL_TARGET_TEMPLATE`), so **no
re-assembly** is needed — the manifest already stores `target_subtask`/`target_memory`
separately. The change is a matched pair: update the template **and** the inference parser
(`openpi.policies.mem_policy.MEMPolicy._parse_hl_output`), plus a round-trip test
(`format(subtask, memory)` → parse → recovers both). The separator must be robust: memory
is emitted last and the `Subtask:`/`Memory:` markers are unlikely to appear inside the
field values; the exact separator is finalized with the round-trip test.

### 5.2 Generation eval decode (in-graph KV-cache, not vLLM)

`predict_subtask_and_memory` currently re-runs the full prefix+generated forward every
step (O(T²)) — that recompute, not the absence of vLLM, is the cost. openpi already has
KV-cache autoregressive decoding: `gemma.py` supports `__call__(..., kv_cache)` +
`decode_logits`, and `pi0_mem.sample_actions` fills a prefix KV cache once and steps
token-by-token. We add a **KV-cache greedy decode for HL text** that reuses that exact
`gemma.llm` graph: fill the prefix cache from `embed_prefix_hl`, then decode one token at a
time against the cache until `<eos>` / `max_new_tokens`. This is bit-faithful to training
and needs no weight export. Faithfulness is pinned by an equivalence test: KV-cache decode
== the naive loop on a small fixed input (see §9).

## 6. Configuration

HL configs live in a **dedicated module** (e.g. `src/openpi/training/config_hl.py`),
**not** the RLDS-coupled `TrainConfig` registry the LL session edits — this keeps the two
threads from colliding in `config.py`. A lightweight `HLTrainConfig` dataclass carries:
model (`Pi0MEMConfig(lora=True)`), `weight_loader`, `freeze_filter`, manifest+frames paths,
splits-file path, token-length caps, batch size, steps, LR schedule, eval cadence,
checkpoint dir, wandb project, seed.

Two registered configs, identical except init weights:

| Config | Init weights |
|---|---|
| `pi0_mem_hl_fr3_base`  | `CheckpointWeightLoader(pi05_base/params)` |
| `pi0_mem_hl_fr3_droid` | `CheckpointWeightLoader(pi05_droid/params)` |

Both LoRA, both on the shared `splits/fr3_splits.json`. On nano4 the two arms run on
disjoint GPU sets in one sbatch job (same pattern as the video-eval sbatch), or as two
jobs — decided in the implementation plan.

## 7. Debugging gates & success criteria

This is a debugging harness, so the acceptance signals are the deliverable:

1. **Overfit-a-batch** — train on a single fixed batch; HL loss must drop to ~0. Proves
   the data→loss→grad→update loop is correct before any GPU-hours on the full set. Runs
   as both a unit test (dummy Pi0MEM variant) and a `--overfit-batch` mode of the script.
2. **Learning** — on the full train set, `dev_seen` CE decreases and its generation
   exact-match rises materially above chance.
3. **Generalization readout** — `dev_unseen` metrics tracked; the `dev_seen`−`dev_unseen`
   gap is reported and interpreted (large gap ⇒ memorization).
4. **Arm comparison** — both arms complete and are evaluated on the test set; `base` vs
   `droid` init compared on test CE + generation metrics.

## 8. Isolation (contamination avoidance)

The two threads are cleanly separated by branch; the risk is a shared mutable workspace.

- **Code.** This work lives on `mem-hl-training` (off `robomind-hl-assembler`), which does
  **not** contain the video-encoder-eval configs. Local worktree at
  `/home/chungyili/Codes/openpi-hl`.
- **nano4.** The existing checkout (`/work/roboleon1295/openpi`, on
  `robomind-hl-assembler`) currently has a *mixed, dirty* working tree (video-eval sbatch
  and serving files staged/modified alongside HL work). To avoid entangling the HL run
  with that state, the HL run uses a **separate worktree/checkout** on nano4 pinned to
  `mem-hl-training`. The exact mechanism (git worktree vs fresh clone, and reusing the
  assembled `data/` dirs read-only vs symlink) is settled in the implementation plan.
- **Runtime.** Own checkpoint dir, own wandb project (`mem-hl-training`), disjoint GPU
  sets from any concurrent LL/video-eval job. Shared read-only: the assembled data dirs
  and `HF_HOME` cache.
- **We never edit** `scripts/train.py`, LL configs, video-eval configs, or the RLDS
  `DataConfig` machinery.

## 9. Testing (TDD)

- Unit test: `train_hl` step on the dummy Pi0MEM variant over a tiny synthetic HL batch —
  assert overfit-a-batch converges (loss → ~0 over N steps).
- Unit test: split builder — given a synthetic manifest, assert (a) no `episode_id`
  appears in two slices, (b) `dev_unseen` tasks are absent from `train`, (c) determinism
  under fixed seed.
- Unit test: NL target round-trip — `HL_TARGET_TEMPLATE.format(subtask, memory)` parsed by
  `_parse_hl_output` recovers `(subtask, memory)` exactly, including tricky values
  (punctuation, empty memory).
- Unit test: KV-cache decode equivalence — the KV-cache HL decode produces the same token
  sequence as the naive `predict_subtask_and_memory` loop on a small fixed input (dummy
  Pi0MEM variant).
- Reuse existing `robomind_hl_test.py` coverage for the dataset/collate transform.

## 10. Risks & open questions

- **Test-set narrowness.** franka_1rgb is only 2 bread tasks; strong test numbers reflect
  bread-family + embodiment transfer, not broad generalization. Interpret accordingly.
- **NL parsing robustness.** The NL framing (§5.1) is slightly less bulletproof than tag
  delimiters; mitigated by emitting memory last and by the round-trip test. If a robust
  separator proves awkward, a single rare literal delimiter remains a fallback.
- **Generation eval cost.** Addressed by the in-graph KV-cache decode (§5.2); still run on
  a small fixed dev subset at a coarse cadence since decode is inherently sequential.
- **Task canonicalization correctness** — verified by inspection before freezing splits
  (§4).
- **Frame I/O throughput.** ~69k small frames on `/work`; if I/O-bound, raise loader
  workers / prefetch. Not expected to dominate at debugging scale.
