# HL-only Training Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testing/debugging harness that LoRA-finetunes the Pi0MEM high-level policy (`compute_loss_hl`) on RoboMIND Franka data, with faithful eval, on an isolated branch.

**Architecture:** A dedicated `scripts/train_hl.py` reuses openpi's mesh/FSDP/checkpoint/wandb scaffolding but swaps the data path (`RobomindHLDataset` → `Observation`) and the loss (`compute_loss_hl`). Training-time logic lives in an importable `src/openpi/training/hl_training.py`; configs live in a standalone `src/openpi/training/config_hl.py` (kept out of the RLDS-coupled `TrainConfig` registry the LL session edits). Target text switches from XML tags to natural language; generation eval uses an in-graph KV-cache greedy decode.

**Tech Stack:** JAX / Flax NNX, optax, orbax checkpoints, wandb, sentencepiece (PaliGemma tokenizer), tyro CLI, pytest. Package manager: `uv` (`uv run ...`).

## Global Constraints

- **Isolation (do NOT touch):** `scripts/train.py`, any LL config in `src/openpi/training/config.py`, the video-encoder-eval configs, and the RLDS `DataConfig` machinery. All HL code is new files or the specific modifications named below (`robomind_hl.py`, `mem_policy.py`, `pi0_mem.py`).
- **Branch:** all work on `mem-hl-training` in worktree `/home/chungyili/Codes/openpi-hl`. Verify with `git branch --show-current` → `mem-hl-training` before committing.
- **Run unit tests with:** `CUDA_VISIBLE_DEVICES="" uv run pytest <path> -v` from the worktree root. This machine runs other GPU jobs — the dummy-model unit tests (Tasks 1-7) MUST be forced to CPU so JAX never grabs the shared GPU. (First run may need `uv sync`; see Task 0.)
- **GPU smoke runs go to nano4, NOT this machine.** Any real-model run (Task 8's overfit gate on `gemma_2b`, Task 9's training) runs in the nano4 `openpi-hl` checkout, prefixed with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`. Do not use gradient checkpointing.
- **Finetune scope (strict LoRA-only):** LoRA on PaliGemma LLM only. Model config for real runs: `Pi0MEMConfig(pi05=True, action_dim=32, action_horizon=50, num_video_frames=6, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", lora=True)`. Freeze filter freezes **all non-LoRA params** = `nnx.Not(nnx_utils.PathRegex(".*lora.*"))` — the SigLIP image encoder and base LLM are frozen; only LLM LoRA adapters train. Do NOT use `get_freeze_filter()` (it would leave SigLIP trainable).
- **Data on nano4 (read-only):** `/work/roboleon1295/openpi/data/robomind_hl_fr3` (train/dev, franka_3rgb) and `/work/roboleon1295/openpi/data/robomind_hl` (test, franka_1rgb). Each dir has `manifest.jsonl` + `frames/`.
- **wandb:** `project_name="mem-hl-training"`. wandb is authenticated on the cluster; keep enabled for real runs, disabled in tests.
- **Tokenizer:** `gs://big_vision/paligemma_tokenizer.model` via `openpi.shared.download.maybe_download(..., gs={"token": "anon"})`. EOS id = 1, PAD id = 0, BOS id = 2.

---

## File Structure

- **Create** `src/openpi/training/hl_splits.py` — episode-based, task-level split builder + JSON I/O (pure functions).
- **Create** `src/openpi/training/hl_splits_test.py` — split builder tests.
- **Create** `scripts/build_hl_splits.py` — CLI to build & write `splits/fr3_splits.json`.
- **Modify** `src/openpi/training/robomind_hl.py` — NL `HL_TARGET_TEMPLATE`; `RobomindHLDataset` episode-set filter.
- **Modify** `src/openpi/training/robomind_hl_test.py` — cover NL template + filter.
- **Modify** `src/openpi/policies/mem_policy.py` — `_parse_hl_output` → NL parser.
- **Create** `src/openpi/policies/mem_policy_test.py` — parser round-trip tests.
- **Modify** `src/openpi/models/pi0_mem.py` — add `predict_subtask_and_memory_cached` (KV-cache decode).
- **Modify** `src/openpi/models/pi0_mem_test.py` — KV-cache-vs-naive equivalence test.
- **Create** `src/openpi/training/config_hl.py` — `HLTrainConfig`, two-arm registry, `get_config`, `cli`.
- **Create** `src/openpi/training/config_hl_test.py` — config resolution + freeze-filter test.
- **Create** `src/openpi/training/hl_training.py` — `init_hl_train_state`, `hl_train_step`, `make_hl_batch_iterator`, `evaluate_hl`.
- **Create** `src/openpi/training/hl_training_test.py` — overfit-a-batch + eval-metric tests.
- **Create** `scripts/train_hl.py` — `main(HLTrainConfig)` entrypoint + `--overfit-batch`.
- **Create** `scripts/nchc/train_hl_fr3.sbatch` — two-arm nano4 job.

---

## Task 0: Baseline — sync env and confirm existing tests pass

**Files:** none (environment only)

- [ ] **Step 1: Confirm branch and sync deps**

Run:
```bash
cd /home/chungyili/Codes/openpi-hl
git branch --show-current          # expect: mem-hl-training
uv sync
```
Expected: sync completes; branch is `mem-hl-training`.

- [ ] **Step 2: Run the existing MEM + transform tests as a baseline**

Run:
```bash
uv run pytest src/openpi/models/pi0_mem_test.py src/openpi/training/robomind_hl_test.py -q
```
Expected: all pass. If any fail, STOP and report — do not build on a red baseline.

---

## Task 1: Natural-language HL target format + parser

Switch the HL target serialization from XML tags to NL, and update the matched inference parser. Applied at load time — no data re-assembly.

**Files:**
- Modify: `src/openpi/training/robomind_hl.py:31` (`HL_TARGET_TEMPLATE`)
- Modify: `src/openpi/policies/mem_policy.py:169-184` (`_parse_hl_output`)
- Test: `src/openpi/policies/mem_policy_test.py` (create)

**Interfaces:**
- Produces: `HL_TARGET_TEMPLATE` (str) formatting `subtask`/`memory` → NL string; `MEMPolicy._parse_hl_output(text) -> tuple[str, str]` parsing that NL string back.

- [ ] **Step 1: Write the failing round-trip test**

Create `src/openpi/policies/mem_policy_test.py`:
```python
import pytest

from openpi.policies.mem_policy import MEMPolicy
from openpi.training.robomind_hl import HL_TARGET_TEMPLATE


@pytest.mark.parametrize(
    ("subtask", "memory"),
    [
        ("move towards the lid of the trash bin", "(none yet)"),
        ("pick up the red block", "already opened the drawer; grasp failed once"),
        ("place bread on the table", ""),
        ("push the drawer closed.", "cabinet: closed. drawer: open"),
    ],
)
def test_nl_target_round_trip(subtask, memory):
    text = HL_TARGET_TEMPLATE.format(subtask=subtask, memory=memory)
    parsed_subtask, parsed_memory = MEMPolicy._parse_hl_output(text)
    assert parsed_subtask == subtask.strip()
    assert parsed_memory == memory.strip()


def test_parse_missing_fields_returns_empty():
    assert MEMPolicy._parse_hl_output("garbage with no markers") == ("", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/policies/mem_policy_test.py -v`
Expected: FAIL — current template/parser use `<subtask>`/`<memory>` tags, so `test_nl_target_round_trip` mismatches.

- [ ] **Step 3: Update the target template**

In `src/openpi/training/robomind_hl.py`, replace line 31:
```python
HL_TARGET_TEMPLATE = "Subtask: {subtask} Memory: {memory}"
```
(The trailing separator between subtask and memory is the literal ` Memory: ` marker; `_encode` appends EOS.)

- [ ] **Step 4: Update the parser to NL**

In `src/openpi/policies/mem_policy.py`, replace the body of `_parse_hl_output` (lines 169-184) with:
```python
    @staticmethod
    def _parse_hl_output(text: str) -> tuple[str, str]:
        """Extract subtask and memory from HL output text.

        Expected format (natural language)::

            Subtask: pick up the cup Memory: already opened the drawer

        Memory is emitted last; we split on the first ``Memory:`` marker after the
        ``Subtask:`` marker. Returns ("", "") if the markers are absent.
        """
        m = re.search(r"Subtask:\s*(.*?)\s*Memory:\s*(.*)\Z", text, re.DOTALL)
        if not m:
            return "", ""
        return m.group(1).strip(), m.group(2).strip()
```
(`re` is already imported at `mem_policy.py:15`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/openpi/policies/mem_policy_test.py -v`
Expected: PASS (all cases).

- [ ] **Step 6: Commit**

```bash
cd /home/chungyili/Codes/openpi-hl
git add src/openpi/training/robomind_hl.py src/openpi/policies/mem_policy.py src/openpi/policies/mem_policy_test.py
git commit -m "feat(mem): NL HL target framing + matched parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Episode-based, task-level split builder

Group fr3 manifest rows by `episode_id`, hold out whole tasks (`dev_unseen`) and held-out episodes of trained tasks (`dev_seen`); persist to JSON.

**Files:**
- Create: `src/openpi/training/hl_splits.py`
- Create: `src/openpi/training/hl_splits_test.py`
- Create: `scripts/build_hl_splits.py`

**Interfaces:**
- Produces:
  - `canonical_task(episode_id: str) -> str`
  - `build_splits(episode_ids: list[str], *, seed: int, n_dev_unseen_tasks: int, n_dev_seen_episodes: int) -> dict` returning `{"seed", "dev_unseen_tasks": list[str], "dev_seen_episodes": list[str], "train_episodes": list[str]}`.
  - `save_splits(path, splits: dict) -> None`, `load_splits(path) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `src/openpi/training/hl_splits_test.py`:
```python
import json

from openpi.training import hl_splits


def _episode_ids():
    # 6 tasks x 5 episodes = 30 episodes; task encoded as <date>_<task>_<run>.
    ids = []
    for task in ["close_trash", "open_drawer", "pick_pear", "place_bread", "cap_lid", "in_block"]:
        for run in range(5):
            ids.append(f"h5/241021_{task}_{run}/success_episodes/train/x_{run}")
    return ids


def test_canonical_task_strips_date_and_run():
    assert hl_splits.canonical_task("h5/241021_close_trash_1/success/train/x") == "close_trash"
    assert hl_splits.canonical_task("h5/241022_open_drawer/success/train/x") == "open_drawer"


def test_build_splits_no_leakage_and_deterministic():
    ids = _episode_ids()
    s1 = hl_splits.build_splits(ids, seed=0, n_dev_unseen_tasks=2, n_dev_seen_episodes=3)
    s2 = hl_splits.build_splits(ids, seed=0, n_dev_unseen_tasks=2, n_dev_seen_episodes=3)
    assert s1 == s2  # determinism under fixed seed

    train = set(s1["train_episodes"])
    dev_seen = set(s1["dev_seen_episodes"])
    dev_unseen_tasks = set(s1["dev_unseen_tasks"])

    # (a) no episode appears in two slices
    assert train.isdisjoint(dev_seen)
    dev_unseen_eps = {e for e in ids if hl_splits.canonical_task(e) in dev_unseen_tasks}
    assert dev_unseen_eps.isdisjoint(train)
    assert dev_unseen_eps.isdisjoint(dev_seen)

    # (b) dev_unseen tasks never appear in train
    assert all(hl_splits.canonical_task(e) not in dev_unseen_tasks for e in train)

    # (c) sizes
    assert len(dev_unseen_tasks) == 2
    assert len(dev_seen) == 3


def test_save_load_round_trip(tmp_path):
    ids = _episode_ids()
    s = hl_splits.build_splits(ids, seed=1, n_dev_unseen_tasks=1, n_dev_seen_episodes=2)
    p = tmp_path / "splits.json"
    hl_splits.save_splits(p, s)
    assert hl_splits.load_splits(p) == s
    assert json.loads(p.read_text())["seed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/training/hl_splits_test.py -v`
Expected: FAIL — `hl_splits` module does not exist.

- [ ] **Step 3: Implement `hl_splits.py`**

Create `src/openpi/training/hl_splits.py`:
```python
"""Episode-based, task-level train/dev split builder for HL training.

A RoboMIND manifest row carries an ``episode_id`` shaped
``.../<date>_<task>_<run>/...``. We group by canonical task, hold out whole tasks
(``dev_unseen``) and held-out episodes of the remaining tasks (``dev_seen``), and
keep the rest for training. Splits are grouped by episode (never by subtask row) so
no episode straddles two slices.
"""

import json
import pathlib
import random


def canonical_task(episode_id: str) -> str:
    """Map an episode_id to its canonical task name.

    Strips the leading ``<date>`` token and a trailing ``_<run>`` integer from the
    task segment (the second ``/``-separated component).
    """
    seg = episode_id.split("/")[1]
    parts = seg.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]           # drop date
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]          # drop run index
    return "_".join(parts)


def build_splits(
    episode_ids: list[str],
    *,
    seed: int,
    n_dev_unseen_tasks: int,
    n_dev_seen_episodes: int,
) -> dict:
    """Build a task-level, episode-grouped split. See module docstring."""
    rng = random.Random(seed)

    tasks_to_eps: dict[str, list[str]] = {}
    for e in sorted(set(episode_ids)):
        tasks_to_eps.setdefault(canonical_task(e), []).append(e)

    all_tasks = sorted(tasks_to_eps)
    if n_dev_unseen_tasks >= len(all_tasks):
        raise ValueError(f"n_dev_unseen_tasks={n_dev_unseen_tasks} >= #tasks={len(all_tasks)}")
    dev_unseen_tasks = sorted(rng.sample(all_tasks, n_dev_unseen_tasks))

    trained_tasks = [t for t in all_tasks if t not in set(dev_unseen_tasks)]
    trained_eps = [e for t in trained_tasks for e in tasks_to_eps[t]]
    trained_eps_sorted = sorted(trained_eps)
    if n_dev_seen_episodes >= len(trained_eps_sorted):
        raise ValueError("n_dev_seen_episodes too large for the trained pool")
    dev_seen = sorted(rng.sample(trained_eps_sorted, n_dev_seen_episodes))
    train = sorted(set(trained_eps_sorted) - set(dev_seen))

    return {
        "seed": seed,
        "dev_unseen_tasks": dev_unseen_tasks,
        "dev_seen_episodes": dev_seen,
        "train_episodes": train,
    }


def save_splits(path, splits: dict) -> None:
    pathlib.Path(path).write_text(json.dumps(splits, indent=2))


def load_splits(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/openpi/training/hl_splits_test.py -v`
Expected: PASS.

- [ ] **Step 5: Add the `build_hl_splits.py` CLI**

Create `scripts/build_hl_splits.py`:
```python
"""Build splits/fr3_splits.json from the fr3 manifest (episode-based, task-level).

Usage:
    uv run python scripts/build_hl_splits.py \
        --manifest data/robomind_hl_fr3/manifest.jsonl \
        --out splits/fr3_splits.json \
        --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40
"""

import argparse
import collections
import json
import pathlib

from openpi.training import hl_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="splits/fr3_splits.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-dev-unseen-tasks", type=int, default=9)
    p.add_argument("--n-dev-seen-episodes", type=int, default=40)
    args = p.parse_args()

    episode_ids = [
        json.loads(line)["episode_id"]
        for line in pathlib.Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    splits = hl_splits.build_splits(
        episode_ids,
        seed=args.seed,
        n_dev_unseen_tasks=args.n_dev_unseen_tasks,
        n_dev_seen_episodes=args.n_dev_seen_episodes,
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hl_splits.save_splits(out, splits)

    # Human-verifiable summary of the task canonicalization + split sizes.
    tasks = collections.Counter(hl_splits.canonical_task(e) for e in episode_ids)
    print(f"total episodes={len(set(episode_ids))} canonical_tasks={len(tasks)}")
    print(f"dev_unseen_tasks ({len(splits['dev_unseen_tasks'])}): {splits['dev_unseen_tasks']}")
    print(f"dev_seen_episodes={len(splits['dev_seen_episodes'])} train_episodes={len(splits['train_episodes'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/hl_splits.py src/openpi/training/hl_splits_test.py scripts/build_hl_splits.py
git commit -m "feat(mem): episode-based task-level HL split builder + CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: RobomindHLDataset episode filter

Let the dataset load only a chosen set of episodes (so train / dev_seen / dev_unseen / test are read from the same manifest via different episode sets).

**Files:**
- Modify: `src/openpi/training/robomind_hl.py:107-135` (`RobomindHLDataset.__init__`)
- Test: `src/openpi/training/robomind_hl_test.py` (extend)

**Interfaces:**
- Consumes: `HL_TARGET_TEMPLATE` (Task 1).
- Produces: `RobomindHLDataset(manifest_path, frames_dir, tokenizer, *, episode_ids: set[str] | None = None, ...)`. When `episode_ids` is given, only rows whose `row["episode_id"]` is in the set are kept.

- [ ] **Step 1: Write the failing test**

Add to `src/openpi/training/robomind_hl_test.py`:
```python
import json

from openpi.training.robomind_hl import RobomindHLDataset


class _FakeTok:
    def encode(self, text):
        return [ord(c) % 100 for c in text]


def _write_manifest(tmp_path):
    rows = [
        {"episode_id": "h5/241021_taskA_0/s/train/x", "subtask_index": "1", "frame": "0",
         "update": "False", "goal": "g", "input_memory": "(none yet)",
         "target_subtask": "do a", "target_memory": "(none yet)", "success": "True",
         "image": "frames/a.png"},
        {"episode_id": "h5/241021_taskB_0/s/train/x", "subtask_index": "1", "frame": "0",
         "update": "False", "goal": "g", "input_memory": "(none yet)",
         "target_subtask": "do b", "target_memory": "(none yet)", "success": "True",
         "image": "frames/b.png"},
    ]
    mpath = tmp_path / "manifest.jsonl"
    mpath.write_text("\n".join(json.dumps(r) for r in rows))
    return mpath


def test_dataset_episode_filter(tmp_path):
    mpath = _write_manifest(tmp_path)
    ds_all = RobomindHLDataset(mpath, tmp_path, tokenizer=_FakeTok())
    assert len(ds_all) == 2

    ds_a = RobomindHLDataset(
        mpath, tmp_path, tokenizer=_FakeTok(),
        episode_ids={"h5/241021_taskA_0/s/train/x"},
    )
    assert len(ds_a) == 1
    assert ds_a.rows[0]["episode_id"] == "h5/241021_taskA_0/s/train/x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/training/robomind_hl_test.py::test_dataset_episode_filter -v`
Expected: FAIL — `RobomindHLDataset` has no `episode_ids` parameter.

- [ ] **Step 3: Add the filter parameter**

In `src/openpi/training/robomind_hl.py`, modify `RobomindHLDataset.__init__` to accept `episode_ids` and filter rows. Change the signature (line ~110) to add `episode_ids: set[str] | None = None` after `tokenizer`, and replace the final `self.rows = [...]` assignment (line 129) with:
```python
        rows = [json.loads(line) for line in pathlib.Path(manifest_path).read_text().splitlines() if line.strip()]
        if episode_ids is not None:
            rows = [r for r in rows if r["episode_id"] in episode_ids]
        self.rows = rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/openpi/training/robomind_hl_test.py::test_dataset_episode_filter -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/robomind_hl.py src/openpi/training/robomind_hl_test.py
git commit -m "feat(mem): RobomindHLDataset episode-set filter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: In-graph KV-cache HL decode

Add a KV-cache greedy decode for HL text that reuses the `gemma.llm` graph the action sampler uses, pinned bit-faithful to the naive loop by an equivalence test.

**Files:**
- Modify: `src/openpi/models/pi0_mem.py` (add method after `predict_subtask_and_memory`, ~line 415)
- Test: `src/openpi/models/pi0_mem_test.py` (extend)

**Interfaces:**
- Consumes: `self.embed_prefix_hl`, `self.PaliGemma.llm` (`__call__(embedded, positions, mask, *, kv_cache)` + `method="embed"` + `method="decode_logits"`), `make_attn_mask`, `_PALIGEMMA_BOS_TOKEN_ID`, `_PALIGEMMA_EOS_TOKEN_ID`.
- Produces: `Pi0MEM.predict_subtask_and_memory_cached(rng, observation, *, max_new_tokens=64) -> at.Int[at.Array, "b t"]` — same return contract as `predict_subtask_and_memory` (generated ids, BOS stripped), but O(T) via KV cache.

- [ ] **Step 1: Write the failing equivalence test**

Add to `src/openpi/models/pi0_mem_test.py`:
```python
def test_kv_cache_decode_matches_naive():
    key = jax.random.key(0)
    config = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)
    obs = config.fake_obs(2)

    naive = model.predict_subtask_and_memory(key, obs, max_new_tokens=8)
    cached = model.predict_subtask_and_memory_cached(key, obs, max_new_tokens=8)
    assert naive.shape == cached.shape
    assert jnp.array_equal(naive, cached)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/models/pi0_mem_test.py::test_kv_cache_decode_matches_naive -v`
Expected: FAIL — `predict_subtask_and_memory_cached` does not exist.

- [ ] **Step 3: Implement the KV-cache decode**

In `src/openpi/models/pi0_mem.py`, add this method immediately after `predict_subtask_and_memory` (after line ~415). It fills the prefix cache once, then decodes one token at a time, appending to the KV cache each step (mirroring `sample_actions`' cache usage, but for the PaliGemma expert and single-token steps):
```python
    def predict_subtask_and_memory_cached(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_new_tokens: int = 64,
    ) -> at.Int[at.Array, "b t"]:
        """KV-cache greedy decode of subtask+memory text (O(T), faithful to the naive loop).

        Fills the HL prefix KV cache once, then decodes token-by-token feeding only the
        newest token against the cache. Returns generated ids with the leading BOS stripped.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_hl(observation)
        prefix_len = prefix_tokens.shape[1]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        # Start decoding from BOS. `cur` is the single most-recent token [b, 1].
        cur = jnp.full((batch_size, 1), _PALIGEMMA_BOS_TOKEN_ID, dtype=jnp.int32)
        generated = cur
        # Number of prefix (real) tokens per batch row; new tokens attend to all of them.
        n_prefix = jnp.sum(prefix_mask, axis=1)  # [b]

        for i in range(max_new_tokens):
            tok_emb = self.PaliGemma.llm(cur, method="embed")  # [b, 1, d]
            # This token's absolute position = #prefix tokens + #generated-so-far.
            positions = (n_prefix + i)[:, None]  # [b, 1]
            # It attends to all prefix tokens (per-row mask) and to itself.
            step_mask = jnp.concatenate(
                [prefix_mask, jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1
            )[:, None, :]  # [b, 1, prefix_len + cache_len_so_far... ]
            (out, _), kv_cache = self.PaliGemma.llm(
                [tok_emb, None], positions=positions, mask=step_mask, kv_cache=kv_cache
            )
            logits = self.PaliGemma.llm(out, method="decode_logits")  # [b, 1, vocab]
            cur = jnp.argmax(logits[:, 0, :], axis=-1, keepdims=True).astype(jnp.int32)
            generated = jnp.concatenate([generated, cur], axis=1)

        del prefix_len
        return generated[:, 1:]
```

**Note (TDD):** the incremental `mask`/`positions` bookkeeping is the fiddly part. If Step 4 shows a mismatch, iterate the mask/positions construction against the equivalence test until `jnp.array_equal(naive, cached)` holds — the naive loop in `predict_subtask_and_memory` is the ground truth. Do not change the naive method.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/openpi/models/pi0_mem_test.py::test_kv_cache_decode_matches_naive -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/models/pi0_mem.py src/openpi/models/pi0_mem_test.py
git commit -m "feat(mem): in-graph KV-cache HL decode (equivalence-tested vs naive)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: HL config module (two arms)

Standalone `HLTrainConfig` + registry with the two init arms, decoupled from `config.py`.

**Files:**
- Create: `src/openpi/training/config_hl.py`
- Create: `src/openpi/training/config_hl_test.py`

**Interfaces:**
- Produces:
  - `HLTrainConfig` dataclass with fields consumed by Task 6/8: `name, model (Pi0MEMConfig), weight_loader, optimizer, lr_schedule, ema_decay, grad_accum_steps, seed, batch_size, fsdp_devices, num_train_steps, log_interval, eval_interval, save_interval, keep_period, manifest_train, frames_train, manifest_test, frames_test, splits_path, max_prompt_tokens, max_memory_tokens, max_target_tokens, max_new_tokens, checkpoint_base_dir, project_name, exp_name, wandb_enabled, overwrite, resume`; properties `checkpoint_dir`, `freeze_filter`, `trainable_filter`.
  - `get_config(name) -> HLTrainConfig`, `cli() -> HLTrainConfig`.

- [ ] **Step 1: Write the failing test**

Create `src/openpi/training/config_hl_test.py`:
```python
import flax.nnx as nnx
import jax

from openpi.training import config_hl


def test_two_arms_registered():
    base = config_hl.get_config("pi0_mem_hl_fr3_base")
    droid = config_hl.get_config("pi0_mem_hl_fr3_droid")
    assert base.model.lora and droid.model.lora
    assert "pi05_base" in base.weight_loader.params_path
    assert "pi05_droid" in droid.weight_loader.params_path
    assert base.project_name == "mem-hl-training"


def test_freeze_filter_trains_only_lora():
    # Use nnx.eval_shape so we DON'T allocate a real ~2B gemma_2b model on CPU —
    # the freeze/trainable filters are structural (path-based) and work on the
    # abstract (ShapeDtypeStruct) module.
    cfg = config_hl.get_config("pi0_mem_hl_fr3_base")
    abstract = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    trainable = nnx.state(abstract, cfg.trainable_filter)
    leaves = jax.tree_util.tree_leaves_with_path(trainable)
    assert leaves, "expected some trainable lora params"
    for path, _ in leaves:
        assert "lora" in jax.tree_util.keystr(path), f"non-lora trainable param: {jax.tree_util.keystr(path)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/training/config_hl_test.py -v`
Expected: FAIL — `config_hl` does not exist.

- [ ] **Step 3: Implement `config_hl.py`**

Create `src/openpi/training/config_hl.py`:
```python
"""Standalone HL-only training configs (decoupled from the RLDS TrainConfig registry).

Two arms differ only by init weights: pi05_base vs pi05_droid. LoRA on PaliGemma;
action expert frozen.
"""

import dataclasses
import difflib
import pathlib

import flax.nnx as nnx
import tyro

import openpi.models.pi0_mem_config as pi0_mem_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

# Root for HL data + splits on the training host (nano4). Overridable via CLI.
_DATA_ROOT = "data"


def _model() -> pi0_mem_config.Pi0MEMConfig:
    return pi0_mem_config.Pi0MEMConfig(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        num_video_frames=6,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        lora=True,
    )


@dataclasses.dataclass(frozen=True)
class HLTrainConfig:
    name: tyro.conf.Suppress[str]

    model: pi0_mem_config.Pi0MEMConfig = dataclasses.field(default_factory=_model)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        )
    )
    ema_decay: float | None = None
    grad_accum_steps: int = 1

    seed: int = 42
    batch_size: int = 32
    fsdp_devices: int = 1
    num_train_steps: int = 20_000
    log_interval: int = 50
    eval_interval: int = 1_000
    save_interval: int = 5_000
    keep_period: int | None = 10_000

    # Data
    manifest_train: str = f"{_DATA_ROOT}/robomind_hl_fr3/manifest.jsonl"
    frames_train: str = f"{_DATA_ROOT}/robomind_hl_fr3"
    manifest_test: str = f"{_DATA_ROOT}/robomind_hl/manifest.jsonl"
    frames_test: str = f"{_DATA_ROOT}/robomind_hl"
    splits_path: str = "splits/fr3_splits.json"
    num_workers: int = 8

    # Tokenization / decode
    max_prompt_tokens: int = 48
    max_memory_tokens: int = 128
    max_target_tokens: int = 200
    max_new_tokens: int = 64
    eval_gen_examples: int = 64  # generation-metric subset size per dev slice

    # Bookkeeping
    checkpoint_base_dir: str = "./checkpoints"
    project_name: str = "mem-hl-training"
    exp_name: str = tyro.MISSING
    wandb_enabled: bool = True
    overwrite: bool = False
    resume: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return pathlib.Path(self.checkpoint_base_dir).resolve() / self.project_name / self.exp_name

    @property
    def freeze_filter(self) -> nnx.filterlib.Filter:
        # Strict LoRA-only: freeze every param whose path does NOT contain "lora"
        # (base LLM + SigLIP image encoder + action expert all frozen).
        return nnx.Not(nnx_utils.PathRegex(".*lora.*"))

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))


_CONFIGS = [
    HLTrainConfig(
        name="pi0_mem_hl_fr3_base",
        exp_name="pi0_mem_hl_fr3_base",
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    HLTrainConfig(
        name="pi0_mem_hl_fr3_droid",
        exp_name="pi0_mem_hl_fr3_droid",
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
]
_CONFIGS_DICT = {c.name: c for c in _CONFIGS}


def get_config(config_name: str) -> HLTrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        hint = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"HL config '{config_name}' not found.{hint}")
    return _CONFIGS_DICT[config_name]


def cli() -> HLTrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
```

**Note:** confirm `weight_loaders.CheckpointWeightLoader` exposes its path as `.params_path`; if the attribute is named differently (e.g. `.params_path` vs `.path`), update the test's assertion accordingly. Confirm `_optimizer.AdamW` and `_optimizer.CosineDecaySchedule` exist (they are used across `config.py`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/openpi/training/config_hl_test.py -v`
Expected: PASS. (If `test_two_arms_registered` fails on the path attribute name, fix the assertion to the real field name and re-run.)

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/config_hl.py src/openpi/training/config_hl_test.py
git commit -m "feat(mem): standalone HLTrainConfig with two init arms

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: HL training library — init, batch iterator, train step

Reusable training pieces, testable via an overfit-a-batch gate on the dummy model.

**Files:**
- Create: `src/openpi/training/hl_training.py`
- Create: `src/openpi/training/hl_training_test.py`

**Interfaces:**
- Consumes: `config_hl.HLTrainConfig` (Task 5), `robomind_hl.RobomindHLDataset`/`collate_hl` (Task 3), `openpi.models.model.Observation`, `openpi.training.optimizer`, `openpi.training.utils.TrainState`, `openpi.training.sharding`.
- Produces:
  - `init_hl_train_state(config, rng, mesh, *, resume) -> tuple[TrainState, sharding]`
  - `hl_train_step(config, rng, state, batch) -> tuple[TrainState, dict]` where `batch = (Observation, target_tokens int[b,t], target_mask bool[b,t])`.
  - `make_hl_batch_iterator(dataset, *, batch_size, rng, shuffle=True) -> Iterator[tuple[dict, np.ndarray, np.ndarray]]` (yields `collate_hl` output).

- [ ] **Step 1: Write the failing overfit-a-batch test**

Create `src/openpi/training/hl_training_test.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0_mem_config import Pi0MEMConfig
from openpi.training import hl_training, optimizer, sharding
from openpi.training.utils import TrainState


def _dummy_config():
    # Minimal HLTrainConfig-like object: only fields hl_train_step/init read.
    import dataclasses

    @dataclasses.dataclass
    class _Cfg:
        model = Pi0MEMConfig(paligemma_variant="dummy", action_expert_variant="dummy", lora=True)
        optimizer = optimizer.AdamW()
        lr_schedule = optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-2, decay_steps=200, decay_lr=1e-2)
        ema_decay = None
        grad_accum_steps = 1
        seed = 0
        @property
        def freeze_filter(self):
            return self.model.get_freeze_filter()
        @property
        def trainable_filter(self):
            import flax.nnx as nnx
            return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
    return _Cfg()


def _fixed_batch(model_config, batch_size=2, target_len=8):
    obs = model_config.fake_obs(batch_size)
    obs_dict = { }  # we pass the Observation directly; see hl_train_step contract
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)
    return obs, target_tokens, target_mask


def test_overfit_single_batch_drops_loss():
    cfg = _dummy_config()
    mesh = sharding.make_mesh(1)
    rng = jax.random.key(0)
    state, _ = hl_training.init_hl_train_state(cfg, rng, mesh, resume=False)

    batch = _fixed_batch(cfg.model)
    losses = []
    with sharding.set_mesh(mesh):
        for _ in range(30):
            state, info = hl_training.hl_train_step(cfg, rng, state, batch)
            losses.append(float(info["loss"]))
    assert losses[-1] < 0.5 * losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
```

**Note:** `init_hl_train_state` must accept a config exposing `model`, `optimizer`, `lr_schedule`, `freeze_filter`, `trainable_filter`, `ema_decay`, `grad_accum_steps` (no data/weight loading in the test path — pass `resume=False` and let it random-init the dummy; skip `weight_loader` when the config lacks a real one by guarding on `getattr(config, "weight_loader", None)`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/training/hl_training_test.py::test_overfit_single_batch_drops_loss -v`
Expected: FAIL — `hl_training` does not exist.

- [ ] **Step 3: Implement `hl_training.py` (init + train step + iterator)**

Create `src/openpi/training/hl_training.py`. `init_hl_train_state`/`hl_train_step` mirror `scripts/train.py:init_train_state`/`train_step` (intentionally duplicated to keep HL decoupled from the LL-owned `scripts/train.py`), but the loss is `compute_loss_hl` and the batch is `(Observation, target_tokens, target_mask)`:
```python
"""HL-only training: init, batch iterator, train step, and evaluation.

Mirrors scripts/train.py's nnx init + step pattern, but drives compute_loss_hl over
RobomindHLDataset instead of compute_loss over RLDS. Kept separate from
scripts/train.py (owned by the LL thread) by design.
"""

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_hl_train_state(config, init_rng, mesh, *, resume):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps).gradient_transformation()

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0, params=params, model_def=nnx.graphdef(model), tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    if resume:
        return train_state_shape, state_sharding

    weight_loader = getattr(config, "weight_loader", None)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    if weight_loader is None or isinstance(weight_loader, _weight_loaders.NoOpWeightLoader):
        train_state = jax.jit(init, out_shardings=state_sharding)(init_rng)
        return train_state, state_sharding

    import flax.traverse_util as traverse_util
    loaded = weight_loader.load(train_state_shape.params.to_pure_dict())
    at.check_pytree_equality(expected=train_state_shape.params.to_pure_dict(), got=loaded,
                             check_shapes=True, check_dtypes=True)
    partial = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    train_state = jax.jit(init, donate_argnums=(1,), in_shardings=replicated, out_shardings=state_sharding)(
        init_rng, partial
    )
    return train_state, state_sharding


@at.typecheck
def hl_train_step(config, rng, state: training_utils.TrainState,
                  batch: tuple[_model.Observation, at.Int[at.Array, "b t"], at.Bool[at.Array, "b t"]]):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, target_tokens, target_mask = batch

    def loss_fn(model, rng, obs, tgt, mask):
        return jnp.mean(model.compute_loss_hl(rng, obs, tgt, mask, train=True))

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, target_tokens, target_mask
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
    return new_state, info


def make_hl_batch_iterator(dataset, *, batch_size, rng: np.random.Generator, shuffle=True):
    """Yield collate_hl outputs (obs_dict, target_tokens, target_mask) forever."""
    from openpi.training.robomind_hl import collate_hl

    n = len(dataset)
    order = np.arange(n)
    while True:
        if shuffle:
            rng.shuffle(order)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            yield collate_hl([dataset[int(i)] for i in idx])
```

- [ ] **Step 4: Fix the test's batch construction and run**

The test's `_fixed_batch` passes an `Observation` directly. Update `hl_training_test.py` `_fixed_batch` to build a real `Observation` from the dummy config:
```python
def _fixed_batch(model_config, batch_size=2, target_len=8):
    obs = model_config.fake_obs(batch_size)
    target_tokens = jnp.ones((batch_size, target_len), dtype=jnp.int32)
    target_mask = jnp.ones((batch_size, target_len), dtype=jnp.bool_)
    return obs, target_tokens, target_mask
```
Run: `uv run pytest src/openpi/training/hl_training_test.py::test_overfit_single_batch_drops_loss -v`
Expected: PASS — loss drops to <50% of the initial over 30 steps.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/hl_training.py src/openpi/training/hl_training_test.py
git commit -m "feat(mem): HL training lib (init, train step, iterator) + overfit-batch test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: HL evaluation — CE loss + generation metrics

Add held-out evaluation over a dataset: teacher-forced CE plus greedy-decode subtask/memory exact-match and token accuracy.

**Files:**
- Modify: `src/openpi/training/hl_training.py` (add `evaluate_hl`, `_decode_ids_to_text`)
- Test: `src/openpi/training/hl_training_test.py` (extend)

**Interfaces:**
- Consumes: `hl_train_step` model graph, `predict_subtask_and_memory_cached` (Task 4), `MEMPolicy._parse_hl_output` (Task 1), a sentencepiece tokenizer.
- Produces: `evaluate_hl(model, dataset, tokenizer, *, batch_size, max_new_tokens, gen_examples, rng) -> dict[str, float]` with keys `ce_loss`, `subtask_exact_match`, `memory_exact_match`, `token_accuracy`.

- [ ] **Step 1: Write the failing test**

Add to `src/openpi/training/hl_training_test.py`:
```python
def test_evaluate_hl_returns_metrics():
    cfg = _dummy_config()
    model = cfg.model.create(jax.random.key(0))

    class _Tok:
        def encode(self, text):
            return [ord(c) % 50 + 3 for c in text][:16]
        def decode(self, ids):
            return "".join(chr((i - 3) % 50 + 65) for i in ids)

    class _DS:
        def __len__(self): return 4
        def __getitem__(self, i):
            return {
                "image": np.zeros((*_model.IMAGE_RESOLUTION, 3), np.uint8),
                "image_mask": np.array(True),
                "state": np.zeros(32, np.float32),
                "tokenized_prompt": np.zeros(48, np.int32),
                "tokenized_prompt_mask": np.zeros(48, bool),
                "tokenized_memory": np.zeros(128, np.int32),
                "tokenized_memory_mask": np.zeros(128, bool),
                "target_tokens": np.ones(32, np.int32),
                "target_mask": np.concatenate([np.ones(5, bool), np.zeros(27, bool)]),
            }

    metrics = hl_training.evaluate_hl(
        model, _DS(), _Tok(), batch_size=2, max_new_tokens=4, gen_examples=2,
        rng=jax.random.key(0),
    )
    for k in ["ce_loss", "subtask_exact_match", "memory_exact_match", "token_accuracy"]:
        assert k in metrics and np.isfinite(metrics[k])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/openpi/training/hl_training_test.py::test_evaluate_hl_returns_metrics -v`
Expected: FAIL — `evaluate_hl` does not exist.

- [ ] **Step 3: Implement `evaluate_hl`**

Append to `src/openpi/training/hl_training.py`:
```python
def _decode_ids_to_text(tokenizer, ids: np.ndarray) -> str:
    clean = [int(x) for x in ids.tolist() if int(x) not in (0, 1)]  # drop PAD/EOS
    if not clean:
        return ""
    try:
        return tokenizer.decode(clean)
    except Exception:
        return ""


def evaluate_hl(model, dataset, tokenizer, *, batch_size, max_new_tokens, gen_examples, rng) -> dict:
    """Held-out eval: teacher-forced CE over the whole dataset + generation metrics on a subset."""
    from openpi.policies.mem_policy import MEMPolicy
    from openpi.training.robomind_hl import collate_hl

    model.eval()
    n = len(dataset)

    # --- CE loss over all full batches. ---
    ce_sum, ce_count = 0.0, 0
    for start in range(0, n - batch_size + 1, batch_size):
        obs_dict, tgt, mask = collate_hl([dataset[i] for i in range(start, start + batch_size)])
        obs = _model.Observation.from_dict(obs_dict)
        loss = model.compute_loss_hl(rng, obs, jnp.asarray(tgt), jnp.asarray(mask), train=False)
        ce_sum += float(jnp.sum(loss)); ce_count += loss.shape[0]
    ce_loss = ce_sum / max(ce_count, 1)

    # --- Generation metrics on the first `gen_examples`. ---
    k = min(gen_examples, n)
    sub_hits = mem_hits = tok_hits = tok_total = 0
    for start in range(0, k - batch_size + 1 if k >= batch_size else 0, batch_size) or [0]:
        b = min(batch_size, k - start)
        obs_dict, tgt, mask = collate_hl([dataset[i] for i in range(start, start + b)])
        obs = _model.Observation.from_dict(obs_dict)
        gen = np.asarray(model.predict_subtask_and_memory_cached(rng, obs, max_new_tokens=max_new_tokens))
        for j in range(b):
            gen_text = _decode_ids_to_text(tokenizer, gen[j])
            tgt_text = _decode_ids_to_text(tokenizer, np.asarray(tgt[j]))
            gs, gm = MEMPolicy._parse_hl_output(gen_text)
            ts, tm = MEMPolicy._parse_hl_output(tgt_text)
            sub_hits += int(gs == ts); mem_hits += int(gm == tm)
            L = min(len(gen[j]), int(mask[j].sum()))
            tok_hits += int(np.sum(np.asarray(gen[j])[:L] == np.asarray(tgt[j])[:L]))
            tok_total += L
    denom = max(k, 1)
    return {
        "ce_loss": ce_loss,
        "subtask_exact_match": sub_hits / denom,
        "memory_exact_match": mem_hits / denom,
        "token_accuracy": tok_hits / max(tok_total, 1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/openpi/training/hl_training_test.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/hl_training.py src/openpi/training/hl_training_test.py
git commit -m "feat(mem): HL eval (CE + generation exact-match/token-acc)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: `scripts/train_hl.py` entrypoint

Wire everything into a runnable training loop with periodic eval, checkpoints, and wandb; plus an `--overfit-batch` debug mode.

**Files:**
- Create: `scripts/train_hl.py`

**Interfaces:**
- Consumes: `config_hl.cli`/`get_config`, `hl_training.*` (Tasks 6-7), `hl_splits.load_splits` (Task 2), `robomind_hl.RobomindHLDataset`/`load_paligemma_sp`, `openpi.training.checkpoints`, `openpi.training.sharding`.

- [ ] **Step 1: Implement the entrypoint**

Create `scripts/train_hl.py`:
```python
"""HL-only training entrypoint for Pi0MEM.

Usage:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train_hl.py pi0_mem_hl_fr3_base
    ... scripts/train_hl.py pi0_mem_hl_fr3_base --overfit-batch   # debug gate
"""

import logging
import sys

import jax
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding
from openpi.training import config_hl, hl_splits, hl_training
from openpi.training.robomind_hl import RobomindHLDataset, load_paligemma_sp


def _datasets(config, tokenizer):
    splits = hl_splits.load_splits(config.splits_path)
    dev_unseen_tasks = set(splits["dev_unseen_tasks"])
    train_eps = set(splits["train_episodes"])
    dev_seen_eps = set(splits["dev_seen_episodes"])

    def make(manifest, frames, ep_ids):
        return RobomindHLDataset(
            manifest, frames, tokenizer, episode_ids=ep_ids,
            max_prompt_tokens=config.max_prompt_tokens,
            max_memory_tokens=config.max_memory_tokens,
            max_target_tokens=config.max_target_tokens,
        )

    train = make(config.manifest_train, config.frames_train, train_eps)
    dev_seen = make(config.manifest_train, config.frames_train, dev_seen_eps)
    # dev_unseen: all episodes whose canonical task is held out.
    import json, pathlib
    all_rows = [json.loads(l) for l in pathlib.Path(config.manifest_train).read_text().splitlines() if l.strip()]
    dev_unseen_eps = {r["episode_id"] for r in all_rows if hl_splits.canonical_task(r["episode_id"]) in dev_unseen_tasks}
    dev_unseen = make(config.manifest_train, config.frames_train, dev_unseen_eps)
    test = make(config.manifest_test, config.frames_test, None)
    return train, {"dev_seen": dev_seen, "dev_unseen": dev_unseen}, test


def main(config: config_hl.HLTrainConfig, *, overfit_batch: bool = False):
    logging.basicConfig(level=logging.INFO)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    mesh = sharding.make_mesh(config.fsdp_devices)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    tokenizer = load_paligemma_sp()
    train_ds, dev, test_ds = _datasets(config, tokenizer)
    it = hl_training.make_hl_batch_iterator(
        train_ds, batch_size=config.batch_size, rng=np.random.default_rng(config.seed)
    )

    ckpt_mgr, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )
    wandb.init(
        mode="online" if config.wandb_enabled else "disabled",
        name=config.exp_name, project=config.project_name, config=vars(config),
    )

    state, state_sharding = hl_training.init_hl_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = _checkpoints.restore_state(ckpt_mgr, state, None)

    def _to_batch(collated):
        obs_dict, tgt, mask = collated
        import openpi.models.model as _model
        return _model.Observation.from_dict(obs_dict), jax.numpy.asarray(tgt), jax.numpy.asarray(mask)

    fixed = _to_batch(next(it)) if overfit_batch else None

    for step in tqdm.tqdm(range(int(state.step), config.num_train_steps)):
        batch = fixed if overfit_batch else _to_batch(next(it))
        with sharding.set_mesh(mesh):
            state, info = hl_training.hl_train_step(config, train_rng, state, batch)
        if step % config.log_interval == 0:
            wandb.log({f"train/{k}": float(v) for k, v in info.items()}, step=step)
        if not overfit_batch and step > 0 and step % config.eval_interval == 0:
            import flax.nnx as nnx
            model = nnx.merge(state.model_def, state.params)
            for name, ds in dev.items():
                m = hl_training.evaluate_hl(
                    model, ds, tokenizer, batch_size=config.batch_size,
                    max_new_tokens=config.max_new_tokens, gen_examples=config.eval_gen_examples,
                    rng=train_rng,
                )
                wandb.log({f"{name}/{k}": v for k, v in m.items()}, step=step)
        if not overfit_batch and step > 0 and step % config.save_interval == 0:
            _checkpoints.save_state(ckpt_mgr, state, None, step)

    # Final test eval.
    if not overfit_batch:
        import flax.nnx as nnx
        model = nnx.merge(state.model_def, state.params)
        m = hl_training.evaluate_hl(
            model, test_ds, tokenizer, batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens, gen_examples=config.eval_gen_examples, rng=train_rng,
        )
        wandb.log({f"test/{k}": v for k, v in m.items()}, step=config.num_train_steps)
        logging.info("TEST metrics: %s", m)
    ckpt_mgr.wait_until_finished()


if __name__ == "__main__":
    overfit = "--overfit-batch" in sys.argv
    if overfit:
        sys.argv.remove("--overfit-batch")
    main(config_hl.cli(), overfit_batch=overfit)
```

- [ ] **Step 2: Smoke-test the entrypoint imports and CLI wiring**

Run:
```bash
uv run python -c "import scripts.train_hl" 2>/dev/null || uv run python - <<'PY'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("train_hl", "scripts/train_hl.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("import OK:", hasattr(m, "main"))
PY
```
Expected: `import OK: True` (module imports without executing training).

- [ ] **Step 3: Note — real-model smoke run happens on nano4**

The `gemma_2b` real-model overfit smoke run needs a GPU and the fr3 data, both of which
live on nano4 — and this machine runs other GPU jobs. So there is **no local GPU run**
here; the real-model overfit-a-batch gate is Task 9 Step 4. Confirm only that Step 2's
import smoke passed. Do not run `scripts/train_hl.py` locally against a real config.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_hl.py
git commit -m "feat(mem): scripts/train_hl.py HL training entrypoint (+ overfit-batch mode)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: nano4 orchestration — splits, isolated checkout, two-arm sbatch

Set up the isolated nano4 workspace, build the splits, and launch the two arms.

**Files:**
- Create: `scripts/nchc/train_hl_fr3.sbatch`

- [ ] **Step 1: Create the two-arm sbatch**

Create `scripts/nchc/train_hl_fr3.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=mem_hl_fr3
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
#SBATCH --output=/work/roboleon1295/openpi-hl/logs/mem_hl_fr3_%j.log
set -x
cd /work/roboleon1295/openpi-hl                     # isolated HL worktree/checkout on nano4
export HF_HOME=/work/roboleon1295/hf_cache
mkdir -p logs splits

# Build the shared split once (idempotent; both arms read it).
uv run python scripts/build_hl_splits.py \
  --manifest data/robomind_hl_fr3/manifest.jsonl --out splits/fr3_splits.json \
  --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40

BASE_LOG=logs/mem_hl_fr3_base_${SLURM_JOB_ID}.log
DROID_LOG=logs/mem_hl_fr3_droid_${SLURM_JOB_ID}.log

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run python scripts/train_hl.py pi0_mem_hl_fr3_base  > "$BASE_LOG"  2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run python scripts/train_hl.py pi0_mem_hl_fr3_droid > "$DROID_LOG" 2>&1 &
P2=$!
wait $P1; echo "BASE_EXIT=$?"
wait $P2; echo "DROID_EXIT=$?"
echo "=== base tail ==="; tail -n 40 "$BASE_LOG"
echo "=== droid tail ==="; tail -n 40 "$DROID_LOG"
```

- [ ] **Step 2: Provision the isolated nano4 checkout**

On nano4, create a separate checkout pinned to `mem-hl-training` so the dirty `robomind-hl-assembler` tree is untouched, and point the assembled data at the existing dirs (read-only). Push the branch first from local:
```bash
# local:
cd /home/chungyili/Codes/openpi-hl && git push chungyi mem-hl-training
# nano4 (run via `! ssh nano4 ...` or an interactive shell):
#   git -C /work/roboleon1295 clone --branch mem-hl-training \
#       /work/roboleon1295/openpi /work/roboleon1295/openpi-hl   # or: git worktree add
#   ln -s /work/roboleon1295/openpi/data /work/roboleon1295/openpi-hl/data
#   cd /work/roboleon1295/openpi-hl && uv sync
```
Expected: `openpi-hl` on `mem-hl-training`, `data/` resolves to the assembled manifests, `uv sync` succeeds.

- [ ] **Step 3: Build splits and eyeball the task canonicalization (verification gate)**

On nano4 in `openpi-hl`:
```bash
uv run python scripts/build_hl_splits.py --manifest data/robomind_hl_fr3/manifest.jsonl \
  --out splits/fr3_splits.json --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40
```
Expected: printed `canonical_tasks` count is sane (~60), `dev_unseen_tasks` is a plausible list of 9 distinct manipulation skills, and `train_episodes` ≈ (fr3 episodes − dev_unseen episodes − 40). **Inspect the task list**; if canonicalization mis-merges (e.g. two skills collapse to one name), fix `hl_splits.canonical_task` and rebuild before training.

- [ ] **Step 4: Overfit-a-batch gate on real data before the full run**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train_hl.py pi0_mem_hl_fr3_base \
  --overfit-batch --num-train-steps 200 --wandb-enabled False
```
Expected: training loss drops toward ~0 over 200 steps (proves data→loss→grad→update on the real model). If it does not, STOP and debug before spending the full run.

- [ ] **Step 5: Launch the two arms**

```bash
sbatch scripts/nchc/train_hl_fr3.sbatch
squeue -u $USER -o "%i %P %t %r %S"
```
Expected: job queued/running; two arms write `base`/`droid` logs and log `dev_seen/*`, `dev_unseen/*` metrics to wandb project `mem-hl-training`. Final `test/*` metrics printed at each arm's end.

- [ ] **Step 6: Commit the sbatch**

```bash
git add scripts/nchc/train_hl_fr3.sbatch
git commit -m "feat(mem): nano4 two-arm HL training sbatch (base vs droid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §2 dedicated `train_hl.py` → Task 8. LoRA + action expert frozen → Task 5 (freeze filter test). NL framing → Task 1. KV-cache decode → Task 4. ✅
- Spec §4 episode-based task-level two-slice split + persisted file → Task 2 (+ Task 9 build/verify). ✅
- Spec §5 harness (data→loss→ckpt→wandb, eval CE + generation) → Tasks 6-8. §5.1 NL → Task 1. §5.2 KV-cache → Task 4/7. ✅
- Spec §6 standalone config module + two arms → Task 5. ✅
- Spec §7 debugging gates: overfit-a-batch → Task 6 (unit) + Task 8/9 (`--overfit-batch`); seen/unseen gap → Task 8 eval logging. ✅
- Spec §8 isolation: branch/worktree, separate nano4 checkout, own ckpt+wandb, train.py untouched → Global Constraints + Task 9. ✅
- Spec §9 tests: overfit-batch → Task 6; split builder → Task 2; NL round-trip → Task 1; KV-cache equivalence → Task 4. ✅

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" or stub lines remain; every code step shows the real content. Task-4 and Task-6 include TDD "iterate until the test passes" notes, which are legitimate (the equivalence/overfit tests are the spec).

**3. Type consistency:** `HLTrainConfig` fields referenced by Tasks 6/8 (`model, weight_loader, optimizer, lr_schedule, freeze_filter, trainable_filter, ema_decay, grad_accum_steps, batch_size, fsdp_devices, num_train_steps, eval_interval, save_interval, keep_period, manifest_*, frames_*, splits_path, max_*_tokens, max_new_tokens, eval_gen_examples, checkpoint_dir, project_name, exp_name, wandb_enabled, overwrite, resume`) are all defined in Task 5. `collate_hl` output `(obs_dict, target_tokens, target_mask)` (Task 3 file) is consumed consistently in Tasks 6-8. `predict_subtask_and_memory_cached(rng, obs, *, max_new_tokens)` (Task 4) matches its call in Task 7. `_parse_hl_output` (Task 1) matches Task 7 usage. ✅

**Open verification items folded into tasks (not placeholders):**
- `CheckpointWeightLoader` path attribute name (Task 5 Step 3 note).
- `_optimizer.AdamW` / `CosineDecaySchedule` existence (used across `config.py`; Task 5 note).
- KV-cache mask/positions bookkeeping (Task 4 TDD note; equivalence test is the gate).
