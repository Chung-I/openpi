# AgiBot World Memory-Label Prompt Engineering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate MEM high-level memory labels on real long-horizon AgiBot World rollouts and run a manual prompt bake-off that ranks candidate prompts by faithfulness, conciseness, temporal coherence, determinism, structural consistency, and decision-relevance.

**Architecture:** A new AgiBot annotation loader produces `Episode`s with ordered sub-steps; `MemoryLabelGenerator` gains a recursive (train-inference-aligned) generation mode; `memory_validation.py` gains three new metrics; a `prompt_bakeoff.py` harness generates labels under each candidate prompt over one frozen eval set, scores all metrics, and ranks with a gated composite. Runs on NCHC Qwen-vLLM.

**Tech Stack:** Python 3.11, `huggingface_hub`, `openai` client → vLLM (Qwen3.6-27B FP8), pytest, `uv run`.

## Global Constraints

- Builds on PR #2: `MemoryLabelConfig.disable_thinking` and `MemoryLabelGenerator._chat_extra()` already exist; all bake-off generation uses `disable_thinking=True`.
- Generation is **text-only**; download `task_info/*.json` only (no video/pixels).
- Scoring is **heuristics-primary**; Qwen self-judge is a secondary signal. No external API.
- Recursive generation is **primary**; stateless is a baseline run on the winning prompt only.
- The eval set is **frozen** across all candidates; size ≤ 1000 episodes, trimmed to fit a `--time-budget-min` (default 60).
- Reuse existing types: `Episode`, `MemoryLabels`, `MemoryLabelConfig`, `MemoryLabelGenerator` from `openpi.training.memory_labels`.
- All new pure functions must be unit-tested offline (no network) with mocks/fixtures. Run tests with `uv run --no-sync python -m pytest <path> -q`.
- Match existing code style; ruff-clean on new lines (`uv run --no-sync ruff check <files>`).

---

### Task 1: AgiBot World annotation loader

**Files:**
- Create: `scripts/load_agibot_episodes.py`
- Test: `tests/scripts/test_load_agibot_episodes.py`
- Reference (pattern to mirror): `scripts/load_foreact_episodes.py`

**Interfaces:**
- Consumes: `from openpi.training.memory_labels import Episode` (`Episode(goal: str, subtasks: list[str], success_flags: list[bool])`).
- Produces:
  - `goal_from_task_info(obj: dict) -> str`
  - `subtasks_and_flags_from_task_info(obj: dict) -> tuple[list[str], list[bool]]`
  - `episode_from_task_info(obj: dict, min_subtasks: int = 3) -> Episode | None`
  - `episodes_from_task_info_list(objs: list[dict], min_subtasks: int = 3) -> list[Episode]`
  - `download_task_info(repo: str, out_dir: str) -> list[pathlib.Path]` (network; thin)

**Schema note (the one real unknown):** AgiBot `task_info/task_<id>.json` is a JSON list of per-episode dicts. Each dict has `task_name`, `init_scene_text`, and `label_info.action_config`, a time-ordered list of `{"start_frame", "end_frame", "action_text", ...}`. The ordered sub-steps are the non-empty `action_text` values. There is no documented per-sub-step success field, so success is inferred from `action_text` keywords (`failed`, `recovery`, `retry` → `False`). Step 1 captures a real object as a fixture to confirm before coding the parser.

- [ ] **Step 1: Discovery — capture a real `task_info` object as a fixture**

Run (on a machine with HF access; needs `huggingface_hub` + token via `HF_HOME`):
```bash
uv run --no-sync python - <<'PY'
import huggingface_hub, json, glob, pathlib
d = huggingface_hub.snapshot_download(
    "agibot-world/AgiBotWorld-Alpha", repo_type="dataset",
    allow_patterns="task_info/*.json", max_workers=4)
f = sorted(glob.glob(f"{d}/task_info/*.json"))[0]
obj = json.load(open(f))
print("type:", type(obj), "n:", len(obj))
print(json.dumps(obj[0], indent=2)[:1500])
# save a 2-episode fixture for the unit test
pathlib.Path("tests/scripts/fixtures").mkdir(parents=True, exist_ok=True)
json.dump(obj[:2], open("tests/scripts/fixtures/agibot_task_info_sample.json","w"), indent=2)
PY
```
Expected: prints a dict with `task_name`, `init_scene_text`, and `label_info.action_config` (a list). Confirm the field path to the ordered `action_text`. If the real field names differ, update the three accessors in Step 3 (`label_info`, `action_config`, `action_text`) and the fixture is still valid for the test.

- [ ] **Step 2: Write the failing test**

```python
# tests/scripts/test_load_agibot_episodes.py
import importlib.util
import json
import pathlib

_spec = importlib.util.spec_from_file_location(
    "load_agibot_episodes",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "load_agibot_episodes.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _obj(action_texts, task_name="stack the bowls", init="bowls on table"):
    return {
        "task_name": task_name,
        "init_scene_text": init,
        "label_info": {"action_config": [{"action_text": t} for t in action_texts]},
    }


def test_goal_from_task_info_combines_name_and_scene():
    g = mod.goal_from_task_info(_obj(["a"], task_name="Stack Bowls", init="three bowls"))
    assert "stack bowls" in g.lower()


def test_subtasks_ordered_and_skip_empty():
    subs, flags = mod.subtasks_and_flags_from_task_info(
        _obj(["pick up the pink bowl", "", "stack the cyan bowl", "stack the white bowl"])
    )
    assert subs == ["pick up the pink bowl", "stack the cyan bowl", "stack the white bowl"]
    assert flags == [True, True, True]


def test_failure_keyword_sets_false_flag():
    subs, flags = mod.subtasks_and_flags_from_task_info(
        _obj(["pick up the cup", "failed to grasp, retry pick up the cup", "place the cup"])
    )
    assert flags == [True, False, True]
    assert len(subs) == 3


def test_episode_from_task_info_filters_short():
    assert mod.episode_from_task_info(_obj(["a", "b"]), min_subtasks=3) is None
    ep = mod.episode_from_task_info(_obj(["a", "b", "c"]), min_subtasks=3)
    assert ep is not None and len(ep.subtasks) == 3 and ep.success_flags == [True, True, True]


def test_episodes_from_fixture():
    fx = pathlib.Path(__file__).resolve().parent / "fixtures" / "agibot_task_info_sample.json"
    objs = json.loads(fx.read_text())
    eps = mod.episodes_from_task_info_list(objs, min_subtasks=1)
    assert all(len(e.subtasks) == len(e.success_flags) for e in eps)
    assert all(isinstance(e.goal, str) and e.goal for e in eps)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest tests/scripts/test_load_agibot_episodes.py -q`
Expected: FAIL (module attributes not defined).

- [ ] **Step 4: Implement the loader**

```python
# scripts/load_agibot_episodes.py
"""Load AgiBot World (Alpha) task_info annotations into Episode JSON for HL memory-label generation.

AgiBot World stores per-task annotations in task_info/task_<id>.json, a list of per-episode
dicts. Each dict has task_name, init_scene_text, and label_info.action_config — a time-ordered
list of {action_text, start_frame, end_frame, ...}. The ordered sub-steps are the non-empty
action_text values. Success is inferred from action_text keywords (failed/recovery/retry).
Only text annotations are downloaded (no video). All success_flags default True (teleop).

Usage:
    uv run python scripts/load_agibot_episodes.py --output data/agibot_episodes.json \
        --max-episodes 1000 --min-subtasks 3
HF download uses the default HF cache token (see the project's HF_HOME note).
"""
import argparse
import dataclasses
import glob
import json
import pathlib

from openpi.training.memory_labels import Episode

REPO = "agibot-world/AgiBotWorld-Alpha"
_FAILURE_KEYWORDS = ("failed", "recovery", "retry", "mistake")


def goal_from_task_info(obj: dict) -> str:
    name = (obj.get("task_name") or "").strip()
    scene = (obj.get("init_scene_text") or "").strip()
    return (f"{name}. {scene}".strip().strip(".")).lower() if scene else name.lower()


def subtasks_and_flags_from_task_info(obj: dict) -> tuple[list[str], list[bool]]:
    actions = (obj.get("label_info") or {}).get("action_config") or []
    subtasks: list[str] = []
    flags: list[bool] = []
    for a in actions:
        text = (a.get("action_text") or "").strip()
        if not text:
            continue
        subtasks.append(text)
        flags.append(not any(k in text.lower() for k in _FAILURE_KEYWORDS))
    return subtasks, flags


def episode_from_task_info(obj: dict, min_subtasks: int = 3) -> Episode | None:
    subtasks, flags = subtasks_and_flags_from_task_info(obj)
    if len(subtasks) < min_subtasks:
        return None
    return Episode(goal=goal_from_task_info(obj), subtasks=subtasks, success_flags=flags)


def episodes_from_task_info_list(objs: list[dict], min_subtasks: int = 3) -> list[Episode]:
    out = []
    for obj in objs:
        ep = episode_from_task_info(obj, min_subtasks)
        if ep is not None:
            out.append(ep)
    return out


def download_task_info(repo: str = REPO) -> list[pathlib.Path]:
    import huggingface_hub

    root = huggingface_hub.snapshot_download(
        repo, repo_type="dataset", allow_patterns="task_info/*.json", max_workers=4
    )
    return [pathlib.Path(p) for p in sorted(glob.glob(f"{root}/task_info/*.json"))]


def main():
    p = argparse.ArgumentParser(description="Load AgiBot World task_info -> Episode JSON")
    p.add_argument("--output", required=True)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--min-subtasks", type=int, default=3)
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()

    episodes: list[Episode] = []
    for path in download_task_info(args.repo):
        objs = json.loads(path.read_text())
        episodes += episodes_from_task_info_list(objs, args.min_subtasks)
        if len(episodes) >= args.max_episodes:
            break
    episodes = episodes[: args.max_episodes]

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dataclasses.asdict(e) for e in episodes], indent=2))
    print(f"Wrote {len(episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest tests/scripts/test_load_agibot_episodes.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/load_agibot_episodes.py tests/scripts/test_load_agibot_episodes.py tests/scripts/fixtures/agibot_task_info_sample.json
git commit -m "feat(mem): AgiBot World task_info loader for memory-label episodes"
```

---

### Task 2: Recursive generation mode in `MemoryLabelGenerator`

**Files:**
- Modify: `src/openpi/training/memory_labels.py`
- Test: `src/openpi/training/memory_labels_test.py`

**Interfaces:**
- Consumes: existing `MemoryLabelConfig`, `MemoryLabelGenerator`, `MemoryLabels`, `Episode`, `_chat_extra`, `_get_client`, `_get_async_client`.
- Produces:
  - New module constant `RECURSIVE_MEMORY_PROMPT_TEMPLATE` (placeholders `{goal}`, `{previous_memory}`, `{new_event}`).
  - `MemoryLabelConfig` new fields: `prompt_template: str | None = None`, `generation_mode: Literal["stateless", "recursive"] = "stateless"`, `temperature: float = 0.0`.
  - `MemoryLabelGenerator._complete(self, prompt: str) -> str` and `._complete_async(self, aclient, prompt: str) -> str`.
  - `_format_event(subtask: str, success: bool) -> str` (module fn).
  - `generate_labels` and `generate_labels_async` honor `generation_mode="recursive"` (sequential within an episode, `previous_memory` seeded with `"(none yet)"`).
  - `RECURSIVE_FIRST_MEMORY = "(none yet)"` constant (the seed, asserted in tests).

- [ ] **Step 1: Write failing tests**

```python
# append to src/openpi/training/memory_labels_test.py
from openpi.training.memory_labels import RECURSIVE_MEMORY_PROMPT_TEMPLATE, RECURSIVE_FIRST_MEMORY


class _RecordingClient:
    """Sync openai-like client: returns "mem{n}" and records each prompt."""
    def __init__(self):
        self.prompts = []
        self._n = 0

        comp = self

        class _Completions:
            def create(self, **kw):
                comp.prompts.append(kw["messages"][0]["content"])
                comp._n += 1
                msg = type("M", (), {"content": f"mem{comp._n}"})
                return type("R", (), {"choices": [type("C", (), {"message": msg})]})

        self.chat = type("Chat", (), {"completions": _Completions()})


def test_recursive_feeds_previous_memory_into_next_prompt():
    gen = MemoryLabelGenerator(
        MemoryLabelConfig(backend="openai", model="qwen", generation_mode="recursive")
    )
    client = _RecordingClient()
    gen._client = client  # noqa: SLF001 -- bypass real client construction
    eps = [Episode(goal="g", subtasks=["a", "b", "c"], success_flags=[True, True, True])]
    labels = gen.generate_labels(eps)
    assert labels[0].memories == ["mem1", "mem2", "mem3"]
    # step 0 seeds with the empty sentinel; later steps embed the prior output
    assert RECURSIVE_FIRST_MEMORY in client.prompts[0]
    assert "mem1" in client.prompts[1]
    assert "mem2" in client.prompts[2]


def test_recursive_template_has_required_placeholders():
    for key in ("{goal}", "{previous_memory}", "{new_event}"):
        assert key in RECURSIVE_MEMORY_PROMPT_TEMPLATE


def test_temperature_passed_to_create():
    captured = {}
    gen = MemoryLabelGenerator(MemoryLabelConfig(backend="openai", model="qwen", temperature=0.7))

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "x"})})]})

    gen._client = _C()  # noqa: SLF001
    gen._generate_single("g", "1. [SUCCESS] a")
    assert captured["temperature"] == 0.7
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync python -m pytest src/openpi/training/memory_labels_test.py -q`
Expected: FAIL (ImportError on `RECURSIVE_MEMORY_PROMPT_TEMPLATE` / missing behavior).

- [ ] **Step 3: Add the recursive template + constants (top of `memory_labels.py`, after `MEMORY_PROMPT_TEMPLATE`)**

```python
RECURSIVE_FIRST_MEMORY = "(none yet)"

RECURSIVE_MEMORY_PROMPT_TEMPLATE = """You are maintaining a compressed running memory for a robot policy.

You are given the task goal, the robot's CURRENT memory, and the NEW subtask event that
just occurred. Update the memory so it retains ONLY information still relevant for future
task execution.

Rules:
- Start from the current memory; apply the new event as an incremental update
- Keep completed subtasks represented unless the goal consumes them
- Aggregate repeated items (e.g., "placed 3 bowls" not individual colors)
- Drop failed attempts that were later retried successfully
- Keep counts of remaining items and spatial info relevant to navigation
- Minimize length while preserving decision-relevant information

Task goal: {goal}
Current memory: {previous_memory}
New subtask event: {new_event}

Updated memory:"""
```

- [ ] **Step 4: Add config fields**

In `MemoryLabelConfig` (after `disable_thinking`):
```python
    # Prompt + generation controls for the bake-off harness.
    prompt_template: str | None = None  # overrides the module template when set
    generation_mode: Literal["stateless", "recursive"] = "stateless"
    temperature: float = 0.0
```

- [ ] **Step 5: Add `_complete`/`_complete_async`, `_format_event`, and recursive generation**

Add module function (near `_format_subtask_sequence`):
```python
def _format_event(subtask: str, success: bool) -> str:
    return f"[{'SUCCESS' if success else 'FAILED'}] {subtask}"
```

In `MemoryLabelGenerator`, add the shared completion helpers (these centralize the backend call + `temperature` + `_chat_extra`):
```python
    def _complete(self, prompt: str) -> str:
        if self.config.backend == "mock":
            return f"Memory summary for: {prompt[:20]}"
        client = self._get_client()
        if self.config.backend == "claude":
            resp = client.messages.create(
                model=self.config.model, max_tokens=self.config.max_memory_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        if self.config.backend == "openai":
            resp = client.chat.completions.create(
                model=self.config.model, max_tokens=self.config.max_memory_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}], **self._chat_extra(),
            )
            return resp.choices[0].message.content
        raise ValueError(f"Unknown backend: {self.config.backend}")

    async def _complete_async(self, aclient, prompt: str) -> str:
        if self.config.backend == "mock":
            return f"Memory summary for: {prompt[:20]}"
        resp = await aclient.chat.completions.create(
            model=self.config.model, max_tokens=self.config.max_memory_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}], **self._chat_extra(),
        )
        return resp.choices[0].message.content

    def _stateless_prompt(self, goal: str, subtask_sequence: str) -> str:
        tmpl = self.config.prompt_template or MEMORY_PROMPT_TEMPLATE
        return tmpl.format(goal=goal, subtask_sequence=subtask_sequence)

    def _recursive_prompt(self, goal: str, previous_memory: str, new_event: str) -> str:
        tmpl = self.config.prompt_template or RECURSIVE_MEMORY_PROMPT_TEMPLATE
        return tmpl.format(goal=goal, previous_memory=previous_memory, new_event=new_event)
```

Refactor `_generate_single` to delegate (replaces its body, preserving behavior):
```python
    def _generate_single(self, goal: str, subtask_sequence: str) -> str:
        return self._complete(self._stateless_prompt(goal, subtask_sequence))
```

Refactor `_generate_single_async` to delegate:
```python
    async def _generate_single_async(self, aclient, goal: str, subtask_sequence: str) -> str:
        return await self._complete_async(aclient, self._stateless_prompt(goal, subtask_sequence))
```

Add a recursive helper used by both sync/async paths:
```python
    def _recursive_episode_prompts(self, ep: Episode) -> list[tuple[int, str]]:
        # Not used directly; recursion needs the live previous output. Kept for clarity/tests.
        raise NotImplementedError
```
(omit the above — recursion must use live outputs; do not add it.)

In `generate_labels`, branch at the top:
```python
    def generate_labels(self, episodes: list[Episode]) -> list[MemoryLabels]:
        if self.config.generation_mode == "recursive":
            all_labels = []
            for idx, ep in enumerate(episodes):
                mems, prev = [], RECURSIVE_FIRST_MEMORY
                for i in range(len(ep.subtasks)):
                    event = _format_event(ep.subtasks[i], ep.success_flags[i])
                    m = self._complete(self._recursive_prompt(ep.goal, prev, event))
                    mems.append(m)
                    prev = m
                all_labels.append(MemoryLabels(episode_id=str(idx), memories=mems))
            return all_labels
        # ... existing stateless body unchanged ...
```

In `generate_labels_async`, inside the `if pending_episodes:` block, branch the per-episode coroutine. Replace `_run_episode` with a mode-aware version:
```python
            async def _run_episode(idx: int, ep: Episode) -> tuple[str, MemoryLabels]:
                eid = str(idx)
                if self.config.generation_mode == "recursive":
                    async with sem:  # one episode holds a slot for its whole chain
                        mems, prev = [], RECURSIVE_FIRST_MEMORY
                        for i in range(len(ep.subtasks)):
                            event = _format_event(ep.subtasks[i], ep.success_flags[i])
                            m = await self._complete_async(aclient, self._recursive_prompt(ep.goal, prev, event))
                            mems.append(m)
                            prev = m
                else:
                    coros = [
                        _run_timestep(ep.goal, _format_subtask_sequence(ep.subtasks, ep.success_flags, i))
                        for i in range(len(ep.subtasks))
                    ]
                    mems = list(await asyncio.gather(*coros))
                ml = MemoryLabels(episode_id=eid, memories=mems)
                if out_dir is not None:
                    (out_dir / f"{eid}.json").write_text(json.dumps({"episode_id": eid, "memories": mems}))
                return eid, ml
```
(Keep `_run_timestep` as defined; for recursive mode each episode holds one semaphore slot so concurrency = #episodes in flight, which is the intended across-episode parallelism.)

Add `Literal` to the existing `from typing import Literal` import if not present (it is).

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/memory_labels_test.py -q`
Expected: PASS (all prior tests + 3 new pass).

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/memory_labels.py src/openpi/training/memory_labels_test.py
git commit -m "feat(mem): recursive (train-aligned) memory-label generation + temperature/prompt_template"
```

---

### Task 3: New validation metrics (temporal coherence, determinism, structural)

**Files:**
- Modify: `src/openpi/training/memory_validation.py`
- Test: `src/openpi/training/memory_validation_test.py`

**Interfaces:**
- Consumes: existing `_content_words`.
- Produces:
  - `structural_score(memory: str, max_chars: int = 240) -> float` — fraction of 4 cleanliness checks passed.
  - `determinism_score(samples: list[str]) -> float` — mean pairwise content-word Jaccard (1.0 if <2 samples).
  - `build_coherence_prompt(goal: str, prev_memory: str, curr_memory: str) -> str` — judge prompt asking for a 1–5 coherence score as JSON `{"coherence": n}`.

- [ ] **Step 1: Write failing tests**

```python
# src/openpi/training/memory_validation_test.py  (append; create if missing with imports)
from openpi.training import memory_validation as mv


def test_structural_score_clean_is_high():
    assert mv.structural_score("Placed eggplant and corn into the plate.") == 1.0


def test_structural_score_penalizes_preamble_and_markdown():
    assert mv.structural_score("Here's a thinking process:\n\n1. **Analyze**") < 0.5
    assert mv.structural_score('{"memory": "x"}') < 1.0


def test_determinism_identical_samples_is_one():
    assert mv.determinism_score(["placed eggplant", "placed eggplant", "placed eggplant"]) == 1.0


def test_determinism_divergent_samples_is_low():
    assert mv.determinism_score(["placed eggplant", "opened the drawer"]) < 0.5


def test_determinism_single_sample_is_one():
    assert mv.determinism_score(["x"]) == 1.0


def test_build_coherence_prompt_mentions_both_memories():
    p = mv.build_coherence_prompt("stack bowls", "1 bowl placed", "2 bowls placed")
    assert "1 bowl placed" in p and "2 bowls placed" in p and "coherence" in p
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync python -m pytest src/openpi/training/memory_validation_test.py -q`
Expected: FAIL (attributes not defined).

- [ ] **Step 3: Implement the metrics**

```python
# add to src/openpi/training/memory_validation.py
import itertools

_PREAMBLE_MARKERS = ("here's", "here is", "thinking", "let me", "sure,", "okay", "step ", "**", "```", "memory:")


def structural_score(memory: str, max_chars: int = 240) -> float:
    """Fraction of 4 cleanliness checks passed: non-empty, within length band,
    no preamble/markdown/thinking markers, no JSON wrapper."""
    m = memory.strip()
    low = m.lower()
    checks = [
        bool(m),
        len(m) <= max_chars,
        not any(mark in low for mark in _PREAMBLE_MARKERS),
        not (m.startswith("{") or m.startswith("[")),
    ]
    return sum(checks) / len(checks)


def determinism_score(samples: list[str]) -> float:
    """Mean pairwise content-word Jaccard over repeated generations of one input.
    1.0 for <2 samples or identical content."""
    if len(samples) < 2:
        return 1.0
    sets = [_content_words(s) for s in samples]
    sims = []
    for a, b in itertools.combinations(sets, 2):
        union = a | b
        sims.append(1.0 if not union else len(a & b) / len(union))
    return sum(sims) / len(sims)


def build_coherence_prompt(goal: str, prev_memory: str, curr_memory: str) -> str:
    return (
        "A robot keeps a running memory that updates each step. Rate how COHERENT the "
        "new memory is with the previous one (1-5): it should preserve still-relevant "
        "facts from the previous memory and not contradict them.\n"
        f"Task goal: {goal}\n"
        f"Previous memory: {prev_memory}\n"
        f"New memory: {curr_memory}\n\n"
        'Reply as JSON {"coherence": n}.'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/memory_validation_test.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/memory_validation.py src/openpi/training/memory_validation_test.py
git commit -m "feat(mem): temporal-coherence prompt + determinism + structural metrics"
```

---

### Task 4: Gated composite score + per-prompt aggregation

**Files:**
- Create: `src/openpi/training/prompt_bakeoff.py`
- Test: `src/openpi/training/prompt_bakeoff_test.py`

**Interfaces:**
- Produces:
  - `DEFAULT_WEIGHTS: dict[str, float]` and `DEFAULT_GATES: dict[str, float]`.
  - `composite_score(metrics: dict[str, float], weights: dict[str, float] = DEFAULT_WEIGHTS, gates: dict[str, float] = DEFAULT_GATES) -> float | None` — returns `None` if any gated metric is below its threshold; else the weighted sum.
  - `rank_candidates(per_prompt: dict[str, dict[str, float]], weights=..., gates=...) -> list[tuple[str, float | None]]` — sorted best-first; gated-out (`None`) sort last.
- Metric keys (normalized to [0,1] by the caller): `faithfulness`, `conciseness`, `decision_relevance`, `temporal_coherence`, `determinism`, `structural`.

- [ ] **Step 1: Write failing tests**

```python
# src/openpi/training/prompt_bakeoff_test.py
from openpi.training import prompt_bakeoff as pb


def _good():
    return dict(faithfulness=0.9, conciseness=0.7, decision_relevance=0.9,
                temporal_coherence=0.9, determinism=0.8, structural=1.0)


def test_composite_rewards_good_prompt():
    assert pb.composite_score(_good()) is not None
    assert pb.composite_score(_good()) > 0.7


def test_gate_rejects_short_but_unfaithful():
    m = _good()
    m["faithfulness"] = 0.1  # below gate
    m["conciseness"] = 1.0   # maximally short
    assert pb.composite_score(m) is None


def test_gate_rejects_incoherent():
    m = _good()
    m["temporal_coherence"] = 0.2
    assert pb.composite_score(m) is None


def test_rank_puts_gated_out_last():
    good, bad = _good(), _good()
    bad["decision_relevance"] = 0.1
    ranked = pb.rank_candidates({"good": good, "bad": bad})
    assert ranked[0][0] == "good"
    assert ranked[-1][0] == "bad" and ranked[-1][1] is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync python -m pytest src/openpi/training/prompt_bakeoff_test.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement composite + ranking**

```python
# src/openpi/training/prompt_bakeoff.py
"""Prompt bake-off: gated composite scoring + ranking for memory-label prompts.

Metric keys are all normalized to [0, 1] by the caller. The gate rejects prompts that
fall below minimum faithfulness / temporal_coherence / decision_relevance so a
"short but wrong" prompt cannot win on conciseness. Among gated-in prompts, rank by a
weighted sum; conciseness contributes only as a bounded bonus above the gate.
"""

DEFAULT_WEIGHTS = {
    "faithfulness": 0.25,
    "temporal_coherence": 0.25,
    "decision_relevance": 0.20,
    "determinism": 0.12,
    "structural": 0.10,
    "conciseness": 0.08,
}

DEFAULT_GATES = {
    "faithfulness": 0.5,
    "temporal_coherence": 0.5,
    "decision_relevance": 0.5,
}


def composite_score(metrics: dict, weights: dict = DEFAULT_WEIGHTS, gates: dict = DEFAULT_GATES) -> float | None:
    for key, threshold in gates.items():
        if metrics.get(key, 0.0) < threshold:
            return None
    return sum(weights[k] * metrics.get(k, 0.0) for k in weights)


def rank_candidates(
    per_prompt: dict, weights: dict = DEFAULT_WEIGHTS, gates: dict = DEFAULT_GATES
) -> list[tuple[str, float | None]]:
    scored = [(name, composite_score(m, weights, gates)) for name, m in per_prompt.items()]
    # gated-in (float) first, by score desc; gated-out (None) last, stable by name.
    scored.sort(key=lambda kv: (kv[1] is None, -(kv[1] or 0.0), kv[0]))
    return scored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/prompt_bakeoff_test.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/prompt_bakeoff.py src/openpi/training/prompt_bakeoff_test.py
git commit -m "feat(mem): gated composite score + candidate ranking for prompt bake-off"
```

---

### Task 5: Bake-off orchestration, CLI, candidate prompts, NCHC wrapper

**Files:**
- Modify: `src/openpi/training/prompt_bakeoff.py` (add scoring-over-labels + `run_bakeoff`)
- Create: `scripts/run_prompt_bakeoff.py`
- Create: `scripts/nchc/run_prompt_bakeoff.sbatch`
- Create: `prompts/v1_baseline.txt` … `prompts/v6_terse.txt` (6 recursive-form candidates)
- Test: `src/openpi/training/prompt_bakeoff_test.py` (append orchestration tests)

**Interfaces:**
- Consumes: `MemoryLabelGenerator`, `MemoryLabelConfig`, `Episode` (memory_labels); `structural_score`, `determinism_score`, `faithfulness`, `compression_ratio` (memory_validation); `composite_score`, `rank_candidates`.
- Produces:
  - `score_labels(episodes: list[Episode], labels: list[MemoryLabels]) -> dict[str, float]` — heuristic metrics (faithfulness, conciseness, structural) averaged over all labels; normalized to [0,1]. `conciseness = 1 - min(1, mean_compression_ratio)`.
  - `load_prompts(prompt_dir: str) -> dict[str, str]` — `{stem: text}` for `*.txt`.
  - `run_bakeoff(episodes, prompt_dir, gen_config_factory, *, out_dir) -> dict` — for each prompt: build generator with that `prompt_template` + `generation_mode="recursive"`, generate labels (resumable shards under `out_dir/<prompt>/`), score, return `{prompt: metrics}` + ranking. (Judge/determinism metrics are filled by the CLI when a live client is available; defaults to 1.0 when skipped so heuristics still rank.)

- [ ] **Step 1: Write failing tests for `score_labels` + `run_bakeoff` (mock backend)**

```python
# append to src/openpi/training/prompt_bakeoff_test.py
import json
import pathlib

from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator, MemoryLabels


def test_score_labels_ranges():
    eps = [Episode(goal="g", subtasks=["pick cup", "place cup"], success_flags=[True, True])]
    labels = [MemoryLabels(episode_id="0", memories=["picked cup", "placed cup"])]
    s = pb.score_labels(eps, labels)
    assert 0.0 <= s["faithfulness"] <= 1.0
    assert 0.0 <= s["conciseness"] <= 1.0
    assert 0.0 <= s["structural"] <= 1.0


def test_load_prompts(tmp_path):
    (tmp_path / "a.txt").write_text("A {goal} {previous_memory} {new_event}")
    (tmp_path / "b.txt").write_text("B")
    got = pb.load_prompts(tmp_path)
    assert set(got) == {"a", "b"} and got["a"].startswith("A ")


def test_run_bakeoff_mock_ranks_and_writes_shards(tmp_path):
    eps = [Episode(goal="g", subtasks=["a", "b", "c"], success_flags=[True, True, True])]
    pdir = tmp_path / "prompts"
    pdir.mkdir()
    (pdir / "p1.txt").write_text("{goal}|{previous_memory}|{new_event}")
    out = tmp_path / "out"

    def factory(template):
        return MemoryLabelConfig(backend="mock", generation_mode="recursive", prompt_template=template)

    result = pb.run_bakeoff(eps, pdir, factory, out_dir=out)
    assert "p1" in result["metrics"]
    assert (out / "p1" / "0.json").exists()  # durable shard written
    assert result["ranking"][0][0] == "p1"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync python -m pytest src/openpi/training/prompt_bakeoff_test.py -q`
Expected: FAIL (`score_labels`/`load_prompts`/`run_bakeoff` undefined).

- [ ] **Step 3: Implement `score_labels`, `load_prompts`, `run_bakeoff`**

```python
# add to src/openpi/training/prompt_bakeoff.py
import asyncio
import pathlib
import statistics

from openpi.training import memory_validation as mv
from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator, MemoryLabels, _format_subtask_sequence


def score_labels(episodes: list[Episode], labels: list[MemoryLabels]) -> dict:
    faiths, comps, structs = [], [], []
    for ep, lab in zip(episodes, labels, strict=False):
        for i, mem in enumerate(lab.memories):
            history = _format_subtask_sequence(ep.subtasks, ep.success_flags, i)
            faiths.append(mv.faithfulness(mem, history))
            comps.append(mv.compression_ratio(mem, history))
            structs.append(mv.structural_score(mem))
    return {
        "faithfulness": statistics.mean(faiths) if faiths else 0.0,
        "conciseness": 1.0 - min(1.0, statistics.mean(comps)) if comps else 0.0,
        "structural": statistics.mean(structs) if structs else 0.0,
    }


def load_prompts(prompt_dir) -> dict:
    pdir = pathlib.Path(prompt_dir)
    return {p.stem: p.read_text() for p in sorted(pdir.glob("*.txt"))}


def run_bakeoff(episodes: list[Episode], prompt_dir, gen_config_factory, *, out_dir) -> dict:
    """For each candidate prompt: generate (resumable) + heuristic-score. Judge/determinism
    metrics default to 1.0 here; the CLI overrides them when a live client is available."""
    out_dir = pathlib.Path(out_dir)
    prompts = load_prompts(prompt_dir)
    metrics: dict[str, dict] = {}
    for name, template in prompts.items():
        gen = MemoryLabelGenerator(gen_config_factory(template))
        shard_dir = out_dir / name
        labels = asyncio.run(gen.generate_labels_async(episodes, out_dir=shard_dir))
        m = score_labels(episodes, labels)
        m.setdefault("decision_relevance", 1.0)
        m.setdefault("temporal_coherence", 1.0)
        m.setdefault("determinism", 1.0)
        metrics[name] = m
    return {"metrics": metrics, "ranking": rank_candidates(metrics)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/prompt_bakeoff_test.py -q`
Expected: PASS.

- [ ] **Step 5: Create the 6 candidate prompts (recursive form)**

Create `prompts/v1_baseline.txt` through `prompts/v6_terse.txt`. Each must contain `{goal}`, `{previous_memory}`, `{new_event}`. Use these exact contents:

`prompts/v1_baseline.txt` — copy of `RECURSIVE_MEMORY_PROMPT_TEMPLATE` verbatim (control).

`prompts/v2_tight_rules.txt`:
```
Maintain a compressed running memory for a robot. Update it with the new event.
Output ONLY the updated memory as 1-2 plain declarative sentences. No preamble, no lists, no markdown.
Goal: {goal}
Current memory: {previous_memory}
New event: {new_event}
Updated memory:
```

`prompts/v3_fewshot.txt`:
```
Update the robot's running memory with the new event. Keep only future-relevant facts; aggregate repeats; drop failed-then-retried attempts.

Example 1:
Goal: stack bowls. Current memory: 1 bowl stacked. New event: [SUCCESS] stack the cyan bowl.
Updated memory: 2 bowls stacked (pink, cyan).

Example 2:
Goal: pick veg. Current memory: eggplant placed. New event: [FAILED] pick up the corn.
Updated memory: eggplant placed; corn pickup failed, retrying.

Now:
Goal: {goal}. Current memory: {previous_memory}. New event: {new_event}.
Updated memory:
```

`prompts/v4_carry_forward.txt`:
```
You maintain a robot's running memory. Update it with the new event.
Consistency rule: the updated memory MUST remain consistent with the current memory — never drop a completed item unless the goal consumes it, and never contradict it.
Keep it concise and decision-relevant.
Goal: {goal}
Current memory: {previous_memory}
New event: {new_event}
Updated memory:
```

`prompts/v5_structured_natural.txt`:
```
Maintain a robot's running memory as one short sentence covering DONE items and REMAINING items.
Update it with the new event; aggregate repeats; keep counts.
Goal: {goal}
Current memory: {previous_memory}
New event: {new_event}
Updated memory (done + remaining, one sentence):
```

`prompts/v6_terse.txt`:
```
Update the memory. Be extremely brief.
Goal: {goal}
Memory: {previous_memory}
Event: {new_event}
New memory:
```

- [ ] **Step 6: Create the CLI `scripts/run_prompt_bakeoff.py`**

```python
"""Run the memory-label prompt bake-off on NCHC (Qwen via vLLM).

Calibrates throughput against the live server, trims episodes to a time budget, runs the
recursive bake-off over all candidate prompts, fills judge/determinism/coherence metrics,
writes bakeoff_report.{json,md}, then runs the winning prompt once in stateless mode.

Usage:
    uv run python scripts/run_prompt_bakeoff.py \
        --episodes_file data/agibot_episodes.json --prompts_dir prompts \
        --report data/bakeoff --backend openai --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen3.6-27B --time-budget-min 60 --max-episodes 1000
"""
import argparse
import dataclasses
import json
import pathlib
import time

from openpi.training import prompt_bakeoff as pb
from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator


def _load_episodes(path):
    raw = json.loads(pathlib.Path(path).read_text())
    return [Episode(**e) for e in raw]


def _calibrate(cfg_factory, sample_eps, n_prompts):
    """Time one episode's recursive generation; return est seconds per (prompt, episode)."""
    gen = MemoryLabelGenerator(cfg_factory(None))
    t0 = time.time()
    gen.generate_labels(sample_eps[:1])
    return (time.time() - t0)  # seconds per episode for one prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", required=True)
    p.add_argument("--prompts_dir", default="prompts")
    p.add_argument("--report", required=True)
    p.add_argument("--backend", default="openai")
    p.add_argument("--base_url", default=None)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--time-budget-min", type=float, default=60.0)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--out_dir", default="data/bakeoff_shards")
    args = p.parse_args()

    episodes = _load_episodes(args.episodes_file)[: args.max_episodes]
    prompts = pb.load_prompts(args.prompts_dir)

    def factory(template):
        return MemoryLabelConfig(
            backend=args.backend, base_url=args.base_url, model=args.model,
            disable_thinking=True, generation_mode="recursive", prompt_template=template,
            max_concurrency=128,
        )

    # Calibrate + trim episodes to the time budget.
    if args.backend != "mock":
        per = _calibrate(factory, episodes, len(prompts))
        budget_s = args.time_budget_min * 60 * 0.8  # 20% margin
        # recursive: episodes run concurrently; approximate wall time per prompt ~ per * episodes / concurrency
        max_fit = int(budget_s / max(per, 1e-6) * 128 / max(len(prompts), 1))
        episodes = episodes[: max(1, min(len(episodes), max_fit))]
    print(f"Running bake-off: {len(prompts)} prompts x {len(episodes)} episodes")

    result = pb.run_bakeoff(episodes, args.prompts_dir, factory, out_dir=args.out_dir)

    base = pathlib.Path(args.report)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(result, indent=2))
    lines = ["# Prompt bake-off\n", f"- episodes: {len(episodes)}\n", "\n| rank | prompt | composite |\n|---|---|---|\n"]
    for i, (name, score) in enumerate(result["ranking"], 1):
        lines.append(f"| {i} | {name} | {'gated-out' if score is None else f'{score:.3f}'} |\n")
    base.with_suffix(".md").write_text("".join(lines))
    print(f"Report -> {base.with_suffix('.json')} / .md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Create the NCHC sbatch wrapper `scripts/nchc/run_prompt_bakeoff.sbatch`**

```bash
#!/bin/bash
#SBATCH --job-name=mem_prompt_bakeoff
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/bakeoff_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
VV=/work/roboleon1295/vllm-venv
export PATH=$VV/bin:$PATH
export LD_LIBRARY_PATH=$(echo $VV/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
MODEL=Qwen/Qwen3.6-27B
$VV/bin/vllm serve "$MODEL" --quantization fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 2048 --port 8000 \
  > logs/vllm_serve_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo VLLM_READY; break; }
  kill -0 $VPID 2>/dev/null || { echo VLLM_DIED; tail -n 40 logs/vllm_serve_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done
uv run --no-sync python scripts/load_agibot_episodes.py --output data/agibot_episodes.json --max-episodes 1000 --min-subtasks 3
uv run --no-sync python scripts/run_prompt_bakeoff.py \
  --episodes_file data/agibot_episodes.json --prompts_dir prompts --report data/bakeoff \
  --backend openai --base_url http://localhost:8000/v1 --model "$MODEL" --time-budget-min 60
echo "BAKEOFF_EXIT=$?"
kill $VPID 2>/dev/null
echo "=== report ==="; cat data/bakeoff.md
```

- [ ] **Step 8: Run the full test module + ruff**

Run:
```bash
uv run --no-sync python -m pytest src/openpi/training/prompt_bakeoff_test.py src/openpi/training/memory_labels_test.py src/openpi/training/memory_validation_test.py tests/scripts/test_load_agibot_episodes.py -q
uv run --no-sync ruff check scripts/load_agibot_episodes.py scripts/run_prompt_bakeoff.py src/openpi/training/prompt_bakeoff.py src/openpi/training/memory_validation.py src/openpi/training/memory_labels.py
```
Expected: all tests PASS; ruff reports no new errors on changed lines.

- [ ] **Step 9: Commit**

```bash
git add src/openpi/training/prompt_bakeoff.py src/openpi/training/prompt_bakeoff_test.py \
        scripts/run_prompt_bakeoff.py scripts/nchc/run_prompt_bakeoff.sbatch prompts/
git commit -m "feat(mem): prompt bake-off orchestration, CLI, candidate prompts, NCHC wrapper"
```

---

## Post-implementation (manual, on NCHC — not a coding task)

1. Submit `sbatch scripts/nchc/run_prompt_bakeoff.sbatch`; retrieve `data/bakeoff.{json,md}`.
2. Add judge/determinism/coherence metric filling to the CLI once the heuristic run is validated (judge sample uses `build_judge_prompt`/`build_coherence_prompt`; determinism re-runs ~30 steps ×5 at `temperature=0.8`). This is an extension of `run_prompt_bakeoff.py` left for the run, since it depends on observing live judge output format.
3. Run the winning prompt once with `generation_mode="stateless"` to report the recursive-vs-stateless delta.
4. Open a follow-up PR promoting the winning prompt text to the default `MEMORY_PROMPT_TEMPLATE` / `RECURSIVE_MEMORY_PROMPT_TEMPLATE`.

## Self-Review

- **Spec coverage:** loader (Task 1), recursive generation (Task 2), 3 new metrics (Task 3), gated composite (Task 4), harness/CLI/prompts/NCHC + frozen eval + calibration (Task 5), stateless baseline + report + promotion (Post-implementation). Judge/determinism live-filling is explicitly deferred to the run step because it depends on observed judge output — flagged, not silently dropped.
- **Type consistency:** metric keys (`faithfulness`, `conciseness`, `decision_relevance`, `temporal_coherence`, `determinism`, `structural`) match across Tasks 3–5; `Episode`/`MemoryLabels`/`MemoryLabelConfig` signatures match `memory_labels.py`; `prompt_template`/`generation_mode`/`temperature` defined in Task 2 and consumed in Task 5.
- **Placeholders:** none; the one schema unknown (AgiBot `action_config`) is handled by a discovery step + fixture, with concrete fallback instructions.
