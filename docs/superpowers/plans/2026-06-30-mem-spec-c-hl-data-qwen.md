# Spec C — HL data generation via Qwen@NCHC + validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate validated compressed memory-transition labels for MEM HL training from a subtask-annotated dataset (ForeAct), using an open-weights Qwen3.6-27B served via vLLM on NCHC (OpenAI-compatible, single GPU), with a high-concurrency async client and a validation report.

**Architecture:** Extend the existing `memory_labels.py` engine with a `base_url` (vLLM endpoint) + an async, concurrency-bounded, checkpointed generation path. Add a ForeAct→`Episode` loader, a validation module (CPU heuristics + sampled LLM-judge + reconstruction probe), and an NCHC vLLM Slurm serving script. Generate+validate only — feeding labels into training is a later spec.

**Tech Stack:** Python, `openai` async client, vLLM (served, FP8, prefix caching), HuggingFace `datasets`, pytest, `uv`.

## Global Constraints

- Run via `uv run`; for tests that import tensorflow-touching modules use `uv run --no-sync` (the env is set; do not `uv sync`/`uv pip install`).
- Branch: `mem-fidelity-fixes` (Specs A/A2/B committed).
- Scope: **generate + validate only** — no `hl_targets`→`compute_loss` wiring.
- Backend: Qwen via the existing `openai` backend + a new `base_url` (vLLM is OpenAI-compatible). vLLM ignores the API key → fall back to `"EMPTY"`.
- All-success v1: ForeAct loader sets `success_flags=[True]*len(subtasks)`.
- Keep `claude`/`mock` backends unchanged; `local` stays `NotImplementedError`.
- vLLM serve flags (deepwiki best-practice): `--quantization fp8 --enable-prefix-caching --gpu-memory-utilization 0.95 --max-num-seqs 256 --max-num-batched-tokens 16384 --max-model-len 2048`.
- Existing dataclasses: `Episode(goal: str, subtasks: list[str], success_flags: list[bool])`, `MemoryLabels(episode_id: str, memories: list[str])`. `MEMORY_PROMPT_TEMPLATE.format(goal=, subtask_sequence=)`. `_format_subtask_sequence(subtasks, success_flags, up_to_index) -> str`.

---

### Task 1: base_url + max_concurrency config

**Files:**
- Modify: `src/openpi/training/memory_labels.py` (`MemoryLabelConfig`, `_get_client`)
- Test: `src/openpi/training/memory_labels_test.py`

**Interfaces:**
- Produces: `MemoryLabelConfig.base_url: str | None = None`, `MemoryLabelConfig.max_concurrency: int = 64`. The `openai` sync client is built as `openai.OpenAI(base_url=config.base_url, api_key=<env> or "EMPTY")`.

- [ ] **Step 1: Write the failing test**

Append to `src/openpi/training/memory_labels_test.py`:

```python
def test_openai_client_uses_base_url(monkeypatch):
    import openai
    captured = {}

    class _FakeOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = MemoryLabelConfig(backend="openai", base_url="http://localhost:8000/v1", model="qwen")
    gen = MemoryLabelGenerator(config)
    gen._get_client()
    assert captured["base_url"] == "http://localhost:8000/v1"
    assert captured["api_key"] == "EMPTY"  # vLLM ignores key; fall back when env unset


def test_config_has_concurrency_default():
    assert MemoryLabelConfig().max_concurrency == 64
```

- [ ] **Step 2: Run to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_labels_test.py::test_openai_client_uses_base_url src/openpi/training/memory_labels_test.py::test_config_has_concurrency_default -v`
Expected: FAIL — `base_url`/`max_concurrency` not accepted; client built without `base_url`.

- [ ] **Step 3: Add the fields and base_url wiring**

In `MemoryLabelConfig` add:

```python
    base_url: str | None = None
    max_concurrency: int = 64
```

In `_get_client`, replace the `openai` branch with:

```python
        elif self.config.backend == "openai":
            import os

            import openai

            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=os.environ.get(self.config.resolved_api_key_env) or "EMPTY",
            )
```

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_labels_test.py -v`
Expected: PASS (new + existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/memory_labels.py src/openpi/training/memory_labels_test.py
git commit -m "$(cat <<'EOF'
feat(memory_labels): base_url + max_concurrency for vLLM/Qwen backend (Spec C)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 2: Async, checkpointed generation

**Files:**
- Modify: `src/openpi/training/memory_labels.py` (add `_get_async_client`, `_generate_single_async`, `generate_labels_async`)
- Test: `src/openpi/training/memory_labels_test.py`

**Interfaces:**
- Consumes: `MemoryLabelConfig.max_concurrency` (Task 1), `_format_subtask_sequence`, `Episode`, `MemoryLabels`.
- Produces: `async MemoryLabelGenerator.generate_labels_async(episodes: list[Episode], out_dir: str | pathlib.Path | None = None) -> list[MemoryLabels]` — issues all (episode, timestep) requests under an `asyncio.Semaphore(max_concurrency)`; when `out_dir` is given, writes one `<episode_id>.json` shard per episode and skips episodes whose shard already exists (resume).

- [ ] **Step 1: Write the failing tests**

Append to `src/openpi/training/memory_labels_test.py`:

```python
import asyncio
import json
import pathlib


def test_generate_labels_async_matches_mock():
    config = MemoryLabelConfig(backend="mock", max_concurrency=4)
    gen = MemoryLabelGenerator(config)
    eps = [Episode(goal="g", subtasks=["a", "b", "c"], success_flags=[True, True, True])]
    labels = asyncio.run(gen.generate_labels_async(eps))
    assert len(labels) == 1
    assert len(labels[0].memories) == 3
    assert all(isinstance(m, str) for m in labels[0].memories)


def test_generate_labels_async_respects_concurrency(monkeypatch):
    config = MemoryLabelConfig(backend="openai", base_url="http://x/v1", model="qwen", max_concurrency=2)
    gen = MemoryLabelGenerator(config)

    state = {"in_flight": 0, "max": 0}

    class _Msg:
        def __init__(self, c): self.message = type("M", (), {"content": c})
    class _Resp:
        def __init__(self, c): self.choices = [_Msg(c)]
    class _Completions:
        async def create(self, **kw):
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.01)
            state["in_flight"] -= 1
            return _Resp("mem")
    class _Chat:
        completions = _Completions()
    class _FakeAsync:
        chat = _Chat()

    monkeypatch.setattr(gen, "_get_async_client", lambda: _FakeAsync())
    eps = [Episode(goal="g", subtasks=[str(i) for i in range(8)], success_flags=[True] * 8)]
    asyncio.run(gen.generate_labels_async(eps))
    assert state["max"] <= 2  # never exceeds max_concurrency


def test_generate_labels_async_resumes(tmp_path):
    config = MemoryLabelConfig(backend="mock")
    gen = MemoryLabelGenerator(config)
    eps = [
        Episode(goal="g0", subtasks=["a"], success_flags=[True]),
        Episode(goal="g1", subtasks=["b"], success_flags=[True]),
    ]
    # Pre-seed episode 0's shard with a sentinel; it must NOT be regenerated.
    (tmp_path / "0.json").write_text(json.dumps({"episode_id": "0", "memories": ["SENTINEL"]}))
    labels = asyncio.run(gen.generate_labels_async(eps, out_dir=tmp_path))
    assert labels[0].memories == ["SENTINEL"]            # resumed, not overwritten
    assert (tmp_path / "1.json").exists()                # episode 1 generated
    assert labels[1].memories[0].startswith("Memory summary")
```

- [ ] **Step 2: Run to verify they fail**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_labels_test.py -k async_ -v`
Expected: FAIL — `AttributeError: ... has no attribute 'generate_labels_async'`.

- [ ] **Step 3: Implement the async path**

Add `import asyncio`, `import json`, `import pathlib` at the top of `memory_labels.py` (if absent). Add these methods to `MemoryLabelGenerator`:

```python
    def _get_async_client(self):
        if self.config.backend == "mock":
            return None
        if self.config.backend == "openai":
            import os

            import openai

            return openai.AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=os.environ.get(self.config.resolved_api_key_env) or "EMPTY",
            )
        raise NotImplementedError(f"async generation not supported for backend {self.config.backend}")

    async def _generate_single_async(self, aclient, goal: str, subtask_sequence: str) -> str:
        if self.config.backend == "mock":
            return f"Memory summary for: {goal}"
        prompt = MEMORY_PROMPT_TEMPLATE.format(goal=goal, subtask_sequence=subtask_sequence)
        resp = await aclient.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_memory_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    async def generate_labels_async(self, episodes, out_dir=None):
        """Concurrency-bounded, resumable async generation. With out_dir, writes one
        <episode_id>.json shard per episode and skips episodes already written."""
        out_dir = pathlib.Path(out_dir) if out_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, MemoryLabels] = {}
        pending: list[tuple[str, int, str, str]] = []
        for idx, ep in enumerate(episodes):
            eid = str(idx)
            shard = (out_dir / f"{eid}.json") if out_dir is not None else None
            if shard is not None and shard.exists():
                data = json.loads(shard.read_text())
                results[eid] = MemoryLabels(episode_id=data["episode_id"], memories=data["memories"])
                continue
            for i in range(len(ep.subtasks)):
                seq = _format_subtask_sequence(ep.subtasks, ep.success_flags, i)
                pending.append((eid, i, ep.goal, seq))

        aclient = self._get_async_client()
        sem = asyncio.Semaphore(self.config.max_concurrency)

        async def _run(eid, i, goal, seq):
            async with sem:
                mem = await self._generate_single_async(aclient, goal, seq)
            return eid, i, mem

        done = await asyncio.gather(*[_run(*p) for p in pending])
        by_ep: dict[str, dict[int, str]] = {}
        for eid, i, mem in done:
            by_ep.setdefault(eid, {})[i] = mem

        for idx, ep in enumerate(episodes):
            eid = str(idx)
            if eid in results:
                continue
            mems = [by_ep[eid][i] for i in range(len(ep.subtasks))]
            results[eid] = MemoryLabels(episode_id=eid, memories=mems)
            if out_dir is not None:
                (out_dir / f"{eid}.json").write_text(json.dumps({"episode_id": eid, "memories": mems}))

        return [results[str(idx)] for idx in range(len(episodes))]
```

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_labels_test.py -v`
Expected: PASS (async match, concurrency-cap, resume + existing).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/memory_labels.py src/openpi/training/memory_labels_test.py
git commit -m "$(cat <<'EOF'
feat(memory_labels): async concurrency-bounded, resumable generation (Spec C)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 3: ForeAct → Episode loader

**Files:**
- Create: `scripts/load_foreact_episodes.py`
- Test: `tests/scripts/test_load_foreact_episodes.py` (create dir if absent)

**Interfaces:**
- Produces: `load_foreact_episodes.episodes_from_records(records: list[dict]) -> list[Episode]` — maps ForeAct-shaped records (`{"goal": str, "subtasks": [str, ...]}`) to `Episode` with `success_flags=[True]*len`. The CLI `main()` reads the HF dataset, calls this, and writes `episodes.json`.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_load_foreact_episodes.py`:

```python
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "load_foreact_episodes",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "load_foreact_episodes.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_episodes_from_records_all_success():
    records = [
        {"goal": "make rice", "subtasks": ["get rice", "cook rice"]},
        {"goal": "clean", "subtasks": ["wipe"]},
    ]
    eps = mod.episodes_from_records(records)
    assert len(eps) == 2
    assert eps[0].goal == "make rice"
    assert eps[0].subtasks == ["get rice", "cook rice"]
    assert eps[0].success_flags == [True, True]
    assert eps[1].success_flags == [True]
```

- [ ] **Step 2: Run to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest tests/scripts/test_load_foreact_episodes.py -v`
Expected: FAIL — file/function does not exist.

- [ ] **Step 3: Implement the loader**

Create `scripts/load_foreact_episodes.py`:

```python
"""Load ForeAct subtask-annotated episodes into Episode JSON for memory-label generation.

Usage:
    uv run python scripts/load_foreact_episodes.py \
        --hf_dataset mit-han-lab/ForeActDataset --split train --output data/episodes.json

NOTE: ForeAct's exact field names must be confirmed against the dataset card; adjust
the field mapping in `episodes_from_records` / `main` if the keys differ. HF download
needs a valid HF token (see the project's HF_HOME note).
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training.memory_labels import Episode


def episodes_from_records(records: list[dict]) -> list[Episode]:
    """Map ForeAct-shaped records to Episodes (all subtasks marked success for v1)."""
    episodes = []
    for r in records:
        subtasks = list(r["subtasks"])
        episodes.append(
            Episode(goal=r["goal"], subtasks=subtasks, success_flags=[True] * len(subtasks))
        )
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Load ForeAct episodes -> Episode JSON")
    parser.add_argument("--hf_dataset", type=str, default="mit-han-lab/ForeActDataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import datasets

    ds = datasets.load_dataset(args.hf_dataset, split=args.split)
    # Map dataset columns to {goal, subtasks}. Adjust keys to the actual ForeAct schema.
    records = [{"goal": ex["goal"], "subtasks": ex["subtasks"]} for ex in ds]
    episodes = episodes_from_records(records)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dataclasses.asdict(e) for e in episodes], indent=2))
    print(f"Wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest tests/scripts/test_load_foreact_episodes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/load_foreact_episodes.py tests/scripts/test_load_foreact_episodes.py
git commit -m "$(cat <<'EOF'
feat(scripts): ForeAct -> Episode loader for HL data (Spec C)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 4: Validation module + script

**Files:**
- Create: `src/openpi/training/memory_validation.py`
- Create: `scripts/validate_memory_labels.py`
- Test: `src/openpi/training/memory_validation_test.py`

**Interfaces:**
- Produces:
  - `compression_ratio(memory: str, cumulative_subtasks: str) -> float`
  - `faithfulness(memory: str, history: str) -> float` (fraction of memory content-words present in history; 1.0 if memory has no content words)
  - `build_judge_prompt(goal: str, history: str, memory: str) -> str`
  - `build_reconstruction_prompt(goal: str, memory: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `src/openpi/training/memory_validation_test.py`:

```python
from openpi.training import memory_validation as mv


def test_compression_ratio():
    assert mv.compression_ratio("short", "a much longer cumulative history string") < 1.0
    assert mv.compression_ratio("", "abc") == 0.0


def test_faithfulness_flags_hallucination():
    history = "placed a plate in the cabinet and wiped the counter"
    faithful = mv.faithfulness("placed plate in cabinet", history)
    halluc = mv.faithfulness("launched a rocket to mars", history)
    assert faithful > 0.8
    assert halluc < 0.5


def test_faithfulness_empty_memory_is_one():
    assert mv.faithfulness("", "anything here") == 1.0


def test_prompt_builders_include_inputs():
    jp = mv.build_judge_prompt(goal="clean kitchen", history="1. wipe", memory="wiped once")
    assert "clean kitchen" in jp and "wiped once" in jp and "1. wipe" in jp
    rp = mv.build_reconstruction_prompt(goal="clean kitchen", memory="wiped once")
    assert "clean kitchen" in rp and "wiped once" in rp
```

- [ ] **Step 2: Run to verify they fail**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_validation_test.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the validation module**

Create `src/openpi/training/memory_validation.py`:

```python
"""Validation metrics + prompts for generated memory labels (Spec C)."""

import re

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "in", "on", "of", "for", "with", "at", "is",
    "was", "i", "it", "into", "from", "by", "that", "this", "then", "have", "has",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS}


def compression_ratio(memory: str, cumulative_subtasks: str) -> float:
    """len(memory)/len(cumulative_subtasks); 0 for empty memory, capped denom at 1."""
    return len(memory) / max(len(cumulative_subtasks), 1)


def faithfulness(memory: str, history: str) -> float:
    """Fraction of memory content-words that appear in the subtask history.

    Low values suggest the memory mentions entities not in the history (possible
    hallucination). Returns 1.0 when the memory has no content words.
    """
    mw = _content_words(memory)
    if not mw:
        return 1.0
    hw = _content_words(history)
    return len(mw & hw) / len(mw)


def build_judge_prompt(goal: str, history: str, memory: str) -> str:
    return (
        "Rate the following robot memory summary on three axes, each 1-5.\n"
        f"Task goal: {goal}\n"
        f"Subtask history:\n{history}\n"
        f"Memory summary: {memory}\n\n"
        "Score FAITHFULNESS (no invented facts), DECISION_RELEVANCE (keeps info needed "
        "for future steps), and CONCISENESS. Reply as JSON "
        '{"faithfulness": n, "decision_relevance": n, "conciseness": n}.'
    )


def build_reconstruction_prompt(goal: str, memory: str) -> str:
    return (
        "Given a robot's task goal and its current memory of what it has done, predict "
        "the single next subtask it should perform.\n"
        f"Task goal: {goal}\n"
        f"Memory: {memory}\n"
        "Next subtask:"
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_validation_test.py -v`
Expected: PASS.

- [ ] **Step 5: Add the validation CLI**

Create `scripts/validate_memory_labels.py`:

```python
"""Validate generated memory labels: full-set heuristics + sampled LLM-judge + reconstruction probe.

Usage:
    uv run python scripts/validate_memory_labels.py \
        --episodes_file data/episodes.json --labels_file data/memory_labels.json \
        --report data/validation_report --backend openai --base_url http://localhost:8000/v1 \
        --model qwen --sample 50

Heuristics run on every label (CPU). The LLM-judge and reconstruction probe run on a
random sample via the same backend/endpoint. Use --backend mock to dry-run wiring.
"""

import argparse
import json
import pathlib
import random
import statistics

from openpi.training import memory_validation as mv
from openpi.training.memory_labels import MemoryLabelConfig, MemoryLabelGenerator, _format_subtask_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--report", required=True, help="output path prefix (.json/.md)")
    p.add_argument("--backend", default="mock")
    p.add_argument("--base_url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args()

    episodes = json.loads(pathlib.Path(args.episodes_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())

    comps, faiths, flagged = [], [], []
    for ep, lab in zip(episodes, labels, strict=False):
        subtasks, flags = ep["subtasks"], ep["success_flags"]
        for i, memory in enumerate(lab["memories"]):
            history = _format_subtask_sequence(subtasks, flags, i)
            comps.append(mv.compression_ratio(memory, history))
            f = mv.faithfulness(memory, history)
            faiths.append(f)
            if f < 0.5:
                flagged.append({"episode_id": lab["episode_id"], "timestep": i, "memory": memory})

    report = {
        "n_labels": len(comps),
        "compression_ratio_mean": statistics.mean(comps) if comps else 0.0,
        "faithfulness_mean": statistics.mean(faiths) if faiths else 0.0,
        "n_flagged_low_faithfulness": len(flagged),
        "flagged_examples": flagged[:20],
    }

    cfg = MemoryLabelConfig(backend=args.backend, base_url=args.base_url, model=args.model or "qwen")
    gen = MemoryLabelGenerator(cfg)
    client = gen._get_client()  # noqa: SLF001 -- reuse the same backend for judge/probe
    # Sampled judge + reconstruction probe (skipped for mock).
    pairs = [(ep, lab, i) for ep, lab in zip(episodes, labels, strict=False) for i in range(len(lab["memories"]))]
    sample = random.sample(pairs, min(args.sample, len(pairs)))
    judge_scores, probe = [], {"n": 0, "match": 0}
    if args.backend != "mock":
        for ep, lab, i in sample:
            history = _format_subtask_sequence(ep["subtasks"], ep["success_flags"], i)
            memory = lab["memories"][i]
            jp = mv.build_judge_prompt(ep["goal"], history, memory)
            jr = client.chat.completions.create(model=cfg.model, max_tokens=64,
                                                messages=[{"role": "user", "content": jp}])
            judge_scores.append(jr.choices[0].message.content)
            if i + 1 < len(ep["subtasks"]):
                rp = mv.build_reconstruction_prompt(ep["goal"], memory)
                rr = client.chat.completions.create(model=cfg.model, max_tokens=32,
                                                    messages=[{"role": "user", "content": rp}])
                pred = rr.choices[0].message.content.strip().lower()
                probe["n"] += 1
                if ep["subtasks"][i + 1].lower() in pred or pred in ep["subtasks"][i + 1].lower():
                    probe["match"] += 1
    report["judge_sample"] = judge_scores
    report["reconstruction_probe"] = probe

    base = pathlib.Path(args.report)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(report, indent=2))
    base.with_suffix(".md").write_text(
        f"# Memory label validation\n\n"
        f"- labels: {report['n_labels']}\n"
        f"- compression ratio (mean): {report['compression_ratio_mean']:.3f}\n"
        f"- faithfulness (mean): {report['faithfulness_mean']:.3f}\n"
        f"- low-faithfulness flagged: {report['n_flagged_low_faithfulness']}\n"
        f"- reconstruction probe: {probe['match']}/{probe['n']} match\n"
    )
    print(f"Validation report written to {base.with_suffix('.json')} / .md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the module tests once more (regression) + commit**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest src/openpi/training/memory_validation_test.py -v`
Expected: PASS.

```bash
git add src/openpi/training/memory_validation.py src/openpi/training/memory_validation_test.py scripts/validate_memory_labels.py
git commit -m "$(cat <<'EOF'
feat(memory_validation): heuristics + LLM-judge/probe + validate CLI (Spec C)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

### Task 5: NCHC serving script + generate CLI wiring + mock end-to-end

**Files:**
- Create: `scripts/nchc/serve_qwen_vllm.sh`
- Modify: `scripts/generate_memory_labels.py` (add `--base_url`, `--max_concurrency`, `--out_dir`, async path + shard aggregation)
- Test: `tests/scripts/test_generate_memory_labels_e2e.py`

**Interfaces:**
- Consumes: `generate_labels_async` (Task 2), `episodes_from_records`/`episodes.json` (Task 3), validation (Task 4).
- Produces: a `pi0_mem` HL-data pipeline runnable end-to-end with `--backend mock` (no model needed).

- [ ] **Step 1: Write the failing end-to-end test**

Create `tests/scripts/test_generate_memory_labels_e2e.py`:

```python
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_generate_memory_labels_mock_e2e(tmp_path):
    episodes = [{"goal": "g", "subtasks": ["a", "b"], "success_flags": [True, True]}]
    ep_file = tmp_path / "episodes.json"
    ep_file.write_text(json.dumps(episodes))
    out = tmp_path / "labels.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_memory_labels.py"),
         "--episodes_file", str(ep_file), "--backend", "mock", "--output", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    labels = json.loads(out.read_text())
    assert len(labels) == 1
    assert len(labels[0]["memories"]) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest tests/scripts/test_generate_memory_labels_e2e.py -v`
Expected: it may PASS already (the existing CLI handles mock). If it passes, proceed to Step 3 to add the new flags; if it fails, Step 3 fixes it. Either way, Step 3 adds the async/base_url wiring the test in Step 4 needs.

- [ ] **Step 3: Wire async + new flags into the generate CLI**

In `scripts/generate_memory_labels.py`, add args and use the async path. Add to the parser:

```python
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=None, help="per-episode shard dir (resume)")
```

Build the config with the new fields:

```python
    config = MemoryLabelConfig(
        backend=args.backend,
        model=args.model or MemoryLabelConfig().model,
        max_memory_tokens=args.max_memory_tokens,
        base_url=args.base_url,
        max_concurrency=args.max_concurrency,
    )
```

Replace the synchronous `generator.generate_labels(episodes)` call with the async path:

```python
    import asyncio

    generator = MemoryLabelGenerator(config)
    labels = asyncio.run(generator.generate_labels_async(episodes, out_dir=args.out_dir))
```

(The shard files in `out_dir` are the resume checkpoint; the aggregated `--output` JSON is still written by the existing serialization block.)

- [ ] **Step 4: Add the NCHC vLLM serve script**

Create `scripts/nchc/serve_qwen_vllm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen_vllm
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

# Serve Qwen3.6-27B (Dense) via vLLM (OpenAI-compatible) on one NCHC GPU.
# Point the generator at http://<node>:8000/v1 (run it on the same node or via srun).
# Tune --max-num-seqs / --max-num-batched-tokens / --gpu-memory-utilization with
# vLLM's auto_tune.sh if needed. Requires the Qwen weights (HF download uses HF_HOME token).
set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"   # set QWEN_MODEL to the exact served id
PORT="${VLLM_PORT:-8000}"

vllm serve "$MODEL" \
    --quantization fp8 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --max-model-len 2048 \
    --port "$PORT"
```

Make it executable: `chmod +x scripts/nchc/serve_qwen_vllm.sh`.

- [ ] **Step 5: Run the end-to-end mock test**

Run: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync pytest tests/scripts/test_generate_memory_labels_e2e.py -v`
Expected: PASS — full mock pipeline (episodes JSON → async mock generation → labels JSON) works.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_memory_labels.py scripts/nchc/serve_qwen_vllm.sh tests/scripts/test_generate_memory_labels_e2e.py
git commit -m "$(cat <<'EOF'
feat(scripts): vLLM serve script + async generate CLI wiring (Spec C)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PcxJCdBRpZNjB3KQixA359
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Qwen backend (base_url) + concurrency config (C.2) → Task 1.
- Async, prefix-cache-friendly, checkpointed generation (C.2) → Task 2.
- ForeAct→Episode loader, all-success (C.4) → Task 3.
- Validation: heuristics + judge + reconstruction probe (C.5) → Task 4.
- NCHC vLLM serving + generate CLI wiring + mock e2e (C.3, C.6) → Task 5.
- Manual NCHC run / verification → documented in spec C.6 + serve script header (no task; needs cluster).

**Placeholder scan:** No TBD/TODO; each code step has complete code; run steps have exact commands + expected output. The `QWEN_MODEL`/ForeAct-schema notes are deliberate "confirm the real id/columns at run time" guardrails with the binding behavior stated, not placeholders. Task 5 Step 2 explicitly handles the "may already pass" case rather than asserting a guaranteed RED.

**Type consistency:** `MemoryLabelConfig` fields `base_url`/`max_concurrency` used identically in Tasks 1/2/5. `generate_labels_async(episodes, out_dir=None) -> list[MemoryLabels]` matches its call in Task 5. `episodes_from_records(records) -> list[Episode]` (Task 3) matches its test. `compression_ratio`/`faithfulness`/`build_judge_prompt`/`build_reconstruction_prompt` signatures match between `memory_validation.py` (Task 4) and the validate CLI. `Episode`/`MemoryLabels` field names (`goal`/`subtasks`/`success_flags`, `episode_id`/`memories`) consistent throughout.
