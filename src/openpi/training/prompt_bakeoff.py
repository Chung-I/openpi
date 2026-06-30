"""Prompt bake-off: gated composite scoring + ranking for memory-label prompts.

Metric keys are all normalized to [0, 1] by the caller. The gate rejects prompts that
fall below minimum faithfulness / temporal_coherence / decision_relevance so a
"short but wrong" prompt cannot win on conciseness. Among gated-in prompts, rank by a
weighted sum; conciseness contributes only as a bounded bonus above the gate.
"""

import asyncio
import pathlib
import statistics

from openpi.training import memory_validation as mv
from openpi.training.memory_labels import Episode
from openpi.training.memory_labels import MemoryLabelGenerator
from openpi.training.memory_labels import MemoryLabels
from openpi.training.memory_labels import _format_subtask_sequence

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


def score_labels(episodes: list[Episode], labels: list[MemoryLabels]) -> dict:
    """Heuristic metrics (faithfulness, conciseness, structural) averaged over all labels."""
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
    """Return {stem: text} for all *.txt files in prompt_dir."""
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
