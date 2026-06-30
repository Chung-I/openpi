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
