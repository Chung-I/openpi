"""Validation metrics + prompts for generated memory labels (Spec C)."""

import itertools
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


_PREAMBLE_MARKERS = ("here's", "here is", "let me", "sure,", "okay", "i'll", "i will", "memory:")
_THINKING_MARKERS = ("thinking", "step ", "first,", "analyze", "reasoning")
_MARKDOWN_MARKERS = ("**", "```", "##", "- ")


def structural_score(memory: str, max_chars: int = 240) -> float:
    """Fraction of 6 cleanliness checks passed: non-empty, within length band, and free of
    preamble / thinking / markdown markers and a JSON wrapper. Clean prose scores 1.0;
    each independent dirtiness category lowers the score, so heavily-dirty output scores low."""
    m = memory.strip()
    low = m.lower()
    checks = [
        bool(m),
        len(m) <= max_chars,
        not any(k in low for k in _PREAMBLE_MARKERS),
        not any(k in low for k in _THINKING_MARKERS),
        not any(k in m for k in _MARKDOWN_MARKERS),
        not m.startswith(("{", "[")),
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
