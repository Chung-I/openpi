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
