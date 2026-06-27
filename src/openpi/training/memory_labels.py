import dataclasses
import logging
from typing import Literal

logger = logging.getLogger("openpi")

MEMORY_PROMPT_TEMPLATE = """You are generating compressed memory summaries for a robot policy.

Given the task goal and the sequence of subtask events so far, produce a
summary that retains ONLY information still relevant for future task execution.

Rules:
- Remove details about completed subtasks that don't affect future decisions
- Aggregate repeated items (e.g., "placed 3 bowls in cabinet" not individual colors)
- Remove failed attempts that were later retried successfully
- Keep spatial information relevant to navigation
- Keep counts of remaining items
- Minimize length while preserving decision-relevant information

Task goal: {goal}
Subtask events so far:
{subtask_sequence}

Compressed memory summary:"""


@dataclasses.dataclass
class MemoryLabelConfig:
    backend: Literal["claude", "openai", "local", "mock"] = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_memory_tokens: int = 128
    batch_size: int = 32
    api_key_env: str = "ANTHROPIC_API_KEY"


def _format_subtask_sequence(subtasks: list[dict], up_to_index: int) -> str:
    lines = []
    for i, st in enumerate(subtasks[: up_to_index + 1]):
        status = "SUCCESS" if st["success"] else "FAILED"
        lines.append(f"{i+1}. [{status}] {st['text']} (t={st['timestamp']:.1f}s)")
    return "\n".join(lines)


class MemoryLabelGenerator:
    def __init__(self, config: MemoryLabelConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.config.backend == "mock":
            return None
        elif self.config.backend == "claude":
            import os

            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ.get(self.config.api_key_env))
        elif self.config.backend == "openai":
            import os

            import openai

            self._client = openai.OpenAI(api_key=os.environ.get(self.config.api_key_env))
        elif self.config.backend == "local":
            raise NotImplementedError("Local LLM backend not yet implemented")
        return self._client

    def _generate_single(self, goal: str, subtask_sequence: str) -> str:
        prompt = MEMORY_PROMPT_TEMPLATE.format(goal=goal, subtask_sequence=subtask_sequence)

        if self.config.backend == "mock":
            return f"Memory summary for: {goal}"

        client = self._get_client()

        if self.config.backend == "claude":
            response = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        if self.config.backend == "openai":
            response = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown backend: {self.config.backend}")

    def generate_labels(self, episodes: list[dict]) -> list[list[str]]:
        """Generate memory labels for each timestep in each episode.

        Args:
            episodes: List of episode dicts, each with 'goal' and 'subtasks' keys.
                     'subtasks' is a list of dicts with 'text', 'success', 'timestamp'.

        Returns:
            List of lists of memory strings, one per subtask per episode.
        """
        all_labels = []
        for episode in episodes:
            goal = episode["goal"]
            subtasks = episode["subtasks"]
            episode_labels = []
            for i in range(len(subtasks)):
                seq = _format_subtask_sequence(subtasks, i)
                memory = self._generate_single(goal, seq)
                episode_labels.append(memory)
            all_labels.append(episode_labels)
        return all_labels
