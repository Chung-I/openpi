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
class Episode:
    goal: str
    subtasks: list[str]  # ordered subtask descriptions
    success_flags: list[bool]  # per-subtask success/failure


@dataclasses.dataclass
class MemoryLabels:
    episode_id: str
    memories: list[str]  # one compressed memory per timestep


@dataclasses.dataclass
class MemoryLabelConfig:
    backend: Literal["claude", "openai", "local", "mock"] = "claude"
    model: str = "claude-sonnet-4-20250514"
    max_memory_tokens: int = 128
    batch_size: int = 32
    api_key_env: str = ""
    base_url: str | None = None
    max_concurrency: int = 64

    @property
    def resolved_api_key_env(self) -> str:
        if self.api_key_env:  # user explicitly set it
            return self.api_key_env
        if self.backend == "openai":
            return "OPENAI_API_KEY"
        return "ANTHROPIC_API_KEY"


def _format_subtask_sequence(subtasks: list[str], success_flags: list[bool], up_to_index: int) -> str:
    lines = []
    for i in range(up_to_index + 1):
        status = "SUCCESS" if success_flags[i] else "FAILED"
        lines.append(f"{i+1}. [{status}] {subtasks[i]}")
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

            self._client = anthropic.Anthropic(api_key=os.environ.get(self.config.resolved_api_key_env))
        elif self.config.backend == "openai":
            import os

            import openai

            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=os.environ.get(self.config.resolved_api_key_env) or "EMPTY",
            )
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

    def generate_labels(self, episodes: list[Episode]) -> list[MemoryLabels]:
        """Generate memory labels for each timestep in each episode.

        Args:
            episodes: List of Episode objects with goal, subtasks, and success_flags.

        Returns:
            List of MemoryLabels, one per episode, each containing one compressed
            memory string per subtask timestep.
        """
        all_labels = []
        for idx, episode in enumerate(episodes):
            episode_labels = []
            for i in range(len(episode.subtasks)):
                seq = _format_subtask_sequence(episode.subtasks, episode.success_flags, i)
                memory = self._generate_single(episode.goal, seq)
                episode_labels.append(memory)
            all_labels.append(MemoryLabels(episode_id=str(idx), memories=episode_labels))
        return all_labels
