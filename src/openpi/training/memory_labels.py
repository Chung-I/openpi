import asyncio
import dataclasses
import json
import logging
import os
import pathlib
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

RECURSIVE_FIRST_MEMORY = "(none yet)"

RECURSIVE_MEMORY_PROMPT_TEMPLATE = """You are labeling the language memory for a robot's high-level policy. The memory is a concise
FIRST-PERSON summary ("I ...") of the semantic events so far that are still relevant for
finishing the task. Given the goal, the current memory, and the new subtask event (with a
SUCCESS/FAILED indicator), output the updated memory.

Rules:
- Fold the new event into the current memory as an incremental first-person update.
- Record ONLY successful subtasks. If the new event is [FAILED], output the current memory
  UNCHANGED - failed attempts are discarded; the memory does not move until the subtask succeeds.
- Keep the minimal set of still-relevant information. Compress and aggregate: prefer counts and
  locations over per-object attributes - e.g., "I placed three bowls in the top right cabinet"
  rather than listing each bowl's color.
- Drop details no longer needed for future steps; keep state that affects completion (e.g., a
  drawer/fridge left open that must be closed).
- If the current memory is "(none yet)", begin the summary from this event.

Example (compression):
Goal: put the bowls in the cabinet. Current memory: I placed a light green bowl and a dark blue bowl in the top right cabinet.
Subtask: [SUCCESS] place the bright yellow bowl in the top right cabinet.
Updated memory: I placed three bowls in the top right cabinet.

Example (failed attempt -> no update):
Goal: pick the vegetables. Current memory: I picked up the eggplant and placed it in the plate.
Subtask: [FAILED] pick up the corn.
Updated memory: I picked up the eggplant and placed it in the plate.

Goal: {goal}
Current memory: {previous_memory}
Subtask: {new_event}
Updated memory:"""


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
    # For "thinking"/reasoning models served via vLLM (e.g. Qwen3): suppress chain-of-thought
    # so the label is the final compressed memory only. Only affects the openai backend.
    disable_thinking: bool = False
    # Prompt + generation controls for the bake-off harness.
    prompt_template: str | None = None  # overrides the module template when set
    generation_mode: Literal["stateless", "recursive"] = "recursive"
    temperature: float = 0.0

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
        lines.append(f"{i + 1}. [{status}] {subtasks[i]}")
    return "\n".join(lines)


def _format_event(subtask: str, success: bool) -> str:  # noqa: FBT001
    return f"[{'SUCCESS' if success else 'FAILED'}] {subtask}"


class MemoryLabelGenerator:
    def __init__(self, config: MemoryLabelConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.config.backend == "mock":
            return None
        if self.config.backend == "claude":
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

    def _complete(self, prompt: str) -> str:
        if self.config.backend == "mock":
            # mock: content derives from the rendered prompt prefix, not the goal
            return f"Memory summary for: {prompt[:20]}"
        client = self._get_client()
        if self.config.backend == "claude":
            resp = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        if self.config.backend == "openai":
            resp = client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_memory_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
                **self._chat_extra(),
            )
            return resp.choices[0].message.content
        raise ValueError(f"Unknown backend: {self.config.backend}")

    async def _complete_async(self, aclient, prompt: str) -> str:
        if self.config.backend == "mock":
            return f"Memory summary for: {prompt[:20]}"
        resp = await aclient.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_memory_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
            **self._chat_extra(),
        )
        return resp.choices[0].message.content

    def _stateless_prompt(self, goal: str, subtask_sequence: str) -> str:
        tmpl = self.config.prompt_template or MEMORY_PROMPT_TEMPLATE
        return tmpl.format(goal=goal, subtask_sequence=subtask_sequence)

    def _recursive_prompt(self, goal: str, previous_memory: str, new_event: str) -> str:
        tmpl = self.config.prompt_template or RECURSIVE_MEMORY_PROMPT_TEMPLATE
        return tmpl.format(goal=goal, previous_memory=previous_memory, new_event=new_event)

    def _generate_single(self, goal: str, subtask_sequence: str) -> str:
        return self._complete(self._stateless_prompt(goal, subtask_sequence))

    def _chat_extra(self) -> dict:
        """Extra kwargs for openai chat.completions.create (e.g. suppress reasoning)."""
        if self.config.disable_thinking:
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        return {}

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
        return await self._complete_async(aclient, self._stateless_prompt(goal, subtask_sequence))

    async def generate_labels_async(
        self, episodes: list[Episode], out_dir: str | os.PathLike | None = None
    ) -> list[MemoryLabels]:
        """Concurrency-bounded, resumable async generation. With out_dir, writes one
        <episode_id>.json shard per episode AS SOON AS that episode's timesteps finish,
        so a mid-run crash keeps already-completed episodes. Episodes already on disk are
        loaded and skipped (cross-run resume). Shards are POSITIONAL (episode_id =
        str(index)) — do not reuse out_dir across a different or reordered episode set."""
        out_dir = pathlib.Path(out_dir) if out_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, MemoryLabels] = {}
        pending_episodes: list[tuple[int, Episode]] = []

        for idx, ep in enumerate(episodes):
            eid = str(idx)
            shard = (out_dir / f"{eid}.json") if out_dir is not None else None
            if shard is not None and shard.exists():
                data = json.loads(shard.read_text())
                results[eid] = MemoryLabels(episode_id=data["episode_id"], memories=data["memories"])
            else:
                pending_episodes.append((idx, ep))

        if pending_episodes:
            aclient = self._get_async_client()
            sem = asyncio.Semaphore(self.config.max_concurrency)

            async def _run_timestep(goal: str, seq: str) -> str:
                async with sem:
                    return await self._generate_single_async(aclient, goal, seq)

            async def _run_episode(idx: int, ep: Episode) -> tuple[str, MemoryLabels]:
                eid = str(idx)
                if self.config.generation_mode == "recursive":
                    async with sem:  # one episode holds a slot for its whole chain
                        mems: list[str] = []
                        prev = RECURSIVE_FIRST_MEMORY
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

            episode_results = await asyncio.gather(*[_run_episode(idx, ep) for idx, ep in pending_episodes])
            results.update(episode_results)

        return [results[str(idx)] for idx in range(len(episodes))]

    def generate_labels(self, episodes: list[Episode]) -> list[MemoryLabels]:
        """Generate memory labels for each timestep in each episode.

        Args:
            episodes: List of Episode objects with goal, subtasks, and success_flags.

        Returns:
            List of MemoryLabels, one per episode, each containing one compressed
            memory string per subtask timestep.
        """
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
        all_labels = []
        for idx, episode in enumerate(episodes):
            episode_labels = []
            for i in range(len(episode.subtasks)):
                seq = _format_subtask_sequence(episode.subtasks, episode.success_flags, i)
                memory = self._generate_single(episode.goal, seq)
                episode_labels.append(memory)
            all_labels.append(MemoryLabels(episode_id=str(idx), memories=episode_labels))
        return all_labels
