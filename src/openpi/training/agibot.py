"""AgiBot World (task_info/task_<id>.json) -> MEM HL records via explicit action_config spans (Shape B).

Unlike RoboCOIN/Galaxea (Shape A: a per-frame index column that must be run-length-encoded), AgiBot
ships an EXPLICIT per-episode `label_info.action_config` list of {start_frame, end_frame,
action_text, skill} spans -- there is nothing to RLE, we just read the spans directly. Spans are
exclusive-end and contiguous (`end_frame[i] == start_frame[i+1]`, 0 gaps); `frame_ranges` here are
the inclusive last-active-frame (`end_frame - 1`), matching robocoin/galaxea/robomind's convention.
AgiBot ships no explicit success/failure field, so success is inferred from FAILURE_KEYWORDS
appearing in action_text (e.g. failure-recovery subtasks like "grasp failed, retry ..."). This
mirrors `robomind.record_from_annotation`'s structure (explicit start/end spans) rather than
robocoin's RLE. The result is the SAME record shape (id, goal, subtasks, frame_ranges,
success_flags) as `robomind.RobomindRecord`, so `robomind.build_samples` and the memory-label
generator reuse it unchanged.

This module is the single source of truth for the action_config parsing schema (kept_action_spans)
and the failure-keyword success predicate (is_success); `scripts/load_agibot_episodes.py` reuses
both rather than re-deriving them.

Each `task_info/task_<task_id>.json` is a flat, individually-downloadable file (a JSON array of
episode entries) -- no tar download needed for this Stage-A (text-only) loader; `task_id` is not
present in the JSON itself and must be supplied by the caller (filename/CLI arg).
"""

import dataclasses
import json
import pathlib

from openpi.training.memory_labels import Episode

# Keywords whose presence in an action_text mark that subtask as a failure/recovery attempt rather
# than a successful one. Exact match with scripts/load_agibot_episodes.py's (former) _FAILURE_KEYWORDS.
FAILURE_KEYWORDS = ("failed", "recovery", "retry", "mistake")


@dataclasses.dataclass
class AgibotRecord:
    id: str
    goal: str
    subtasks: list[str]
    frame_ranges: list[tuple[int, int]]
    success_flags: list[bool]


def kept_action_spans(entry: dict) -> list[tuple[str, int, int]]:
    """Yield (action_text.strip(), start_frame, end_frame) for every label_info.action_config item
    whose action_text is non-empty after strip. This is the shared schema-knowledge (key
    `label_info`, field `action_text`, drop-empty) reused by scripts/load_agibot_episodes.py."""
    action_config = (entry.get("label_info") or {}).get("action_config") or []
    spans: list[tuple[str, int, int]] = []
    for c in action_config:
        text = (c.get("action_text") or "").strip()
        if not text:
            continue
        spans.append((text, int(c["start_frame"]), int(c["end_frame"])))
    return spans


def is_success(action_text: str) -> bool:
    """False if action_text contains a failure/recovery keyword (case-insensitive)."""
    return not any(k in action_text.lower() for k in FAILURE_KEYWORDS)


def record_from_episode(repo: str, task_id, entry: dict) -> AgibotRecord | None:
    """Build one MEM record from a single task_info entry (None if it has no kept subtasks)."""
    spans = kept_action_spans(entry)
    if not spans:
        return None
    subtasks = [text for text, _, _ in spans]
    ranges = [(start, end - 1) for _, start, end in spans]  # exclusive-end -> inclusive
    success_flags = [is_success(text) for text, _, _ in spans]
    return AgibotRecord(
        id=f"{repo}/task_{task_id}/episode_{entry['episode_id']}",
        goal=(entry.get("task_name") or "").strip(),
        subtasks=subtasks,
        frame_ranges=ranges,
        success_flags=success_flags,
    )


def records_from_task_json(repo: str, task_id, path, max_episodes: int | None = None) -> list[AgibotRecord]:
    """Build records from a task_info/task_<task_id>.json file (a JSON array of episode entries)."""
    entries = json.loads(pathlib.Path(path).read_text())[:max_episodes]
    out = []
    for entry in entries:
        r = record_from_episode(repo, task_id, entry)
        if r is not None:
            out.append(r)
    return out


def records_from_repo(repo: str, task_id, max_episodes: int | None = None) -> list[AgibotRecord]:
    """Download one task_info/task_<task_id>.json from `repo` and build records from it."""
    import huggingface_hub

    path = huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=f"task_info/task_{task_id}.json")
    return records_from_task_json(repo, task_id, path, max_episodes=max_episodes)


def to_episode(rec: AgibotRecord) -> Episode:
    return Episode(goal=rec.goal, subtasks=list(rec.subtasks), success_flags=list(rec.success_flags))
