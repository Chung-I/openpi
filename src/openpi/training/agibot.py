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
import re
import tarfile

from openpi.training.memory_labels import Episode

_HEAD_VIDEO_MEMBER_RE = re.compile(r"^(?P<episode_id>[^/]+)/videos/(?P<video_key>[^/]+)$")

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


def extract_task_head_videos(
    tar_path: str | pathlib.Path,
    dest_dir: str | pathlib.Path,
    video_key: str = "head_color.mp4",
    episode_ids: set[str] | None = None,
) -> pathlib.Path:
    """Stream an AgiBot observation `.tar` (`observations/{task_id}/*.tar`, per-episode dirs of the
    form `<episode_id>/videos/<cam>.mp4` + `<episode_id>/depth/*.png`) and extract ONLY the
    `<episode_id>/videos/{video_key}` members -- skip depth PNGs and other cameras, which dominate
    the tar's size -- into `dest_dir/<episode_id>/videos/{video_key}`. If `episode_ids` is given
    (as strings), extract only those episodes. Idempotent: an already-extracted file (same dest
    path already present) is left untouched and its member is skipped without re-reading tar bytes
    for it. Returns `dest_dir`."""
    dest_dir = pathlib.Path(dest_dir)
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            m = _HEAD_VIDEO_MEMBER_RE.match(member.name)
            if not m or m.group("video_key") != video_key:
                continue
            episode_id = m.group("episode_id")
            if episode_ids is not None and episode_id not in episode_ids:
                continue
            out_path = dest_dir / episode_id / "videos" / video_key
            if out_path.exists():
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, out_path.open("wb") as dst:
                dst.write(src.read())
    return dest_dir


def download_task_tars(repo: str, task_id, cache_dir: str | pathlib.Path | None = None) -> list[pathlib.Path]:
    """Download every `observations/{task_id}/*.tar` shard from `repo` to the local HF cache
    (or `cache_dir` if given) and return their local paths.

    Size caveat: per-task observation tars can be up to ~48GB and a task may span multiple shard
    tars; this downloads each matched shard whole (no partial/streaming download of a single
    video member) -- scope which tasks you call this for. Pair with `extract_task_head_videos`
    to keep only head-cam video on disk after extraction; the downloaded tar itself can then be
    deleted."""
    import huggingface_hub

    api = huggingface_hub.HfApi()
    prefix = f"observations/{task_id}/"
    filenames = [
        f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".tar")
    ]
    paths = []
    for filename in filenames:
        p = huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=filename, cache_dir=cache_dir)
        paths.append(pathlib.Path(p))
    return paths
