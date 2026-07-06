"""RoboCOIN (LeRobot v2.1) -> MEM HL records via per-frame subtask RLE (Shape A).

RoboCOIN episode parquets carry a per-frame `subtask_annotation` int32 vector (shape [5]; column
0 is the primary subtask, the rest are concurrent dual-arm labels we ignore for v1). Contiguous
runs of the primary column ARE the subtask segments -- there are no explicit start/end fields, so
we run-length-encode them (`rle_segments`). Each integer maps to text via the group's
`annotations/subtask_annotations.jsonl` vocab; the sentinel subtasks "Abnormal"/"End"/"null" are
dropped. The result is the SAME record shape (id, goal, subtasks, frame_ranges, success_flags) as
`robomind.RobomindRecord`, so `robomind.build_samples` and the memory-label generator reuse
unchanged. RoboCOIN ships no clean success labels, so every kept subtask is marked success for v1.
"""

import dataclasses
import json
import pathlib

import numpy as np

from openpi.training.memory_labels import Episode

# Vocab strings that are not real subtasks (case-insensitive). Filtered by TEXT, not index, since
# the integer assignment is per-group and could differ across RoboCOIN task repos.
SENTINELS = frozenset({"abnormal", "end", "null"})


@dataclasses.dataclass
class RobocoinRecord:
    id: str
    goal: str
    subtasks: list[str]
    frame_ranges: list[tuple[int, int]]
    success_flags: list[bool]


def primary_labels(arr) -> np.ndarray:
    """The primary (column-0) subtask label per frame; accepts a 1-D or (N, k) array."""
    a = np.asarray(arr)
    return a[:, 0] if a.ndim == 2 else a


def rle_segments(labels) -> list[tuple[int, int, int]]:
    """Contiguous runs of `labels` as (value, start, end) half-open spans."""
    a = np.asarray(labels)
    if a.size == 0:
        return []
    bounds = [0, *[i for i in range(1, len(a)) if a[i] != a[i - 1]], len(a)]
    return [(int(a[bounds[k]]), bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def is_sentinel(text: str) -> bool:
    return text.strip().lower() in SENTINELS


def record_from_episode(episode_id: str, subtask_annotation, goal: str, vocab: dict[int, str]) -> RobocoinRecord | None:
    """Build one MEM record from an episode's per-frame subtask labels (None if no real subtasks)."""
    subtasks: list[str] = []
    ranges: list[tuple[int, int]] = []
    for val, start, end in rle_segments(primary_labels(subtask_annotation)):
        text = vocab[val].strip()
        if is_sentinel(text):
            continue
        subtasks.append(text)
        ranges.append((start, end - 1))  # inclusive boundary: last frame the subtask is active
    if not subtasks:
        return None
    return RobocoinRecord(
        id=episode_id, goal=goal.strip(), subtasks=subtasks, frame_ranges=ranges, success_flags=[True] * len(subtasks)
    )


def to_episode(rec: RobocoinRecord) -> Episode:
    return Episode(goal=rec.goal, subtasks=list(rec.subtasks), success_flags=list(rec.success_flags))


def load_vocab(path) -> dict[int, str]:
    """subtask_index -> subtask text from a RoboCOIN annotations/subtask_annotations.jsonl."""
    rows = (json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip())
    return {r["subtask_index"]: r["subtask"] for r in rows}


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def _parquet_rel(info: dict, episode_index: int) -> str:
    chunk = episode_index // int(info.get("chunks_size", 1000))
    return info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)


def _read_primary_labels(parquet_path) -> np.ndarray:
    """Primary (column-0) per-frame subtask label from an episode parquet."""
    import pyarrow.parquet as pq

    col = pq.read_table(parquet_path, columns=["subtask_annotation"])["subtask_annotation"].to_pylist()
    return np.array([row[0] for row in col], dtype=np.int64)


def _build_records(info, vocab, episodes, resolve_parquet, repo: str) -> list[RobocoinRecord]:
    """Shared record builder; `resolve_parquet(rel)` returns a readable parquet path for a rel path."""
    out: list[RobocoinRecord] = []
    for ep in episodes:
        idx = int(ep["episode_index"])
        goal = "; ".join(t.strip() for t in ep.get("tasks", []) if t.strip())
        labels = _read_primary_labels(resolve_parquet(_parquet_rel(info, idx)))
        rec = record_from_episode(f"{repo}/episode_{idx:06d}", labels, goal, vocab)
        if rec is not None:
            out.append(rec)
    return out


def records_from_local(root, max_episodes: int | None = None, repo: str = "RoboCOIN") -> list[RobocoinRecord]:
    """Build MEM records from an extracted RoboCOIN/LeRobot dir (meta/ + annotations/ + data/)."""
    root = pathlib.Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    vocab = load_vocab(root / "annotations" / "subtask_annotations.jsonl")
    episodes = _read_jsonl(root / "meta" / "episodes.jsonl")[:max_episodes]
    return _build_records(info, vocab, episodes, lambda rel: root / rel, repo)


def records_from_repo(repo: str, max_episodes: int | None = None) -> list[RobocoinRecord]:
    """Build records from a RoboCOIN task repo, downloading only the parquets needed.

    Fetches the small meta/annotation files, then pulls each episode's parquet on demand -- so
    `max_episodes=N` downloads N parquets, not the whole (often GB-scale) repo.
    """
    import huggingface_hub

    def _dl(filename: str):
        return pathlib.Path(huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=filename))

    info = json.loads(_dl("meta/info.json").read_text())
    vocab = load_vocab(_dl("annotations/subtask_annotations.jsonl"))
    episodes = _read_jsonl(_dl("meta/episodes.jsonl"))[:max_episodes]
    return _build_records(info, vocab, episodes, _dl, repo)
