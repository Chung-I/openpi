"""Galaxea Open-World Dataset (LeRobot v2.1) -> MEM HL records via per-frame task_index RLE (Shape A).

Unlike RoboCOIN's per-frame vector column, Galaxea's episode parquets carry two per-frame SCALAR
int64 columns: `task_index` (the fine-grained subtask) and `coarse_task_index` (the episode's
overall goal, constant within an episode). Contiguous runs of `task_index` ARE the subtask segments,
so we reuse `robocoin.rle_segments` to find them; `coarse_task_index`'s most-common non-sentinel
value gives the goal. Each integer maps to text via the archive's `meta/tasks.jsonl` vocab (reused
via `robocoin.load_vocab` with Galaxea's own task_index/task field names) -- vocab strings are
bilingual `中文@English` (some rows are English-only), so `english()` extracts the English side.
Sentinel vocab strings "null"/"qualified"/"unqualified" are dropped. The result is the SAME record
shape (id, goal, subtasks, frame_ranges, success_flags) as `robomind.RobomindRecord`, so
`robomind.build_samples` and the memory-label generator reuse it unchanged. Galaxea ships no clean
success labels (quality_index is pre-filtered to approved episodes), so every kept subtask is marked
success for v1.

Galaxea packages one task per self-contained `lerobot/<archive_name>.tar.gz` (own meta/data/videos,
single chunk-000) -- unlike RoboCOIN's per-file parquet layout, there is no lazy per-episode fetch;
the whole archive must be downloaded and extracted before any episode can be read.
"""

import collections
import dataclasses
import json
import pathlib

from openpi.training import robocoin as rc
from openpi.training.memory_labels import Episode

# Vocab strings that are not real subtasks/goals (case-insensitive), checked on the
# English-normalized text: "null" is robocoin's "no annotation" sentinel; "qualified"/"unqualified"
# are quality-index strings that leak into the same tasks.jsonl vocab.
GALAXEA_SENTINELS = frozenset({"null", "qualified", "unqualified"})


@dataclasses.dataclass
class GalaxeaRecord:
    id: str
    goal: str
    subtasks: list[str]
    frame_ranges: list[tuple[int, int]]
    success_flags: list[bool]


def english(text: str) -> str:
    """The English side of a bilingual `中文@English` vocab string; unchanged if there's no '@'."""
    if "@" in text:
        return text.split("@", 1)[1].strip()
    return text.strip()


def is_sentinel(text: str) -> bool:
    return text.strip().lower() in GALAXEA_SENTINELS


def _goal_from_coarse(coarse_task_index, vocab: dict[int, str]) -> str:
    """The most-common non-sentinel `coarse_task_index` value's English text ("" if none).

    `coarse_task_index` is constant within an episode in practice, but we take the most-common
    value to be robust to a stray frame reading a different (or sentinel) value.
    """
    counts = collections.Counter(int(v) for v in coarse_task_index)
    candidates = [(count, val) for val, count in counts.items() if not is_sentinel(english(vocab[val]))]
    if not candidates:
        return ""
    candidates.sort(key=lambda item: -item[0])
    return english(vocab[candidates[0][1]])


def record_from_episode(episode_id: str, task_index, coarse_task_index, vocab: dict[int, str]) -> GalaxeaRecord | None:
    """Build one MEM record from an episode's per-frame task_index/coarse_task_index columns."""
    subtasks: list[str] = []
    ranges: list[tuple[int, int]] = []
    for val, start, end in rc.rle_segments(task_index):
        text = english(vocab[val])
        if not text or is_sentinel(text):
            continue
        subtasks.append(text)
        ranges.append((start, end - 1))  # inclusive boundary: last frame the subtask is active
    if not subtasks:
        return None
    return GalaxeaRecord(
        id=episode_id,
        goal=_goal_from_coarse(coarse_task_index, vocab),
        subtasks=subtasks,
        frame_ranges=ranges,
        success_flags=[True] * len(subtasks),
    )


def to_episode(rec: GalaxeaRecord) -> Episode:
    return Episode(goal=rec.goal, subtasks=list(rec.subtasks), success_flags=list(rec.success_flags))




def _read_task_columns(parquet_path) -> tuple[list[int], list[int]]:
    """(task_index, coarse_task_index) scalar per-frame columns from an episode parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["task_index", "coarse_task_index"])
    return table["task_index"].to_pylist(), table["coarse_task_index"].to_pylist()


def _build_records(info, vocab, episodes, resolve_parquet, archive_name: str) -> list[GalaxeaRecord]:
    """Shared record builder; `resolve_parquet(rel)` returns a readable parquet path for a rel path."""
    out: list[GalaxeaRecord] = []
    for ep in episodes:
        idx = int(ep["episode_index"])
        task_index, coarse_task_index = _read_task_columns(resolve_parquet(rc._parquet_rel(info, idx)))  # noqa: SLF001
        rec = record_from_episode(f"{archive_name}/episode_{idx:06d}", task_index, coarse_task_index, vocab)
        if rec is not None:
            out.append(rec)
    return out


def records_from_local(extracted_root, archive_name: str, max_episodes: int | None = None) -> list[GalaxeaRecord]:
    """Build MEM records from an extracted Galaxea/LeRobot archive dir (meta/ + data/)."""
    root = pathlib.Path(extracted_root) / archive_name
    info = json.loads((root / "meta" / "info.json").read_text())
    vocab = rc.load_vocab(root / "meta" / "tasks.jsonl", index_key="task_index", text_key="task")
    episodes = rc.read_jsonl(root / "meta" / "episodes.jsonl")[:max_episodes]
    return _build_records(info, vocab, episodes, lambda rel: root / rel, archive_name)


def records_from_repo(repo: str, archive_name: str, cache_dir, max_episodes: int | None = None) -> list[GalaxeaRecord]:
    """Download+extract one Galaxea task archive (`lerobot/{archive_name}.tar.gz`), then build records.

    Whole-archive download is unavoidable: Galaxea packages one self-contained LeRobot dataset per
    task as a single tar.gz, so there is no lazy per-episode/per-file fetch like RoboCOIN's. Reuses
    an already-extracted archive under `cache_dir` if present.
    """
    import tarfile

    import huggingface_hub

    extracted_root = pathlib.Path(cache_dir) / "extracted"
    if not (extracted_root / archive_name / "meta" / "info.json").exists():
        tar_path = huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=f"lerobot/{archive_name}.tar.gz")
        extracted_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extracted_root)  # trusted first-party (gated HF) dataset archive
    return records_from_local(extracted_root, archive_name, max_episodes=max_episodes)
