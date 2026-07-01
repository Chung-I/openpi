"""RoboMIND (h5_franka_1rgb) -> MEM HL training-data assembler.

Parses the subtask annotations (static/language_description_annotation_json/h5_franka_1rgb.json),
builds 1 Hz HL samples aligned to observations, reads frames from the per-episode HDF5, and
writes a manifest + frames. Memory labels are produced separately by MemoryLabelGenerator.
"""

import dataclasses
import json
import pathlib
import re

from openpi.training.memory_labels import Episode

REPO = "x-humanoid-robomind/RoboMIND"
ANNOTATION = "static/language_description_annotation_json/h5_franka_1rgb.json"
_FRAME_RE = re.compile(r"(\d+)\.jpg$")


@dataclasses.dataclass
class RobomindRecord:
    id: str
    goal: str
    subtasks: list[str]
    frame_ranges: list[tuple[int, int]]
    success_flags: list[bool]


def frame_index(name: str) -> int:
    m = _FRAME_RE.search(name or "")
    if not m:
        raise ValueError(f"no frame index in {name!r}")
    return int(m.group(1))


def record_from_annotation(obj: dict) -> RobomindRecord | None:
    resp = obj.get("response") or {}
    subtasks, ranges = [], []
    for st in resp.get("steps") or []:
        text = (st.get("step_description") or "").strip()
        if not text:
            continue
        subtasks.append(text)
        ranges.append((frame_index(st["start_frame"]), frame_index(st["end_frame"])))
    if not subtasks:
        return None
    return RobomindRecord(
        id=obj["id"],
        goal=(resp.get("task_summary") or "").strip(),
        subtasks=subtasks,
        frame_ranges=ranges,
        success_flags=[True] * len(subtasks),
    )


def records_from_annotation_list(objs: list[dict]) -> list[RobomindRecord]:
    out = []
    for o in objs:
        r = record_from_annotation(o)
        if r is not None:
            out.append(r)
    return out


def to_episode(rec: RobomindRecord) -> Episode:
    return Episode(goal=rec.goal, subtasks=list(rec.subtasks), success_flags=list(rec.success_flags))


def download_annotation(repo: str = REPO, filename: str = ANNOTATION) -> list[dict]:
    import huggingface_hub

    p = huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=filename)
    return json.loads(pathlib.Path(p).read_text())
