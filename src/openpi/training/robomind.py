"""RoboMIND (h5_franka_1rgb) -> MEM HL training-data assembler.

Parses the subtask annotations (static/language_description_annotation_json/h5_franka_1rgb.json),
builds 1 Hz HL samples aligned to observations, reads frames from the per-episode HDF5, and
writes a manifest + frames. Memory labels are produced separately by MemoryLabelGenerator.
"""

import dataclasses
import json
import pathlib
import re

import numpy as np

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


def _mk(eid, goal, idx, m_prev, success, frame, update, tgt_sub, tgt_mem):
    return {
        "episode_id": eid, "subtask_index": idx + 1, "frame": int(frame), "update": update,
        "goal": goal, "input_memory": m_prev, "target_subtask": tgt_sub,
        "target_memory": tgt_mem, "success": success,
    }


def build_samples(
    record: dict, memories: list[str], fps: float, sample_hz: float = 1.0, first_memory: str = "(none yet)"
) -> list[dict]:
    """1 Hz HL samples: within a subtask the target is (l_i, m_{i-1}) with no update; at the
    boundary frame e_i it is (l_{i+1}, m_i) on success or (l_i, m_{i-1}) on failure (no update).
    Last subtask advances to "done"."""
    subtasks, ranges, flags = record["subtasks"], record["frame_ranges"], record["success_flags"]
    goal, eid, n = record["goal"], record["id"], len(record["subtasks"])
    if len(memories) != n:
        raise ValueError(f"memories ({len(memories)}) != subtasks ({n}) for {eid}")
    stride = max(1, round(fps / sample_hz))
    samples = []
    for idx in range(n):
        s, e = int(ranges[idx][0]), int(ranges[idx][1])
        m_prev = first_memory if idx == 0 else memories[idx - 1]
        l_cur = subtasks[idx]
        l_next = subtasks[idx + 1] if idx + 1 < n else "done"
        for f in range(s, e, stride):
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], f, False, l_cur, m_prev))  # noqa: FBT003, PERF401
        if flags[idx]:
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], e, True, l_next, memories[idx]))  # noqa: FBT003
        else:
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], e, False, l_cur, m_prev))  # noqa: FBT003
    return samples


def _task_of(record_id: str) -> str:
    return record_id.split("/")[1]


# RoboMIND h5_franka_1rgb stores each camera_top frame as a flat uint8 HxWx3 RGB buffer (not JPEG).
_RAW_H, _RAW_W = 720, 1280


def _decode_resize(raw, size: int = 224) -> np.ndarray:
    import cv2

    arr = np.asarray(raw)
    if arr.ndim == 1:
        if arr.size >= 2 and arr[0] == 0xFF and arr[1] == 0xD8:  # JPEG magic -> encoded bytes
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"cv2.imdecode failed on 1-D input of shape {arr.shape}")
            arr = decoded[:, :, ::-1]  # BGR -> RGB
        elif arr.size == _RAW_H * _RAW_W * 3:  # flat raw RGB frame
            arr = arr.reshape(_RAW_H, _RAW_W, 3)
        else:
            raise ValueError(f"unrecognized 1-D frame buffer of size {arr.size} (not JPEG, not {_RAW_H}x{_RAW_W}x3)")
    return cv2.resize(arr, (size, size)).astype(np.uint8)


def read_fps(h5_path: str, default: float) -> float:
    import h5py

    with h5py.File(h5_path, "r") as f:
        for key in ("fps", "control_freq", "frequency", "hz"):
            if key in f.attrs:
                return float(f.attrs[key])
    return float(default)


def read_camera_top_frames(h5_path: str, indices: list[int], size: int = 224) -> dict[int, np.ndarray]:
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        ds = f["observations/rgb_images/camera_top"]
        last = len(ds) - 1
        for idx in indices:
            out[idx] = _decode_resize(ds[min(max(idx, 0), last)], size)
    return out


def assemble(
    records: list[dict], labels: list[dict], *, out_dir, cache_dir,
    fps_default: float = 10.0, sample_hz: float = 1.0, max_episodes: int | None = None,
) -> list[dict]:
    import shutil

    from PIL import Image

    out_dir = pathlib.Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = len(records) if max_episodes is None else min(max_episodes, len(records))
    if len(labels) < n:
        raise ValueError(f"labels ({len(labels)}) shorter than records ({n}) - records[i] must pair with labels[i]")
    # Validate records<->labels pairing on the original index, then process grouped by task so each
    # ~150GB task archive is fetched+extracted exactly once (episodes sharing a task reuse the extract).
    items = []
    for i in range(n):
        rec = records[i]
        lab = labels[i]
        if lab.get("episode_id") not in (None, str(i)):
            raise ValueError(f"label {i} episode_id={lab.get('episode_id')!r} != str({i}); records/labels misaligned")
        items.append((rec, lab, _task_of(rec["id"])))
    items.sort(key=lambda it: it[2])  # stable -> contiguous task groups, original order within a group
    rows = []
    prev_task = None
    for rec, lab, task in items:
        if prev_task is not None and task != prev_task:
            shutil.rmtree(pathlib.Path(cache_dir) / prev_task, ignore_errors=True)
        prev_task = task
        try:
            mems = lab["memories"]
            h5 = fetch_task_hdf5(REPO, rec["id"], cache_dir)
            fps = read_fps(h5, fps_default)
            samples = build_samples(rec, mems, fps, sample_hz)
            imgs = read_camera_top_frames(h5, sorted({s["frame"] for s in samples}))
            stem = rec["id"].replace("/", "_")
            ep_rows = []
            for s in samples:
                rel = f"frames/{stem}_{s['frame']}.jpg"
                Image.fromarray(imgs[s["frame"]]).save(out_dir / rel)
                s["image"] = rel
                ep_rows.append(s)
        except Exception as e:  # skip a bad episode, keep the run going
            print(f"skip {rec['id']}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.extend(ep_rows)
    if prev_task is not None:
        shutil.rmtree(pathlib.Path(cache_dir) / prev_task, ignore_errors=True)
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} HL samples -> {out_dir / 'manifest.jsonl'}")
    return rows


def fetch_task_hdf5(repo: str, record_id: str, cache_dir: str) -> str:
    """Download the record's task-tar parts, reassemble+extract into cache_dir/<task>, and return
    the episode's trajectory.hdf5. Caches per task (guarded by a .done sentinel); the caller deletes
    cache_dir/<task> when done."""
    import glob
    import shutil
    import tarfile

    import huggingface_hub

    task = _task_of(record_id)
    dest = pathlib.Path(cache_dir) / task
    done = dest / ".done"
    if not done.exists():
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)  # clean a stale partial extract
        files = huggingface_hub.list_repo_files(repo, repo_type="dataset")
        parts = sorted(f for f in files if f"/{task}.tar.gz.part-" in f)
        if not parts:
            raise FileNotFoundError(f"no tar parts for task {task}")
        dest.mkdir(parents=True, exist_ok=True)
        local = [huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=p) for p in parts]
        tar_path = dest / f"{task}.tar.gz"
        with open(tar_path, "wb") as out:
            for lp in local:
                with open(lp, "rb") as pf:
                    shutil.copyfileobj(pf, out)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest)
        tar_path.unlink()
        done.touch()
    ts = record_id.split("/")[-2]
    hits = glob.glob(f"{dest}/**/{ts}/data/trajectory.hdf5", recursive=True)
    if not hits:
        raise FileNotFoundError(f"trajectory.hdf5 not found for {record_id}")
    return hits[0]
