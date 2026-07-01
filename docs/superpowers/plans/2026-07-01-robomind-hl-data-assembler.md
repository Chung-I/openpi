# MEM HL Training-Data Assembler (RoboMIND Franka) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn RoboMIND `h5_franka_1rgb` subtask annotations + HDF5 frames + generated memory labels into HL training samples — each pairing an observation `o_t` with `(goal g, prev memory m_t)` and targets `(next subtask l_{t+1}, updated memory m_{t+1})` — written as a manifest + frames.

**Architecture:** Core library `src/openpi/training/robomind.py` (annotation parsing, the pure `build_samples` 1 Hz construction, HDF5 frame access, and the `assemble` orchestrator) plus two thin CLI scripts. Memory-label generation reuses the existing `MemoryLabelGenerator` (recursive, v7_paper) and is not re-implemented.

**Tech Stack:** Python 3.11, `huggingface_hub`, `h5py`, `opencv-python`, `pillow`, `numpy`, pytest, `uv run`.

## Global Constraints

- Builds on merged MEM + PR #4: the default `RECURSIVE_MEMORY_PROMPT_TEMPLATE` is the paper-faithful v7 prompt; `MemoryLabelGenerator` supports `generation_mode="recursive"`. Reuse `Episode` from `openpi.training.memory_labels`.
- HL conditions on a **single `o_t`** (one `camera_top` frame per sample), not a window.
- **1 Hz** observation sampling; the boundary/update sample is at frame `e_i` (subtask `i`'s `end_frame`).
- Construction (verbatim): within subtask `i` (`s_i ≤ frame < e_i`) → target `(l_i, m_{i-1})`, no update; boundary (`frame = e_i`, success) → target `(l_{i+1}, m_i)`, update; last subtask boundary → `("done", m_N)`; **failed** subtask boundary → no-update `(l_i, m_{i-1})`. `m_0 = "(none yet)"`.
- Failure-aware but success-only data (RoboMIND `failure_data/` deferred).
- Selective per-task fetch: download a task's `.tar.gz.part-*`, reassemble, extract, read frames, then the caller deletes (bounded disk).
- Output: `data/robomind_hl/manifest.jsonl` (one JSON object per line) + `data/robomind_hl/frames/`.
- Annotation source: repo `x-humanoid-robomind/RoboMIND`, file `static/language_description_annotation_json/h5_franka_1rgb.json`.
- Run tests with `uv run --no-sync python -m pytest <path> -q`; keep new lines ruff-clean (`uv run --no-sync ruff check <files>`).

---

### Task 1: RoboMIND annotation parsing

**Files:**
- Create: `src/openpi/training/robomind.py`
- Test: `src/openpi/training/robomind_test.py`

**Interfaces:**
- Consumes: `from openpi.training.memory_labels import Episode` (`Episode(goal, subtasks, success_flags)`).
- Produces:
  - `REPO = "x-humanoid-robomind/RoboMIND"`, `ANNOTATION = "static/language_description_annotation_json/h5_franka_1rgb.json"`
  - `@dataclass RobomindRecord(id: str, goal: str, subtasks: list[str], frame_ranges: list[tuple[int, int]], success_flags: list[bool])`
  - `frame_index(name: str) -> int`  (`"camera_top_0023.jpg" -> 23`)
  - `record_from_annotation(obj: dict) -> RobomindRecord | None`
  - `records_from_annotation_list(objs: list[dict]) -> list[RobomindRecord]`
  - `to_episode(rec: RobomindRecord) -> Episode`
  - `download_annotation(repo=REPO, filename=ANNOTATION) -> list[dict]`  (network; thin)

- [ ] **Step 1: Write the failing test**

```python
# src/openpi/training/robomind_test.py
from openpi.training import robomind as rm


def _obj():
    return {
        "id": "h5_franka_1rgb/bread_in_basket/success_episodes/train/1016_161244/data",
        "response": {
            "task_summary": "placing bread into a basket",
            "steps": [
                {"step_description": "move towards the bread", "start_frame": "camera_top_0000.jpg", "end_frame": "camera_top_0023.jpg"},
                {"step_description": "grab the bread", "start_frame": "camera_top_0030.jpg", "end_frame": "camera_top_0096.jpg"},
            ],
        },
    }


def test_frame_index():
    assert rm.frame_index("camera_top_0023.jpg") == 23


def test_record_parses_goal_subtasks_ranges():
    r = rm.record_from_annotation(_obj())
    assert r.goal == "placing bread into a basket"
    assert r.subtasks == ["move towards the bread", "grab the bread"]
    assert r.frame_ranges == [(0, 23), (30, 96)]
    assert r.success_flags == [True, True]


def test_to_episode_view():
    ep = rm.to_episode(rm.record_from_annotation(_obj()))
    assert ep.goal == "placing bread into a basket"
    assert ep.subtasks == ["move towards the bread", "grab the bread"]
    assert ep.success_flags == [True, True]


def test_record_none_when_no_steps():
    assert rm.record_from_annotation({"id": "x", "response": {"task_summary": "g", "steps": []}}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: FAIL (module `openpi.training.robomind` does not exist).

- [ ] **Step 3: Implement the module (parsing part)**

```python
# src/openpi/training/robomind.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/robomind.py src/openpi/training/robomind_test.py
git commit -m "feat(mem): RoboMIND annotation parsing for HL assembler"
```

---

### Task 2: `build_samples` — the 1 Hz construction

**Files:**
- Modify: `src/openpi/training/robomind.py`
- Test: `src/openpi/training/robomind_test.py`

**Interfaces:**
- Consumes: a record as a plain dict (`{"id","goal","subtasks","frame_ranges","success_flags"}` — matches `dataclasses.asdict(RobomindRecord)`), and `memories = [m_1..m_N]`.
- Produces:
  - `build_samples(record: dict, memories: list[str], fps: float, sample_hz: float = 1.0, first_memory: str = "(none yet)") -> list[dict]` — each dict has `episode_id, subtask_index (1-based), frame, update, goal, input_memory, target_subtask, target_memory, success` (no `image` yet).

- [ ] **Step 1: Write the failing tests**

```python
# append to src/openpi/training/robomind_test.py
import pytest


def _rec():
    return {"id": "ep1", "goal": "g", "subtasks": ["a", "b", "c"],
            "frame_ranges": [[0, 10], [10, 20], [20, 30]], "success_flags": [True, True, True]}


def test_within_subtask_samples_are_no_update():
    s = rm.build_samples(_rec(), ["m1", "m2", "m3"], fps=10, sample_hz=1)  # stride = 10
    w = [x for x in s if x["subtask_index"] == 1 and not x["update"]]
    assert w[0]["frame"] == 0
    assert w[0]["target_subtask"] == "a"
    assert w[0]["input_memory"] == "(none yet)"
    assert w[0]["target_memory"] == "(none yet)"


def test_boundary_updates_memory_and_advances_subtask():
    s = rm.build_samples(_rec(), ["m1", "m2", "m3"], fps=10, sample_hz=1)
    b1 = next(x for x in s if x["subtask_index"] == 1 and x["update"])
    assert b1["frame"] == 10
    assert b1["target_subtask"] == "b"
    assert b1["target_memory"] == "m1"
    assert b1["input_memory"] == "(none yet)"
    b2 = next(x for x in s if x["subtask_index"] == 2 and x["update"])
    assert b2["target_subtask"] == "c"
    assert b2["target_memory"] == "m2"
    assert b2["input_memory"] == "m1"


def test_last_subtask_boundary_is_done():
    s = rm.build_samples(_rec(), ["m1", "m2", "m3"], fps=10, sample_hz=1)
    b3 = next(x for x in s if x["subtask_index"] == 3 and x["update"])
    assert b3["target_subtask"] == "done"
    assert b3["target_memory"] == "m3"


def test_failed_subtask_boundary_is_no_update():
    rec = _rec()
    rec["success_flags"] = [True, False, True]
    s = rm.build_samples(rec, ["m1", "m1", "m3"], fps=10, sample_hz=1)  # failed subtask 2 -> m2 == m1
    assert [x for x in s if x["subtask_index"] == 2 and x["update"]] == []  # no update on failure
    end2 = next(x for x in s if x["subtask_index"] == 2 and x["frame"] == 20)
    assert end2["target_subtask"] == "b"
    assert end2["target_memory"] == "m1"
    assert end2["update"] is False


def test_memories_length_mismatch_raises():
    with pytest.raises(ValueError):
        rm.build_samples(_rec(), ["m1", "m2"], fps=10)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: FAIL (`build_samples` not defined).

- [ ] **Step 3: Implement `build_samples` (+ `_mk`)**

```python
# add to src/openpi/training/robomind.py
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
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], f, False, l_cur, m_prev))
        if flags[idx]:
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], e, True, l_next, memories[idx]))
        else:
            samples.append(_mk(eid, goal, idx, m_prev, flags[idx], e, False, l_cur, m_prev))
    return samples
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/training/robomind.py src/openpi/training/robomind_test.py
git commit -m "feat(mem): build_samples 1Hz HL construction (vision-conditioned when-to-update)"
```

---

### Task 3: HDF5 frame access

**Files:**
- Modify: `src/openpi/training/robomind.py`, `pyproject.toml`
- Test: `src/openpi/training/robomind_test.py`

**Interfaces:**
- Produces:
  - `_task_of(record_id: str) -> str`  (`".../bread_in_basket/success_episodes/..." -> "bread_in_basket"`)
  - `_decode_resize(raw, size: int = 224) -> np.ndarray`  (encoded-JPEG bytes or raw HWC array -> `size x size x 3` uint8)
  - `read_fps(h5_path: str, default: float) -> float`  (from HDF5 `attrs`; else default)
  - `read_camera_top_frames(h5_path: str, indices: list[int], size: int = 224) -> dict[int, np.ndarray]`
  - `fetch_task_hdf5(repo: str, record_id: str, cache_dir: str) -> str`  (network; download task parts -> reassemble -> extract -> return `trajectory.hdf5` path)

**Discovery note:** RoboMIND `rgb_images/camera_top` may store per-frame **encoded JPEG bytes** (1-D uint8) or **raw HWC arrays**; `_decode_resize` handles both. `read_fps` checks common attr keys. Step 1 confirms against one real HDF5 if HF access is available; otherwise the code's both-paths handling + the synthetic-h5 tests below are sufficient (the frame layout is re-confirmed at the first cluster run).

- [ ] **Step 1: (discovery, best-effort) confirm HDF5 layout**

Run (only if HF access is available; otherwise skip — the implementation handles both encodings):
```bash
uv run --no-sync python - <<'PY'
import huggingface_hub, glob, tarfile, h5py, pathlib
repo = "x-humanoid-robomind/RoboMIND"
files = huggingface_hub.list_repo_files(repo, repo_type="dataset")
parts = sorted(f for f in files if "/bread_in_basket.tar.gz.part-" in f)[:3]  # partial peek
print("parts:", parts[:3])
PY
```
Expected: prints task-tar part paths. (Full extraction happens at the cluster run.)

- [ ] **Step 2: Write the failing tests (synthetic HDF5, no network)**

```python
# append to src/openpi/training/robomind_test.py
import numpy as np


def test_task_of():
    assert rm._task_of("h5_franka_1rgb/bread_in_basket/success_episodes/train/1016_161244/data") == "bread_in_basket"


def test_decode_resize_raw_array():
    out = rm._decode_resize(np.zeros((10, 12, 3), np.uint8), size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_read_fps_and_frames_from_h5(tmp_path):
    import h5py
    p = tmp_path / "trajectory.hdf5"
    with h5py.File(p, "w") as f:
        g = f.create_group("rgb_images")
        g.create_dataset("camera_top", data=np.zeros((5, 8, 8, 3), np.uint8))
        f.attrs["fps"] = 10.0
    assert rm.read_fps(str(p), default=3.0) == 10.0
    frames = rm.read_camera_top_frames(str(p), [0, 4, 99], size=224)  # 99 clamps to last
    assert set(frames) == {0, 4, 99}
    assert frames[0].shape == (224, 224, 3)


def test_read_fps_default_when_missing(tmp_path):
    import h5py
    p = tmp_path / "t.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("rgb_images").create_dataset("camera_top", data=np.zeros((2, 4, 4, 3), np.uint8))
    assert rm.read_fps(str(p), default=7.5) == 7.5
```

- [ ] **Step 3: Add h5py to dependencies**

In `pyproject.toml`, in the `dependencies = [` list, add after `"nltk>=3.9",`:
```toml
    "h5py>=3.11",
```
Then: `uv sync` (or the reviewer/runner installs it). Expected: h5py resolves.

- [ ] **Step 4: Implement frame access**

```python
# add to src/openpi/training/robomind.py
import numpy as np


def _task_of(record_id: str) -> str:
    return record_id.split("/")[1]


def _decode_resize(raw, size: int = 224):
    import cv2

    arr = np.asarray(raw)
    if arr.ndim == 1:  # encoded JPEG bytes
        arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR -> RGB
    return cv2.resize(arr, (size, size)).astype(np.uint8)


def read_fps(h5_path: str, default: float) -> float:
    import h5py

    with h5py.File(h5_path, "r") as f:
        for key in ("fps", "control_freq", "frequency", "hz"):
            if key in f.attrs:
                return float(f.attrs[key])
    return float(default)


def read_camera_top_frames(h5_path: str, indices: list[int], size: int = 224) -> dict:
    import h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        ds = f["rgb_images/camera_top"]
        last = len(ds) - 1
        for idx in indices:
            out[idx] = _decode_resize(ds[min(max(idx, 0), last)], size)
    return out


def fetch_task_hdf5(repo: str, record_id: str, cache_dir: str) -> str:
    """Download the record's task-tar parts, reassemble+extract into cache_dir/<task>, and return
    the episode's trajectory.hdf5. Caches per task; the caller deletes cache_dir/<task> when done."""
    import glob
    import tarfile

    import huggingface_hub

    task = _task_of(record_id)
    dest = pathlib.Path(cache_dir) / task
    if not dest.exists():
        files = huggingface_hub.list_repo_files(repo, repo_type="dataset")
        parts = sorted(f for f in files if f"/{task}.tar.gz.part-" in f)
        if not parts:
            raise FileNotFoundError(f"no tar parts for task {task}")
        dest.mkdir(parents=True, exist_ok=True)
        local = [huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=p) for p in parts]
        tar_path = dest / f"{task}.tar.gz"
        with open(tar_path, "wb") as out:
            for lp in local:
                out.write(pathlib.Path(lp).read_bytes())
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(dest)  # noqa: S202 -- trusted dataset archive
        tar_path.unlink()
    ts = record_id.split("/")[-2]
    hits = glob.glob(f"{dest}/**/{ts}/data/trajectory.hdf5", recursive=True)
    if not hits:
        raise FileNotFoundError(f"trajectory.hdf5 not found for {record_id}")
    return hits[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: PASS (13 passed).

- [ ] **Step 6: Commit**

```bash
git add src/openpi/training/robomind.py src/openpi/training/robomind_test.py pyproject.toml
git commit -m "feat(mem): RoboMIND HDF5 frame access (selective fetch + decode/resize) + h5py dep"
```

---

### Task 4: `assemble` orchestrator + CLI scripts

**Files:**
- Modify: `src/openpi/training/robomind.py`
- Create: `scripts/load_robomind_episodes.py`, `scripts/assemble_hl_data.py`
- Test: `src/openpi/training/robomind_test.py`

**Interfaces:**
- Consumes: `build_samples`, `fetch_task_hdf5`, `read_fps`, `read_camera_top_frames` (this module); `records` (list of record dicts) and `labels` (list of `{"episode_id","memories"}`, positionally aligned to records).
- Produces:
  - `assemble(records: list[dict], labels: list[dict], *, out_dir, cache_dir, fps_default: float = 10.0, sample_hz: float = 1.0, max_episodes: int | None = None) -> list[dict]` — writes `out_dir/manifest.jsonl` + `out_dir/frames/*.jpg`, returns the sample rows (each with an added `image` relative path).
  - CLI `scripts/load_robomind_episodes.py` (writes `records.json` + `episodes.json`) and `scripts/assemble_hl_data.py` (runs `assemble`).

- [ ] **Step 1: Write the failing integration test (mocked I/O)**

```python
# append to src/openpi/training/robomind_test.py
def test_assemble_writes_manifest_and_frames(tmp_path, monkeypatch):
    records = [{"id": "h5_franka_1rgb/t/success_episodes/train/1/data", "goal": "g",
                "subtasks": ["a", "b"], "frame_ranges": [[0, 10], [10, 20]], "success_flags": [True, True]}]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(rm, "fetch_task_hdf5", lambda repo, rid, cache: "fake.hdf5")
    monkeypatch.setattr(rm, "read_fps", lambda h5, default: 10.0)
    monkeypatch.setattr(rm, "read_camera_top_frames",
                        lambda h5, idxs, size=224: {i: np.zeros((size, size, 3), np.uint8) for i in idxs})
    rows = rm.assemble(records, labels, out_dir=tmp_path, cache_dir=tmp_path / "cache", max_episodes=1)
    manifest = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    import json as _json
    first = _json.loads(manifest[0])
    assert first["image"].startswith("frames/")
    assert (tmp_path / first["image"]).exists()
    # a boundary update sample is present with the advanced target
    assert any(r["update"] and r["target_subtask"] == "b" and r["target_memory"] == "m1" for r in rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py::test_assemble_writes_manifest_and_frames -q`
Expected: FAIL (`assemble` not defined).

- [ ] **Step 3: Implement `assemble`**

```python
# add to src/openpi/training/robomind.py
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
    rows = []
    for i in range(n):
        rec = records[i]
        mems = labels[i]["memories"]
        task = _task_of(rec["id"])
        try:
            h5 = fetch_task_hdf5(REPO, rec["id"], cache_dir)
            fps = read_fps(h5, fps_default)
            samples = build_samples(rec, mems, fps, sample_hz)
            imgs = read_camera_top_frames(h5, sorted({s["frame"] for s in samples}))
        except Exception as e:  # noqa: BLE001 -- skip a bad episode, keep the run going
            print(f"skip {rec['id']}: {type(e).__name__}: {e}", flush=True)
            continue
        stem = rec["id"].replace("/", "_")
        for s in samples:
            rel = f"frames/{stem}_{s['frame']}.jpg"
            Image.fromarray(imgs[s["frame"]]).save(out_dir / rel)
            s["image"] = rel
            rows.append(s)
        # extract-and-discard: free the task cache once this episode's task is consumed
        shutil.rmtree(pathlib.Path(cache_dir) / task, ignore_errors=True)
    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} HL samples -> {out_dir / 'manifest.jsonl'}")
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q`
Expected: PASS (14 passed).

- [ ] **Step 5: Create the CLI scripts**

`scripts/load_robomind_episodes.py`:
```python
"""Download RoboMIND h5_franka_1rgb annotations -> records.json + episodes.json (for label gen)."""
import argparse
import dataclasses
import json
import pathlib

from openpi.training import robomind as rm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records-out", required=True)
    p.add_argument("--episodes-out", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--repo", default=rm.REPO)
    p.add_argument("--annotation", default=rm.ANNOTATION)
    args = p.parse_args()

    recs = rm.records_from_annotation_list(rm.download_annotation(args.repo, args.annotation))
    if args.max_episodes:
        recs = recs[: args.max_episodes]
    pathlib.Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.records_out).write_text(json.dumps([dataclasses.asdict(r) for r in recs], indent=2))
    pathlib.Path(args.episodes_out).write_text(
        json.dumps([dataclasses.asdict(rm.to_episode(r)) for r in recs], indent=2)
    )
    print(f"Wrote {len(recs)} records -> {args.records_out} ; episodes -> {args.episodes_out}")


if __name__ == "__main__":
    main()
```

`scripts/assemble_hl_data.py`:
```python
"""Assemble RoboMIND HL training samples (records + memory labels + frames) -> manifest + frames.

Stage A first (reuse the existing generator to produce labels):
    uv run python scripts/generate_memory_labels.py --episodes_file data/robomind_episodes.json \
        --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B \
        --output data/robomind_labels.json --disable_thinking   # recursive v7 by default
Then:
    uv run python scripts/assemble_hl_data.py --records_file data/robomind_records.json \
        --labels_file data/robomind_labels.json --out_dir data/robomind_hl
"""
import argparse
import json
import pathlib

from openpi.training import robomind as rm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--out_dir", default="data/robomind_hl")
    p.add_argument("--cache_dir", default="data/robomind_cache")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--sample-hz", type=float, default=1.0)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    records = json.loads(pathlib.Path(args.records_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())
    rm.assemble(records, labels, out_dir=args.out_dir, cache_dir=args.cache_dir,
                fps_default=args.fps, sample_hz=args.sample_hz, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify scripts import + full module + ruff**

Run:
```bash
uv run --no-sync python -c "import ast; ast.parse(open('scripts/load_robomind_episodes.py').read()); ast.parse(open('scripts/assemble_hl_data.py').read()); print('scripts OK')"
uv run --no-sync python -m pytest src/openpi/training/robomind_test.py -q
uv run --no-sync ruff check src/openpi/training/robomind.py src/openpi/training/robomind_test.py scripts/load_robomind_episodes.py scripts/assemble_hl_data.py
```
Expected: scripts OK; 14 passed; ruff clean on changed lines.

- [ ] **Step 7: Commit**

```bash
git add src/openpi/training/robomind.py src/openpi/training/robomind_test.py \
        scripts/load_robomind_episodes.py scripts/assemble_hl_data.py
git commit -m "feat(mem): HL data assembler orchestrator + CLIs (RoboMIND -> manifest + frames)"
```

---

## Post-implementation (manual, on NCHC — not a coding task)

1. `uv run python scripts/load_robomind_episodes.py --records-out data/robomind_records.json --episodes-out data/robomind_episodes.json` (161 episodes; text-only).
2. Serve Qwen; `generate_memory_labels.py` on `robomind_episodes.json` (recursive v7) -> `robomind_labels.json`.
3. `uv run python scripts/assemble_hl_data.py --records_file ... --labels_file ...` on a compute node with HF access + disk for the sampled task-tars; confirm `manifest.jsonl` rows + `frames/` images look right; confirm the real HDF5 `camera_top` encoding + fps (adjust `--fps` / `_decode_resize` if the discovery step was skipped).
4. Follow-up specs: ingest `failure_data/`; manifest -> `Observation` transform + pi0_mem HL training.

## Self-Review

- **Spec coverage:** parsing (Task 1), the 1 Hz construction incl. failure no-update (Task 2), frame access + selective fetch + h5py (Task 3), orchestrator + manifest + extract-discard + CLIs (Task 4), Stage-A reuse of the existing generator (Post-implementation). Manifest schema fields match the spec.
- **Type consistency:** `RobomindRecord`/`dataclasses.asdict` dict keys (`subtasks`, `frame_ranges`, `success_flags`, `id`, `goal`) are consumed identically by `build_samples`/`assemble`; `memories` is `list[str]` from `labels[i]["memories"]`; sample dict keys are stable across Tasks 2 and 4.
- **Placeholders:** none; the one real unknown (HDF5 `camera_top` encoding + fps attr) is handled by `_decode_resize`'s both-paths logic + the discovery step + the cluster-run confirmation.
