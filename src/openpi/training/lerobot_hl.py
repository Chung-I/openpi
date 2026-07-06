"""LeRobot-mp4 (RoboCOIN) -> MEM HL training-data assembler.

Stage A (`robocoin.records_from_repo`) produces records shaped like `robomind.RobomindRecord`
(id, goal, subtasks, frame_ranges, success_flags); Stage B (memory-label generation) produces
per-episode memory strings. This module is Stage C: it reads frames from a LeRobot v2.1 dataset's
per-episode mp4 videos (RoboCOIN is H.264, 30fps), reuses `robomind.build_samples` (dataset-agnostic
sampling) to build 1 Hz HL samples, and writes a manifest + frames -- the SAME output contract as
`robomind.assemble`, so downstream training code is dataset-agnostic.

Generic across LeRobot datasets that expose the standard layout
`videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4` (`VIDEO_PATH_TEMPLATE`
below is only the fallback default -- `assemble` prefers `info["video_path"]` from the dataset's own
`meta/info.json` when present, since layouts vary across datasets); RoboCOIN uses the default layout
with H.264 video fetched per-episode from an HF repo (`repo=`); Galaxea reuses this module with a
different `video_key`, AV1 video (`decode_backend="av"`, decoded via PyAV since cv2 silently fails on
AV1), and a locally-extracted archive (`video_root=`, no HF download -- Galaxea ships one whole task
archive per repo file, already extracted by Stage A).
"""

import json
import pathlib
import re

import numpy as np

from openpi.training import robomind as rm

VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_EPISODE_RE = re.compile(r"episode_(\d+)$")


def episode_index_of(record_id: str) -> int:
    m = _EPISODE_RE.search(record_id)
    if not m:
        raise ValueError(f"no episode index in {record_id!r}")
    return int(m.group(1))


def _iter_frames_cv2(mp4_path: str | pathlib.Path):
    """Yield decoded frames (RGB, in decode order) from an mp4 via cv2. Sequential decode
    (cap.read() forward) since seeking is unreliable on some H.264 streams."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame[:, :, ::-1]  # BGR -> RGB
    finally:
        cap.release()


def _iter_frames_av(mp4_path: str | pathlib.Path):
    """Yield decoded frames (RGB, in decode order) from an mp4 via PyAV. Needed for codecs cv2
    cannot decode (e.g. Galaxea's AV1 head-cam video, where cv2.VideoCapture opens the file but
    every read() silently returns False). `frame.to_ndarray(format="rgb24")` is already RGB."""
    import av

    container = av.open(str(mp4_path))
    try:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


_BACKENDS = {"cv2": _iter_frames_cv2, "av": _iter_frames_av}


def _frames_from_iterator(mp4_path: str | pathlib.Path, frames, indices: list[int], size: int) -> dict[int, np.ndarray]:
    """Shared clamp/keying/resize logic for `read_video_frames` backends. `frames` is an iterator
    yielding decoded RGB frames in order (already RGB -- backends do any BGR->RGB conversion
    themselves before yielding). Correctness does NOT depend on any codec/container frame-count
    metadata (unreliable for some streams -- can be 0 or wrong): indices past EOF are discovered by
    decoding and clamp to the last successfully decoded frame (mirrors
    robomind.read_camera_top_frames). Returns RGB uint8 frames, letterbox-resized to size x size,
    keyed by the ORIGINAL requested indices."""
    wanted = sorted({max(int(i), 0) for i in indices})
    max_target = wanted[-1]
    by_pos: dict[int, np.ndarray] = {}
    pos = -1
    last_frame = None
    for frame in frames:
        pos += 1
        last_frame = frame
        if pos in wanted and pos not in by_pos:
            by_pos[pos] = rm._resize_with_pad(np.ascontiguousarray(frame), size)  # noqa: SLF001
        if pos >= max_target:
            break
    if last_frame is None:
        raise ValueError(f"failed to decode any frame from {mp4_path}")
    if pos not in by_pos:  # EOF hit before max_target -> `pos` is the true last decoded frame
        by_pos[pos] = rm._resize_with_pad(np.ascontiguousarray(last_frame), size)  # noqa: SLF001
    return {int(i): by_pos.get(max(int(i), 0), by_pos[pos]) for i in indices}


def read_video_frames(
    mp4_path: str | pathlib.Path, indices: list[int], size: int = 224, backend: str = "cv2"
) -> dict[int, np.ndarray]:
    """Decode the requested frame indices from an mp4. `backend="cv2"` (default) keeps RoboCOIN's
    proven H.264 path unchanged; `backend="av"` uses PyAV, needed for codecs cv2 cannot decode (e.g.
    Galaxea's AV1 head-cam video). Both backends share the same contract via `_frames_from_iterator`:
    returns RGB uint8 frames, letterbox-resized to size x size, keyed by the ORIGINAL requested
    indices, with out-of-range indices clamped to the last successfully decoded frame (EOF-driven,
    not metadata-driven), and raising ValueError if nothing decodes."""
    if not indices:
        return {}
    try:
        iter_fn = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown video decode backend {backend!r}; expected one of {sorted(_BACKENDS)}") from None
    return _frames_from_iterator(mp4_path, iter_fn(mp4_path), indices, size)


def _download_meta(repo: str, filename: str = "meta/info.json") -> str:
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=filename)


def _download_video(repo: str, rel: str) -> str:
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=rel)


def assemble(
    records: list[dict],
    labels: list[dict],
    *,
    out_dir,
    repo: str | None = None,
    video_root: str | pathlib.Path | None = None,
    video_key: str = "observation.images.cam_head_rgb",
    video_path_template: str | None = None,
    sample_hz: float = 1.0,
    max_episodes: int | None = None,
    fps_default: float = 30.0,
    decode_backend: str = "cv2",
    max_samples_per_subtask: int | None = None,
    sample_jitter: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    """Assemble HL samples + frames from a LeRobot v2.1 dataset. Exactly one of `repo` (RoboCOIN:
    download meta/info.json + per-episode mp4 from an HF dataset repo) / `video_root` (Galaxea: a
    locally-extracted LeRobot-layout archive dir with meta/info.json; AgiBot: a locally-extracted
    RAW (non-LeRobot) layout dir with NO meta/info.json -- resolve mp4s under it directly, no
    download) must be given.

    `video_path_template`, if given, takes precedence over `info.get("video_path")` and the
    `VIDEO_PATH_TEMPLATE` default (in that order) -- lets AgiBot's raw `{episode_index}/videos/
    {video_key}` layout coexist with Galaxea's info.json-driven template and RoboCOIN's default.

    `max_samples_per_subtask`, if given, caps the number of within-subtask samples per subtask to
    counter duration bias (see `robomind.build_samples` for details). `sample_jitter`/`seed` add
    deterministic generation-time noise to within-subtask sampling times (see `robomind.build_samples`
    for details); default `sample_jitter=0.0` is off and byte-identical to the pre-jitter grid."""
    from PIL import Image

    if (repo is None) == (video_root is None):
        raise ValueError("exactly one of repo/video_root must be given")

    out_dir = pathlib.Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n = len(records) if max_episodes is None else min(max_episodes, len(records))
    if len(labels) < n:
        raise ValueError(f"labels ({len(labels)}) shorter than records ({n}) - records[i] must pair with labels[i]")
    for i in range(n):
        eid = labels[i].get("episode_id")
        if eid not in (None, str(i)):
            raise ValueError(f"label {i} episode_id={eid!r} != str({i}); records/labels misaligned")

    if video_root is not None:
        video_root = pathlib.Path(video_root)
        info_path = video_root / "meta" / "info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
    else:
        info = json.loads(pathlib.Path(_download_meta(repo)).read_text())
    fps = float(info.get("fps", fps_default))
    chunks_size = int(info.get("chunks_size", 1000))
    video_path_template = video_path_template or info.get("video_path", VIDEO_PATH_TEMPLATE)

    rows = []
    for i in range(n):
        rec = records[i]
        lab = labels[i]
        try:
            idx = episode_index_of(rec["id"])
            chunk = idx // chunks_size
            rel = video_path_template.format(episode_chunk=chunk, video_key=video_key, episode_index=idx)
            mp4_path = video_root / rel if video_root is not None else _download_video(repo, rel)
            samples = rm.build_samples(
                rec, lab["memories"], fps, sample_hz, max_samples_per_subtask=max_samples_per_subtask,
                sample_jitter=sample_jitter, seed=seed,
            )
            imgs = read_video_frames(mp4_path, sorted({s["frame"] for s in samples}), backend=decode_backend)
            stem = rec["id"].replace("/", "_")
            ep_rows = []
            for s in samples:
                rel_img = f"frames/{stem}_{s['frame']}.jpg"
                Image.fromarray(imgs[s["frame"]]).save(out_dir / rel_img)
                s["image"] = rel_img
                ep_rows.append(s)
        except Exception as e:  # skip a bad episode, keep the run going
            print(f"skip {rec['id']}: {type(e).__name__}: {e}", flush=True)
            continue
        rows.extend(ep_rows)

    if n > 0 and not rows:
        raise RuntimeError(
            f"all {n} episodes were skipped - 0 HL samples produced (check repo/video_key/video_path template)"
        )

    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} HL samples -> {out_dir / 'manifest.jsonl'}")
    return rows
