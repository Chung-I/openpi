"""LeRobot-mp4 (RoboCOIN) -> MEM HL training-data assembler.

Stage A (`robocoin.records_from_repo`) produces records shaped like `robomind.RobomindRecord`
(id, goal, subtasks, frame_ranges, success_flags); Stage B (memory-label generation) produces
per-episode memory strings. This module is Stage C: it reads frames from a LeRobot v2.1 dataset's
per-episode mp4 videos (RoboCOIN is H.264, 30fps), reuses `robomind.build_samples` (dataset-agnostic
sampling) to build 1 Hz HL samples, and writes a manifest + frames -- the SAME output contract as
`robomind.assemble`, so downstream training code is dataset-agnostic.

Generic across LeRobot datasets that expose the standard layout
`videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`; RoboCOIN uses it
now, Galaxea will reuse this module later with a different `video_key`.
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


def read_video_frames(mp4_path, indices: list[int], size: int = 224) -> dict[int, np.ndarray]:
    """Decode the requested frame indices from an mp4. Sequential decode (cap.read() forward to
    each wanted index) since seeking is unreliable on some H.264 streams. Clamps out-of-range
    indices to the last frame (mirrors robomind.read_camera_top_frames). Returns RGB uint8 frames,
    letterbox-resized to size x size."""
    import cv2

    cap = cv2.VideoCapture(str(mp4_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last = max(total - 1, 0)
        clamped = [min(max(int(i), 0), last) for i in indices]
        decoded: dict[int, np.ndarray] = {}
        pos = -1
        frame = None
        for target in sorted(set(clamped)):
            while pos < target:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"failed to decode frame {pos + 1} from {mp4_path}")
                pos += 1
            rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
            decoded[target] = rm._resize_with_pad(rgb, size)  # noqa: SLF001
        return {int(i): decoded[c] for i, c in zip(indices, clamped, strict=True)}
    finally:
        cap.release()


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
    repo: str,
    video_key: str = "observation.images.cam_head_rgb",
    sample_hz: float = 1.0,
    max_episodes: int | None = None,
    fps_default: float = 30.0,
) -> list[dict]:
    from PIL import Image

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

    info = json.loads(pathlib.Path(_download_meta(repo)).read_text())
    fps = float(info.get("fps", fps_default))
    chunks_size = int(info.get("chunks_size", 1000))

    rows = []
    for i in range(n):
        rec = records[i]
        lab = labels[i]
        try:
            idx = episode_index_of(rec["id"])
            chunk = idx // chunks_size
            rel = VIDEO_PATH_TEMPLATE.format(episode_chunk=chunk, video_key=video_key, episode_index=idx)
            mp4_path = _download_video(repo, rel)
            samples = rm.build_samples(rec, lab["memories"], fps, sample_hz)
            imgs = read_video_frames(mp4_path, sorted({s["frame"] for s in samples}))
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

    (out_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} HL samples -> {out_dir / 'manifest.jsonl'}")
    return rows
