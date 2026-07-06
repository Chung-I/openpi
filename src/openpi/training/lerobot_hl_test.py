import json

import numpy as np
import pytest

from openpi.training import lerobot_hl as lh


def test_episode_index_of_parses_suffix():
    assert lh.episode_index_of("RoboCOIN/episode_000042") == 42
    assert lh.episode_index_of("some/repo/episode_000000") == 0


def test_episode_index_of_raises_without_match():
    with pytest.raises(ValueError, match="no episode index"):
        lh.episode_index_of("RoboCOIN/not_an_episode")


def _rec(idx=0):
    return {
        "id": f"RoboCOIN/episode_{idx:06d}",
        "goal": "g",
        "subtasks": ["a", "b"],
        "frame_ranges": [[0, 10], [10, 20]],
        "success_flags": [True, True],
    }


def _fake_video(idxs, size=224):
    return {i: np.zeros((size, size, 3), np.uint8) for i in idxs}


def test_assemble_raises_on_misaligned_labels(tmp_path):
    records = [_rec(0)]
    labels = [{"episode_id": "7", "memories": ["m1"]}]  # episode_id != str(0)
    with pytest.raises(ValueError, match="misaligned"):
        lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo")


def test_assemble_raises_when_labels_shorter_than_records(tmp_path):
    records = [_rec(0), _rec(1)]
    labels = [{"episode_id": "0", "memories": ["m1"]}]  # only 1 label for 2 records
    with pytest.raises(ValueError, match="shorter"):
        lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo")


def test_assemble_writes_manifest_and_frames(tmp_path, monkeypatch):
    records = [_rec(0)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(lh, "_download_meta", lambda repo: _write_info(tmp_path))
    monkeypatch.setattr(lh, "_download_video", lambda repo, rel: "fake.mp4")
    monkeypatch.setattr(lh, "read_video_frames", lambda path, idxs, size=224: _fake_video(idxs, size))

    rows = lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo", max_episodes=1)

    manifest = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    first = json.loads(manifest[0])
    assert first["image"].startswith("frames/")
    assert (tmp_path / first["image"]).exists()
    # a boundary update sample is present with the advanced target
    assert any(r["update"] and r["target_subtask"] == "b" and r["target_memory"] == "m1" for r in rows)


def test_assemble_skips_bad_episode_without_partial_rows(tmp_path, monkeypatch):
    records = [_rec(0), _rec(1)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}, {"episode_id": "1", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(lh, "_download_meta", lambda repo: _write_info(tmp_path))

    def _dl_video(repo, rel):
        if "000001" in rel:
            raise RuntimeError("boom")
        return "fake.mp4"

    monkeypatch.setattr(lh, "_download_video", _dl_video)
    monkeypatch.setattr(lh, "read_video_frames", lambda path, idxs, size=224: _fake_video(idxs, size))

    rows = lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo")

    manifest = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    assert all(r["episode_id"] == "RoboCOIN/episode_000000" for r in rows)


def _write_info(tmp_path):
    p = tmp_path / "info.json"
    p.write_text(json.dumps({"fps": 30.0, "chunks_size": 1000, "video_path": lh.VIDEO_PATH_TEMPLATE}))
    return str(p)


def _open_writer(path, size=(64, 64), fps=10):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        writer.release()
        return None
    return writer


def test_read_video_frames_decodes_requested_indices_and_clamps(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "synthetic.mp4"
    n_frames = 10
    size = (64, 64)
    writer = _open_writer(path, size)
    if writer is None:
        pytest.skip("cv2 VideoWriter could not open mp4v codec")
    try:
        for i in range(n_frames):
            brightness = int(i * 20)
            frame = np.full((size[1], size[0], 3), brightness, np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    frames = lh.read_video_frames(str(path), [0, 5, 9, 20], size=64)  # 20 is out-of-range -> clamps to last (9)
    assert set(frames) == {0, 5, 9, 20}
    assert frames[0].shape == (64, 64, 3)
    assert frames[0].dtype == np.uint8
    # monotonic brightness across decoded frames
    b0 = float(frames[0][32, 32].mean())
    b5 = float(frames[5][32, 32].mean())
    b9 = float(frames[9][32, 32].mean())
    assert b0 < b5 < b9
    # out-of-range index clamps to the last frame
    assert np.array_equal(frames[20], frames[9])


def test_read_video_frames_converts_bgr_to_rgb(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "color.mp4"
    size = (64, 64)
    writer = _open_writer(path, size)
    if writer is None:
        pytest.skip("cv2 VideoWriter could not open mp4v codec")
    try:
        # BGR pixel: B=10, G=20, R=200 (reddish) -- written as-is since VideoWriter expects BGR input.
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:, :] = (10, 20, 200)
        writer.write(frame)
    finally:
        writer.release()

    frames = lh.read_video_frames(str(path), [0], size=64)
    pixel = frames[0][32, 32]
    # after BGR->RGB conversion the red channel (index 0) should be the highest and blue (index 2) lowest
    assert int(pixel[0]) > int(pixel[1]) > int(pixel[2])
    assert int(pixel[0]) > 150
    assert int(pixel[2]) < 60
