import json
import pathlib

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
    monkeypatch.setattr(lh, "read_video_frames", lambda path, idxs, size=224, **kwargs: _fake_video(idxs, size))

    rows = lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo", max_episodes=1)

    manifest = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    first = json.loads(manifest[0])
    assert first["image"].startswith("frames/")
    assert (tmp_path / first["image"]).exists()
    # a boundary update sample is present with the advanced target
    assert any(r["update"] and r["target_subtask"] == "b" and r["target_memory"] == "m1" for r in rows)


def test_assemble_uses_video_path_template_from_info_json(tmp_path, monkeypatch):
    """info.json's video_path template must be honored, not the module's default constant --
    the default is only a fallback. Uses a custom template shaped differently from the default
    to prove the value actually came from info.json."""
    records = [_rec(0)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    custom_template = "custom/{episode_chunk:03d}/{video_key}/ep{episode_index}.mp4"
    assert custom_template != lh.VIDEO_PATH_TEMPLATE

    def _write_custom_info():
        p = tmp_path / "info_custom.json"
        p.write_text(json.dumps({"fps": 30.0, "chunks_size": 1000, "video_path": custom_template}))
        return str(p)

    monkeypatch.setattr(lh, "_download_meta", lambda repo: _write_custom_info())
    requested = {}

    def _dl_video(repo, rel):
        requested["rel"] = rel
        return "fake.mp4"

    monkeypatch.setattr(lh, "_download_video", _dl_video)
    monkeypatch.setattr(lh, "read_video_frames", lambda path, idxs, size=224, **kwargs: _fake_video(idxs, size))

    lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo", max_episodes=1)

    expected_rel = custom_template.format(episode_chunk=0, video_key="observation.images.cam_head_rgb", episode_index=0)
    default_rel = lh.VIDEO_PATH_TEMPLATE.format(
        episode_chunk=0, video_key="observation.images.cam_head_rgb", episode_index=0
    )
    assert requested["rel"] == expected_rel
    assert requested["rel"] != default_rel


def test_assemble_skips_bad_episode_without_partial_rows(tmp_path, monkeypatch):
    records = [_rec(0), _rec(1)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}, {"episode_id": "1", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(lh, "_download_meta", lambda repo: _write_info(tmp_path))

    def _dl_video(repo, rel):
        if "000001" in rel:
            raise RuntimeError("boom")
        return "fake.mp4"

    monkeypatch.setattr(lh, "_download_video", _dl_video)
    monkeypatch.setattr(lh, "read_video_frames", lambda path, idxs, size=224, **kwargs: _fake_video(idxs, size))

    rows = lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo")

    manifest = (tmp_path / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    assert all(r["episode_id"] == "RoboCOIN/episode_000000" for r in rows)


def test_assemble_raises_when_all_episodes_skipped(tmp_path, monkeypatch):
    """If all episodes fail, assemble must raise RuntimeError before writing an empty manifest."""
    records = [_rec(0), _rec(1)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}, {"episode_id": "1", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(lh, "_download_meta", lambda repo: _write_info(tmp_path))
    monkeypatch.setattr(lh, "_download_video", lambda repo, rel: "fake.mp4")
    # Always raise to simulate all episodes failing
    monkeypatch.setattr(
        lh,
        "read_video_frames",
        lambda path, idxs, size=224, **kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")),
    )

    with pytest.raises(RuntimeError, match="all.*skipped"):
        lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo")

    # Manifest should NOT have been written when all episodes were skipped
    assert not (tmp_path / "manifest.jsonl").exists()


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


def test_read_video_frames_clamp_ignores_bogus_frame_count_metadata(monkeypatch):
    """CAP_PROP_FRAME_COUNT can be 0 or wrong for some H.264 containers. Correctness must come from
    EOF discovered by decoding, not from that metadata -- simulate a capture that lies about having
    zero frames, then confirm in-range and out-of-range indices still resolve correctly."""
    cv2 = pytest.importorskip("cv2")
    frames = [np.full((64, 64, 3), i * 40, np.uint8) for i in range(5)]

    class _FakeCap:
        def __init__(self, _path):
            self.pos = -1

        def get(self, prop):
            return 0  # lies: metadata claims zero frames regardless of what's asked

        def read(self):
            self.pos += 1
            if self.pos < len(frames):
                return True, frames[self.pos].copy()
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _FakeCap)

    result = lh.read_video_frames("ignored.mp4", [0, 2, 4, 100], size=64)

    assert set(result) == {0, 2, 4, 100}
    # out-of-range clamps to the last actually-decoded frame, not to the (bogus) metadata count
    assert np.array_equal(result[100], result[4])
    assert not np.array_equal(result[0], result[4])


def test_read_video_frames_raises_on_undecodable_video(monkeypatch):
    """Zero frames ever decoded is a genuine error and must still raise, not silently degenerate."""
    cv2 = pytest.importorskip("cv2")

    class _EmptyCap:
        def __init__(self, _path):
            pass

        def get(self, prop):
            return 0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _EmptyCap)

    with pytest.raises(ValueError, match="failed to decode"):
        lh.read_video_frames("empty.mp4", [0, 3], size=64)


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


def test_read_video_frames_av_backend_agrees_with_cv2_on_same_file(tmp_path):
    """AV1 is impractical to encode in a unit test; PyAV also decodes H.264/mpeg4, so this proves the
    av backend is contract-equivalent to cv2 on a file both can read: same requested-index keying,
    same EOF-driven clamp to the last decoded frame, same RGB uint8 [size,size,3] shape, and (for a
    flat-color synthetic video) the same decoded pixel values."""
    pytest.importorskip("cv2")
    pytest.importorskip("av")
    path = tmp_path / "synthetic_av.mp4"
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

    idxs = [0, 5, 9, 20]  # 20 is out-of-range -> clamps to last decoded frame (9)
    cv2_frames = lh.read_video_frames(str(path), idxs, size=64, backend="cv2")
    av_frames = lh.read_video_frames(str(path), idxs, size=64, backend="av")

    assert set(av_frames) == set(cv2_frames) == {0, 5, 9, 20}
    for i in idxs:
        assert av_frames[i].shape == (64, 64, 3)
        assert av_frames[i].dtype == np.uint8
    # out-of-range index clamps to the last decoded frame on both backends
    assert np.array_equal(av_frames[20], av_frames[9])
    # flat-color synthetic frames -> both backends should decode near-identical pixel values
    for i in idxs:
        assert np.allclose(av_frames[i].astype(int), cv2_frames[i].astype(int), atol=10)


def test_read_video_frames_unknown_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        lh.read_video_frames("ignored.mp4", [0], backend="bogus")


_GALAXEA_HEAD_MP4 = (
    "/tmp/claude-1000/-home-chungyili-Codes-openpi/3c93bc44-003f-4620-92a3-ab05543c599a/scratchpad/galaxea/"
    "extracted/Turn_On_Off_The_Light_20250619_001/videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
)


def test_read_video_frames_av_backend_decodes_real_galaxea_av1():
    """Optional real-AV1 integration test -- skipped when the scratchpad fixture isn't present
    (not part of CI)."""
    pytest.importorskip("av")
    path = pathlib.Path(_GALAXEA_HEAD_MP4)
    if not path.exists():
        pytest.skip(f"real Galaxea fixture not present at {path}")

    frames = lh.read_video_frames(str(path), [0, 1, 2], size=224, backend="av")

    assert set(frames) == {0, 1, 2}
    for f in frames.values():
        assert f.shape == (224, 224, 3)
        assert f.dtype == np.uint8
    assert not np.array_equal(frames[0], frames[1])


def _write_galaxea_video_root(tmp_path, video_key="observation.images.head_rgb", fps=15.0):
    """A tiny local LeRobot-shaped fixture: meta/info.json + a real small mp4 under
    videos/chunk-000/<video_key>/episode_000000.mp4."""
    root = tmp_path / "video_root"
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True)
    meta_dir.joinpath("info.json").write_text(
        json.dumps({"fps": fps, "chunks_size": 1000, "video_path": lh.VIDEO_PATH_TEMPLATE})
    )
    video_dir = root / "videos" / "chunk-000" / video_key
    video_dir.mkdir(parents=True)
    mp4_path = video_dir / "episode_000000.mp4"
    writer = _open_writer(mp4_path, (64, 64))
    if writer is None:
        return None
    try:
        for i in range(10):
            writer.write(np.full((64, 64, 3), i * 20, np.uint8))
    finally:
        writer.release()
    return root


def test_assemble_with_video_root_reads_locally_without_download(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    video_root = _write_galaxea_video_root(tmp_path)
    if video_root is None:
        pytest.skip("cv2 VideoWriter could not open mp4v codec")

    def _boom(*args, **kwargs):
        raise AssertionError("must not download when video_root is given")

    monkeypatch.setattr(lh, "_download_meta", _boom)
    monkeypatch.setattr(lh, "_download_video", _boom)

    records = [_rec(0)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    out_dir = tmp_path / "out"
    rows = lh.assemble(
        records,
        labels,
        out_dir=out_dir,
        video_root=video_root,
        video_key="observation.images.head_rgb",
        decode_backend="av",
        max_episodes=1,
    )

    manifest = (out_dir / "manifest.jsonl").read_text().strip().splitlines()
    assert len(manifest) == len(rows)
    assert len(rows) > 0
    first = json.loads(manifest[0])
    assert first["image"].startswith("frames/")
    assert (out_dir / first["image"]).exists()


def test_assemble_requires_exactly_one_of_repo_or_video_root(tmp_path):
    records = [_rec(0)]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    with pytest.raises(ValueError, match="exactly one"):
        lh.assemble(records, labels, out_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        lh.assemble(records, labels, out_dir=tmp_path, repo="fake/repo", video_root=tmp_path)
