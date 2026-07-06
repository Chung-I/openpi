import pytest

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
    with pytest.raises(ValueError, match="memories"):
        rm.build_samples(_rec(), ["m1", "m2"], fps=10)


def _long_rec():
    return {"id": "ep1", "goal": "g", "subtasks": ["a"], "frame_ranges": [[0, 100]], "success_flags": [True]}


def test_cap_reduces_long_subtask_to_evenly_spaced_within_samples():
    s = rm.build_samples(_long_rec(), ["m1"], fps=10, sample_hz=1, max_samples_per_subtask=3)
    w = [x for x in s if not x["update"]]
    b = [x for x in s if x["update"]]
    assert [x["frame"] for x in w] == [0, 40, 90]  # evenly spread, retains first + last within-frame
    assert len(b) == 1


def test_cap_leaves_under_cap_subtask_unchanged():
    rec = {"id": "ep1", "goal": "g", "subtasks": ["a"], "frame_ranges": [[0, 20]], "success_flags": [True]}
    s = rm.build_samples(rec, ["m1"], fps=10, sample_hz=1, max_samples_per_subtask=5)
    w = [x for x in s if not x["update"]]
    b = [x for x in s if x["update"]]
    assert [x["frame"] for x in w] == [0, 10]
    assert len(b) == 1


def test_default_none_identical_to_uncapped():
    rec = _long_rec()
    uncapped = rm.build_samples(rec, ["m1"], fps=10, sample_hz=1)
    default = rm.build_samples(rec, ["m1"], fps=10, sample_hz=1, max_samples_per_subtask=None)
    assert default == uncapped


def test_cap_does_not_touch_boundary_sample():
    s = rm.build_samples(_long_rec(), ["m1"], fps=10, sample_hz=1, max_samples_per_subtask=3)
    b = next(x for x in s if x["update"])
    assert b["frame"] == 100
    assert b["update"] is True
    assert b["target_subtask"] == "done"
    assert b["target_memory"] == "m1"


def test_cap_applied_per_subtask_independently():
    rec = {
        "id": "ep1", "goal": "g", "subtasks": ["a", "b"],
        "frame_ranges": [[0, 100], [100, 120]], "success_flags": [True, True],
    }
    s = rm.build_samples(rec, ["m1", "m2"], fps=10, sample_hz=1, max_samples_per_subtask=3)
    w1 = [x["frame"] for x in s if x["subtask_index"] == 1 and not x["update"]]
    w2 = [x["frame"] for x in s if x["subtask_index"] == 2 and not x["update"]]
    assert w1 == [0, 40, 90]  # capped to 3
    assert w2 == [100, 110]  # under cap (2 within-samples), unchanged


def test_task_of():
    assert rm._task_of("h5_franka_1rgb/bread_in_basket/success_episodes/train/1016_161244/data") == "bread_in_basket"  # noqa: SLF001


def test_decode_resize_raw_array():
    import numpy as np
    out = rm._decode_resize(np.zeros((10, 12, 3), np.uint8), size=224)  # noqa: SLF001
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def _write_traj_h5(path, n_frames=5, *, fps=None):
    """Build a fixture matching real RoboMIND h5_franka_1rgb layout: observations/rgb_images/camera_top
    holds one flat uint8 720x1280x3 RGB buffer per frame (variable-length object dtype)."""
    import h5py
    import numpy as np
    with h5py.File(path, "w") as f:
        g = f.create_group("observations/rgb_images")
        ds = g.create_dataset("camera_top", shape=(n_frames,), dtype=h5py.vlen_dtype(np.uint8))
        for i in range(n_frames):
            ds[i] = np.full(rm._RAW_H * rm._RAW_W * 3, i % 256, np.uint8)  # noqa: SLF001
        if fps is not None:
            f.attrs["fps"] = fps


def test_read_fps_and_frames_from_h5(tmp_path):
    import numpy as np
    p = tmp_path / "trajectory.hdf5"
    _write_traj_h5(p, n_frames=5, fps=10.0)
    assert rm.read_fps(str(p), default=3.0) == 10.0
    frames = rm.read_camera_top_frames(str(p), [0, 4, 99], size=224)  # 99 clamps to last
    assert set(frames) == {0, 4, 99}
    assert frames[0].shape == (224, 224, 3)
    assert frames[0].dtype == np.uint8


def test_read_fps_default_when_missing(tmp_path):
    p = tmp_path / "t.hdf5"
    _write_traj_h5(p, n_frames=2, fps=None)
    assert rm.read_fps(str(p), default=7.5) == 7.5


def test_decode_resize_raw_flat():
    import numpy as np
    flat = np.full(rm._RAW_H * rm._RAW_W * 3, 200, np.uint8)  # noqa: SLF001 -- real RoboMIND 720x1280 layout
    out = rm._decode_resize(flat, size=224)  # noqa: SLF001
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8
    # letterbox (aspect-preserving + pad): content in the center, black pad at top/bottom
    assert out[112, 112].tolist() == [200, 200, 200]
    assert out[0, 0].tolist() == [0, 0, 0]


def test_decode_resize_raw_flat_640x480():
    import numpy as np
    # some franka_3rgb tasks store camera_top as flat 480x640x3 raw RGB (921600 bytes), not JPEG
    flat = np.full(480 * 640 * 3, 150, np.uint8)
    out = rm._decode_resize(flat, size=224)  # noqa: SLF001
    assert out.shape == (224, 224, 3)
    assert out[112, 112].tolist() == [150, 150, 150]  # center = content
    assert out[0, 0].tolist() == [0, 0, 0]  # corner = pad


def test_decode_resize_raises_on_bad_jpeg():
    import numpy as np
    # JPEG magic (FF D8) but corrupt payload -> imdecode returns None
    with pytest.raises(ValueError, match="cv2.imdecode failed"):
        rm._decode_resize(np.array([0xFF, 0xD8, 0, 0, 0], dtype=np.uint8))  # noqa: SLF001


def test_decode_resize_raises_on_unrecognized_1d():
    import numpy as np
    # no JPEG magic, wrong size for a raw frame -> explicit error, not a silent misread
    with pytest.raises(ValueError, match="unrecognized"):
        rm._decode_resize(np.array([0, 1, 2, 3, 4], dtype=np.uint8))  # noqa: SLF001


def test_assemble_raises_on_misaligned_labels(tmp_path):
    import pytest

    records = [{"id": "h5_franka_1rgb/t/success_episodes/train/1/data", "goal": "g",
                "subtasks": ["a"], "frame_ranges": [[0, 5]], "success_flags": [True]}]
    labels = [{"episode_id": "7", "memories": ["m1"]}]  # episode_id != str(0)
    with pytest.raises(ValueError, match="misaligned"):
        rm.assemble(records, labels, out_dir=tmp_path, cache_dir=tmp_path / "c")


def test_assemble_raises_when_labels_shorter_than_records(tmp_path):
    import pytest

    records = [{"id": f"h5_franka_1rgb/t/success_episodes/train/{k}/data", "goal": "g",
                "subtasks": ["a"], "frame_ranges": [[0, 5]], "success_flags": [True]} for k in range(2)]
    labels = [{"episode_id": "0", "memories": ["m1"]}]  # only 1 label for 2 records
    with pytest.raises(ValueError, match="shorter"):
        rm.assemble(records, labels, out_dir=tmp_path, cache_dir=tmp_path / "c")


def test_assemble_writes_manifest_and_frames(tmp_path, monkeypatch):
    import numpy as np

    records = [{"id": "h5_franka_1rgb/t/success_episodes/train/1/data", "goal": "g",
                "subtasks": ["a", "b"], "frame_ranges": [[0, 10], [10, 20]], "success_flags": [True, True]}]
    labels = [{"episode_id": "0", "memories": ["m1", "m2"]}]
    monkeypatch.setattr(rm, "fetch_task_frames_h5",
                        lambda repo, task, rids, cache, **kw: {r.split("/")[-2]: "fake.hdf5" for r in rids})
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


def test_assemble_cleans_cache_per_task_not_per_episode(tmp_path, monkeypatch):
    import shutil

    import numpy as np

    # two episodes share task "t1", one is "t2"; interleaved on purpose so only task-grouping (not
    # input order) yields per-task cleanup. rmtree must fire once per distinct task, not per episode,
    # or same-task episodes would re-extract the ~150GB archive each time.
    records = [
        {"id": "h5_franka_1rgb/t1/success_episodes/train/1/data", "goal": "g",
         "subtasks": ["a"], "frame_ranges": [[0, 5]], "success_flags": [True]},
        {"id": "h5_franka_1rgb/t2/success_episodes/train/2/data", "goal": "g",
         "subtasks": ["a"], "frame_ranges": [[0, 5]], "success_flags": [True]},
        {"id": "h5_franka_1rgb/t1/success_episodes/train/3/data", "goal": "g",
         "subtasks": ["a"], "frame_ranges": [[0, 5]], "success_flags": [True]},
    ]
    labels = [{"episode_id": str(i), "memories": ["m1"]} for i in range(3)]
    removed = []
    monkeypatch.setattr(shutil, "rmtree", lambda p, **kw: removed.append(str(p)))
    monkeypatch.setattr(rm, "fetch_task_frames_h5",
                        lambda repo, task, rids, cache, **kw: {r.split("/")[-2]: f"fake_{task}.hdf5" for r in rids})
    monkeypatch.setattr(rm, "read_fps", lambda h5, default: 10.0)
    monkeypatch.setattr(rm, "read_camera_top_frames",
                        lambda h5, idxs, size=224: {i: np.zeros((size, size, 3), np.uint8) for i in idxs})
    rm.assemble(records, labels, out_dir=tmp_path, cache_dir=tmp_path / "cache")
    # exactly two cleanups (one per distinct task), each ending in the task name -- not three (per episode)
    assert len(removed) == 2
    assert removed[0].endswith("t1")
    assert removed[1].endswith("t2")


def test_fetch_task_frames_h5_extracts_only_wanted(tmp_path):
    import io
    import pathlib
    import tarfile

    # build a real tar.gz holding two episodes' trajectory.hdf5, split into 2 byte-parts like RoboMIND
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tf:
        for ts in ("111", "222"):
            data = f"h5-{ts}".encode()
            info = tarfile.TarInfo(f"mytask/success_episodes/train/{ts}/data/trajectory.hdf5")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    payload = raw.getvalue()
    mid = len(payload) // 2
    p0 = tmp_path / "mytask.tar.gz.part-aa"
    p0.write_bytes(payload[:mid])
    p1 = tmp_path / "mytask.tar.gz.part-ab"
    p1.write_bytes(payload[mid:])
    cache = tmp_path / "cache"
    rid1 = "emb/mytask/success_episodes/train/111/data"
    rid2 = "emb/mytask/success_episodes/train/222/data"

    # only "111" requested -> only its hdf5 is extracted, "222" is skipped
    out = rm.fetch_task_frames_h5("repo", "mytask", [rid1], cache, part_paths=[str(p0), str(p1)])
    assert set(out) == {"111"}
    assert pathlib.Path(out["111"]).read_bytes() == b"h5-111"
    import glob as _glob
    assert not _glob.glob(f"{cache}/**/222/data/trajectory.hdf5", recursive=True)

    # a later call for "222" extracts it; the already-present "111" is reused without re-streaming
    out2 = rm.fetch_task_frames_h5("repo", "mytask", [rid1, rid2], cache,
                                   part_paths=[str(p0), str(p1)], cleanup_parts=True)
    assert set(out2) == {"111", "222"}
    assert pathlib.Path(out2["222"]).read_bytes() == b"h5-222"
    # caller-supplied part_paths are never deleted, even with cleanup_parts=True
    assert p0.exists()
    assert p1.exists()


def test_member_of():
    rid = "h5_franka_1rgb/bread_in_basket/success_episodes/train/1016_161244/data"
    assert rm._member_of(rid) == "bread_in_basket/success_episodes/train/1016_161244/data/trajectory.hdf5"  # noqa: SLF001


def test_download_task_parts_preserves_order(monkeypatch):
    import huggingface_hub
    # unsorted repo listing across embodiments; only mytask's parts, returned in part order
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo, repo_type: [
        "b/h5x/mytask.tar.gz.part-ab", "b/h5x/other.tar.gz.part-aa", "b/h5x/mytask.tar.gz.part-aa",
        "b/h5x/mytask.tar.gz.part-ac",
    ])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, repo_type, filename: f"/cache/{filename}")
    out = rm._download_task_parts("repo", "mytask", max_workers=4)  # noqa: SLF001
    assert out == ["/cache/b/h5x/mytask.tar.gz.part-aa", "/cache/b/h5x/mytask.tar.gz.part-ab",
                   "/cache/b/h5x/mytask.tar.gz.part-ac"]


def test_delete_part_blobs(tmp_path):
    # HF cache layout: a snapshot symlink pointing at a content-addressed blob; both must be removed
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "deadbeef"
    blob.write_bytes(b"x" * 1000)
    link = tmp_path / "task.tar.gz.part-aa"
    link.symlink_to(blob)
    rm._delete_part_blobs([str(link)])  # noqa: SLF001
    assert not link.exists()
    assert not blob.exists()
