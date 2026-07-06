import glob
import json
import pathlib
import tarfile

import pytest

from openpi.training import agibot as ab

REPO = "agibot-world/AgiBotWorld-Alpha"


def _entry(episode_id=685046, task_name="Pickup items in the supermarket", action_config=None):
    if action_config is None:
        action_config = [
            {
                "start_frame": 0,
                "end_frame": 323,
                "action_text": "Retrieve shiitake mushroom from the shelf.",
                "skill": "Pick",
            },
            {
                "start_frame": 323,
                "end_frame": 444,
                "action_text": "Place the held shiitake mushroom into the plastic bag in the shopping cart.",
                "skill": "Place",
            },
        ]
    return {
        "episode_id": episode_id,
        "task_name": task_name,
        "init_scene_text": "The robot is positioned in front of the fruit stand in the supermarket environment.",
        "label_info": {"action_config": action_config},
    }


# --- record_from_episode -------------------------------------------------------------------------


def test_record_from_episode_parses_goal_subtasks_ranges():
    r = ab.record_from_episode(REPO, 327, _entry())
    assert r.id == "agibot-world/AgiBotWorld-Alpha/task_327/episode_685046"
    assert r.goal == "Pickup items in the supermarket"
    assert r.subtasks == [
        "Retrieve shiitake mushroom from the shelf.",
        "Place the held shiitake mushroom into the plastic bag in the shopping cart.",
    ]
    assert r.frame_ranges == [(0, 322), (323, 443)]
    assert r.success_flags == [True, True]


def test_record_none_when_action_config_empty():
    entry = _entry(action_config=[])
    assert ab.record_from_episode(REPO, 327, entry) is None


def test_record_none_when_label_info_missing():
    entry = _entry()
    del entry["label_info"]
    assert ab.record_from_episode(REPO, 327, entry) is None


def test_record_from_episode_marks_failure_keyword_subtask_as_unsuccessful():
    entry = _entry(
        action_config=[
            {"start_frame": 0, "end_frame": 100, "action_text": "Pick the apple.", "skill": "Pick"},
            {
                "start_frame": 100,
                "end_frame": 200,
                "action_text": "Grasp failed, attempting recovery of the apple.",
                "skill": "Pick",
            },
            {"start_frame": 200, "end_frame": 300, "action_text": "Place the apple in the cart.", "skill": "Place"},
        ]
    )
    r = ab.record_from_episode(REPO, 327, entry)
    assert r.subtasks == [
        "Pick the apple.",
        "Grasp failed, attempting recovery of the apple.",
        "Place the apple in the cart.",
    ]
    assert r.frame_ranges == [(0, 99), (100, 199), (200, 299)]
    assert r.success_flags == [True, False, True]


def test_record_drops_empty_and_whitespace_action_text_keeps_ranges_aligned():
    entry = _entry(
        action_config=[
            {"start_frame": 0, "end_frame": 100, "action_text": "Pick the apple.", "skill": "Pick"},
            {"start_frame": 100, "end_frame": 150, "action_text": "   ", "skill": "Pick"},  # dropped
            {"start_frame": 150, "end_frame": 300, "action_text": "Place the apple in the cart.", "skill": "Place"},
        ]
    )
    r = ab.record_from_episode(REPO, 327, entry)
    assert r.subtasks == ["Pick the apple.", "Place the apple in the cart."]
    assert r.frame_ranges == [(0, 99), (150, 299)]
    assert r.success_flags == [True, True]


def test_to_episode_view():
    ep = ab.to_episode(ab.record_from_episode(REPO, 327, _entry()))
    assert ep.goal == "Pickup items in the supermarket"
    assert ep.subtasks == [
        "Retrieve shiitake mushroom from the shelf.",
        "Place the held shiitake mushroom into the plastic bag in the shopping cart.",
    ]
    assert ep.success_flags == [True, True]


# --- records_from_task_json (json-array fixture) --------------------------------------------------


def test_records_from_task_json_reads_array(tmp_path):
    path = tmp_path / "task_327.json"
    path.write_text(json.dumps([_entry(episode_id=1), _entry(episode_id=2, task_name="Other task")]))
    recs = ab.records_from_task_json(REPO, 327, path)
    assert len(recs) == 2
    assert recs[0].id == "agibot-world/AgiBotWorld-Alpha/task_327/episode_1"
    assert recs[1].id == "agibot-world/AgiBotWorld-Alpha/task_327/episode_2"
    assert recs[1].goal == "Other task"


def test_records_from_task_json_max_episodes(tmp_path):
    path = tmp_path / "task_327.json"
    path.write_text(json.dumps([_entry(episode_id=i) for i in range(5)]))
    recs = ab.records_from_task_json(REPO, 327, path, max_episodes=2)
    assert len(recs) == 2


def test_records_from_task_json_skips_none_entries(tmp_path):
    path = tmp_path / "task_327.json"
    objs = [_entry(episode_id=1, action_config=[]), _entry(episode_id=2)]
    path.write_text(json.dumps(objs))
    recs = ab.records_from_task_json(REPO, 327, path)
    assert len(recs) == 1
    assert recs[0].id.endswith("episode_2")


# --- optional integration test against a REAL cached task_info file (not in CI) ------------------

_REAL_CANDIDATES = glob.glob(
    str(
        pathlib.Path.home()
        / ".cache/huggingface/hub/datasets--agibot-world--AgiBotWorld-Alpha/snapshots/*/task_info/task_327.json"
    )
)


@pytest.mark.skipif(not _REAL_CANDIDATES, reason="real AgiBot task_327.json not cached locally")
def test_records_from_task_json_real_task_327():
    recs = ab.records_from_task_json(REPO, 327, _REAL_CANDIDATES[0], max_episodes=5)
    assert recs
    for r in recs:
        assert r.goal
        assert r.subtasks
        assert len(r.subtasks) == len(r.frame_ranges) == len(r.success_flags)
        assert all(r.success_flags)
        # exclusive-end action_config spans are contiguous -> inclusive-end frame_ranges are too
        for (_, end), (start, _) in zip(r.frame_ranges, r.frame_ranges[1:], strict=False):
            assert end + 1 == start


# --- extract_task_head_videos ---------------------------------------------------------------------


def _build_task_tar(tar_path, episode_ids=(685046, 685047)):
    """A plain (uncompressed) tar shaped like an AgiBot observations/<task_id>/<shard>.tar: per
    episode, a head_color.mp4 (to extract) plus decoys (a depth PNG + another camera's mp4) that
    must be skipped."""
    with tarfile.open(tar_path, "w") as tf:
        for eid in episode_ids:
            _add_bytes(tf, f"{eid}/videos/head_color.mp4", f"head-video-bytes-{eid}".encode())
            _add_bytes(tf, f"{eid}/depth/0.png", b"depth-png-bytes")
            _add_bytes(tf, f"{eid}/videos/hand_left_color.mp4", b"hand-video-bytes")
    return tar_path


def _add_bytes(tf, name, data):
    import io

    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_extract_task_head_videos_extracts_only_head_cam(tmp_path):
    tar_path = _build_task_tar(tmp_path / "shard.tar")
    dest_dir = tmp_path / "extracted"

    result = ab.extract_task_head_videos(tar_path, dest_dir)

    assert result == dest_dir
    for eid in (685046, 685047):
        head = dest_dir / str(eid) / "videos" / "head_color.mp4"
        assert head.exists()
        assert head.read_bytes() == f"head-video-bytes-{eid}".encode()
        assert not (dest_dir / str(eid) / "depth").exists()
        assert not (dest_dir / str(eid) / "videos" / "hand_left_color.mp4").exists()


def test_extract_task_head_videos_is_idempotent(tmp_path):
    tar_path = _build_task_tar(tmp_path / "shard.tar")
    dest_dir = tmp_path / "extracted"

    ab.extract_task_head_videos(tar_path, dest_dir)
    head = dest_dir / "685046" / "videos" / "head_color.mp4"
    mtime_first = head.stat().st_mtime_ns

    # rerun must not error and must leave the already-extracted file intact
    ab.extract_task_head_videos(tar_path, dest_dir)
    assert head.exists()
    assert head.read_bytes() == b"head-video-bytes-685046"
    assert head.stat().st_mtime_ns == mtime_first


def test_extract_task_head_videos_filters_by_episode_ids(tmp_path):
    tar_path = _build_task_tar(tmp_path / "shard.tar")
    dest_dir = tmp_path / "extracted"

    ab.extract_task_head_videos(tar_path, dest_dir, episode_ids={"685046"})

    assert (dest_dir / "685046" / "videos" / "head_color.mp4").exists()
    assert not (dest_dir / "685047").exists()
