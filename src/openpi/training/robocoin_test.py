import json

import numpy as np
import pytest

from openpi.training import robocoin as rc

# Minimal stand-in for annotations/subtask_annotations.jsonl (subtask_index -> text).
VOCAB = {
    0: "Abnormal",
    1: "grab plate",
    2: "place plate",
    3: "press switch",
    4: "place in microwave",
    5: "End",
    6: "twist timer",
    7: "close door",
    8: "null",
}


def _expand(pairs):
    """(label, count) runs -> flat per-frame primary-label array."""
    out = []
    for v, c in pairs:
        out += [v] * c
    return np.array(out, dtype=np.int32)


def test_rle_segments_basic():
    assert rc.rle_segments([3, 3, 1, 1, 1, 4]) == [(3, 0, 2), (1, 2, 5), (4, 5, 6)]


def test_rle_segments_empty():
    assert rc.rle_segments([]) == []


def test_rle_segments_recurring_label_kept_separate():
    # a subtask that recurs later is a distinct segment, not merged
    assert rc.rle_segments([3, 1, 3]) == [(3, 0, 1), (1, 1, 2), (3, 2, 3)]


def test_primary_labels_takes_first_column():
    arr = np.array([[3, 8, 8, 8, 8], [1, 8, 8, 8, 8]], np.int32)  # 5-wide multi-label vector
    assert rc.primary_labels(arr).tolist() == [3, 1]


def test_record_maps_vocab_and_builds_inclusive_ranges():
    labels = _expand([(3, 10), (1, 5), (4, 20)])  # runs [0,10) [10,15) [15,35)
    r = rc.record_from_episode("ep0", labels, goal="Heat the burger", vocab=VOCAB)
    assert r.id == "ep0"
    assert r.goal == "Heat the burger"
    assert r.subtasks == ["press switch", "grab plate", "place in microwave"]
    # frame_ranges use the run's last active frame (end_exclusive - 1) as the boundary
    assert r.frame_ranges == [(0, 9), (10, 14), (15, 34)]
    assert r.success_flags == [True, True, True]


def test_record_drops_trailing_end_and_null_sentinels():
    labels = _expand([(3, 10), (1, 5), (5, 4), (8, 6)])  # ... End, null trail
    r = rc.record_from_episode("ep0", labels, "g", VOCAB)
    assert r.subtasks == ["press switch", "grab plate"]
    assert r.frame_ranges == [(0, 9), (10, 14)]


def test_record_drops_abnormal_midsequence():
    labels = _expand([(3, 10), (0, 3), (1, 5)])  # Abnormal [10,13) dropped, neighbours kept
    r = rc.record_from_episode("ep0", labels, "g", VOCAB)
    assert r.subtasks == ["press switch", "grab plate"]
    assert r.frame_ranges == [(0, 9), (13, 17)]


def test_record_none_when_all_sentinel():
    assert rc.record_from_episode("ep0", _expand([(5, 3), (8, 4)]), "g", VOCAB) is None


def test_record_recurring_subtask_kept_as_ordered_events():
    r = rc.record_from_episode("ep0", _expand([(3, 5), (1, 5), (3, 5)]), "g", VOCAB)
    assert r.subtasks == ["press switch", "grab plate", "press switch"]


def test_record_accepts_multilabel_vector():
    arr = np.stack([_expand([(3, 4), (1, 3)]), _expand([(8, 4), (8, 3)])], axis=1)  # (7, 2) primary=col0
    r = rc.record_from_episode("ep0", arr, "g", VOCAB)
    assert r.subtasks == ["press switch", "grab plate"]


def test_to_episode_view():
    ep = rc.to_episode(rc.record_from_episode("ep0", _expand([(3, 10), (1, 5)]), "g", VOCAB))
    assert ep.goal == "g"
    assert ep.subtasks == ["press switch", "grab plate"]
    assert ep.success_flags == [True, True]


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        rc.record_from_episode("ep0", _expand([(99, 3)]), "g", VOCAB)


def test_load_vocab(tmp_path):
    p = tmp_path / "subtask_annotations.jsonl"
    p.write_text('{"subtask_index": 0, "subtask": "Abnormal"}\n{"subtask_index": 1, "subtask": "grab plate"}\n')
    assert rc.load_vocab(p) == {0: "Abnormal", 1: "grab plate"}


def _write_lerobot_fixture(root, episodes):
    """Build a minimal RoboCOIN/LeRobot v2.1 dir: meta + annotations + one parquet per episode.

    `episodes` = list of (tasks: list[str], runs: list[(label, count)]).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {"chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
        )
    )
    (root / "annotations" / "subtask_annotations.jsonl").write_text(
        "".join(f'{{"subtask_index": {i}, "subtask": {json.dumps(t)}}}\n' for i, t in VOCAB.items())
    )
    ep_lines = []
    for idx, (tasks, runs) in enumerate(episodes):
        prim = _expand(runs)
        # store as the real 5-wide list<int32> vector (cols 1-4 filled with the null sentinel index)
        vec = np.stack([prim, *([np.full_like(prim, 8)] * 4)], axis=1)
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table({"subtask_annotation": pa.array(list(vec.tolist()), type=pa.list_(pa.int32()))}),
            root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet",
        )
        ep_lines.append(json.dumps({"episode_index": idx, "tasks": tasks, "length": len(prim)}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")


def test_records_from_local_reads_parquet_and_goal(tmp_path):
    _write_lerobot_fixture(
        tmp_path,
        [
            (["Heat the burger"], [(3, 10), (1, 5), (5, 4), (8, 6)]),  # ...End,null trail dropped
            (["Serve the plate"], [(2, 8), (4, 4)]),
        ],
    )
    recs = rc.records_from_local(tmp_path)
    assert len(recs) == 2
    assert recs[0].id.endswith("episode_000000")
    assert recs[0].goal == "Heat the burger"
    assert recs[0].subtasks == ["press switch", "grab plate"]
    assert recs[0].frame_ranges == [(0, 9), (10, 14)]
    assert recs[1].goal == "Serve the plate"
    assert recs[1].subtasks == ["place plate", "place in microwave"]


def test_records_from_local_max_episodes(tmp_path):
    _write_lerobot_fixture(tmp_path, [(["a"], [(3, 3)]), (["b"], [(1, 3)]), (["c"], [(4, 3)])])
    assert len(rc.records_from_local(tmp_path, max_episodes=2)) == 2


def test_records_from_local_skips_all_sentinel_episode(tmp_path):
    _write_lerobot_fixture(tmp_path, [(["a"], [(5, 3), (8, 4)]), (["b"], [(1, 3)])])  # first is all-sentinel
    recs = rc.records_from_local(tmp_path)
    assert len(recs) == 1
    assert recs[0].goal == "b"


def test_records_from_repo_downloads_scope_only_needed_files(tmp_path, monkeypatch):
    """Regression: records_from_repo must NOT snapshot the whole GB-scale repo.

    Should download only: meta/info.json, annotations/subtask_annotations.jsonl,
    meta/episodes.jsonl, and exactly one parquet per episode up to max_episodes.
    """
    # Build a small fixture with 3 episodes in tmp_path.
    _write_lerobot_fixture(
        tmp_path,
        [
            (["task 0"], [(3, 5), (1, 3)]),
            (["task 1"], [(1, 4)]),
            (["task 2"], [(2, 3)]),
        ],
    )

    # Track every hf_hub_download call.
    downloaded_files = []

    def fake_hf_download(repo: str, repo_type: str, filename: str):
        """Return a pathlib.Path to the local fixture file."""
        downloaded_files.append(filename)
        return tmp_path / filename

    def fake_snapshot_download(*args, **kwargs):
        """Snapshot download should never be called."""
        raise RuntimeError("snapshot_download should not be called by records_from_repo")

    # Monkeypatch the hub functions.
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    # Call records_from_repo with max_episodes=2 (should fetch only first 2 episodes).
    recs = rc.records_from_repo("test_repo", max_episodes=2)

    # Verify we got 2 records (episodes 0 and 1 both have non-sentinel subtasks).
    assert len(recs) == 2
    assert recs[0].id.endswith("episode_000000")
    assert recs[1].id.endswith("episode_000001")

    # Verify the exact set of downloaded files.
    expected_files = {
        "meta/info.json",
        "annotations/subtask_annotations.jsonl",
        "meta/episodes.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "data/chunk-000/episode_000001.parquet",
    }
    assert set(downloaded_files) == expected_files
    assert len(downloaded_files) == 5  # Exactly 5 downloads.
    # Verify episode 2 was NOT downloaded.
    assert "episode_000002.parquet" not in str(downloaded_files)
