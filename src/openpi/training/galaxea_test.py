import json
import pathlib

import pytest

from openpi.training import galaxea as gx

# Minimal stand-in for a Galaxea meta/tasks.jsonl vocab (task_index -> bilingual "中文@English" text;
# some rows -- sentinels and one coarse goal -- are English-only, matching the real archives).
GALAXEA_VOCAB = {
    0: "左手把房间的灯打开@Turn on the room light with your left hand.",
    1: "左手把房间的灯关闭@Turn off the room light with your left hand.",
    2: "null",
    3: "qualified",
    4: "unqualified",
    5: "turn on off the light",
    6: "把箱子放好@Put the box away.",
}


def _expand(pairs):
    """(value, count) runs -> flat per-frame list."""
    out = []
    for v, c in pairs:
        out += [v] * c
    return out


# --- english() ---------------------------------------------------------------------------------


def test_english_bilingual_takes_english_side():
    assert (
        gx.english("左手把房间的灯打开@Turn on the room light with your left hand.")
        == "Turn on the room light with your left hand."
    )


def test_english_english_only_unchanged():
    assert gx.english("turn on off the light") == "turn on off the light"


def test_english_strips_whitespace():
    assert gx.english("中文@  Turn on the light. ") == "Turn on the light."


# --- record_from_episode -------------------------------------------------------------------------


def test_record_from_episode_rle_and_english_extraction():
    task_index = _expand([(0, 69), (1, 63)])
    coarse = [5] * len(task_index)
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.id == "ep0"
    assert r.subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
    ]
    assert r.frame_ranges == [(0, 68), (69, 131)]
    assert r.goal == "turn on off the light"
    assert r.success_flags == [True, True]


def test_record_drops_null_sentinel_midsequence():
    task_index = _expand([(0, 10), (2, 5), (1, 8)])  # null run in the middle dropped
    coarse = [5] * len(task_index)
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
    ]
    assert r.frame_ranges == [(0, 9), (15, 22)]


def test_record_drops_qualified_and_unqualified_sentinels():
    task_index = _expand([(0, 10), (3, 4), (4, 4)])
    coarse = [5] * len(task_index)
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.subtasks == ["Turn on the room light with your left hand."]


def test_record_drops_empty_text_from_bilingual_vocab():
    """Regression: vocab "中文@" (nothing after @) makes english() return "", which should be skipped."""
    vocab_with_empty = GALAXEA_VOCAB.copy()
    vocab_with_empty[7] = "中文@"  # empty English side
    task_index = _expand([(0, 5), (7, 3), (1, 5)])
    coarse = [5] * len(task_index)
    r = gx.record_from_episode("ep0", task_index, coarse, vocab_with_empty)
    # The empty-text segment (value 7, frames 5-7) should be dropped
    assert r.subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
    ]
    # Frame ranges should skip over the empty-text segment
    assert r.frame_ranges == [(0, 4), (8, 12)]


def test_record_none_when_all_task_index_null():
    task_index = _expand([(2, 5), (2, 3)])
    coarse = [5] * len(task_index)
    assert gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB) is None


def test_record_recurring_subtask_kept_as_ordered_events():
    task_index = _expand([(0, 5), (1, 5), (0, 5)])
    coarse = [5] * len(task_index)
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
        "Turn on the room light with your left hand.",
    ]


def test_goal_from_constant_coarse_task_index():
    task_index = _expand([(0, 5)])
    coarse = [5] * 5
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.goal == "turn on off the light"


def test_goal_most_common_wins_over_stray_value():
    task_index = _expand([(0, 5)])
    coarse = [5, 5, 5, 5, 6]  # one stray frame reading the other coarse task
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.goal == "turn on off the light"


def test_goal_empty_when_all_coarse_sentinel():
    task_index = _expand([(0, 5)])
    coarse = [3] * 5  # "qualified" sentinel only
    r = gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB)
    assert r.goal == ""


def test_to_episode_view():
    task_index = _expand([(0, 5), (1, 5)])
    coarse = [5] * len(task_index)
    ep = gx.to_episode(gx.record_from_episode("ep0", task_index, coarse, GALAXEA_VOCAB))
    assert ep.goal == "turn on off the light"
    assert ep.subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
    ]
    assert ep.success_flags == [True, True]


# --- records_from_local (pyarrow fixture) ----------------------------------------------------


def _write_galaxea_fixture(root, archive_name, episodes):
    """Build a minimal extracted Galaxea/LeRobot v2.1 archive dir: meta + one parquet per episode.

    `episodes` = list of (task_runs: list[(value, count)], coarse_values: list[int] per-frame).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    archive_root = root / archive_name
    (archive_root / "meta").mkdir(parents=True)
    (archive_root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "fps": 15,
            }
        )
    )
    (archive_root / "meta" / "tasks.jsonl").write_text(
        "".join(f'{{"task_index": {i}, "task": {json.dumps(t)}}}\n' for i, t in GALAXEA_VOCAB.items())
    )
    ep_lines = []
    for idx, (task_runs, coarse_values) in enumerate(episodes):
        task_index = _expand(task_runs)
        assert len(coarse_values) == len(task_index)
        (archive_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array(task_index, type=pa.int64()),
                    "coarse_task_index": pa.array(coarse_values, type=pa.int64()),
                }
            ),
            archive_root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet",
        )
        ep_lines.append(json.dumps({"episode_index": idx, "length": len(task_index)}))
    (archive_root / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")


def test_records_from_local_reads_parquet_and_goal(tmp_path):
    _write_galaxea_fixture(
        tmp_path,
        "TestArchive",
        [
            ([(0, 69), (1, 63)], [5] * 132),
            ([(0, 10), (1, 10)], [6] * 20),
        ],
    )
    recs = gx.records_from_local(tmp_path, "TestArchive")
    assert len(recs) == 2
    assert recs[0].id == "TestArchive/episode_000000"
    assert recs[0].goal == "turn on off the light"
    assert recs[0].subtasks == [
        "Turn on the room light with your left hand.",
        "Turn off the room light with your left hand.",
    ]
    assert recs[0].frame_ranges == [(0, 68), (69, 131)]
    assert recs[1].id == "TestArchive/episode_000001"
    assert recs[1].goal == "Put the box away."


def test_records_from_local_max_episodes(tmp_path):
    _write_galaxea_fixture(
        tmp_path,
        "TestArchive",
        [
            ([(0, 3)], [5] * 3),
            ([(1, 3)], [5] * 3),
            ([(0, 3)], [5] * 3),
        ],
    )
    assert len(gx.records_from_local(tmp_path, "TestArchive", max_episodes=2)) == 2


def test_records_from_local_skips_all_sentinel_episode(tmp_path):
    _write_galaxea_fixture(
        tmp_path,
        "TestArchive",
        [
            ([(2, 5)], [5] * 5),  # all null -> dropped
            ([(1, 3)], [5] * 3),
        ],
    )
    recs = gx.records_from_local(tmp_path, "TestArchive")
    assert len(recs) == 1
    assert recs[0].id.endswith("episode_000001")


# --- optional integration test against a REAL extracted archive (not in CI) -------------------

_REAL_ROOT = pathlib.Path(
    "/tmp/claude-1000/-home-chungyili-Codes-openpi/3c93bc44-003f-4620-92a3-ab05543c599a/scratchpad/galaxea/extracted"
)


@pytest.mark.skipif(not _REAL_ROOT.exists(), reason="real Galaxea archive not present outside scratchpad")
def test_records_from_local_real_turn_on_off_the_light():
    recs = gx.records_from_local(_REAL_ROOT, "Turn_On_Off_The_Light_20250619_001", max_episodes=1)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.id == "Turn_On_Off_The_Light_20250619_001/episode_000000"
    assert len(rec.subtasks) == 2
    assert rec.subtasks[0].lower().startswith("turn on the room light")
    assert rec.subtasks[1].lower().startswith("turn off the room light")
    assert rec.goal.lower() == "turn on off the light"
