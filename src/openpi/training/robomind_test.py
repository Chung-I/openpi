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
