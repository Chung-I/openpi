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
