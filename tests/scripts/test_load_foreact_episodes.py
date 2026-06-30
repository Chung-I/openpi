import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "load_foreact_episodes",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "load_foreact_episodes.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_goal_from_group_drops_leading_date():
    assert mod.goal_from_group("20251102_Pick_Veg") == "pick veg"
    assert mod.goal_from_group("Stack_Bowls") == "stack bowls"


def test_subtasks_from_task_rows_orders_dedups_drops_terminal():
    rows = [
        {"task_index": 2, "task": "Pick up the corn and place it into the plate."},
        {"task_index": 0, "task": "Pick up the eggplant and place it into the plate."},
        {"task_index": 1, "task": "Finish."},
        {"task_index": 3, "task": "pick up the eggplant and place it into the plate."},  # case dup of idx 0
        {"task_index": 4, "task": "Done"},
    ]
    assert mod.subtasks_from_task_rows(rows) == [
        "Pick up the eggplant and place it into the plate.",  # task_index 0 first
        "Pick up the corn and place it into the plate.",
    ]


def test_episode_from_task_rows_all_success():
    rows = [
        {"task_index": 1, "task": "wipe the counter."},
        {"task_index": 0, "task": "rinse the plate."},
        {"task_index": 2, "task": "Finish."},
    ]
    ep = mod.episode_from_task_rows("20251104_Clean_Kitchen", rows)
    assert ep.goal == "clean kitchen"
    assert ep.subtasks == ["rinse the plate.", "wipe the counter."]
    assert ep.success_flags == [True, True]


def test_episode_from_task_rows_none_when_only_terminal():
    assert mod.episode_from_task_rows("g", [{"task_index": 0, "task": "Finish."}]) is None
