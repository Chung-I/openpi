import importlib.util
import json
import pathlib

_spec = importlib.util.spec_from_file_location(
    "load_agibot_episodes",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "load_agibot_episodes.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _obj(action_texts, task_name="stack the bowls", init="bowls on table"):
    return {
        "task_name": task_name,
        "init_scene_text": init,
        "label_info": {"action_config": [{"action_text": t} for t in action_texts]},
    }


def test_goal_from_task_info_combines_name_and_scene():
    g = mod.goal_from_task_info(_obj(["a"], task_name="Stack Bowls", init="three bowls"))
    assert "stack bowls" in g.lower()


def test_subtasks_ordered_and_skip_empty():
    subs, flags = mod.subtasks_and_flags_from_task_info(
        _obj(["pick up the pink bowl", "", "stack the cyan bowl", "stack the white bowl"])
    )
    assert subs == ["pick up the pink bowl", "stack the cyan bowl", "stack the white bowl"]
    assert flags == [True, True, True]


def test_failure_keyword_sets_false_flag():
    subs, flags = mod.subtasks_and_flags_from_task_info(
        _obj(["pick up the cup", "failed to grasp, retry pick up the cup", "place the cup"])
    )
    assert flags == [True, False, True]
    assert len(subs) == 3


def test_episode_from_task_info_filters_short():
    assert mod.episode_from_task_info(_obj(["a", "b"]), min_subtasks=3) is None
    ep = mod.episode_from_task_info(_obj(["a", "b", "c"]), min_subtasks=3)
    assert ep is not None and len(ep.subtasks) == 3 and ep.success_flags == [True, True, True]  # noqa: PT018


def test_episodes_from_fixture():
    fx = pathlib.Path(__file__).resolve().parent / "fixtures" / "agibot_task_info_sample.json"
    objs = json.loads(fx.read_text())
    eps = mod.episodes_from_task_info_list(objs, min_subtasks=1)
    assert all(len(e.subtasks) == len(e.success_flags) for e in eps)
    assert all(isinstance(e.goal, str) and e.goal for e in eps)
