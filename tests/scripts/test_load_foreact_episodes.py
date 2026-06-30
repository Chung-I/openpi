import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "load_foreact_episodes",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "load_foreact_episodes.py",
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_episodes_from_records_all_success():
    records = [
        {"goal": "make rice", "subtasks": ["get rice", "cook rice"]},
        {"goal": "clean", "subtasks": ["wipe"]},
    ]
    eps = mod.episodes_from_records(records)
    assert len(eps) == 2
    assert eps[0].goal == "make rice"
    assert eps[0].subtasks == ["get rice", "cook rice"]
    assert eps[0].success_flags == [True, True]
    assert eps[1].success_flags == [True]
