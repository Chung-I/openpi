"""Load AgiBot World (Alpha) task_info annotations into Episode JSON for HL memory-label generation.

AgiBot World stores per-task annotations in task_info/task_<id>.json, a list of per-episode
dicts. Each dict has task_name, init_scene_text, and label_info.action_config — a time-ordered
list of {action_text, start_frame, end_frame, ...}. The ordered sub-steps are the non-empty
action_text values. Success is inferred from action_text keywords (failed/recovery/retry).
Only text annotations are downloaded (no video). All success_flags default True (teleop).

Usage:
    uv run python scripts/load_agibot_episodes.py --output data/agibot_episodes.json \
        --max-episodes 1000 --min-subtasks 3
HF download uses the default HF cache token (see the project's HF_HOME note).
"""
import argparse
import dataclasses
import glob
import json
import pathlib

from openpi.training.memory_labels import Episode

REPO = "agibot-world/AgiBotWorld-Alpha"
_FAILURE_KEYWORDS = ("failed", "recovery", "retry", "mistake")


def goal_from_task_info(obj: dict) -> str:
    name = (obj.get("task_name") or "").strip()
    scene = (obj.get("init_scene_text") or "").strip()
    parts = [p for p in (name, scene) if p]
    return ". ".join(parts).lower()


def subtasks_and_flags_from_task_info(obj: dict) -> tuple[list[str], list[bool]]:
    actions = (obj.get("label_info") or {}).get("action_config") or []
    subtasks: list[str] = []
    flags: list[bool] = []
    for a in actions:
        text = (a.get("action_text") or "").strip()
        if not text:
            continue
        subtasks.append(text)
        flags.append(not any(k in text.lower() for k in _FAILURE_KEYWORDS))
    return subtasks, flags


def episode_from_task_info(obj: dict, min_subtasks: int = 3) -> Episode | None:
    subtasks, flags = subtasks_and_flags_from_task_info(obj)
    if len(subtasks) < min_subtasks:
        return None
    return Episode(goal=goal_from_task_info(obj), subtasks=subtasks, success_flags=flags)


def episodes_from_task_info_list(objs: list[dict], min_subtasks: int = 3) -> list[Episode]:
    out = []
    for obj in objs:
        ep = episode_from_task_info(obj, min_subtasks)
        if ep is not None:
            out.append(ep)
    return out


def download_task_info(repo: str = REPO, out_dir: str | None = None) -> list[pathlib.Path]:
    import huggingface_hub

    root = huggingface_hub.snapshot_download(
        repo, repo_type="dataset", allow_patterns="task_info/*.json", local_dir=out_dir, max_workers=4
    )
    return [pathlib.Path(p) for p in sorted(glob.glob(f"{root}/task_info/*.json"))]


def main():
    p = argparse.ArgumentParser(description="Load AgiBot World task_info -> Episode JSON")
    p.add_argument("--output", required=True)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--min-subtasks", type=int, default=3)
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()

    episodes: list[Episode] = []
    for path in download_task_info(args.repo):
        objs = json.loads(path.read_text())
        episodes += episodes_from_task_info_list(objs, args.min_subtasks)
        if len(episodes) >= args.max_episodes:
            break
    episodes = episodes[: args.max_episodes]

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dataclasses.asdict(e) for e in episodes], indent=2))
    print(f"Wrote {len(episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
