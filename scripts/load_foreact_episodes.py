"""Load ForeAct (LeRobot v2) subtask annotations into Episode JSON for HL memory-label generation.

ForeAct (`mit-han-lab/ForeActDataset`) is a set of LeRobot task-group sub-datasets. Each
LeRobot "episode" is a single subtask (`meta/episodes.jsonl`: {"episode_index",
"tasks": [str], "length"}), and `meta/tasks.jsonl` (task_index -> task text) is the
group's subtask repertoire, with "Finish."/"Done." as terminal markers. A MEM Episode =
one task-group: goal derived from the group dir name, subtasks = the group's distinct
subtasks (task_index order, case-deduped, terminal markers dropped). All subtasks are
marked success for v1 (ForeAct is teleoperated, no failure labels).

Usage:
    uv run python scripts/load_foreact_episodes.py --output data/foreact_episodes.json \
        --groups 20251102_Pick_Veg 20251104_Stack_Bowls
HF download uses the default HF cache token (see the project's HF_HOME note).
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training.memory_labels import Episode

REPO = "mit-han-lab/ForeActDataset"
_TERMINAL = {"finish", "done"}


def goal_from_group(group: str) -> str:
    """Derive a task goal from a ForeAct task-group dir name, dropping a leading date token.

    e.g. "20251102_Pick_Veg" -> "pick veg"; "Stack_Bowls" -> "stack bowls".
    """
    parts = group.split("_", 1)
    name = parts[1] if parts and parts[0].isdigit() and len(parts) > 1 else group
    return name.replace("_", " ").strip().lower()


def subtasks_from_task_rows(rows: list[dict]) -> list[str]:
    """Ordered, case-deduped subtask texts from LeRobot tasks.jsonl rows (terminal markers dropped)."""
    seen: set[str] = set()
    out: list[str] = []
    for row in sorted(rows, key=lambda r: r["task_index"]):
        task = (row.get("task") or "").strip()
        key = task.lower().rstrip(".")
        if not task or key in _TERMINAL or key in seen:
            continue
        seen.add(key)
        out.append(task)
    return out


def episode_from_task_rows(group: str, rows: list[dict]) -> Episode | None:
    """Build one MEM Episode from a task-group's tasks.jsonl rows (None if no subtasks)."""
    subtasks = subtasks_from_task_rows(rows)
    if not subtasks:
        return None
    return Episode(goal=goal_from_group(group), subtasks=subtasks, success_flags=[True] * len(subtasks))


def _read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def episodes_from_group(group: str, repo: str = REPO) -> list[Episode]:
    """Download a task-group's tasks.jsonl from HF and build its MEM Episode."""
    import huggingface_hub

    path = huggingface_hub.hf_hub_download(repo, repo_type="dataset", filename=f"{group}/meta/tasks.jsonl")
    ep = episode_from_task_rows(group, _read_jsonl(path))
    return [ep] if ep else []


def list_groups(repo: str = REPO) -> list[str]:
    """List the task-group dirs in the ForeAct repo (those with a meta/tasks.jsonl)."""
    import huggingface_hub

    files = huggingface_hub.list_repo_files(repo, repo_type="dataset")
    return sorted({f.split("/")[0] for f in files if "/meta/tasks.jsonl" in f})


def main():
    parser = argparse.ArgumentParser(description="Load ForeAct (LeRobot) task-groups -> Episode JSON")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--groups", nargs="*", default=None, help="task-group dirs; default = all groups in the repo")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--repo", type=str, default=REPO)
    args = parser.parse_args()

    groups = args.groups or list_groups(args.repo)
    if args.max_groups:
        groups = groups[: args.max_groups]

    episodes: list[Episode] = []
    for g in groups:
        eps = episodes_from_group(g, args.repo)
        if eps:
            print(f"{g}: goal='{eps[0].goal}' subtasks={len(eps[0].subtasks)}")
        episodes += eps

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dataclasses.asdict(e) for e in episodes], indent=2))
    print(f"Wrote {len(episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
