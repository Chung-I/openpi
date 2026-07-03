"""Episode-based, task-level train/dev split builder for HL training.

A RoboMIND manifest row carries an ``episode_id`` shaped
``.../<date>_<task>_<run>/...``. We group by canonical task, hold out whole tasks
(``dev_unseen``) and held-out episodes of the remaining tasks (``dev_seen``), and
keep the rest for training. Splits are grouped by episode (never by subtask row) so
no episode straddles two slices.
"""

import json
import pathlib
import random


def canonical_task(episode_id: str) -> str:
    """Map an episode_id to its canonical task name.

    Strips the leading ``<date>`` token and a trailing ``_<run>`` integer from the
    task segment (the second ``/``-separated component).
    """
    seg = episode_id.split("/")[1]
    parts = seg.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]           # drop date
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]          # drop run index
    return "_".join(parts)


def build_splits(
    episode_ids: list[str],
    *,
    seed: int,
    n_dev_unseen_tasks: int,
    n_dev_seen_episodes: int,
) -> dict:
    """Build a task-level, episode-grouped split. See module docstring."""
    rng = random.Random(seed)

    tasks_to_eps: dict[str, list[str]] = {}
    for e in sorted(set(episode_ids)):
        tasks_to_eps.setdefault(canonical_task(e), []).append(e)

    all_tasks = sorted(tasks_to_eps)
    if n_dev_unseen_tasks >= len(all_tasks):
        raise ValueError(f"n_dev_unseen_tasks={n_dev_unseen_tasks} >= #tasks={len(all_tasks)}")
    dev_unseen_tasks = sorted(rng.sample(all_tasks, n_dev_unseen_tasks))

    trained_tasks = [t for t in all_tasks if t not in set(dev_unseen_tasks)]
    trained_eps = [e for t in trained_tasks for e in tasks_to_eps[t]]
    trained_eps_sorted = sorted(trained_eps)
    if n_dev_seen_episodes >= len(trained_eps_sorted):
        raise ValueError("n_dev_seen_episodes too large for the trained pool")
    dev_seen = sorted(rng.sample(trained_eps_sorted, n_dev_seen_episodes))
    train = sorted(set(trained_eps_sorted) - set(dev_seen))

    return {
        "seed": seed,
        "dev_unseen_tasks": dev_unseen_tasks,
        "dev_seen_episodes": dev_seen,
        "train_episodes": train,
    }


def save_splits(path, splits: dict) -> None:
    pathlib.Path(path).write_text(json.dumps(splits, indent=2))


def load_splits(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())
