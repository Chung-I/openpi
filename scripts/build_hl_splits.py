"""Build splits/fr3_splits.json from the fr3 manifest (episode-based, task-level).

Usage:
    uv run python scripts/build_hl_splits.py \
        --manifest data/robomind_hl_fr3/manifest.jsonl \
        --out splits/fr3_splits.json \
        --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40
"""

import argparse
import collections
import json
import pathlib

from openpi.training import hl_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="splits/fr3_splits.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-dev-unseen-tasks", type=int, default=9)
    p.add_argument("--n-dev-seen-episodes", type=int, default=40)
    args = p.parse_args()

    episode_ids = [
        json.loads(line)["episode_id"]
        for line in pathlib.Path(args.manifest).read_text().splitlines()
        if line.strip()
    ]
    splits = hl_splits.build_splits(
        episode_ids,
        seed=args.seed,
        n_dev_unseen_tasks=args.n_dev_unseen_tasks,
        n_dev_seen_episodes=args.n_dev_seen_episodes,
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hl_splits.save_splits(out, splits)

    # Human-verifiable summary of the task canonicalization + split sizes.
    tasks = collections.Counter(hl_splits.canonical_task(e) for e in episode_ids)
    print(f"total episodes={len(set(episode_ids))} canonical_tasks={len(tasks)}")
    print(f"dev_unseen_tasks ({len(splits['dev_unseen_tasks'])}): {splits['dev_unseen_tasks']}")
    print(f"dev_seen_episodes={len(splits['dev_seen_episodes'])} train_episodes={len(splits['train_episodes'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
