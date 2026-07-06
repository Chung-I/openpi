"""Download a Galaxea Open-World task archive -> records.json + episodes.json (for MEM label gen).

Galaxea stores per-frame `task_index` (fine subtask) and `coarse_task_index` (goal) scalar columns
in each episode parquet (Shape A); this run-length-encodes `task_index` into ordered subtask
segments with frame spans, and takes the goal from `coarse_task_index`. Then:
    uv run python scripts/generate_memory_labels.py --episodes_file data/galaxea_episodes.json ...
and a Stage-C frame-decode step consume the records exactly like RoboCOIN/RoboMIND.

Usage:
    uv run python scripts/load_galaxea_episodes.py --repo OpenGalaxea/Galaxea-Open-World-Dataset \
        --archive Turn_On_Off_The_Light_20250619_001 \
        --records-out data/galaxea_records.json --episodes-out data/galaxea_episodes.json
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training import galaxea as gx


def main():
    p = argparse.ArgumentParser(description="Load a Galaxea task archive -> records + episodes JSON")
    p.add_argument("--repo", required=True, help="e.g. OpenGalaxea/Galaxea-Open-World-Dataset")
    p.add_argument("--archive", required=True, help="e.g. Turn_On_Off_The_Light_20250619_001")
    p.add_argument("--records-out", required=True)
    p.add_argument("--episodes-out", required=True)
    p.add_argument("--cache-dir", default="data/galaxea_cache")
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    recs = gx.records_from_repo(args.repo, args.archive, args.cache_dir, max_episodes=args.max_episodes)
    pathlib.Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.episodes_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.records_out).write_text(json.dumps([dataclasses.asdict(r) for r in recs], indent=2))
    pathlib.Path(args.episodes_out).write_text(
        json.dumps([dataclasses.asdict(gx.to_episode(r)) for r in recs], indent=2)
    )
    print(f"Wrote {len(recs)} records -> {args.records_out} ; episodes -> {args.episodes_out}")


if __name__ == "__main__":
    main()
