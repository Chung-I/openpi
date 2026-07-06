"""Download a RoboCOIN (LeRobot v2.1) task repo -> records.json + episodes.json (for MEM label gen).

RoboCOIN stores per-frame subtask indices in each episode parquet's `subtask_annotation` column
(Shape A); this run-length-encodes them into ordered subtask segments with frame spans. Then:
    uv run python scripts/generate_memory_labels.py --episodes_file data/robocoin_episodes.json ...
and a LeRobot-mp4 assemble step (Stage C) consume the records exactly like RoboMIND.

Usage:
    uv run python scripts/load_robocoin_episodes.py --repo RoboCOIN/Agilex_Cobot_Magic_heat_burger \
        --records-out data/robocoin_records.json --episodes-out data/robocoin_episodes.json
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training import robocoin as rc


def main():
    p = argparse.ArgumentParser(description="Load a RoboCOIN task repo -> records + episodes JSON")
    p.add_argument("--repo", required=True, help="e.g. RoboCOIN/Agilex_Cobot_Magic_heat_burger")
    p.add_argument("--records-out", required=True)
    p.add_argument("--episodes-out", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    recs = rc.records_from_repo(args.repo, max_episodes=args.max_episodes)
    pathlib.Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.episodes_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.records_out).write_text(json.dumps([dataclasses.asdict(r) for r in recs], indent=2))
    pathlib.Path(args.episodes_out).write_text(
        json.dumps([dataclasses.asdict(rc.to_episode(r)) for r in recs], indent=2)
    )
    print(f"Wrote {len(recs)} records -> {args.records_out} ; episodes -> {args.episodes_out}")


if __name__ == "__main__":
    main()
