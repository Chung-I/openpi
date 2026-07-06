"""Download one AgiBot World task_info/task_<id>.json -> records.json + episodes.json (for MEM label gen).

AgiBot World ships EXPLICIT per-episode subtask spans in `label_info.action_config` (Shape B, unlike
RoboCOIN/Galaxea's per-frame index RLE): each entry is {start_frame, end_frame (exclusive),
action_text, skill}. This reads the spans directly (no RLE needed) and takes the goal from
`task_name`. Then:
    uv run python scripts/generate_memory_labels.py --episodes_file data/agibot_records_episodes.json ...
and a later Stage-C video-decode step consume the records exactly like RoboCOIN/RoboMIND/Galaxea.

NOTE: this is a DIFFERENT tool than `scripts/load_agibot_episodes.py` (which builds a frozen episode
set for the memory-label prompt bake-off, `Episode`-only, no frame_ranges). This one produces the
full `(id, goal, subtasks, frame_ranges, success_flags)` record shape needed for HL sample assembly.

Usage:
    uv run python scripts/load_agibot_task_episodes.py --repo agibot-world/AgiBotWorld-Alpha \
        --task-id 327 \
        --records-out data/agibot_records.json --episodes-out data/agibot_episodes.json
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training import agibot as ab


def main():
    p = argparse.ArgumentParser(description="Load one AgiBot World task_info file -> records + episodes JSON")
    p.add_argument("--repo", required=True, help="e.g. agibot-world/AgiBotWorld-Alpha")
    p.add_argument("--task-id", required=True, help="e.g. 327")
    p.add_argument("--records-out", required=True)
    p.add_argument("--episodes-out", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    recs = ab.records_from_repo(args.repo, args.task_id, max_episodes=args.max_episodes)
    pathlib.Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.episodes_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.records_out).write_text(json.dumps([dataclasses.asdict(r) for r in recs], indent=2))
    pathlib.Path(args.episodes_out).write_text(
        json.dumps([dataclasses.asdict(ab.to_episode(r)) for r in recs], indent=2)
    )
    print(f"Wrote {len(recs)} records -> {args.records_out} ; episodes -> {args.episodes_out}")


if __name__ == "__main__":
    main()
