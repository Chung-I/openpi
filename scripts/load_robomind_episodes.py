"""Download RoboMIND h5_franka_1rgb annotations -> records.json + episodes.json (for label gen)."""
import argparse
import dataclasses
import json
import pathlib

from openpi.training import robomind as rm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records-out", required=True)
    p.add_argument("--episodes-out", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--repo", default=rm.REPO)
    p.add_argument("--annotation", default=rm.ANNOTATION)
    args = p.parse_args()

    recs = rm.records_from_annotation_list(rm.download_annotation(args.repo, args.annotation))
    if args.max_episodes:
        recs = recs[: args.max_episodes]
    pathlib.Path(args.records_out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.records_out).write_text(json.dumps([dataclasses.asdict(r) for r in recs], indent=2))
    pathlib.Path(args.episodes_out).write_text(
        json.dumps([dataclasses.asdict(rm.to_episode(r)) for r in recs], indent=2)
    )
    print(f"Wrote {len(recs)} records -> {args.records_out} ; episodes -> {args.episodes_out}")


if __name__ == "__main__":
    main()
