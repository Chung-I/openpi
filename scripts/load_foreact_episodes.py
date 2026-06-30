"""Load ForeAct subtask-annotated episodes into Episode JSON for memory-label generation.

Usage:
    uv run python scripts/load_foreact_episodes.py \
        --hf_dataset mit-han-lab/ForeActDataset --split train --output data/episodes.json

NOTE: ForeAct's exact field names must be confirmed against the dataset card; adjust
the field mapping in `episodes_from_records` / `main` if the keys differ. HF download
needs a valid HF token (see the project's HF_HOME note).
"""

import argparse
import dataclasses
import json
import pathlib

from openpi.training.memory_labels import Episode


def episodes_from_records(records: list[dict]) -> list[Episode]:
    """Map ForeAct-shaped records to Episodes (all subtasks marked success for v1)."""
    episodes = []
    for r in records:
        subtasks = list(r["subtasks"])
        episodes.append(
            Episode(goal=r["goal"], subtasks=subtasks, success_flags=[True] * len(subtasks))
        )
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Load ForeAct episodes -> Episode JSON")
    parser.add_argument("--hf_dataset", type=str, default="mit-han-lab/ForeActDataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    import datasets

    ds = datasets.load_dataset(args.hf_dataset, split=args.split)
    # Map dataset columns to {goal, subtasks}. Adjust keys to the actual ForeAct schema.
    records = [{"goal": ex["goal"], "subtasks": ex["subtasks"]} for ex in ds]
    episodes = episodes_from_records(records)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([dataclasses.asdict(e) for e in episodes], indent=2))
    print(f"Wrote {len(episodes)} episodes to {out}")


if __name__ == "__main__":
    main()
