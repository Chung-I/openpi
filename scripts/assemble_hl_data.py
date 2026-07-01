"""Assemble RoboMIND HL training samples (records + memory labels + frames) -> manifest + frames.

Stage A first (reuse the existing generator to produce labels):
    uv run python scripts/generate_memory_labels.py --episodes_file data/robomind_episodes.json \
        --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-27B \
        --output data/robomind_labels.json --disable_thinking   # recursive v7 by default
Then:
    uv run python scripts/assemble_hl_data.py --records_file data/robomind_records.json \
        --labels_file data/robomind_labels.json --out_dir data/robomind_hl
"""
import argparse
import json
import pathlib

from openpi.training import robomind as rm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--out_dir", default="data/robomind_hl")
    p.add_argument("--cache_dir", default="data/robomind_cache")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--sample-hz", type=float, default=1.0)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    records = json.loads(pathlib.Path(args.records_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())
    rm.assemble(records, labels, out_dir=args.out_dir, cache_dir=args.cache_dir,
                fps_default=args.fps, sample_hz=args.sample_hz, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
