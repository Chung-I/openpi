"""Assemble RoboCOIN HL training samples (records + memory labels + frames) -> manifest + frames.

Stage A first (build records from a RoboCOIN task repo):
    uv run python -c "
    import json
    from openpi.training import robocoin
    records = robocoin.records_from_repo('some-org/RoboCOIN-task', max_episodes=100)
    json.dump([r.__dict__ for r in records], open('data/robocoin_records.json', 'w'))
    "
Then Stage B (reuse the existing generator to produce labels):
    uv run python scripts/generate_memory_labels.py --episodes_file data/robocoin_episodes.json \
        --backend openai --base_url http://localhost:8000/v1 --model Qwen/Qwen2.5-VL-72B-Instruct \
        --output data/robocoin_labels.json --disable_thinking
Then Stage C (this script):
    uv run python scripts/assemble_robocoin_hl.py --records_file data/robocoin_records.json \
        --labels_file data/robocoin_labels.json --repo some-org/RoboCOIN-task --out_dir data/robocoin_hl
"""
import argparse
import json
import pathlib

from openpi.training import lerobot_hl as lh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--repo", required=True, help="HF dataset repo id, e.g. some-org/RoboCOIN-task")
    p.add_argument("--out_dir", default="data/robocoin_hl")
    p.add_argument("--video-key", default="observation.images.cam_head_rgb")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--sample-hz", type=float, default=1.0)
    p.add_argument("--max-episodes", type=int, default=None)
    args = p.parse_args()

    records = json.loads(pathlib.Path(args.records_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())
    lh.assemble(
        records, labels, out_dir=args.out_dir, repo=args.repo, video_key=args.video_key,
        sample_hz=args.sample_hz, max_episodes=args.max_episodes, fps_default=args.fps,
    )


if __name__ == "__main__":
    main()
