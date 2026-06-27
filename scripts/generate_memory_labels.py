"""Generate memory compression labels for MEM training.

Usage:
    uv run python scripts/generate_memory_labels.py \
        --episodes_file data/episodes.json \
        --backend claude \
        --output data/memory_labels.json
"""

import argparse
import json
import logging
import pathlib

from openpi.training.memory_labels import Episode, MemoryLabelConfig, MemoryLabelGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate memory compression labels for MEM")
    parser.add_argument("--episodes_file", type=str, required=True, help="Path to JSON file with episode data")
    parser.add_argument(
        "--backend",
        type=str,
        default="claude",
        choices=["claude", "openai", "local", "mock"],
    )
    parser.add_argument("--model", type=str, default=None, help="Model name for the LLM backend")
    parser.add_argument("--output", type=str, required=True, help="Output path for memory labels JSON")
    parser.add_argument("--max_memory_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.model:
        config = MemoryLabelConfig(
            backend=args.backend,
            model=args.model,
            max_memory_tokens=args.max_memory_tokens,
        )
    else:
        config = MemoryLabelConfig(
            backend=args.backend,
            max_memory_tokens=args.max_memory_tokens,
        )

    with open(args.episodes_file) as f:
        episodes = [Episode(goal=ep["goal"], subtasks=ep["subtasks"], success_flags=ep["success_flags"]) for ep in json.load(f)]

    logger.info(f"Generating labels for {len(episodes)} episodes with backend={args.backend}")
    generator = MemoryLabelGenerator(config)
    labels = generator.generate_labels(episodes)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([{"episode_id": label.episode_id, "memories": label.memories} for label in labels], f, indent=2)

    logger.info(f"Labels written to {output_path}")


if __name__ == "__main__":
    main()
