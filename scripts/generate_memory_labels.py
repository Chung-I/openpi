"""Generate memory compression labels for MEM training.

Usage:
    uv run python scripts/generate_memory_labels.py \
        --episodes_file data/episodes.json \
        --backend claude \
        --output data/memory_labels.json

    # Async path with vLLM endpoint (Qwen on NCHC):
    uv run python scripts/generate_memory_labels.py \
        --episodes_file data/episodes.json \
        --backend openai \
        --base_url http://<node>:8000/v1 \
        --max_concurrency 64 \
        --out_dir data/shards \
        --output data/memory_labels.json
"""

import argparse
import asyncio
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
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_concurrency", type=int, default=64)
    parser.add_argument("--out_dir", type=str, default=None, help="per-episode shard dir (resume)")
    parser.add_argument("--generation_mode", type=str, default="stateless", choices=["stateless", "recursive"])
    parser.add_argument("--disable_thinking", action="store_true")
    args = parser.parse_args()

    config = MemoryLabelConfig(
        backend=args.backend,
        model=args.model or MemoryLabelConfig().model,
        max_memory_tokens=args.max_memory_tokens,
        base_url=args.base_url,
        max_concurrency=args.max_concurrency,
        generation_mode=args.generation_mode,
        disable_thinking=args.disable_thinking,
    )

    with open(args.episodes_file) as f:
        episodes = [Episode(goal=ep["goal"], subtasks=ep["subtasks"], success_flags=ep["success_flags"]) for ep in json.load(f)]

    logger.info(f"Generating labels for {len(episodes)} episodes with backend={args.backend}")
    generator = MemoryLabelGenerator(config)
    labels = asyncio.run(generator.generate_labels_async(episodes, out_dir=args.out_dir))

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([{"episode_id": label.episode_id, "memories": label.memories} for label in labels], f, indent=2)

    logger.info(f"Labels written to {output_path}")


if __name__ == "__main__":
    main()
