"""Run the memory-label prompt bake-off on NCHC (Qwen via vLLM).

Calibrates throughput against the live server, trims episodes to a time budget, runs the
recursive bake-off over all candidate prompts, fills judge/determinism/coherence metrics,
writes bakeoff_report.{json,md}, then runs the winning prompt once in stateless mode.

Usage:
    uv run python scripts/run_prompt_bakeoff.py \
        --episodes_file data/agibot_episodes.json --prompts_dir prompts \
        --report data/bakeoff --backend openai --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen3.6-27B --time-budget-min 60 --max-episodes 1000
"""
import argparse
import json
import pathlib
import time

from openpi.training import prompt_bakeoff as pb
from openpi.training.memory_labels import Episode
from openpi.training.memory_labels import MemoryLabelConfig
from openpi.training.memory_labels import MemoryLabelGenerator


def _load_episodes(path):
    raw = json.loads(pathlib.Path(path).read_text())
    return [Episode(**e) for e in raw]


def _calibrate(cfg_factory, sample_eps, n_prompts):
    """Time one episode's recursive generation; return est seconds per (prompt, episode)."""
    gen = MemoryLabelGenerator(cfg_factory(None))
    t0 = time.time()
    gen.generate_labels(sample_eps[:1])
    return time.time() - t0  # seconds per episode for one prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", required=True)
    p.add_argument("--prompts_dir", default="prompts")
    p.add_argument("--report", required=True)
    p.add_argument("--backend", default="openai")
    p.add_argument("--base_url", default=None)
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--time-budget-min", type=float, default=60.0)
    p.add_argument("--max-episodes", type=int, default=1000)
    p.add_argument("--out_dir", default="data/bakeoff_shards")
    args = p.parse_args()

    episodes = _load_episodes(args.episodes_file)[: args.max_episodes]
    prompts = pb.load_prompts(args.prompts_dir)

    def factory(template):
        return MemoryLabelConfig(
            backend=args.backend, base_url=args.base_url, model=args.model,
            disable_thinking=True, generation_mode="recursive", prompt_template=template,
            max_concurrency=128,
        )

    # Calibrate + trim episodes to the time budget.
    if args.backend != "mock":
        per = _calibrate(factory, episodes, len(prompts))
        budget_s = args.time_budget_min * 60 * 0.8  # 20% margin
        # recursive: episodes run concurrently; approximate wall time per prompt ~ per * episodes / concurrency
        max_fit = int(budget_s / max(per, 1e-6) * 128 / max(len(prompts), 1))
        episodes = episodes[: max(1, min(len(episodes), max_fit))]
    print(f"Running bake-off: {len(prompts)} prompts x {len(episodes)} episodes")

    result = pb.run_bakeoff(episodes, args.prompts_dir, factory, out_dir=args.out_dir)

    base = pathlib.Path(args.report)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(result, indent=2))
    lines = ["# Prompt bake-off\n", f"- episodes: {len(episodes)}\n", "\n| rank | prompt | composite |\n|---|---|---|\n"]
    for i, (name, score) in enumerate(result["ranking"], 1):
        lines.append(f"| {i} | {name} | {'gated-out' if score is None else f'{score:.3f}'} |\n")
    base.with_suffix(".md").write_text("".join(lines))
    print(f"Report -> {base.with_suffix('.json')} / .md")


if __name__ == "__main__":
    main()
