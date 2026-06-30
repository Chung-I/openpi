"""Validate generated memory labels: full-set heuristics + sampled LLM-judge + reconstruction probe.

Usage:
    uv run python scripts/validate_memory_labels.py \
        --episodes_file data/episodes.json --labels_file data/memory_labels.json \
        --report data/validation_report --backend openai --base_url http://localhost:8000/v1 \
        --model qwen --sample 50

Heuristics run on every label (CPU). The LLM-judge and reconstruction probe run on a
random sample via the same backend/endpoint. Use --backend mock to dry-run wiring.
"""

import argparse
import json
import pathlib
import random
import statistics

from openpi.training import memory_validation as mv
from openpi.training.memory_labels import MemoryLabelConfig, MemoryLabelGenerator, _format_subtask_sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", required=True)
    p.add_argument("--labels_file", required=True)
    p.add_argument("--report", required=True, help="output path prefix (.json/.md)")
    p.add_argument("--backend", default="mock")
    p.add_argument("--base_url", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--sample", type=int, default=50)
    args = p.parse_args()

    episodes = json.loads(pathlib.Path(args.episodes_file).read_text())
    labels = json.loads(pathlib.Path(args.labels_file).read_text())

    comps, faiths, flagged = [], [], []
    for ep, lab in zip(episodes, labels, strict=False):
        subtasks, flags = ep["subtasks"], ep["success_flags"]
        for i, memory in enumerate(lab["memories"]):
            history = _format_subtask_sequence(subtasks, flags, i)
            comps.append(mv.compression_ratio(memory, history))
            f = mv.faithfulness(memory, history)
            faiths.append(f)
            if f < 0.5:
                flagged.append({"episode_id": lab["episode_id"], "timestep": i, "memory": memory})

    report = {
        "n_labels": len(comps),
        "compression_ratio_mean": statistics.mean(comps) if comps else 0.0,
        "faithfulness_mean": statistics.mean(faiths) if faiths else 0.0,
        "n_flagged_low_faithfulness": len(flagged),
        "flagged_examples": flagged[:20],
    }

    cfg = MemoryLabelConfig(backend=args.backend, base_url=args.base_url, model=args.model or "qwen")
    gen = MemoryLabelGenerator(cfg)
    client = gen._get_client()  # noqa: SLF001 -- reuse the same backend for judge/probe
    # Sampled judge + reconstruction probe (skipped for mock).
    pairs = [(ep, lab, i) for ep, lab in zip(episodes, labels, strict=False) for i in range(len(lab["memories"]))]
    sample = random.sample(pairs, min(args.sample, len(pairs)))
    judge_scores, probe = [], {"n": 0, "match": 0}
    if args.backend != "mock":
        for ep, lab, i in sample:
            history = _format_subtask_sequence(ep["subtasks"], ep["success_flags"], i)
            memory = lab["memories"][i]
            jp = mv.build_judge_prompt(ep["goal"], history, memory)
            jr = client.chat.completions.create(model=cfg.model, max_tokens=64,
                                                messages=[{"role": "user", "content": jp}])
            judge_scores.append(jr.choices[0].message.content)
            if i + 1 < len(ep["subtasks"]):
                rp = mv.build_reconstruction_prompt(ep["goal"], memory)
                rr = client.chat.completions.create(model=cfg.model, max_tokens=32,
                                                    messages=[{"role": "user", "content": rp}])
                pred = rr.choices[0].message.content.strip().lower()
                probe["n"] += 1
                if ep["subtasks"][i + 1].lower() in pred or pred in ep["subtasks"][i + 1].lower():
                    probe["match"] += 1
    report["judge_sample"] = judge_scores
    report["reconstruction_probe"] = probe

    base = pathlib.Path(args.report)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(report, indent=2))
    base.with_suffix(".md").write_text(
        f"# Memory label validation\n\n"
        f"- labels: {report['n_labels']}\n"
        f"- compression ratio (mean): {report['compression_ratio_mean']:.3f}\n"
        f"- faithfulness (mean): {report['faithfulness_mean']:.3f}\n"
        f"- low-faithfulness flagged: {report['n_flagged_low_faithfulness']}\n"
        f"- reconstruction probe: {probe['match']}/{probe['n']} match\n"
    )
    print(f"Validation report written to {base.with_suffix('.json')} / .md")


if __name__ == "__main__":
    main()
