"""Live-fill (async judge/coherence/determinism) + paired permutation significance for the bake-off.

Async-batches the Qwen judge + determinism calls (semaphore-bounded) and reuses the generated
label shards (no regeneration). Recomputes dense per-label heuristics (faithfulness/conciseness/
structural, and failure_invariance when the slice contains [FAILED] steps), fills
decision_relevance + temporal_coherence (judge) and determinism (resample) on a COMMON sample
(paired across prompts), computes the gated composite + ranking, and runs paired sign-flip
permutation tests of the top prompt vs the others. Run after run_prompt_bakeoff (shards exist).
"""

import argparse
import asyncio
import json
import pathlib
import random
import re
import statistics

import numpy as np

from openpi.training import memory_validation as mv
from openpi.training import prompt_bakeoff as pb
from openpi.training.memory_labels import RECURSIVE_FIRST_MEMORY
from openpi.training.memory_labels import Episode
from openpi.training.memory_labels import MemoryLabelConfig
from openpi.training.memory_labels import MemoryLabelGenerator
from openpi.training.memory_labels import _format_event
from openpi.training.memory_labels import _format_subtask_sequence

_JSON = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse(raw):
    m = _JSON.search(raw or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _num(d, key):
    v = (d or {}).get(key)
    return float(v) / 5.0 if isinstance(v, int | float) else None


def perm_test(a, b, n_perm=10000, seed=0, batch=2000):
    """Paired two-sided sign-flip permutation test on mean(a-b). Returns (mean_diff, p, n)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    while done < n_perm:
        k = min(batch, n_perm - done)
        signs = rng.integers(0, 2, size=(k, n)) * 2 - 1
        ge += int(np.sum(np.abs((d[None, :] * signs).mean(axis=1)) >= abs(obs) - 1e-12))
        done += k
    return obs, (ge + 1) / (n_perm + 1), n


async def run(args):
    episodes = [Episode(**e) for e in json.loads(pathlib.Path(args.episodes_file).read_text())]
    prompts = pb.load_prompts(args.prompts_dir)
    labels = {
        name: [
            json.loads((pathlib.Path(args.shards_dir) / name / f"{i}.json").read_text())["memories"]
            for i in range(len(episodes))
        ]
        for name in prompts
    }
    order = [(e, i) for e, ep in enumerate(episodes) for i in range(len(ep.subtasks))]
    rng = random.Random(args.seed)
    judge_pos = rng.sample(order, min(args.judge_sample, len(order)))
    det_pos = rng.sample(order, min(args.det_sample, len(order)))

    cfg = MemoryLabelConfig(backend="openai", base_url=args.base_url, model=args.model, disable_thinking=True)
    gen = MemoryLabelGenerator(cfg)
    aclient = gen._get_async_client()  # noqa: SLF001
    extra = gen._chat_extra()  # noqa: SLF001
    sem = asyncio.Semaphore(args.concurrency)

    async def complete(text, temperature=0.0, max_tokens=128):
        async with sem:
            r = await aclient.chat.completions.create(
                model=args.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": text}],
                **extra,
            )
        return r.choices[0].message.content or ""

    # Dense per-label heuristics (+ failure_invariance when the slice has failures)
    heur = {}
    for name in prompts:
        f, c, s = [], [], []
        for e, i in order:
            ep = episodes[e]
            m = labels[name][e][i]
            h = _format_subtask_sequence(ep.subtasks, ep.success_flags, i)
            f.append(mv.faithfulness(m, h))
            c.append(1.0 - min(1.0, mv.compression_ratio(m, h)))
            s.append(mv.structural_score(m))
        heur[name] = {"faithfulness": f, "conciseness": c, "structural": s}
    dense = ["faithfulness", "conciseness", "structural"]
    if any(not ok for ep in episodes for ok in ep.success_flags):
        dense.append("failure_invariance")
        for name in prompts:
            heur[name]["failure_invariance"] = [
                x
                for e, ep in enumerate(episodes)
                for x in mv.failure_invariance_scores(labels[name][e], ep.success_flags, RECURSIVE_FIRST_MEMORY)
            ]

    # Judge (decision_relevance) + coherence, async-batched across prompts+positions
    dr = {n: {} for n in prompts}
    coh = {n: {} for n in prompts}
    jobs, meta = [], []
    for name in prompts:
        for e, i in judge_pos:
            ep = episodes[e]
            m = labels[name][e][i]
            h = _format_subtask_sequence(ep.subtasks, ep.success_flags, i)
            jobs.append(complete(mv.build_judge_prompt(ep.goal, h, m), 0.0, 96))
            meta.append(("dr", name, (e, i)))
            if i > 0:
                jobs.append(complete(mv.build_coherence_prompt(ep.goal, labels[name][e][i - 1], m), 0.0, 64))
                meta.append(("coh", name, (e, i)))
    for (kind, name, pos), raw in zip(meta, await asyncio.gather(*jobs), strict=True):
        v = _num(_parse(raw), "decision_relevance" if kind == "dr" else "coherence")
        if v is not None:
            (dr if kind == "dr" else coh)[name][pos] = v

    # Determinism: async resample each common position at temp 0.8
    det = {n: {} for n in prompts}
    dgens = {
        name: MemoryLabelGenerator(
            MemoryLabelConfig(
                backend="openai",
                base_url=args.base_url,
                model=args.model,
                disable_thinking=True,
                generation_mode="recursive",
                prompt_template=tmpl,
                temperature=0.8,
            )
        )
        for name, tmpl in prompts.items()
    }
    det_jobs, det_meta = [], []
    for name in prompts:
        for e, i in det_pos:
            ep = episodes[e]
            prev = RECURSIVE_FIRST_MEMORY if i == 0 else labels[name][e][i - 1]
            pr = dgens[name]._recursive_prompt(ep.goal, prev, _format_event(ep.subtasks[i], ep.success_flags[i]))  # noqa: SLF001
            det_jobs.append(asyncio.gather(*[complete(pr, 0.8, 128) for _ in range(args.det_resamples)]))
            det_meta.append((name, (e, i)))
    for (name, pos), samples in zip(det_meta, await asyncio.gather(*det_jobs), strict=True):
        det[name][pos] = mv.determinism_score(list(samples))
    for name in prompts:
        print(f"filled {name}: dr={len(dr[name])} coh={len(coh[name])} det={len(det[name])}", flush=True)

    # Aggregate per-prompt metrics + gated composite ranking
    metrics = {}
    for name in prompts:
        metrics[name] = {
            "faithfulness": statistics.mean(heur[name]["faithfulness"]),
            "conciseness": statistics.mean(heur[name]["conciseness"]),
            "structural": statistics.mean(heur[name]["structural"]),
            "decision_relevance": statistics.mean(dr[name].values()) if dr[name] else 0.0,
            "temporal_coherence": statistics.mean(coh[name].values()) if coh[name] else 0.0,
            "determinism": statistics.mean(det[name].values()) if det[name] else 0.0,
        }
        if "failure_invariance" in dense:
            metrics[name]["failure_invariance"] = statistics.mean(heur[name]["failure_invariance"])
    ranking = pb.rank_candidates(metrics)
    winner = ranking[0][0]

    sig = []
    for other, _ in ranking[1:]:
        for metric in dense:
            sig.append((other, metric, *perm_test(heur[winner][metric], heur[other][metric], args.n_perm, args.seed)))  # noqa: PERF401
        for metric, store in (("decision_relevance", dr), ("temporal_coherence", coh), ("determinism", det)):
            common = [k for k in store[winner] if k in store[other]]
            a = [store[winner][k] for k in common]
            b = [store[other][k] for k in common]
            sig.append((other, metric, *perm_test(a, b, args.n_perm, args.seed)))

    _write_report(args, metrics, ranking, winner, sig, order, judge_pos, det_pos, dense, episodes)


def _write_report(args, metrics, ranking, winner, sig, order, judge_pos, det_pos, dense, episodes):
    has_fi = "failure_invariance" in dense
    out = pathlib.Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "ranking": ranking,
                "winner": winner,
                "significance": [{"vs": o, "metric": m, "mean_diff": md, "p": pv, "n": n} for o, m, md, pv, n in sig],
                "n_episodes": len(episodes),
                "n_labels": len(order),
                "judge_sample": len(judge_pos),
                "det_sample": len(det_pos),
            },
            indent=2,
        )
    )
    header = "| rank | prompt | faith | concise | struct | dec_rel | coher | determ | composite |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [
        f"# Prompt bake-off: {len(episodes)} episodes, {len(order)} labels; "
        f"judge n={len(judge_pos)}, det n={len(det_pos)}x{args.det_resamples}\n\n",
        "## Ranking (composite; failure_invariance shown when the slice has failures)\n\n",
        header + (" fail_inv |\n" if has_fi else "\n"),
        sep + ("---|\n" if has_fi else "\n"),
    ]
    for r, (name, comp) in enumerate(ranking, 1):
        m = metrics[name]
        cs = "gated-out" if comp is None else f"{comp:.3f}"
        row = (
            f"| {r} | {name} | {m['faithfulness']:.3f} | {m['conciseness']:.3f} | {m['structural']:.3f} | "
            f"{m['decision_relevance']:.3f} | {m['temporal_coherence']:.3f} | {m['determinism']:.3f} | {cs} |"
        )
        lines.append(row + (f" {m.get('failure_invariance', 1.0):.3f} |\n" if has_fi else "\n"))
    lines.append(f"\n## Paired permutation tests ({args.n_perm} perms, two-sided), winner = **{winner}**\n\n")
    lines.append("| vs | metric | mean_diff (winner-other) | p | n |\n|---|---|---|---|---|\n")
    for o, m, md, pv, n in sig:
        star = " ***" if pv < 0.001 else (" **" if pv < 0.01 else (" *" if pv < 0.05 else ""))
        lines.append(f"| {o} | {m} | {md:+.4f} | {pv:.4f}{star} | {n} |\n")
    out.with_suffix(".md").write_text("".join(lines))
    print(f"Wrote {out.with_suffix('.json')} / .md ; winner={winner}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes_file", required=True)
    p.add_argument("--prompts_dir", default="prompts")
    p.add_argument("--shards_dir", default="data/bakeoff_shards")
    p.add_argument("--report", default="data/bakeoff_full")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B")
    p.add_argument("--judge_sample", type=int, default=120)
    p.add_argument("--det_sample", type=int, default=30)
    p.add_argument("--det_resamples", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--n_perm", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
