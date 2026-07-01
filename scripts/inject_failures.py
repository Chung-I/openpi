"""Inject fail-then-retry sequences into Episodes to test the 'discard failed attempts' rule.

For each episode, insert a [FAILED] copy of a subtask immediately before its [SUCCESS]
occurrence (mirroring the MEM paper's "pick bowl -> pick bowl -> place bowl"). Produces
episodes whose success_flags contain False entries, so the memory-label 'no update on
failure' behavior becomes measurable (see memory_validation.failure_invariance_scores).

Usage:
    uv run python scripts/inject_failures.py --input data/agibot_episodes.json \
        --output data/agibot_episodes_fail.json --max-fail-per-episode 2
"""

import argparse
import dataclasses
import json
import pathlib
import random

from openpi.training.memory_labels import Episode


def inject(ep: Episode, n_fail: int, rng: random.Random) -> Episode:
    """Insert `n_fail` failed attempts, each a copy of a subtask right before its success."""
    subs, flags = list(ep.subtasks), list(ep.success_flags)
    k = min(n_fail, len(subs))
    for pos in sorted(rng.sample(range(len(subs)), k), reverse=True):
        subs.insert(pos, subs[pos])  # the failed attempt is a copy right before the success
        flags.insert(pos, False)
    return Episode(goal=ep.goal, subtasks=subs, success_flags=flags)


def main():
    p = argparse.ArgumentParser(description="Inject fail-then-retry sequences into Episode JSON")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-fail-per-episode", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    eps = [Episode(**e) for e in json.loads(pathlib.Path(args.input).read_text())]
    out = [inject(e, rng.randint(1, args.max_fail_per_episode), rng) for e in eps]
    n_fail = sum(1 for e in out for f in e.success_flags if not f)
    pathlib.Path(args.output).write_text(json.dumps([dataclasses.asdict(e) for e in out], indent=2))
    print(f"Wrote {len(out)} episodes with {n_fail} injected failures -> {args.output}")


if __name__ == "__main__":
    main()
