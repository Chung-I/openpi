"""Log the 15k per-update bucketed HL eval results to wandb as one comparison Table.

Parses the raw eval_hl_metrics logs (no hand-transcription) and logs a single summary run
to the `mem-hl-training` project, so the comparison lives alongside the training runs.

Usage (on nano4, wandb authenticated):
    uv run --no-sync python scripts/log_bucketed_wandb.py
"""

import pathlib
import re

import wandb

# friendly-name -> eval_hl_metrics log file (all @ step 15000, gen-examples 64/bucket)
LOGS = {
    "A_baseline": "logs/eval3_A.log",
    "B_unfrozen_siglip": "logs/eval3_B.log",
    "upsample": "logs/eval3_ups.log",
    "base_q_question": "logs/eval_baseq_15k.log",
    "B_plus_upsample": "logs/eval_visups_15k.log",
}
_METRICS = ("subtask_exact_match", "memory_exact_match", "token_accuracy")


def parse(path):
    """-> list of (split, bucket, n, subtask_em, memory_em, token_acc)."""
    txt = pathlib.Path(path).read_text()
    rows = []
    for sm in re.finditer(r"\[(\w+)\]\s+n_update=(\d+)\s+n_noupdate=(\d+)(.*?)(?=\n\[|\Z)", txt, re.S):
        split, n_upd, n_noupd, body = sm.group(1), int(sm.group(2)), int(sm.group(3)), sm.group(4)
        vals = {
            mm.group(1): (float(mm.group(2)), float(mm.group(3)))
            for mm in re.finditer(r"(\w+)\s+update=([\d.]+)\s+noupdate=([\d.]+)", body)
        }
        for bucket, n in (("update", n_upd), ("noupdate", n_noupd)):
            i = 0 if bucket == "update" else 1
            rows.append((split, bucket, n, *[vals[m][i] for m in _METRICS]))
    return rows


def main():
    wandb.init(
        project="mem-hl-training",
        name="bucketed-eval-15k-summary",
        job_type="eval",
        config={"step": 15000, "gen_examples_per_bucket": 64},
    )
    cols = ["config", "split", "update_bucket", "n", "subtask_em", "memory_em", "token_acc"]
    table = wandb.Table(columns=cols)
    for name, path in LOGS.items():
        for split, bucket, n, sub, mem, tok in parse(path):
            table.add_data(name, split, bucket, n, sub, mem, tok)
            # key headline scalar: transition (update=True) subtask EM per config+split
            if bucket == "update":
                wandb.summary[f"{split}/update_subtask_em/{name}"] = sub
    wandb.log({"bucketed_eval_15k": table})
    wandb.finish()
    print("logged bucketed_eval_15k table to wandb project mem-hl-training")


if __name__ == "__main__":
    main()
