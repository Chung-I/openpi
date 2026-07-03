#!/bin/bash
#SBATCH --job-name=mem_hl_fr3
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
#SBATCH --output=/work/roboleon1295/openpi-hl/logs/mem_hl_fr3_%j.log
set -x
cd /work/roboleon1295/openpi-hl                     # isolated HL worktree/checkout on nano4
export HF_HOME=/work/roboleon1295/hf_cache
mkdir -p logs splits

# Build the shared split once (idempotent; both arms read it).
uv run python scripts/build_hl_splits.py \
  --manifest data/robomind_hl_fr3/manifest.jsonl --out splits/fr3_splits.json \
  --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40

BASE_LOG=logs/mem_hl_fr3_base_${SLURM_JOB_ID}.log
DROID_LOG=logs/mem_hl_fr3_droid_${SLURM_JOB_ID}.log

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run python scripts/train_hl.py pi0_mem_hl_fr3_base  > "$BASE_LOG"  2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run python scripts/train_hl.py pi0_mem_hl_fr3_droid > "$DROID_LOG" 2>&1 &
P2=$!
wait $P1; echo "BASE_EXIT=$?"
wait $P2; echo "DROID_EXIT=$?"
echo "=== base tail ==="; tail -n 40 "$BASE_LOG"
echo "=== droid tail ==="; tail -n 40 "$DROID_LOG"
