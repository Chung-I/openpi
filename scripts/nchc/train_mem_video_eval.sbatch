#!/bin/bash
#SBATCH --job-name=mem_video_eval
#SBATCH --account=MST114563
#SBATCH --partition=gpu            # REPLACE with the 8-GPU partition from `sinfo -o "%P %G"`
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_video_eval_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache

K6_LOG=logs/mem_video_eval_k6_${SLURM_JOB_ID}.log
K1_LOG=logs/mem_video_eval_k1_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast if the remap/regex is wrong (before any GPU-hours). ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify || { echo "K6 WEIGHT CHECK FAILED"; exit 1; }
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k1_verify || { echo "K1 WEIGHT CHECK FAILED"; exit 1; }

# --- Two arms in parallel: 4 GPUs each, disjoint device sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k6_verify > "$K6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k1_verify > "$K1_LOG" 2>&1 &
P1=$!

wait $P6; echo "K6_EXIT=$?"
wait $P1; echo "K1_EXIT=$?"

echo "=== K6 tail ==="; tail -n 40 "$K6_LOG"
echo "=== K1 tail ==="; tail -n 40 "$K1_LOG"
