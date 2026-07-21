#!/bin/bash
#SBATCH --job-name=trunc
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/trunc_%j.log
set -x
cd /work/roboleon1295/openpi

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache

A6_LOG=logs/trunc6_${SLURM_JOB_ID}.log
A18_LOG=logs/trunc18_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast before burning any GPU-hours. ---
for cfg in pi05_droid_jointpos_trunc6 pi05_droid_jointpos_trunc18; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
    scripts/check_truncation.py "$cfg" || { echo "WEIGHT CHECK FAILED: $cfg"; exit 1; }
done

# --- Two arms in parallel: 4 GPUs each, disjoint device sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc6 > "$A6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc18 > "$A18_LOG" 2>&1 &
P18=$!

wait $P6;  echo "TRUNC6_EXIT=$?"
wait $P18; echo "TRUNC18_EXIT=$?"

echo "=== trunc6 tail ===";  tail -n 40 "$A6_LOG"
echo "=== trunc18 tail ==="; tail -n 40 "$A18_LOG"
