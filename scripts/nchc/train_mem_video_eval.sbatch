#!/bin/bash
#SBATCH --job-name=mem_video_eval
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_video_eval_%j.log
set -x
cd /work/roboleon1295/openpi

# --- env: checkpoint cache (jointpos base already downloaded here), GCS/HF certs,
#     jax/tmp caches, bounded DROID shuffle buffer (250k default OOMs host RAM),
#     offline wandb on compute node (sync from login later). ---
export OPENPI_DATA_HOME=/work/roboleon1295/.openpi_cache
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
export DROID_SHUFFLE_BUFFER=50000
export WANDB_MODE=offline
export WANDB_DIR=/work/roboleon1295/wandb; mkdir -p "$WANDB_DIR"

K6_LOG=logs/mem_video_eval_k6_${SLURM_JOB_ID}.log
K1_LOG=logs/mem_video_eval_k1_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast if the remap/regex is wrong (before any GPU-hours). ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify || { echo "K6 WEIGHT CHECK FAILED"; exit 1; }
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k1_verify || { echo "K1 WEIGHT CHECK FAILED"; exit 1; }

# --- Two arms in parallel: 4 GPUs each (fsdp_devices=1 -> 4-way data parallel), disjoint sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k6_verify > "$K6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run --no-sync python scripts/train.py pi0_mem_droid_k1_verify > "$K1_LOG" 2>&1 &
P1=$!

wait $P6; echo "K6_EXIT=$?"
wait $P1; echo "K1_EXIT=$?"
echo "=== K6 tail ==="; tail -n 30 "$K6_LOG"
echo "=== K1 tail ==="; tail -n 30 "$K1_LOG"
