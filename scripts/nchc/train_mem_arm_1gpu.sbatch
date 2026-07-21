#!/bin/bash
#SBATCH --job-name=mem_arm
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_arm_%j.log
set -x
CFG="${1:?usage: sbatch train_mem_arm_1gpu.sbatch <config_name>}"
cd /work/roboleon1295/openpi
export OPENPI_DATA_HOME=/work/roboleon1295/.openpi_cache
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"; export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
export DROID_SHUFFLE_BUFFER=25000
export WANDB_MODE=offline
export WANDB_DIR=/work/roboleon1295/wandb; mkdir -p "$WANDB_DIR"

echo "NODE=$(hostname) CONFIG=$CFG"
# Pre-launch gate: fail fast if the weight remap/regex is wrong.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py "$CFG" || { echo "WEIGHT CHECK FAILED"; exit 1; }
# Single GPU (fsdp_devices=1 in config -> no sharding, full model on this GPU).
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync python scripts/train.py "$CFG"
echo "TRAIN_EXIT=$?"
