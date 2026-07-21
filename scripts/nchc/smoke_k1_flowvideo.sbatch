#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=3:50:00
#SBATCH --job-name=k1_smoke
#SBATCH --output=/work/roboleon1295/openpi/logs/k1_smoke_%j.log
set -x
cd /work/roboleon1295/openpi
export OPENPI_DATA_HOME=/work/roboleon1295/.openpi_cache
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"; export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
export DROID_SHUFFLE_BUFFER=25000
export WANDB_MODE=offline; export WANDB_DIR=/work/roboleon1295/wandb; mkdir -p "$WANDB_DIR"
# SMOKE: flow-video K=1, 1000 steps, to validate the recipe before the full 20k.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync python scripts/train.py pi0_mem_droid_k1_verify \
  --exp-name pi0_mem_droid_k1_flowvideo_smoke --num-train-steps 1000 --save-interval 500 --overwrite
echo "SMOKE_EXIT=$?"
