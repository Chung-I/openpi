#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --time=3:55:00
#SBATCH --output=/work/roboleon1295/openpi/logs/dev_%x_%j.log
set -x
CFG="${1:?config}"; EXP="${2:?exp}"
cd /work/roboleon1295/openpi
export OPENPI_DATA_HOME=/work/roboleon1295/.openpi_cache HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"; export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
export DROID_SHUFFLE_BUFFER=25000
export WANDB_MODE=offline; export WANDB_DIR=/work/roboleon1295/wandb; mkdir -p "$WANDB_DIR"
# Dev-window run: full 20k target, killed at ~4h; saves every 1000 -> last complete ckpt survives.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync python scripts/train.py "$CFG" \
  --exp-name "$EXP" --num-train-steps 20000 --save-interval 1000 --keep-period 5000 --overwrite
echo "DEV_EXIT=$?"
