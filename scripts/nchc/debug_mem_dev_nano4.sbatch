#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --job-name=mem_ve_debug
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_ve_debug_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
set -x
CONFIG="${1:-pi0_mem_droid_k6_verify}"
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export WANDB_MODE=offline
export DROID_SHUFFLE_BUFFER=2000
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name=debug_1gpu --overwrite --fsdp-devices=1 --batch-size=16 --num-train-steps=50 --save-interval=50
echo "TRAIN_EXIT=$?"
