#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH --time=12:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_ve_%x_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
set -x
CONFIG="${1:?usage: sbatch --job-name=<name> this.sbatch <config_name>}"
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export WANDB_MODE=offline
export DROID_SHUFFLE_BUFFER=50000
# TF GCS (DROID tf.data streaming) needs a CA bundle; the node has no /etc/ssl/certs/ca-certificates.crt.
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/nchc/check_mem_droid_weights.py "$CONFIG" || { echo "WEIGHT CHECK FAILED"; exit 1; }
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" --batch-size=${BATCH:-32}
echo "TRAIN_EXIT=$?"
