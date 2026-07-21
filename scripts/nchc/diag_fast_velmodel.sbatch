#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --job-name=velmodel
#SBATCH --output=/work/roboleon1295/openpi/logs/velmodel_%j.log
set -x
cd /work/roboleon1295/openpi
export OPENPI_DATA_HOME=/work/roboleon1295/.openpi_cache HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"; export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
export DROID_SHUFFLE_BUFFER=1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync python scripts/nchc/diag_fast_velmodel.py
echo "DIAG_EXIT=$?"
