#!/bin/bash
# Train ONE truncation arm on 4 GPUs.
#
#   sbatch scripts/nchc/train_truncation_one.sbatch pi05_droid_jointpos_trunc6_fullrank
#
# The two-arm script (train_truncation.sbatch) exists for the paired LoRA experiment; this
# one is for escalation runs where only a single arm needs training.
#SBATCH --job-name=trunc1
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
# 600G for a single arm. The tf.data shuffle buffer holds 250k DROID timesteps and fills
# PROGRESSIVELY, reaching ~200G RSS per training process -- a too-small allocation does not
# crash, it thrashes with the GPUs at 0% while Slurm still reports RUNNING.
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/trunc1_%j.log
set -x

CONFIG="${1:?usage: train_truncation_one.sbatch <config_name>}"

OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home: the binding limit there is a per-user quota, not disk space.
# NOTE: this also moves where wandb looks for credentials.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"
export HF_HOME=/work/roboleon1295/hf_cache
if [ ! -f "$HOME/.netrc" ]; then
  echo "FATAL: $HOME/.netrc missing -- wandb would run unauthenticated. Copy it first."
  exit 1
fi

# TF's GCS client (tf.data streaming DROID RLDS) needs a CA bundle; these nodes have no
# /etc/ssl/certs/ca-certificates.crt. Without this the run dies ~60s in with libcurl code 77.
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then echo "FATAL: no certifi CA bundle (got '$CA_BUNDLE')"; exit 1; fi
export CURL_CA_BUNDLE="$CA_BUNDLE" SSL_CERT_FILE="$CA_BUNDLE" GCS_CA_BUNDLE="$CA_BUNDLE"

mkdir -p logs
LOG="logs/${CONFIG}_${SLURM_JOB_ID}.log"

# Pre-launch gate: a model/loader depth mismatch fails here in seconds rather than after an
# 11 GiB checkpoint download and a queue wait.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
  scripts/check_truncation.py "$CONFIG" || { echo "WEIGHT CHECK FAILED: $CONFIG"; exit 1; }

XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" > "$LOG" 2>&1
EXIT=$?
echo "TRAIN_EXIT=$EXIT"
echo "=== tail ==="; tail -n 40 "$LOG"
exit "$EXIT"
