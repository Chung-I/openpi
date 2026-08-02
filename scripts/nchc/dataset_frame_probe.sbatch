#!/bin/bash
# Task 10 successor: dataset-frame prediction probe (serving-path vs training-drift
# discriminator). CPU-only job -- streams a few DROID RLDS frames from GCS and queries
# the ALREADY-RUNNING serve jobs over websocket (inference happens on their GPUs).
#
#   sbatch scripts/nchc/dataset_frame_probe.sbatch \
#     [baseline_host:port] [vlash_host:port] [statecond_host:port]
#
# Defaults match serve jobs 229287 / 229288 / 229758 (2026-08-02). Pass the current
# nodes if any serve job was resubmitted elsewhere.
#
#SBATCH --job-name=frame_probe
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/frame_probe_%j.log
set -x

BASELINE="${1:-25a-hgpn002:8000}"
VLASH="${2:-25a-hgpn004:8001}"
STATECOND="${3:-25a-hgpn002:8002}"

OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home (100G quota).
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.cache"
export XDG_CACHE_HOME="$HOME/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"
export PYTHONUNBUFFERED=1

# TF's GCS client needs a CA bundle (same recipe as vlash_droid_train.sbatch).
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then
  echo "FATAL: could not locate a certifi CA bundle (got '$CA_BUNDLE')."
  exit 1
fi
export CURL_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export GCS_CA_BUNDLE="$CA_BUNDLE"

# CPU-only: hide the GPU from JAX/TF (we only stream data + call remote servers).
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

mkdir -p /work/roboleon1295/stage0

uv run --no-sync python scripts/diag_dataset_frame_probe.py \
  --frames 6 --chunk 15 --k 5 \
  --server "baseline=$BASELINE" \
  --server "vlash=$VLASH" \
  --server "statecond=$STATECOND" \
  --out "/work/roboleon1295/stage0/dataset_frame_probe_${SLURM_JOB_ID}.json"
EXIT_CODE=$?
echo "PROBE_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
