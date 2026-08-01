#!/bin/bash
#SBATCH --job-name=vlash_repro
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
# Single-GPU, 12-step smoke test -- generous but within the dev partition's per-GPU fair-use
# cap (200G/12 cores per GPU, per Global Constraints). Sized well above what 12 steps actually
# need because the RLDS tf.data shuffle buffer fills PROGRESSIVELY from the start regardless
# of how few steps we run for, and train_truncation.sbatch's 8-GPU run blew past a too-small
# allocation and thrashed silently (RUNNING, 0% GPU) for hours before that was diagnosed.
#SBATCH --mem=190G
#SBATCH --time=00:40:00
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/vlash_repro_%j.log
set -x

# Repo root is overridable so this can run from a different worktree without editing the
# script on the cluster. Slurm exports the submitting environment by default, so:
#   OPENPI_ROOT=/some/other/worktree \
#     sbatch --output=/some/other/worktree/logs/vlash_repro_%j.log \
#       scripts/nchc/vlash_droid_repro.sbatch
# --output must be overridden on the command line too: Slurm opens that path before this
# script runs, so a #SBATCH directive cannot reference OPENPI_ROOT.
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache
# Pin the cache to the single canonical location; uv's cache follows HOME, and the HOME
# override above would otherwise silently start a SECOND uv cache under jobhome.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"
# This job runs with wandb disabled (--no-wandb-enabled below), so no ~/.netrc requirement
# here -- unlike train_truncation.sbatch, which needs wandb auth for real training runs.

export PYTHONUNBUFFERED=1

# TF's GCS client (tf.data streaming the DROID RLDS shards from gs://gresearch/robotics) needs
# a CA bundle, and these nodes have no /etc/ssl/certs/ca-certificates.crt -- without this the
# run dies ~60s in with "libcurl code 77 ... Problem with the SSL CA cert". Derived from the
# venv rather than hardcoded so it follows OPENPI_ROOT. This worktree has its OWN venv (see
# docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 3 section) -- do not point
# UV_PROJECT_ENVIRONMENT at /work/roboleon1295/openpi/.venv, which is synced against a
# different branch's lockfile.
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then
  echo "FATAL: could not locate a certifi CA bundle (got '$CA_BUNDLE'); TF GCS reads would fail."
  exit 1
fi
export CURL_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export GCS_CA_BUNDLE="$CA_BUNDLE"

mkdir -p logs

# --- 12-step repro of pi05_droid_jointpos_vlash: config + RLDS window + offset transform. ---
# --no-enable-async-checkpointing: on this single-GPU job, orbax's async
# device-to-host-transfer + background save was observed to hang indefinitely at step 6's
# checkpoint save (zero bytes written, zero GPU util, main thread stuck on a futex, for 10+
# minutes) -- see docs/superpowers/plans/2026-08-01-vlash-droid-notes.md (Task 3 section).
# Async checkpointing matters for throughput on real multi-GPU production runs; this repro is a
# single GPU / 12-step smoke test, so paying for a synchronous (slower but reliable) save here
# is the right tradeoff. Re-evaluate this flag when moving to the real multi-GPU training job.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py pi05_droid_jointpos_vlash \
  --exp-name=vlash_repro \
  --overwrite \
  --batch-size=4 \
  --num-train-steps=12 \
  --log-interval=1 \
  --save-interval=6 \
  --no-wandb-enabled \
  --no-enable-async-checkpointing
EXIT_CODE=$?

echo "TRAIN_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
