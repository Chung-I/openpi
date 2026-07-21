#!/bin/bash
#SBATCH --job-name=trunc
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/trunc_%j.log
set -x
# Repo root is overridable so this can run from a git worktree without editing the script
# on the cluster. Slurm exports the submitting environment by default, so:
#   OPENPI_ROOT=/work/roboleon1295/openpi-trunc \
#     sbatch --output=/work/roboleon1295/openpi-trunc/logs/trunc_%j.log \
#       scripts/nchc/train_truncation.sbatch
# --output must be overridden on the command line too: Slurm opens that path before this
# script runs, so a #SBATCH directive cannot reference OPENPI_ROOT.
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi}"
# Without this guard, a missing/unmounted repo dir would leave the shell in whatever
# directory it started in and every relative path below (logs/, scripts/...) would run
# somewhere silently wrong instead of failing loudly.
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
# NOTE: this also moves where wandb looks for credentials. ~/.netrc must already exist
# under the new HOME or wandb runs unauthenticated and the run is not tracked --
# see docs/eval/TRUNCATION.md section 2.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
if [ ! -f "$HOME/.netrc" ]; then
  echo "FATAL: $HOME/.netrc missing -- wandb would run unauthenticated. Copy it first."
  exit 1
fi
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache
# Pin the cache to the single canonical location. uv's cache follows HOME, and the HOME
# override above would otherwise silently start a SECOND uv cache under jobhome, re-
# downloading everything. /home is off limits regardless: the binding limit there is a
# per-user quota, not filesystem space, and a full quota fails builds mid-run.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"

# TF's GCS client (tf.data streaming the DROID RLDS shards) needs a CA bundle, and these
# nodes have no /etc/ssl/certs/ca-certificates.crt -- it is a RHEL-family image carrying
# ca-bundle.crt instead, while libcurl defaults to the Debian path. Without this the run
# dies ~60s in with "libcurl code 77 ... Problem with the SSL CA cert". Derived from the
# venv rather than hardcoded so it follows OPENPI_ROOT.
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then
  echo "FATAL: could not locate a certifi CA bundle (got '$CA_BUNDLE'); TF GCS reads would fail."
  exit 1
fi
export CURL_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export GCS_CA_BUNDLE="$CA_BUNDLE"

# Slurm opens --output (trunc_%j.log) before this script runs, so it can't be created here --
# see docs/eval/TRUNCATION.md section 2. These per-arm logs, however, are only ever opened by
# this script body, so they do need to be created here.
mkdir -p logs

A6_LOG=logs/trunc6_${SLURM_JOB_ID}.log
A18_LOG=logs/trunc18_${SLURM_JOB_ID}.log

# --- Pre-launch gate: fail fast before burning any GPU-hours. ---
for cfg in pi05_droid_jointpos_trunc6 pi05_droid_jointpos_trunc18; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python \
    scripts/check_truncation.py "$cfg" || { echo "WEIGHT CHECK FAILED: $cfg"; exit 1; }
done

# --- Two arms in parallel: 4 GPUs each, disjoint device sets. ---
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc6 > "$A6_LOG" 2>&1 &
P6=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/train.py pi05_droid_jointpos_trunc18 > "$A18_LOG" 2>&1 &
P18=$!

wait $P6;  TRUNC6_EXIT=$?;  echo "TRUNC6_EXIT=$TRUNC6_EXIT"
wait $P18; TRUNC18_EXIT=$?; echo "TRUNC18_EXIT=$TRUNC18_EXIT"

echo "=== trunc6 tail ===";  tail -n 40 "$A6_LOG"
echo "=== trunc18 tail ==="; tail -n 40 "$A18_LOG"

# The script's own exit status must reflect the arms' status: ending on `tail` (which almost
# always succeeds) would otherwise report COMPLETED to Slurm even if an arm died at step 300.
if [ "$TRUNC6_EXIT" -ne 0 ] || [ "$TRUNC18_EXIT" -ne 0 ]; then
  echo "FATAL: at least one arm failed (trunc6=$TRUNC6_EXIT, trunc18=$TRUNC18_EXIT)"
  exit 1
fi
