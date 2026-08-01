#!/bin/bash
# Serve the RELEASED (non-vlash) pi05_droid_jointpos checkpoint for a RoboLab smoke test,
# reached over a plain SSH -L tunnel from the operator's local machine (no tailscale --
# unlike scripts/nchc/serve_truncation.sbatch on chungyi/pi05-layer-truncation, this is a
# dev-partition job and the compute node is reachable directly through nano4's login node).
#
#   sbatch scripts/nchc/vlash_droid_serve.sbatch [config_name] [checkpoint_dir] [port]
#
# Defaults serve the released baseline:
#   config_name=pi05_droid_jointpos
#   checkpoint_dir=/work/roboleon1295/checkpoints/pi05_droid_jointpos
#   port=8000
#
# Local bring-up once the job is RUNNING:
#   sacct -j <jobid> --format=JobID,State,NodeList -n
#   grep SERVER_NODE logs/vlash_serve_<jobid>.log
#   ssh -f -N -L 8000:<node>:8000 nano4
#
#SBATCH --job-name=vlash_serve
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/vlash_serve_%j.log
set -x

CONFIG="${1:-pi05_droid_jointpos}"
CKPT="${2:-/work/roboleon1295/checkpoints/pi05_droid_jointpos}"
PORT="${3:-8000}"

# Repo root is overridable, mirroring vlash_droid_repro.sbatch's convention (this worktree has
# its own venv -- see docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 3 section).
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"

export PYTHONUNBUFFERED=1

# TF/GCS CA bundle: needed if the checkpoint or its assets are ever pointed at a gs:// path;
# harmless no-op for the local-checkpoint case, cheap insurance either way (same fix as
# vlash_droid_repro.sbatch and serve_truncation.sbatch).
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then
  echo "FATAL: could not locate a certifi CA bundle (got '$CA_BUNDLE')."
  exit 1
fi
export CURL_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export GCS_CA_BUNDLE="$CA_BUNDLE"

mkdir -p logs

# Printed FIRST (before any potentially slow model load/compile) so the operator can grep the
# log immediately after the job transitions to RUNNING and open the SSH tunnel without waiting
# for the full ~minutes-long JAX/checkpoint warmup.
echo "SERVER_NODE:$(hostname)"
echo "SERVING config=$CONFIG ckpt=$CKPT port=$PORT"

# --port must precede the `policy:checkpoint` subcommand -- tyro applies trailing args to the
# preceding subcommand, and the reversed order fails to parse (see serve_truncation.sbatch).
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/serve_policy.py --port "$PORT" \
  policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CKPT"
EXIT_CODE=$?

echo "SERVE_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
