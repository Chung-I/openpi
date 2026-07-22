#!/bin/bash
# Serve one layer-truncation arm for RoboLab evaluation.
#
#   sbatch scripts/nchc/serve_truncation.sbatch <config_name> <checkpoint_dir> [port] [ts_hostname]
#
# e.g. sbatch scripts/nchc/serve_truncation.sbatch \
#        pi05_droid_jointpos_trunc6 \
#        checkpoints/pi05_droid_jointpos_trunc6/pi05_droid_jointpos_trunc6/19999 \
#        8000 nchc-trunc6
#
# The RoboLab sim client runs on the operator's local machine and reaches this server over
# tailscale; only the policy runs here. Serving needs no MEM session server -- see
# docs/eval/TRUNCATION.md section 3.
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --job-name=trunc_serve
#SBATCH --output=/work/roboleon1295/openpi/logs/trunc_serve_%j.log
set -x

CONFIG="${1:?usage: serve_truncation.sbatch <config_name> <checkpoint_dir> [port] [ts_hostname]}"
CKPT="${2:?usage: serve_truncation.sbatch <config_name> <checkpoint_dir> [port] [ts_hostname]}"
PORT="${3:-8000}"
TS_HOST="${4:-nchc-trunc-serve}"

OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# Keep caches off /home: the binding limit there is a per-user quota, not disk space.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.cache"
export XDG_CACHE_HOME="$HOME/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"
export HF_HOME=/work/roboleon1295/hf_cache
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache

# These nodes have no /etc/ssl/certs/ca-certificates.crt (RHEL-family image), which the
# checkpoint loader needs for any gs:// read. Same fix as the training script.
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then echo "FATAL: no certifi CA bundle (got '$CA_BUNDLE')"; exit 1; fi
export CURL_CA_BUNDLE="$CA_BUNDLE" SSL_CERT_FILE="$CA_BUNDLE" GCS_CA_BUNDLE="$CA_BUNDLE"

echo "NODE=$(hostname)"

# --- tailscale (userspace networking; compute nodes have no inbound route otherwise) ---
TS=/work/roboleon1295/tailscale
SOCK="/work/roboleon1295/ts_${SLURM_JOB_ID}.sock"
rm -f "$SOCK"
"$TS/tailscaled" --tun=userspace-networking \
  --state="/work/roboleon1295/ts_${SLURM_JOB_ID}.state" --socket="$SOCK" \
  >"$OPENPI_ROOT/logs/trunc_serve_tsd_${SLURM_JOB_ID}.log" 2>&1 &
sleep 5
"$TS/tailscale" --socket="$SOCK" up \
  --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" --hostname="$TS_HOST" --reset
echo "TS_UP_EXIT=$?"
for i in $(seq 1 20); do
  IP=$("$TS/tailscale" --socket="$SOCK" ip -4 2>/dev/null | head -1)
  [ -n "$IP" ] && { echo "TAILSCALE_IP=$IP host=$TS_HOST port=$PORT"; break; }
  sleep 3
done

echo "SERVING config=$CONFIG ckpt=$CKPT port=$PORT"
# NOTE: --port is top-level and MUST precede the policy:checkpoint subcommand; tyro applies
# trailing arguments to the preceding subcommand and the reversed order fails to parse.
OPENPI_DATA_HOME=/work/roboleon1295/.cache/openpi XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  uv run --no-sync python scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CKPT"
echo "SERVE_EXIT=$?"

"$TS/tailscale" --socket="$SOCK" logout 2>/dev/null
