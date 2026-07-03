#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --job-name=mem_serve
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_serve_%j.log
set -x
CONFIG="${1:?usage: sbatch serve_mem_session_nano4.sbatch <config> <ckpt_dir> [port]}"
CKPT="${2:?need ckpt_dir}"
PORT="${3:-8000}"
TS=/work/roboleon1295/tailscale
SOCK=/work/roboleon1295/ts.sock
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

echo "NODE=$(hostname)"
# --- tailscale (userspace, no root; validated bring-up) ---
rm -f "$SOCK"
"$TS/tailscaled" --tun=userspace-networking --state=/work/roboleon1295/ts.state --socket="$SOCK" \
  >/work/roboleon1295/openpi/logs/mem_serve_tsd_${SLURM_JOB_ID}.log 2>&1 &
sleep 5
"$TS/tailscale" --socket="$SOCK" up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" \
  --hostname=nchc-mem-serve --reset
echo "TS_UP_EXIT=$?"
for i in $(seq 1 20); do
  IP=$("$TS/tailscale" --socket="$SOCK" ip -4 2>/dev/null | head -1)
  [ -n "$IP" ] && { echo "TAILSCALE_IP=$IP host=nchc-mem-serve"; break; }
  sleep 3
done

# --- serve (foreground; holds the job) ---
echo "SERVING config=$CONFIG port=$PORT"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/serve_mem_session.py \
  --config "$CONFIG" --ckpt-dir "$CKPT" --port "$PORT" --host 127.0.0.1
echo "SERVE_EXIT=$?"
"$TS/tailscale" --socket="$SOCK" logout 2>/dev/null
