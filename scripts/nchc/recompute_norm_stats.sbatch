#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --job-name=mem_normstats
#SBATCH --output=/work/roboleon1295/openpi/logs/normstats_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"
export GCS_CA_BUNDLE="$CURL_CA_BUNDLE"
MAXF="${1:-200000}"
for CFG in pi0_mem_droid_k1_verify pi0_mem_droid_k6_verify; do
  echo "=== compute_norm_stats $CFG (max_frames=$MAXF) ==="
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run --no-sync python scripts/compute_norm_stats.py \
    --config-name "$CFG" --max-frames "$MAXF"
  echo "NORMSTATS_EXIT_${CFG}=$?"
  ls -la assets/$CFG/droid/ 2>&1
done
echo "ALL_DONE"
