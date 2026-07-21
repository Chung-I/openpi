#!/bin/bash
#SBATCH --job-name=mem_ve_wcheck
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_ve_wcheck_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export WANDB_MODE=offline
echo "=== K6 weight check ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k6_verify
echo "K6_CHECK_EXIT=$?"
echo "=== K1 weight check ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/nchc/check_mem_droid_weights.py pi0_mem_droid_k1_verify
echo "K1_CHECK_EXIT=$?"
