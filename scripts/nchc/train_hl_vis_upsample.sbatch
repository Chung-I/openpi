#!/bin/bash
# B + upsample: unfrozen SigLIP AND update-upsampling (both levers together).
#SBATCH --job-name=mem_hl_vis_ups
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leon129506@gmail.com
#SBATCH --output=/work/roboleon1295/openpi-hl/logs/mem_hl_vis_ups_%j.log
set -x
cd /work/roboleon1295/openpi-hl
export HF_HOME=/work/roboleon1295/hf_cache
export TMPDIR=/work/roboleon1295/tmp
mkdir -p logs splits "$TMPDIR"

uv run python scripts/build_hl_splits.py \
  --manifest data/robomind_hl_fr3/manifest.jsonl --out splits/fr3_splits.json \
  --seed 0 --n-dev-unseen-tasks 9 --n-dev-seen-episodes 40

XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run python scripts/train_hl.py pi0_mem_hl_fr3_base_vis_upsample
