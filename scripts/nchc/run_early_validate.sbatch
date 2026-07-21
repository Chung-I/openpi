#!/bin/bash
#SBATCH --job-name=early_validate
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/early_validate_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache NLTK_DATA=/work/roboleon1295/nltk_data
export TMPDIR=/work/roboleon1295/tmp XDG_CACHE_HOME=/work/roboleon1295/.cache HOME=/work/roboleon1295
rm -rf data/robomind_hl_early data/robomind_cache_early
uv run --no-sync python scripts/assemble_hl_data.py \
  --records_file data/robomind_records_early.json --labels_file data/robomind_labels_early.json \
  --out_dir data/robomind_hl_early --cache_dir data/robomind_cache_early --fps 10 --sample-hz 1
echo "ASM_EXIT=$?"
echo "=== manifest ==="; wc -l data/robomind_hl_early/manifest.jsonl 2>/dev/null; head -3 data/robomind_hl_early/manifest.jsonl 2>/dev/null
echo "=== frames ==="; ls data/robomind_hl_early/frames 2>/dev/null | wc -l
rm -rf data/robomind_cache_early
