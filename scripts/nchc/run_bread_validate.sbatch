#!/bin/bash
#SBATCH --job-name=bread_validate
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/bread_validate_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
export NLTK_DATA=/work/roboleon1295/nltk_data
export TMPDIR=/work/roboleon1295/tmp
export XDG_CACHE_HOME=/work/roboleon1295/.cache
export HOME=/work/roboleon1295
rm -rf data/robomind_hl_bread data/robomind_cache_bread
uv run --no-sync python scripts/assemble_hl_data.py \
  --records_file data/robomind_records_bread.json --labels_file data/robomind_labels_bread.json \
  --out_dir data/robomind_hl_bread --cache_dir data/robomind_cache_bread \
  --fps 10 --sample-hz 1 --max-episodes 10
echo "ASM_EXIT=$?"
echo "=== manifest ==="; wc -l data/robomind_hl_bread/manifest.jsonl 2>/dev/null
echo "=== frames ==="; ls data/robomind_hl_bread/frames 2>/dev/null | wc -l
# free the transient 150GB extract
rm -rf data/robomind_cache_bread
