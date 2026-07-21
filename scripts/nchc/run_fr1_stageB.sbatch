#!/bin/bash
#SBATCH --job-name=fr1_stageB
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/fr1_stageB_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache NLTK_DATA=/work/roboleon1295/nltk_data
export TMPDIR=/work/roboleon1295/tmp XDG_CACHE_HOME=/work/roboleon1295/.cache HOME=/work/roboleon1295
rm -rf data/robomind_hl data/robomind_cache
uv run --no-sync python scripts/assemble_hl_data.py \
  --records_file data/robomind_records.json --labels_file data/robomind_labels.json \
  --out_dir data/robomind_hl --cache_dir data/robomind_cache \
  --fps 10 --sample-hz 1 --cleanup-parts
echo "ASM_EXIT=$?"
echo "=== manifest ==="; wc -l data/robomind_hl/manifest.jsonl 2>/dev/null
echo "=== frames ==="; ls data/robomind_hl/frames 2>/dev/null | wc -l
echo "=== output size ==="; du -sh data/robomind_hl 2>/dev/null
rm -rf data/robomind_cache
