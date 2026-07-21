#!/bin/bash
#SBATCH --job-name=fr3_stageB
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/fr3_stageB_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache NLTK_DATA=/work/roboleon1295/nltk_data
export TMPDIR=/work/roboleon1295/tmp XDG_CACHE_HOME=/work/roboleon1295/.cache HOME=/work/roboleon1295
rm -rf data/robomind_hl_fr3 data/robomind_cache_fr3
uv run --no-sync python scripts/assemble_hl_data.py \
  --records_file data/robomind_records_fr3.json --labels_file data/robomind_labels_fr3.json \
  --out_dir data/robomind_hl_fr3 --cache_dir data/robomind_cache_fr3 \
  --fps 10 --sample-hz 1 --cleanup-parts
echo "ASM_EXIT=$?"
echo "=== manifest ==="; wc -l data/robomind_hl_fr3/manifest.jsonl 2>/dev/null
echo "=== frames ==="; ls data/robomind_hl_fr3/frames 2>/dev/null | wc -l
echo "=== output size ==="; du -sh data/robomind_hl_fr3 2>/dev/null
rm -rf data/robomind_cache_fr3
