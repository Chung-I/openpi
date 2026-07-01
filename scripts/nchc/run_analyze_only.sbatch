#!/bin/bash
#SBATCH --job-name=mem_analyze
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/analyze_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
export NLTK_DATA=/work/roboleon1295/nltk_data
VV=/work/roboleon1295/vllm-venv
export PATH=$VV/bin:$PATH
export LD_LIBRARY_PATH=$(echo $VV/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
MODEL=Qwen/Qwen3.6-27B
# Reuse existing shards (do NOT clear/regenerate) — only live-fill judge/det + significance.
$VV/bin/vllm serve "$MODEL" --quantization fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 2048 --port 8000 \
  > logs/vllm_serve_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo VLLM_READY; break; }
  kill -0 $VPID 2>/dev/null || { echo VLLM_DIED; tail -n 40 logs/vllm_serve_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done
time uv run --no-sync python scripts/analyze_bakeoff.py \
  --episodes_file data/agibot_episodes.json --prompts_dir prompts --shards_dir data/bakeoff_shards \
  --report data/bakeoff_full --base_url http://localhost:8000/v1 --model "$MODEL" \
  --judge_sample 120 --det_sample 30 --det_resamples 5 --concurrency 64 --n_perm 10000
echo "ANALYZE_EXIT=$?"
kill $VPID 2>/dev/null
echo "=== FULL report ==="; cat data/bakeoff_full.md
