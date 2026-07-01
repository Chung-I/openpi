#!/bin/bash
#SBATCH --job-name=mem_bakeoff_fail
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/bakeoff_fail_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
export NLTK_DATA=/work/roboleon1295/nltk_data
VV=/work/roboleon1295/vllm-venv
export PATH=$VV/bin:$PATH
export LD_LIBRARY_PATH=$(echo $VV/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
MODEL=Qwen/Qwen3.6-27B
PORT=$((8000 + SLURM_JOB_ID % 500 + 1))
BASE="http://localhost:${PORT}/v1"
# clean any stale vllm of ours on this node + free the port
pkill -u "$USER" -f "vllm serve" 2>/dev/null; sleep 5
rm -rf data/bakeoff_fail_shards data/bakeoff_fail.json data/bakeoff_fail.md data/bakeoff_fail_full.json data/bakeoff_fail_full.md
$VV/bin/vllm serve "$MODEL" --quantization fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 2048 --port "$PORT" \
  > logs/vllm_serve_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo VLLM_READY on $PORT; break; }
  kill -0 $VPID 2>/dev/null || { echo VLLM_DIED; tail -n 40 logs/vllm_serve_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done
uv run --no-sync python scripts/inject_failures.py --input data/agibot_episodes.json --output data/agibot_episodes_fail.json --max-fail-per-episode 2
uv run --no-sync python scripts/run_prompt_bakeoff.py \
  --episodes_file data/agibot_episodes_fail.json --prompts_dir prompts --report data/bakeoff_fail \
  --backend openai --base_url "$BASE" --model "$MODEL" --time-budget-min 60 --out_dir data/bakeoff_fail_shards
echo "GEN_EXIT=$?"
time uv run --no-sync python scripts/analyze_bakeoff.py \
  --episodes_file data/agibot_episodes_fail.json --prompts_dir prompts --shards_dir data/bakeoff_fail_shards \
  --report data/bakeoff_fail_full --base_url "$BASE" --model "$MODEL" \
  --judge_sample 120 --det_sample 30 --det_resamples 5 --concurrency 64 --n_perm 10000
echo "ANALYZE_EXIT=$?"
kill $VPID 2>/dev/null
echo "=== FAILURE-SLICE report ==="; cat data/bakeoff_fail_full.md
