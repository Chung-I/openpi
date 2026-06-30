#!/bin/bash
#SBATCH --job-name=mem_prompt_bakeoff
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/bakeoff_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
VV=/work/roboleon1295/vllm-venv
export PATH=$VV/bin:$PATH
export LD_LIBRARY_PATH=$(echo $VV/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
MODEL=Qwen/Qwen3.6-27B
$VV/bin/vllm serve "$MODEL" --quantization fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 2048 --port 8000 \
  > logs/vllm_serve_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  curl -sf http://localhost:8000/health >/dev/null 2>&1 && { echo VLLM_READY; break; }
  kill -0 $VPID 2>/dev/null || { echo VLLM_DIED; tail -n 40 logs/vllm_serve_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done
uv run --no-sync python scripts/load_agibot_episodes.py --output data/agibot_episodes.json --max-episodes 1000 --min-subtasks 3
uv run --no-sync python scripts/run_prompt_bakeoff.py \
  --episodes_file data/agibot_episodes.json --prompts_dir prompts --report data/bakeoff \
  --backend openai --base_url http://localhost:8000/v1 --model "$MODEL" --time-budget-min 60
echo "BAKEOFF_EXIT=$?"
kill $VPID 2>/dev/null
echo "=== report ==="; cat data/bakeoff.md
