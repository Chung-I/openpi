#!/bin/bash
#SBATCH --job-name=fr3_assemble
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/fr3_assemble_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache
export NLTK_DATA=/work/roboleon1295/nltk_data
export VLLM_CACHE_ROOT=/work/roboleon1295/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/work/roboleon1295/.cache/torchinductor
export TRITON_CACHE_DIR=/work/roboleon1295/.cache/triton
export TMPDIR=/work/roboleon1295/tmp
export XDG_CACHE_HOME=/work/roboleon1295/.cache
export HOME=/work/roboleon1295
VV=/work/roboleon1295/vllm-venv
export PATH=$VV/bin:$PATH
export LD_LIBRARY_PATH=$(echo $VV/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':'):$LD_LIBRARY_PATH
MODEL=Qwen/Qwen3.6-27B
PORT=$((8000 + SLURM_JOB_ID % 500 + 1)); BASE="http://localhost:${PORT}/v1"
pkill -u "$USER" -f "vllm serve" 2>/dev/null; sleep 5
$VV/bin/vllm serve "$MODEL" --quantization fp8 --enable-prefix-caching \
  --gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 2048 --port "$PORT" \
  > logs/vllm_serve_${SLURM_JOB_ID}.log 2>&1 &
VPID=$!
for i in $(seq 1 240); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && { echo VLLM_READY; break; }
  kill -0 $VPID 2>/dev/null || { echo VLLM_DIED; tail -n 40 logs/vllm_serve_${SLURM_JOB_ID}.log; exit 1; }
  sleep 5
done
# Stage A: recursive v7 memory labels for all 3287 franka_3rgb episodes (sharded -> resumable)
uv run --no-sync python scripts/generate_memory_labels.py \
  --episodes_file data/robomind_episodes_fr3.json --backend openai --base_url "$BASE" --model "$MODEL" \
  --output data/robomind_labels_fr3.json --generation_mode recursive --disable_thinking \
  --max_concurrency 64 --out_dir data/robomind_label_shards_fr3
echo "GEN_EXIT=$?"
pkill -u "$USER" -f "vllm serve" 2>/dev/null; sleep 5
# Stage B: streaming assemble with per-task archive cleanup (quota-safe: peak = one task)
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
