#!/bin/bash
#SBATCH --job-name=robomind_assemble
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/work/roboleon1295/openpi/logs/robomind_%j.log
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
# Stage A: recursive v7 memory labels for all 161 episodes
uv run --no-sync python scripts/generate_memory_labels.py \
  --episodes_file data/robomind_episodes.json --backend openai --base_url "$BASE" --model "$MODEL" \
  --output data/robomind_labels.json --generation_mode recursive --disable_thinking \
  --max_concurrency 64 --out_dir data/robomind_label_shards
echo "GEN_EXIT=$?"
kill $VPID 2>/dev/null
# Stage B: assemble a 10-episode subset (CPU + HF fetch) to validate frames/fps/manifest
rm -rf data/robomind_hl data/robomind_cache
uv run --no-sync python scripts/assemble_hl_data.py \
  --records_file data/robomind_records.json --labels_file data/robomind_labels.json \
  --out_dir data/robomind_hl --cache_dir data/robomind_cache --fps 10 --sample-hz 1 --max-episodes 10
echo "ASM_EXIT=$?"
echo "=== manifest ==="; wc -l data/robomind_hl/manifest.jsonl 2>/dev/null; head -3 data/robomind_hl/manifest.jsonl 2>/dev/null
echo "=== frames ==="; ls data/robomind_hl/frames 2>/dev/null | wc -l; ls data/robomind_hl/frames 2>/dev/null | head -3
