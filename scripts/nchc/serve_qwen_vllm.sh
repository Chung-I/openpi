#!/bin/bash
#SBATCH --job-name=qwen_vllm
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

# Serve Qwen3.6-27B (Dense) via vLLM (OpenAI-compatible) on one NCHC GPU.
# Point the generator at http://<node>:8000/v1 (run it on the same node or via srun).
# Tune --max-num-seqs / --max-num-batched-tokens / --gpu-memory-utilization with
# vLLM's auto_tune.sh if needed. Requires the Qwen weights (HF download uses HF_HOME token).
set -euo pipefail

MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"   # set QWEN_MODEL to the exact served id
PORT="${VLLM_PORT:-8000}"

vllm serve "$MODEL" \
    --quantization fp8 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --max-model-len 2048 \
    --port "$PORT"
