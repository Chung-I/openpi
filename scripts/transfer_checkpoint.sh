#!/bin/bash
# Transfer a checkpoint from Nano5 (NCHC) to cml18 (CMLAB) via this machine as relay.
# Usage: ./scripts/transfer_checkpoint.sh <step_number>
# Example: ./scripts/transfer_checkpoint.sh 5000

set -euo pipefail

STEP="${1:?Usage: $0 <step_number>}"
NANO5_CKPT="/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_stream/pi0_mem_droid_stream/${STEP}"
CML18_CKPT="~/Codes/openpi/checkpoints/pi0_mem_droid_stream/${STEP}"
LOCAL_TMP="/tmp/ckpt_transfer_${STEP}"

echo "=== Checkpoint Transfer: step ${STEP} ==="
echo "  From: nano5:${NANO5_CKPT}"
echo "  To:   cml18:${CML18_CKPT}"

# Step 1: Download from Nano5 to local
echo "[1/3] Downloading from Nano5..."
mkdir -p "${LOCAL_TMP}"
rsync -avz --progress "nano5:${NANO5_CKPT}/" "${LOCAL_TMP}/"

# Step 2: Upload to cml18
echo "[2/3] Uploading to cml18..."
ssh cml18.csie.ntu.edu.tw "mkdir -p ${CML18_CKPT}"
rsync -avz --progress "${LOCAL_TMP}/" "cml18.csie.ntu.edu.tw:${CML18_CKPT}/"

# Step 3: Also transfer assets (norm stats) — needed for inference
NANO5_ASSETS="/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_stream/pi0_mem_droid_stream/${STEP}/assets"
CML18_ASSETS="~/Codes/openpi/checkpoints/pi0_mem_droid_stream/${STEP}/assets"
if ssh nano5 "test -d ${NANO5_ASSETS}"; then
    echo "[3/3] Transferring assets..."
    rsync -avz --progress "nano5:${NANO5_ASSETS}/" "${LOCAL_TMP}/assets/"
    rsync -avz --progress "${LOCAL_TMP}/assets/" "cml18.csie.ntu.edu.tw:${CML18_ASSETS}/"
else
    echo "[3/3] No assets directory in checkpoint (will use config defaults)"
fi

# Cleanup
rm -rf "${LOCAL_TMP}"

echo "=== Transfer complete ==="
echo "To serve: ssh cml18.csie.ntu.edu.tw 'cd ~/Codes/openpi && ./serve_mem.sh checkpoints/pi0_mem_droid_stream/${STEP}'"
