#!/bin/bash
#SBATCH --job-name=vlash-proven
#SBATCH --partition=8gpus
#SBATCH --account=MST114563
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=800G
#SBATCH --time=16:00:00
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/milestone_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leon129506@gmail.com

# Short fine-tune with milestone checkpoints, to measure how fast sim competence
# degrades with DROID fine-tuning (the drift question Probe B could not answer --
# the 30K runs' intermediate checkpoints were rotated away by --keep-period=40000).
#
# 3000 steps, checkpoint every 500 => milestones at 500/1000/1500/2000/2500/2999.
# --keep-period=5000 makes orbax keep every one of them (step % 500 == 0).
#
# SPACE: each full checkpoint is 42G (12G params + 31G train_state) and /work has
# ~298G free. Six full checkpoints (252G) would leave ~46G, close enough to the
# quota to risk wekafs's silent-write-drop failure mode. The companion pruner
# (scripts/nchc/prune_train_state.sh, launched separately) deletes train_state/
# from each completed milestone as the next one lands; only params/ is needed to
# serve a checkpoint for evaluation. Peak footprint with the pruner: ~105G.
#
# Branch note: this run is the first to include the cherry-picked
# fix/droid-rlds-finetune commits (DeltaActions read-only copy + gcsfs download
# fallback) -- the canonical versions of fixes vlash-droid had re-derived.

set -x
module purge
module load miniconda3/26.1.1

export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
export TRITON_CACHE_DIR="$HOME/.triton" XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/.cache/huggingface

# GCS egress: tf.data streams DROID RLDS from gs://gresearch/robotics and the
# cluster's default CA bundle does not verify Google's chain.
CERTIFI_PEM=$(/work/roboleon1295/openpi-vlash/.venv/bin/python -c "import certifi; print(certifi.where())")
export CURL_CA_BUNDLE="$CERTIFI_PEM" SSL_CERT_FILE="$CERTIFI_PEM" GCS_CA_BUNDLE="$CERTIFI_PEM"

OPENPI_ROOT=${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}
cd "$OPENPI_ROOT"

CONFIG=${CONFIG:-pi05_droid_jointpos_statecond_d0}
EXP_NAME=${EXP_NAME:-milestone_3k}
BATCH_SIZE=${BATCH_SIZE:-32}

df -h /work/roboleon1295 | tail -1

XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --batch-size="$BATCH_SIZE" \
  --num-train-steps=20000 \
  --log-interval=100 \
  --save-interval=5000 \
  --keep-period=5000 \
  --fsdp-devices=4
EXIT_CODE=$?

echo "TRAIN_EXIT=$EXIT_CODE"
df -h /work/roboleon1295 | tail -1
ls -d "$OPENPI_ROOT/checkpoints/$CONFIG/$EXP_NAME"/*/ 2>/dev/null
exit "$EXIT_CODE"
