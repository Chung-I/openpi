#!/bin/bash
# Ablation training run for `pi05_droid_jointpos_statecond_d0`: isolates the text-encoded-state
# -> AdaRMS-state-embedding factor (Task 1's state_cond=True) from offset training (Task 2's
# VlashTemporalOffset) by training the SAME AdaRMS-conditioned architecture as
# pi05_droid_jointpos_vlash but ONLY ever at delta=0 (vlash_delta_max=0, no temporal-offset
# augmentation at all). Direct copy of scripts/nchc/vlash_droid_train.sbatch with only the
# config name, job name, wandb run name, and effective-pairs-per-step sizing note changed --
# see that file's comments for the full derivation of every #SBATCH/env setting below
# (--mem=800G, --fsdp-devices=4, CA-bundle GCS egress recipe, shuffle-buffer/async-checkpointing
# defaults, --keep-period=40000 quota reasoning) and
# docs/superpowers/plans/2026-08-01-vlash-droid-notes.md (Task 3/7 sections) for the underlying
# evidence.
#
#   sbatch scripts/nchc/statecond_d0_train.sbatch [config_name] [batch_size] [exp_name]
#
# Default config is `pi05_droid_jointpos_statecond_d0`, NON-shared (vlash_shared_obs=False):
# unlike pi05_droid_jointpos_vlash_shared (which stacks vlash_delta_max+1=4 branches per
# example), a shared-obs arm here would stack exactly ONE (identity, delta=0) branch --
# pointless duplication of the unshared path with none of the KV-broadcast compute saving
# (that saving comes from amortizing the prefix forward pass over MULTIPLE DIFFERENT delta
# branches, which don't exist at delta_max=0). So this run uses the plain (unshared)
# `Pi0.compute_loss` path directly, where `batch_size` IS the number of effective
# (obs, action-chunk) pairs per step (no 4x multiplier like the shared config).
# batch_size=32 total / 4 GPUs = 8/GPU -- 32 effective pairs/step, matching the headline
# shared-obs run's effective batch size for a fair comparison. Measured non-shared throughput
# on 1 GPU (Task 7 equivalence gate, arm A) was ~1.3s/step; expect roughly 0.7-0.9s/step on
# 4 GPUs by the same scaling the shared arm saw (0.42s/step on 1 GPU -> 0.22s/step on 4), i.e.
# ~6-7h for 30,000 steps -- comfortably inside the 24h cap.
#
#SBATCH --job-name=statecond_d0_headline
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
# Same sizing as vlash_droid_train.sbatch, same reasoning: the Task 7 equivalence-gate job
# measured ~184GB baseline overhead (checkpoint-save transient + JAX/XLA + tf.data) at
# --mem=190G/1-GPU before OOMing at the FINAL checkpoint save specifically; the full
# 250_000-timestep shuffle buffer adds another ~75GB, for an estimated ~260GB peak. --mem=800G
# leaves >3x margin and was confirmed sufficient across multiple checkpoint rotations on the
# actual 4xH200/FSDP headline run (job 228672).
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=leon129506@gmail.com
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/statecond_d0_headline_%j.log
set -x

CONFIG="${1:-pi05_droid_jointpos_statecond_d0}"
BATCH_SIZE="${2:-32}"
EXP_NAME="${3:-statecond_d0_headline}"

# Repo root is overridable, mirroring vlash_droid_repro.sbatch/vlash_droid_train.sbatch's
# convention (this worktree has its own venv -- see
# docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 3 section).
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# --- Quota check FIRST, before burning any GPU-hours: wekafs at full quota drops writes
# silently (Global Constraints), and a 30K-step/24h run is exactly the kind of job that could
# fill /work partway through if a prior task left it tight. Require >100G free, per the brief.
QUOTA_FREE_G=$(df --output=avail -BG /work/roboleon1295 | tail -1 | tr -dc '0-9')
echo "QUOTA_FREE_G=$QUOTA_FREE_G"
if [ -z "$QUOTA_FREE_G" ] || [ "$QUOTA_FREE_G" -lt 100 ]; then
  echo "FATAL: /work/roboleon1295 has ${QUOTA_FREE_G:-unknown}G free, need >100G. Not starting."
  df -h /work/roboleon1295
  exit 1
fi

# Keep caches off /home: it has a 100G quota and a full quota breaks runs mid-flight.
export HOME=/work/roboleon1295/jobhome
mkdir -p "$HOME/.triton" "$HOME/.cache"
if [ ! -f "$HOME/.netrc" ]; then
  echo "FATAL: $HOME/.netrc missing -- wandb would run unauthenticated. Copy it first."
  exit 1
fi
export TRITON_CACHE_DIR="$HOME/.triton"
export XDG_CACHE_HOME="$HOME/.cache"
export HF_HOME=/work/roboleon1295/hf_cache
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/roboleon1295/.cache/uv}"

export PYTHONUNBUFFERED=1

# TF's GCS client (tf.data streaming the DROID RLDS shards from gs://gresearch/robotics) needs
# a CA bundle -- see vlash_droid_repro.sbatch for the full explanation of why.
CA_BUNDLE="$(uv run --no-sync python -c 'import certifi; print(certifi.where())' 2>/dev/null)"
if [ ! -f "$CA_BUNDLE" ]; then
  echo "FATAL: could not locate a certifi CA bundle (got '$CA_BUNDLE'); TF GCS reads would fail."
  exit 1
fi
export CURL_CA_BUNDLE="$CA_BUNDLE"
export SSL_CERT_FILE="$CA_BUNDLE"
export GCS_CA_BUNDLE="$CA_BUNDLE"

mkdir -p logs

echo "STATECOND_D0 ABLATION RUN: config=$CONFIG batch_size=$BATCH_SIZE exp_name=$EXP_NAME"

# --keep-period=40000: exceeds num-train-steps (30000), so orbax's periodic "keep forever"
# milestones never fire -- keeps only ONE checkpoint on disk at steady state (~42GB measured
# for this exact model shape on the headline run), refreshed every --save-interval steps. Does
# NOT eliminate the old+new transient overlap during a rotation (mitigated by --mem headroom,
# not this flag).
#
# shuffle-buffer-size and enable-async-checkpointing are deliberately left at their config
# defaults (250_000 / True) -- see the #SBATCH --mem comment above for why that's safe here.
#
# --fsdp-devices=4: same fix as vlash_droid_train.sbatch's job-228648 postmortem --
# TrainConfig.fsdp_devices defaults to 1 (no FSDP), which would replicate the ENTIRE train_state
# (full finetune, no freeze_filter, ~2.3B params with full Adam state) onto every one of the 4
# GPUs instead of sharding it. sharding.DATA_AXIS = (BATCH_AXIS, FSDP_AXIS) means the training
# batch is always split across all devices regardless of this flag (per-GPU batch stays
# BATCH_SIZE/4 either way) -- only model/optimizer-state sharding changes.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --batch-size="$BATCH_SIZE" \
  --num-train-steps=30000 \
  --log-interval=100 \
  --save-interval=1000 \
  --keep-period=40000 \
  --fsdp-devices=4
EXIT_CODE=$?

echo "TRAIN_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
