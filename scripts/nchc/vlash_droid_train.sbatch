#!/bin/bash
# Task 7 step 3: headline VLASH-on-DROID training run. 4xH200, 30K steps, wandb project
# vlash-droid, per-offset held-out validation loss (val_loss/delta_{0..3}) logged every 1K
# steps via openpi.training.vlash_eval.PerOffsetValLoss.
#
#   sbatch scripts/nchc/vlash_droid_train.sbatch [config_name] [batch_size] [exp_name]
#
# Defaults use the SHARED-obs arm: the Task 7 equivalence gate PASSED (see
# docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 7 section, for the full
# verdict + evidence -- param_norm trajectories matched to <0.001% relative gap throughout
# the run; the shared arm ALSO measured ~3x faster wall-clock per step than the non-shared arm
# at the same effective batch, since KV-broadcast shares the expensive prefix/SigLIP compute
# across the delta_max+1=4 branches). batch_size=8 total / 4 GPUs = 2/GPU;
# VlashAllOffsets stacks all 4 branches per example, so 2/GPU * 4 branches * 4 GPUs = 32
# effective (obs, action-chunk) pairs/step. If the non-shared arm is ever preferred instead
# (e.g. the shared path needs to be ruled out for a real training run), submit with:
#   sbatch scripts/nchc/vlash_droid_train.sbatch pi05_droid_jointpos_vlash 32 headline
# (batch_size=32 total / 4 GPUs = 8/GPU -- same 32 effective pairs/step).
#
#SBATCH --job-name=vlash_headline
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
# Sized from DIRECT measured evidence, not just the truncation branch's precedent: the Task 7
# equivalence-gate job (scripts/nchc/vlash_equivalence_gate.sbatch, --mem=190G,
# shuffle_buffer_size=50000, async checkpointing ON) trained pi05_droid_jointpos_vlash_shared
# to completion (all 2000 steps, healthy loss curve) but was `oom_kill`'d (exit 137,
# sacct MaxRSS=199227620K > the 190G request) at the VERY LAST checkpoint save -- specifically
# while finalizing step 1999's save and concurrently deleting the old step-1000 checkpoint
# (CheckpointManager keeps N+1 checkpoints briefly during any rotation, regardless of
# max_to_keep/keep_period -- this is NOT something --keep-period avoids, see below). That
# means baseline overhead (checkpoint-save transient + JAX/XLA + tf.data, EXCLUDING the
# shuffle buffer, which was only 50000 timesteps/~15GB here) already reached ~184GB for this
# exact model. The headline run's FULL 250_000-timestep shuffle buffer adds another ~75GB on
# top of that measured baseline, i.e. an estimated ~260GB peak -- so --mem is set well above
# that (not just above the truncation branch's 600G/arm precedent) for real margin against a
# repeat of this exact failure mode on a 24h job that would be very costly to lose near the end.
# Async checkpointing and the full shuffle buffer are still left at their config defaults
# (matches the truncation branch's proven recipe for actual training throughput); the mitigation
# here is more --mem, not disabling either flag.
#SBATCH --mem=800G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=leon129506@gmail.com
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/vlash_headline_%j.log
set -x

CONFIG="${1:-pi05_droid_jointpos_vlash_shared}"
BATCH_SIZE="${2:-8}"
EXP_NAME="${3:-headline}"

# Repo root is overridable, mirroring vlash_droid_repro.sbatch's convention (this worktree has
# its own venv -- see docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 3 section).
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

# --- Quota check FIRST, before burning any GPU-hours: wekafs at full quota drops writes
# silently (Global Constraints), and a 30K-step/24h run is exactly the kind of job that could
# fill /work partway through if a prior task left it tight (222G free as of the Task 7 gate,
# down from 274G at Task 0 -- keep watching this trend). Require >100G free, per the brief.
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

echo "HEADLINE RUN: config=$CONFIG batch_size=$BATCH_SIZE exp_name=$EXP_NAME"

# --keep-period=40000: exceeds num-train-steps (30000), so orbax's periodic "keep forever"
# milestones (default keep_period=5000 -> permanent checkpoints at 5000/10000/.../25000 PLUS
# the ever-refreshed max_to_keep=1 latest) never fire. Measured checkpoint size is ~42GB
# (Task 7 gate job, both params + train_state items) -- 5+ permanent milestones would risk the
# /work quota (168G free as of the Task 7 gate, down from 222G at gate start and 274G at
# Task 0 -- watch this trend and prune old experiment checkpoints before resubmitting if it
# keeps falling), so this keeps only ONE checkpoint on disk at steady state, refreshed every
# --save-interval steps. NOTE: this does NOT eliminate the old+new transient overlap during a
# rotation (CheckpointManager finishes writing the new checkpoint before deleting the old one,
# regardless of keep_period) -- that risk is mitigated by the --mem headroom above, not this
# flag; --keep-period only bounds STANDING disk usage between rotations.
#
# shuffle-buffer-size and enable-async-checkpointing are deliberately left at their config
# defaults (250_000 / True) -- see the #SBATCH --mem comment above for why that's safe here.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --batch-size="$BATCH_SIZE" \
  --num-train-steps=30000 \
  --log-interval=100 \
  --save-interval=1000 \
  --keep-period=40000
EXIT_CODE=$?

echo "TRAIN_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
