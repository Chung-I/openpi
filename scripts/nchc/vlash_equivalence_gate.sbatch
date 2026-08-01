#!/bin/bash
# Task 7 step 2: shared-obs equivalence gate. 2K steps, single GPU, same seed, comparing
# pi05_droid_jointpos_vlash (arm A, batch 32) against pi05_droid_jointpos_vlash_shared
# (arm B, batch 8) -- both see 32 effective (obs, action-chunk) pairs/step, since the shared
# arm's VlashAllOffsets stacks all delta_max+1=4 branches per example (8 * 4 = 32). PASS =
# wandb loss-vs-step overlays track within noise band after warmup (~step 300+, relative gap
# <5% typical-loss). See docs/superpowers/plans/2026-08-01-vlash-droid-notes.md (Task 7
# section) for the recorded verdict.
#
#   sbatch scripts/nchc/vlash_equivalence_gate.sbatch [config_name] [batch_size] [exp_name]
#
# Submit BOTH arms (2 separate jobs, same seed via --seed=42 below):
#   sbatch scripts/nchc/vlash_equivalence_gate.sbatch pi05_droid_jointpos_vlash        32 gate_a
#   sbatch scripts/nchc/vlash_equivalence_gate.sbatch pi05_droid_jointpos_vlash_shared  8 gate_b
#
#SBATCH --job-name=vlash_gate
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
# 190G, not the full 200G/GPU fair-use cap -- leaves a little headroom. Sized for:
# shuffle_buffer_size=50000 (~15GB of images, see below) + checkpoint transient (~50-60GB,
# per the Task 3 12-step repro's OOM postmortem) + JAX/XLA overhead, comfortably under budget
# even with async checkpointing enabled (left at its default True here, unlike the Task 3
# repro's --no-enable-async-checkpointing -- this gate job intentionally tests the
# proven-in-production truncation-branch combination (shuffle buffer + async checkpointing
# both on) at a size between Task 3's 5000-timestep smoke buffer and full 250k production
# scale, to gather evidence for the headline run's --mem/flag choices).
#SBATCH --mem=190G
#SBATCH --time=02:00:00
#SBATCH --output=/work/roboleon1295/openpi-vlash/logs/vlash_gate_%j.log
set -x

CONFIG="${1:?Usage: sbatch vlash_equivalence_gate.sbatch <config_name> <batch_size> <exp_name>}"
BATCH_SIZE="${2:?batch_size required}"
EXP_NAME="${3:?exp_name required}"

# Repo root is overridable, mirroring vlash_droid_repro.sbatch's convention (this worktree has
# its own venv -- see docs/superpowers/plans/2026-08-01-vlash-droid-notes.md, Task 3 section).
OPENPI_ROOT="${OPENPI_ROOT:-/work/roboleon1295/openpi-vlash}"
cd "$OPENPI_ROOT" || { echo "FATAL: cd $OPENPI_ROOT failed"; exit 1; }

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

echo "GATE ARM: config=$CONFIG batch_size=$BATCH_SIZE exp_name=$EXP_NAME"

# --seed=42 explicit (matches TrainConfig's default, pinned here so both arms are provably
# run with the SAME seed even if the default ever changes) -- controls weight init and
# flow-matching noise/time draws. NOTE: the temporal-offset sampling itself
# (VlashTemporalOffset's np.random.default_rng()) is intentionally OS-entropy seeded per
# Task 2's design decision, so the two arms will NOT sample bit-identical offset sequences --
# the equivalence gate compares aggregate loss-vs-step behavior, not a bitwise-identical run.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/train.py "$CONFIG" \
  --exp-name="$EXP_NAME" \
  --overwrite \
  --seed=42 \
  --batch-size="$BATCH_SIZE" \
  --num-train-steps=2000 \
  --log-interval=20 \
  --save-interval=1000 \
  --data.shuffle-buffer-size=50000
EXIT_CODE=$?

echo "TRAIN_EXIT=$EXIT_CODE"
exit "$EXIT_CODE"
