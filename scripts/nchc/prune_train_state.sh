#!/bin/bash
# Delete train_state/ from each milestone checkpoint as soon as orbax finalizes it.
#
# Each openpi checkpoint is 42G: 12G params/ + 31G train_state/. Serving an
# evaluation only reads params/ (+ assets/), so train_state is dead weight for
# this experiment's purpose -- but six milestones' worth would be 186G of it,
# enough to push /work toward the quota where wekafs starts silently dropping
# writes (documented failure mode: truncated logs, no error).
#
# Safety: only prunes dirs stamped _CHECKPOINT_METADATA (orbax commit marker;
# in-flight steps live in .orbax-checkpoint-tmp-* dirs), and only ever removes the
# train_state subdirectory -- params/ and assets/ are left intact so every
# milestone stays servable.
#
# Usage: nohup scripts/nchc/prune_train_state.sh <ckpt_root> > prune.log 2>&1 &
#        (exits when the sentinel file <ckpt_root>/.prune_done appears)

set -u
ROOT=${1:?usage: prune_train_state.sh <checkpoint_root>}

while true; do
    # Numeric checkpoint dirs, oldest first; orbax's in-flight
    # .orbax-checkpoint-tmp-* dirs never match this pattern.
    mapfile -t DIRS < <(find "$ROOT" -maxdepth 1 -type d -regextype posix-extended \
        -regex '.*/[0-9]+$' -printf '%f\n' 2>/dev/null | sort -n)

    # Prune EVERY finalized checkpoint, newest included: orbax writes to a tmp
    # dir and renames on completion, and stamps _CHECKPOINT_METADATA when the
    # step is committed -- so its presence means nothing is still writing there.
    # Training never reads back an earlier checkpoint, so dropping train_state
    # costs only resume-from-that-step, which this experiment does not need.
    for d in "${DIRS[@]}"; do
        TS="$ROOT/$d/train_state"
        [ -d "$TS" ] || continue
        [ -e "$ROOT/$d/_CHECKPOINT_METADATA" ] || { echo "SKIP $d (not finalized yet)"; continue; }
        SIZE=$(du -sh "$TS" 2>/dev/null | cut -f1)
        rm -rf "$TS" && echo "PRUNED $d/train_state ($SIZE) free=$(df -h "$ROOT" | tail -1 | awk '{print $4}')"
    done

    [ -f "$ROOT/.prune_done" ] && { echo "PRUNER_EXIT sentinel found"; break; }
    sleep 20
done
