#!/bin/bash
# Delete train_state/ from completed milestone checkpoints while training runs.
#
# Each openpi checkpoint is 42G: 12G params/ + 31G train_state/. Serving an
# evaluation only reads params/ (+ assets/), so train_state is dead weight for
# this experiment's purpose -- but six milestones' worth would be 186G of it,
# enough to push /work toward the quota where wekafs starts silently dropping
# writes (documented failure mode: truncated logs, no error).
#
# Safety: never touches the newest checkpoint dir (orbax may still be writing
# it), skips any dir with an orbax tmp marker, and only ever removes the
# train_state subdirectory -- params/ and assets/ are left intact so every
# milestone stays servable.
#
# Usage: nohup scripts/nchc/prune_train_state.sh <ckpt_root> > prune.log 2>&1 &
#        (exits when the sentinel file <ckpt_root>/.prune_done appears)

set -u
ROOT=${1:?usage: prune_train_state.sh <checkpoint_root>}

while true; do
    # Numeric checkpoint dirs, oldest first; skip in-flight orbax tmp dirs.
    mapfile -t DIRS < <(find "$ROOT" -maxdepth 1 -type d -regextype posix-extended \
        -regex '.*/[0-9]+$' -printf '%f\n' 2>/dev/null | sort -n)

    # Leave the newest alone: training may still be writing it.
    for ((i = 0; i < ${#DIRS[@]} - 1; i++)); do
        TS="$ROOT/${DIRS[$i]}/train_state"
        if [ -d "$TS" ]; then
            SIZE=$(du -sh "$TS" 2>/dev/null | cut -f1)
            rm -rf "$TS" && echo "PRUNED ${DIRS[$i]}/train_state ($SIZE) free=$(df -h "$ROOT" | tail -1 | awk '{print $4}')"
        fi
    done

    [ -f "$ROOT/.prune_done" ] && { echo "PRUNER_EXIT sentinel found"; break; }
    sleep 60
done
