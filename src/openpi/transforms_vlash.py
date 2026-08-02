"""VLASH-on-DROID: temporal-offset transform with jointpos rollforward.

This transform samples a random temporal offset `delta` in `[0, delta_max]` and shifts the
action-chunk window forward by `delta` steps. Because the incoming `actions` are already in
delta-action space (dims 0..6, produced by `transforms.DeltaActions` upstream) while dim 7 is
the *absolute* gripper command, "rolling forward" the state to the new anchor at `t + delta`
requires two different update rules:

- dims 0..6 (joint-position deltas): accumulate (sum) the skipped deltas onto `state`.
- dim 7 (absolute gripper command): replace `state[7]` with the actual gripper command that
  was issued at the new anchor step, i.e. `actions[delta - 1, 7]`.

See `apply_offset` for the pure, testable core and `VlashTemporalOffset` for the
`transforms.DataTransformFn`-compatible wrapper that samples `delta` and applies it.
"""

import collections
import dataclasses
import logging

import numpy as np

from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn

# Task 3 repro-gate debug aid ONLY: prints a histogram of the first `_OFFSET_LOG_SAMPLE_SIZE`
# offsets sampled per process, then goes silent for the rest of the run. Purely diagnostic --
# does not affect the sampled `delta` or any training behavior. Safe to delete this block and
# its call site in `VlashTemporalOffset.__call__` with no effect on correctness.
_OFFSET_LOG_SAMPLE_SIZE = 32
_offset_log_samples: list[int] = []
_offset_log_state = {"done": False}


def _log_first_batch_offsets(delta: int) -> None:
    if _offset_log_state["done"]:
        return
    _offset_log_samples.append(delta)
    if len(_offset_log_samples) >= _OFFSET_LOG_SAMPLE_SIZE:
        histogram = dict(sorted(collections.Counter(_offset_log_samples).items()))
        logging.info(f"[vlash] first {len(_offset_log_samples)} sampled vlash_offset values: {histogram}")
        _offset_log_state["done"] = True


def apply_offset(sample: DataDict, *, delta: int, action_horizon: int) -> DataDict:
    """Shift `sample`'s action window forward by `delta` steps and roll the state forward.

    Args:
        sample: dict with `state` (state_dim,) and `actions` (action_horizon + delta_max,
            action_dim), where actions dims 0..6 are deltas and dim 7 is the absolute gripper
            command. Mutated in place and returned.
        delta: number of leading action steps to skip (0 <= delta <= delta_max).
        action_horizon: length of the returned action chunk (H).

    Returns:
        `sample` with `actions` sliced to `[delta : delta + action_horizon]`, `state` rolled
        forward to the new anchor, and `vlash_offset` set to `delta`.
    """
    state, actions = sample["state"], sample["actions"]

    if delta > 0:
        state = state.copy()
        state[:7] += actions[:delta, :7].sum(axis=0)
        state[7] = actions[delta - 1, 7]

    sample["state"] = state
    sample["actions"] = actions[delta : delta + action_horizon]
    sample["vlash_offset"] = delta
    return sample


def apply_all_offsets(sample: DataDict, *, delta_max: int, action_horizon: int) -> DataDict:
    """Emit EVERY offset branch instead of sampling one (shared-observation training).

    Reuses `apply_offset` per branch. The input `state` (state_dim,) / `actions`
    (action_horizon + delta_max, action_dim) are replaced by stacked per-branch arrays:
    `state` becomes [(delta_max + 1), state_dim] and `actions` becomes
    [(delta_max + 1), action_horizon, action_dim], where branch `delta` is exactly
    `apply_offset(..., delta=delta)`'s output (branch 0 is the identity). The stacked layout
    deliberately keeps the branches under the ORIGINAL keys so the downstream `Normalize` /
    `PadStatesAndActions` transforms (which broadcast over leading dims) apply identically to
    every branch; `SplitVlashBranches` later moves the stack to `vlash_states` and restores a
    scalar-batch `state`.
    """
    state, actions = sample["state"], sample["actions"]
    branches = [
        apply_offset({"state": state, "actions": actions}, delta=delta, action_horizon=action_horizon)
        for delta in range(delta_max + 1)
    ]
    sample["state"] = np.stack([branch["state"] for branch in branches], axis=0)
    sample["actions"] = np.stack([branch["actions"] for branch in branches], axis=0)
    return sample


@dataclasses.dataclass(frozen=True)
class VlashAllOffsets(DataTransformFn):
    """Deterministic all-offsets variant of `VlashTemporalOffset` for shared-obs training.

    Same pipeline position contract as `VlashTemporalOffset` (must run AFTER
    `transforms.DeltaActions`); instead of sampling a single `delta`, it emits all
    `delta_max + 1` branches stacked along a new leading axis. See `apply_all_offsets`.
    """

    delta_max: int
    action_horizon: int

    def __call__(self, data: DataDict) -> DataDict:
        # See VlashTemporalOffset.__call__: inference has no actions to branch over.
        if "actions" not in data:
            return data
        return apply_all_offsets(data, delta_max=self.delta_max, action_horizon=self.action_horizon)


@dataclasses.dataclass(frozen=True)
class SplitVlashBranches(DataTransformFn):
    """Splits the stacked all-offset branches into model-facing fields.

    Must run at the END of the model transforms -- after `Normalize` and
    `PadStatesAndActions` have been applied to the stacked `state` -- so that every branch
    state is normalized/padded identically. Moves the stacked states
    [(delta_max + 1), state_dim] to `vlash_states` (consumed by
    `Pi0.compute_loss_shared_obs` via `Observation.vlash_states`) and restores `state` to
    branch 0 (the delta=0 identity branch, i.e. the actual current state). `actions` stay
    stacked: the model trains on all branches at once.
    """

    def __call__(self, data: DataDict) -> DataDict:
        stacked = data["state"]
        data["vlash_states"] = stacked
        data["state"] = stacked[0]
        return data


@dataclasses.dataclass(frozen=True)
class VlashFixedOffset(DataTransformFn):
    """Applies `apply_offset` with a FORCED delta -- no sampling.

    Task 7 per-offset validation hook: building a held-out batch at each fixed delta in
    `{0, ..., delta_max}` reuses this exact same `apply_offset` core the training-time
    `VlashTemporalOffset` uses, just with `delta` forced instead of drawn from
    `Uniform{0, ..., delta_max}`. Must be inserted at the same pipeline position as
    `VlashTemporalOffset` (right after `transforms.DeltaActions`) -- see
    `RLDSDroidDataConfig.vlash_fixed_delta` in `training/config.py`.
    """

    delta: int
    action_horizon: int

    def __call__(self, data: DataDict) -> DataDict:
        return apply_offset(data, delta=self.delta, action_horizon=self.action_horizon)


@dataclasses.dataclass(frozen=True)
class VlashTemporalOffset(DataTransformFn):
    """Samples a random temporal offset and applies `apply_offset`.

    Must be inserted AFTER `transforms.DeltaActions` in the jointpos transform pipeline, since
    it assumes `actions[..., :7]` are already in delta space.

    Randomness: this transform draws a fresh `np.random.default_rng()` (OS-entropy seeded) on
    every call rather than deriving a seed from the sample. openpi's existing transforms
    (`transforms.py`) contain no RNG/seed convention to mirror — none of them are
    randomized — so there is nothing in the codebase to be consistent with. RLDS map-level
    determinism (the same example always producing the same output) is intentionally NOT
    preserved here: `vlash_offset` is a training-time augmentation that should vary across
    epochs for the same example, analogous to image augmentation transforms, so per-call
    fresh randomness is the desired behavior rather than a shortcut. See
    `docs/superpowers/plans/2026-08-01-vlash-droid-notes.md` (Task 2 section) for the full
    rationale.
    """

    delta_max: int
    action_horizon: int
    # Key under which the sampled offset is recorded in the output sample (for logging /
    # downstream consumption). `apply_offset` itself always writes to the literal key
    # "vlash_offset"; when `rng_key` differs from that default we rename the field afterward
    # so callers can pick a different output key without changing `apply_offset`'s contract.
    rng_key: str = "vlash_offset"

    def __call__(self, data: DataDict) -> DataDict:
        # No-op at inference: the serving pipeline applies data_transforms to observations
        # only, so there are no actions to offset and no future state to roll forward (the
        # eval client performs the rollforward itself). Mirrors the guard `DeltaActions`
        # already carries (`transforms.py`: `if "actions" not in data ...: return data`);
        # without it, serving a training config raises KeyError: 'actions'.
        if "actions" not in data:
            return data
        rng = np.random.default_rng()
        delta = int(rng.integers(0, self.delta_max + 1))
        _log_first_batch_offsets(delta)
        data = apply_offset(data, delta=delta, action_horizon=self.action_horizon)
        if self.rng_key != "vlash_offset":
            data[self.rng_key] = data.pop("vlash_offset")
        return data
