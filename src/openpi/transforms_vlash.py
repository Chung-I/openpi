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

import dataclasses

import numpy as np

from openpi.transforms import DataDict
from openpi.transforms import DataTransformFn


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
        rng = np.random.default_rng()
        delta = int(rng.integers(0, self.delta_max + 1))
        data = apply_offset(data, delta=delta, action_horizon=self.action_horizon)
        if self.rng_key != "vlash_offset":
            data[self.rng_key] = data.pop("vlash_offset")
        return data
