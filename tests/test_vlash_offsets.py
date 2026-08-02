import numpy as np
from openpi.transforms_vlash import VlashFixedOffset, VlashTemporalOffset, apply_offset


def _sample(H=4, dmax=2, adim=8):
    actions = np.zeros((H + dmax, adim), dtype=np.float32)
    actions[:, 0] = np.arange(H + dmax) + 1        # deltas 1,2,3,...
    actions[:, 7] = 0.1 * (np.arange(H + dmax) + 1)  # absolute gripper cmds
    state = np.zeros(adim, dtype=np.float32)
    state[0] = 10.0
    state[7] = 0.5
    return {"state": state, "actions": actions}


def test_offset_zero_identity():
    s = _sample()
    out = apply_offset(dict(s), delta=0, action_horizon=4)
    np.testing.assert_array_equal(out["actions"], s["actions"][:4])
    np.testing.assert_array_equal(out["state"], s["state"])


def test_offset_two_rollforward():
    s = _sample()
    out = apply_offset(dict(s), delta=2, action_horizon=4)
    np.testing.assert_array_equal(out["actions"], s["actions"][2:6])
    assert out["state"][0] == 10.0 + 1 + 2          # accumulated deltas
    assert abs(out["state"][7] - 0.2) < 1e-6         # gripper cmd at delta-1
    assert out["vlash_offset"] == 2


def test_transform_samples_in_range():
    tr = VlashTemporalOffset(delta_max=2, action_horizon=4)
    seen = set()
    for _ in range(200):
        out = tr(_sample())
        seen.add(int(out["vlash_offset"]))
        assert out["actions"].shape == (4, 8)
    assert seen == {0, 1, 2}


def test_fixed_offset_never_varies():
    """VlashFixedOffset (Task 7 per-offset val hook) forces the SAME delta every call --
    matches apply_offset(..., delta=2, ...) exactly, unlike VlashTemporalOffset's sampling."""
    tr = VlashFixedOffset(delta=2, action_horizon=4)
    for _ in range(5):
        out = tr(_sample())
        expected = apply_offset(_sample(), delta=2, action_horizon=4)
        np.testing.assert_array_equal(out["actions"], expected["actions"])
        np.testing.assert_array_equal(out["state"], expected["state"])
        assert out["vlash_offset"] == 2


def test_offset_transforms_noop_without_actions():
    """Serving applies data_transforms to observations only; the offset transforms must
    pass through instead of raising KeyError (DeltaActions carries the same guard)."""
    from openpi.transforms_vlash import VlashAllOffsets

    obs_only = {"state": np.zeros(8, dtype=np.float32)}
    for tr in (VlashTemporalOffset(delta_max=3, action_horizon=15), VlashAllOffsets(delta_max=3, action_horizon=15)):
        out = tr(dict(obs_only))
        assert "actions" not in out
        np.testing.assert_array_equal(out["state"], obs_only["state"])
