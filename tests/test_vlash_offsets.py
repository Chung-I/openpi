import numpy as np
from openpi.transforms_vlash import VlashTemporalOffset, apply_offset


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
