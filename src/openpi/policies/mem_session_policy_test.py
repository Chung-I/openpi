import numpy as np

from openpi.policies.mem_session_policy import MemSessionPolicy


class _StubPolicy:
    """Records the obs dict it was given; returns a fixed action chunk."""

    def __init__(self):
        self.seen = []

    def infer(self, obs):
        self.seen.append(obs)
        return {"actions": np.zeros((16, 8), np.float32)}


def _frame(val):
    return {
        "observation/exterior_image_1_left": np.full((4, 4, 3), val, np.uint8),
        "observation/wrist_image_left": np.full((4, 4, 3), val, np.uint8),
        "observation/joint_position": np.full((7,), val, np.float32),
        "observation/gripper_position": np.full((1,), val, np.float32),
        "prompt": "do the task",
    }


def test_session_isolation_padding_reset_and_shape():
    stub = _StubPolicy()
    sp = MemSessionPolicy(stub, num_video_frames=6)

    # Session A: two frames (values 1 then 2).
    sp.infer({"endpoint": "infer", "session_id": "A", **_frame(1)})
    out = sp.infer({"endpoint": "infer", "session_id": "A", **_frame(2)})

    v = stub.seen[-1]["observation/video_exterior_image_1_left"]
    assert v.shape == (6, 4, 4, 3)          # K-stacked
    assert v[-1, 0, 0, 0] == 2              # newest frame last
    assert v[0, 0, 0, 0] == 1               # left-padded with the oldest (=1)
    vs = stub.seen[-1]["observation/video_joint_position"]
    assert vs.shape == (6, 7)
    assert out["actions"].shape == (16, 8)

    # Session B is independent — never sees A's frames.
    sp.infer({"endpoint": "infer", "session_id": "B", **_frame(9)})
    vb = stub.seen[-1]["observation/video_exterior_image_1_left"]
    assert set(np.unique(vb).tolist()) == {9}

    # Reset only A.
    ack = sp.infer({"endpoint": "reset", "session_ids": ["A"]})
    assert ack == {"ok": True}
    assert "A" not in sp._sessions and "B" in sp._sessions

    # endpoint/session_id are stripped before delegating; prompt is preserved.
    assert "endpoint" not in stub.seen[-1] and "session_id" not in stub.seen[-1]
    assert stub.seen[-1]["prompt"] == "do the task"


def test_k1_single_frame():
    stub = _StubPolicy()
    sp = MemSessionPolicy(stub, num_video_frames=1)
    sp.infer({"endpoint": "infer", "session_id": "X", **_frame(5)})
    v = stub.seen[-1]["observation/video_exterior_image_1_left"]
    assert v.shape == (1, 4, 4, 3) and v[0, 0, 0, 0] == 5


def test_reset_all():
    sp = MemSessionPolicy(_StubPolicy(), num_video_frames=2)
    sp.infer({"endpoint": "infer", "session_id": "A", **_frame(1)})
    sp.infer({"endpoint": "infer", "session_id": "B", **_frame(1)})
    sp.infer({"endpoint": "reset", "session_ids": None})
    assert len(sp._sessions) == 0
