"""DelayedChunkExecutor (examples/libero/main_delay.py): protocol invariants.

Pure-python with a fake client -- no libero/robosuite stack, no model. Pins the
overlap arithmetic and the rtc/* request contract against the server's
alignment convention (policy.py: shift = rtc["executed"]).
"""

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "examples" / "libero"))

from main_delay import DelayedChunkExecutor  # noqa: E402

H = 10


class FakeClient:
    """Returns chunk r on request r: chunk[i] = 100*r + i (scalar actions)."""

    def __init__(self):
        self.requests = []  # captured elements

    def infer(self, element):
        r = len(self.requests)
        self.requests.append(dict(element))
        return {"actions": np.arange(H, dtype=np.float64)[:, None] + 100.0 * r}


def _run(executor, n_steps):
    return [float(executor.act({"observation/state": np.zeros(8)})[0]) for _ in range(n_steps)]


def test_sync_matches_stock_replan_protocol():
    client = FakeClient()
    ex = DelayedChunkExecutor(client, "sync", 0, 5, env_id=1)
    acts = _run(ex, 12)
    # chunk r executes r*100 + [0..4]; request every 5 steps
    assert acts == [0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 200, 201]
    assert len(client.requests) == 3
    assert not any(k.startswith("rtc/") for k in client.requests[0])


def test_delay_overlap_arithmetic():
    """d=2, E=5: cycle 0 executes c0[0:5]; every later cycle executes
    prev[5:7] then new[2:5] -- the committed in-flight overlap, then the new
    chunk from its own index d."""
    client = FakeClient()
    ex = DelayedChunkExecutor(client, "naive", 2, 5, env_id=1)
    acts = _run(ex, 15)
    assert acts[:5] == [0, 1, 2, 3, 4]  # first chunk: no previous, start at 0
    assert acts[5:10] == [5, 6, 102, 103, 104]  # c0[5:7] + c1[2:5]
    assert acts[10:15] == [105, 106, 202, 203, 204]  # c1[5:7] + c2[2:5]


def test_rtc_request_contract():
    client = FakeClient()
    ex = DelayedChunkExecutor(client, "rtc", 3, 5, env_id=42)
    _run(ex, 10)
    first, second = client.requests
    for el in (first, second):
        assert el["rtc/mode"] == "rtc"
        assert el["rtc/env_id"] == 42
        assert el["rtc/inference_delay"] == 3
        assert el["rtc/executed"] == 5  # server aligns prev chunk by this shift
    # pah only sendable once H is known (after the first response): H - E
    assert "rtc/prefix_attention_horizon" not in first
    assert second["rtc/prefix_attention_horizon"] == H - 5


def test_rtc_overlap_matches_naive():
    """RTC changes the server-side sampling, not the client-side execution
    slicing -- the executed indices must be identical to the naive arm."""
    naive = DelayedChunkExecutor(FakeClient(), "naive", 2, 5, env_id=1)
    rtc = DelayedChunkExecutor(FakeClient(), "rtc", 2, 5, env_id=1)
    assert _run(naive, 15) == _run(rtc, 15)


def test_guards():
    with pytest.raises(AssertionError):
        DelayedChunkExecutor(FakeClient(), "sync", 1, 5, env_id=1)  # sync is d0
    with pytest.raises(AssertionError):
        DelayedChunkExecutor(FakeClient(), "naive", 6, 5, env_id=1)  # d > E
    # chunk too short for sustained operation: H=10 < E=8 + d=3 (later cycles
    # would need prev[8:11]); the guard fires on the first request already
    ex = DelayedChunkExecutor(FakeClient(), "naive", 3, 8, env_id=1)
    with pytest.raises(AssertionError, match="too short"):
        ex.act({"s": 0})
