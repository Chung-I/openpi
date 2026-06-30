import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from openpi.training.droid_rlds_dataset import _video_window_indices


def test_video_window_indices_strided_and_current_last():
    idx = _video_window_indices(traj_len=100, num_frames=6, stride=15)
    idx = idx.numpy() if isinstance(idx, tf.Tensor) else np.asarray(idx)
    assert idx.shape == (100, 6)
    # timestep 90: [15, 30, 45, 60, 75, 90], current (90) last
    np.testing.assert_array_equal(idx[90], [15, 30, 45, 60, 75, 90])
    # timestep 0: all clamped to 0 (repeat-oldest)
    np.testing.assert_array_equal(idx[0], [0, 0, 0, 0, 0, 0])
    # current frame is always the last column
    np.testing.assert_array_equal(idx[:, -1], np.arange(100))
