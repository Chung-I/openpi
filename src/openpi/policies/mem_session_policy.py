"""Session-keyed serving wrapper for a MEM DROID policy.

RoboLab runs many parallel envs; each is identified by a per-env ``session_id``
(carried in the request envelope). This policy keeps a per-session rolling buffer
of the last K single-frame DROID observations, synthesizes the ``observation/video_*``
keys the video encoder needs, and delegates to a shared stateless trained ``Policy``
(LL-only). Requests are routed on ``request["endpoint"]``.

HL-ready note: the DROID checkpoints have no trained HL, so this serves LL only via
the standard ``Policy.sample_actions`` path. To enable HL later (with an HL-trained
checkpoint), swap the shared ``Policy`` for a per-session ``MEMPolicy`` here — the
session buffer and routing stay identical.
"""

import collections

import numpy as np

from openpi_client import base_policy as _base_policy

# single-frame request key -> the video_* key DroidInputs consumes
_VIDEO_KEY_MAP = {
    "observation/exterior_image_1_left": "observation/video_exterior_image_1_left",
    "observation/wrist_image_left": "observation/video_wrist_image_left",
    "observation/joint_position": "observation/video_joint_position",
    "observation/gripper_position": "observation/video_gripper_position",
}


class MemSessionPolicy(_base_policy.BasePolicy):
    def __init__(self, policy, num_video_frames: int, max_sessions: int = 256):
        self._policy = policy
        self._k = int(num_video_frames)
        self._max_sessions = int(max_sessions)
        # OrderedDict as an LRU: session_id -> deque(maxlen=K) of frame-piece dicts.
        self._sessions: "collections.OrderedDict[str, collections.deque]" = collections.OrderedDict()

    def _buffer(self, sid: str) -> collections.deque:
        buf = self._sessions.get(sid)
        if buf is None:
            buf = collections.deque(maxlen=self._k)
            self._sessions[sid] = buf
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)  # evict least-recently-used
        else:
            self._sessions.move_to_end(sid)
        return buf

    def infer(self, obs: dict) -> dict:
        if obs.get("endpoint") == "reset":
            sids = obs.get("session_ids")
            if sids is None:
                self._sessions.clear()
            else:
                for sid in sids:
                    self._sessions.pop(sid, None)
            return {"ok": True}

        sid = obs["session_id"]
        buf = self._buffer(sid)
        buf.append({src: np.asarray(obs[src]) for src in _VIDEO_KEY_MAP})

        frames = list(buf)
        if len(frames) < self._k:  # left-pad by repeating the oldest
            frames = [frames[0]] * (self._k - len(frames)) + frames

        request = {k: v for k, v in obs.items() if k not in ("endpoint", "session_id")}
        for src, vkey in _VIDEO_KEY_MAP.items():
            request[vkey] = np.stack([f[src] for f in frames], axis=0)
        return self._policy.infer(request)

    def reset(self) -> None:
        self._sessions.clear()
