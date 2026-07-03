# MEM Video-Encoder Eval — Sub-project C (Serving + Tunnel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve each MEM DROID arm's checkpoint so RoboLab (on the dev-box RTX 5090) can drive it, with the model on an NCHC H200, over a validated tailscale link — one arm at a time.

**Architecture:** A **session-keyed** openpi serving policy (`MemSessionPolicy`) keeps a per-`session_id` rolling K-frame buffer, synthesizes the `observation/video_*` keys the video encoder needs, and delegates to a shared stateless trained `Policy` (LL-only). It rides the **stock** `websocket_policy_server` (statefulness lives in the policy; requests are an `endpoint`-tagged envelope). A RoboLab client subclass (`MemDroidJointposClient`) assigns a per-env `session_id`, sends the envelope, and evicts sessions on `reset(env_id)`. Transport = tailscale (compute node userspace `tailscaled` ↔ dev box), already validated.

**Tech Stack:** openpi (JAX/Flax), `openpi_client` websocket transport, RoboLab `InferenceClient`, tailscale (userspace), SLURM (NCHC).

## Global Constraints

- **Two repos:** openpi at `/home/chungyili/Codes/openpi`; RoboLab at `/home/chungyili/Codes/RoboLab`. Only a **new client subclass** is added to RoboLab (`policies/pi0_family/`) — no RoboLab core change.
- **Request envelope:** `{"endpoint": "infer", "session_id": <uuid>, "observation/exterior_image_1_left", "observation/wrist_image_left", "observation/joint_position", "observation/gripper_position", "prompt"}` → `{"actions": [16,8]}`; `{"endpoint": "reset", "session_ids": [...]|null}` → `{"ok": true}`.
- **LL-only for DROID** (HL untrained): serve `sample_actions` via the standard `Policy`; empty subtask/memory + DROID `prompt` as goal + K video frames. `num_video_frames` (K) is read from the train config (never hard-coded): K=6 (`pi0_mem_droid_k6_verify`), K=1 (`pi0_mem_droid_k1_verify`).
- **Video key map** (single-frame → video key): `exterior_image_1_left→video_exterior_image_1_left`, `wrist_image_left→video_wrist_image_left`, `joint_position→video_joint_position`, `gripper_position→video_gripper_position`.
- **Buffer:** `collections.deque(maxlen=K)` per session; when < K frames, **left-pad by repeating the oldest**; stacked to `[K, …]`.
- **Checkpoints:** `/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k{6,1}_verify/pi0_mem_droid_k{6,1}_verify/9999`.
- **Tailscale (validated):** static binaries at `/work/roboleon1295/tailscale/`; `tailscaled --tun=userspace-networking --state=… --socket=…`; `tailscale up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" --hostname=nchc-mem-serve --reset` (**no `--ephemeral` flag** — it's a key property); inbound forwards to `127.0.0.1`.
- **NCHC env (proven):** `--account=MST114563 --partition=8gpus`, `uv run --no-sync`, `HF_HOME=/work/roboleon1295/huggingface`, `TMPDIR`/`JAX_COMPILATION_CACHE_DIR` on `/work`, `CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.

---

### Task 1: `MemSessionPolicy` (session-keyed serving wrapper) + unit test

**Files:**
- Create: `src/openpi/policies/mem_session_policy.py`
- Test: `src/openpi/policies/mem_session_policy_test.py`

**Interfaces:**
- Consumes: `openpi_client.base_policy.BasePolicy` (abstract `infer(obs: dict) -> dict`); any object with `.infer(obs: dict) -> dict` as the wrapped `policy`.
- Produces: `MemSessionPolicy(policy, num_video_frames: int, max_sessions: int = 256)` with `.infer(request: dict) -> dict`. Used by Task 2.

- [ ] **Step 1: Write the failing test**

Create `src/openpi/policies/mem_session_policy_test.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/openpi/policies/mem_session_policy_test.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'openpi.policies.mem_session_policy'`.

- [ ] **Step 3: Implement `MemSessionPolicy`**

Create `src/openpi/policies/mem_session_policy.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest src/openpi/policies/mem_session_policy_test.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/openpi/policies/mem_session_policy.py src/openpi/policies/mem_session_policy_test.py
git commit -m "feat(mem): MemSessionPolicy — session-keyed K-frame serving wrapper (LL-only)"
```

---

### Task 2: `serve_mem_session.py` (openpi serve script)

**Files:**
- Create: `scripts/serve_mem_session.py`
- Test: (build smoke — no new test file)

**Interfaces:**
- Consumes: `MemSessionPolicy` (Task 1); `openpi.policies.policy_config.create_trained_policy(train_config, ckpt_dir) -> Policy`; `openpi.training.config.get_config(name) -> TrainConfig`; `openpi.serving.websocket_policy_server.WebsocketPolicyServer(policy, host, port)`.
- Produces: a CLI `python scripts/serve_mem_session.py --config <name> --ckpt-dir <dir> --port <p>` serving on `127.0.0.1:port`.

- [ ] **Step 1: Write the serve script**

Create `scripts/serve_mem_session.py`:

```python
"""Serve a MEM DROID checkpoint with per-session K-frame history for RoboLab.

Usage:
    uv run --no-sync python scripts/serve_mem_session.py \
        --config pi0_mem_droid_k6_verify \
        --ckpt-dir /work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k6_verify/pi0_mem_droid_k6_verify/9999 \
        --port 8000
"""

import dataclasses

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.policies.mem_session_policy import MemSessionPolicy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    ckpt_dir: str
    port: int = 8000
    host: str = "127.0.0.1"


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.ckpt_dir)
    served = MemSessionPolicy(policy, num_video_frames=train_config.model.num_video_frames)
    server = websocket_policy_server.WebsocketPolicyServer(served, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
```

- [ ] **Step 2: Build smoke (no model load / no GPU)**

Run: `uv run python -c "import ast; ast.parse(open('scripts/serve_mem_session.py').read()); print('SYNTAX OK')"`
Expected: `SYNTAX OK`.

Run: `uv run python -c "from openpi.policies.mem_session_policy import MemSessionPolicy; from openpi.training import config as c; assert c.get_config('pi0_mem_droid_k6_verify').model.num_video_frames == 6; print('CONFIG OK')"`
Expected: `CONFIG OK` (confirms the config name + K read path the script uses; no checkpoint/GPU needed).

- [ ] **Step 3: Commit**

```bash
git add scripts/serve_mem_session.py
git commit -m "feat(mem): serve_mem_session.py — websocket serving for MEM DROID (session-keyed)"
```

---

### Task 3: `MemDroidJointposClient` (RoboLab client subclass) + unit test

**Files:**
- Modify: `/home/chungyili/Codes/RoboLab/policies/pi0_family/client.py` (append a new class)
- Test: Create `/home/chungyili/Codes/RoboLab/policies/pi0_family/mem_client_test.py`

**Interfaces:**
- Consumes: `Pi0DroidJointposClient` (parent — its `_extract_observation`, `_pack_request`, `self.client.infer`, `reset`).
- Produces: `MemDroidJointposClient(remote_host, remote_port, policy_variant="pi05", ...)` that adds `session_id`+`endpoint` to requests and evicts sessions on `reset(env_id)`. Consumed by Tasks 4–5.

- [ ] **Step 1: Write the failing test**

Create `/home/chungyili/Codes/RoboLab/policies/pi0_family/mem_client_test.py`:

```python
import numpy as np

from policies.pi0_family.client import MemDroidJointposClient


class _FakeWs:
    """Stand-in for the websocket client; records reset envelopes."""

    def __init__(self):
        self.sent = []

    def infer(self, request):
        self.sent.append(request)
        return {"actions": np.zeros((16, 8), np.float32)}


def _make_client():
    # Bypass __init__'s network connect; wire a fake transport.
    c = MemDroidJointposClient.__new__(MemDroidJointposClient)
    from robolab.eval.base_client import InferenceClient
    InferenceClient.__init__(c)
    c.open_loop_horizon = 15
    c._env_session_id = {}
    c.client = _FakeWs()
    return c


def _extracted():
    return {
        "right_image": np.zeros((10, 10, 3), np.uint8),
        "wrist_image": np.zeros((10, 10, 3), np.uint8),
        "joint_position": np.zeros((7,), np.float32),
        "gripper_position": np.zeros((1,), np.float32),
    }


def test_pack_request_adds_session_and_endpoint():
    c = _make_client()
    c._env_session_id[0] = "sid-0"
    req = c._pack_request({**_extracted(), "session_id": "sid-0"}, "pick up cup")
    assert req["endpoint"] == "infer"
    assert req["session_id"] == "sid-0"
    assert req["prompt"] == "pick up cup"
    assert "observation/exterior_image_1_left" in req


def test_reset_sends_eviction_and_clears():
    c = _make_client()
    c._env_session_id = {0: "sid-0", 1: "sid-1"}
    c.reset(env_id=0)
    assert c.client.sent[-1] == {"endpoint": "reset", "session_ids": ["sid-0"]}
    assert 0 not in c._env_session_id and 1 in c._env_session_id
```

- [ ] **Step 2: Run to verify it fails**

Run (in RoboLab): `cd /home/chungyili/Codes/RoboLab && python -m pytest policies/pi0_family/mem_client_test.py -q`
Expected: FAIL — `ImportError: cannot import name 'MemDroidJointposClient'`.

- [ ] **Step 3: Implement the subclass**

Append to `/home/chungyili/Codes/RoboLab/policies/pi0_family/client.py`:

```python
import uuid


class MemDroidJointposClient(Pi0DroidJointposClient):
    """Pi0-DROID client for the MEM video policy.

    Assigns a per-env ``session_id`` so the server can keep independent per-env
    K-frame history, tags each request with ``endpoint="infer"``, and evicts the
    server-side session(s) on ``reset(env_id)`` (called per episode by RoboLab).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._env_session_id: dict[int, str] = {}

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        extracted = super()._extract_observation(raw_obs, env_id=env_id)
        if env_id not in self._env_session_id:
            self._env_session_id[env_id] = str(uuid.uuid4())
        extracted["session_id"] = self._env_session_id[env_id]
        return extracted

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["session_id"] = extracted_obs["session_id"]
        request["endpoint"] = "infer"
        return request

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            session_ids = list(self._env_session_id.values())
        elif env_id in self._env_session_id:
            session_ids = [self._env_session_id[env_id]]
        else:
            session_ids = []
        try:
            self.client.infer({"endpoint": "reset", "session_ids": session_ids or None})
        except Exception as e:  # server-side eviction is best-effort
            logger.warning("[MemDroidJointposClient] reset notify failed: %s", e)
        if env_id is None:
            self._env_session_id.clear()
        else:
            self._env_session_id.pop(env_id, None)
        super().reset(env_id=env_id)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /home/chungyili/Codes/RoboLab && python -m pytest policies/pi0_family/mem_client_test.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit (in the RoboLab repo)**

```bash
cd /home/chungyili/Codes/RoboLab
git add policies/pi0_family/client.py policies/pi0_family/mem_client_test.py
git commit -m "feat: MemDroidJointposClient — per-env session_id + endpoint envelope for MEM serving"
```

---

### Task 4: NCHC serving sbatch (`serve_mem_session_nano4.sbatch`) with tailscale

**Files:**
- Create: `scripts/nchc/serve_mem_session_nano4.sbatch`

**Interfaces:**
- Consumes: `scripts/serve_mem_session.py` (Task 2); the staged tailscale binaries + auth key + validated bring-up.
- Produces: a running server on `127.0.0.1:$PORT` on an NCHC compute node, reachable at `nchc-mem-serve:$PORT` on the tailnet.

- [ ] **Step 1: Write the sbatch**

Create `scripts/nchc/serve_mem_session_nano4.sbatch`:

```bash
#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --job-name=mem_serve
#SBATCH --output=/work/roboleon1295/openpi/logs/mem_serve_%j.log
set -x
CONFIG="${1:?usage: sbatch serve_mem_session_nano4.sbatch <config> <ckpt_dir> [port]}"
CKPT="${2:?need ckpt_dir}"
PORT="${3:-8000}"
TS=/work/roboleon1295/tailscale
SOCK=/work/roboleon1295/ts.sock
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

echo "NODE=$(hostname)"
# --- tailscale (userspace, no root; validated bring-up) ---
rm -f "$SOCK"
"$TS/tailscaled" --tun=userspace-networking --state=/work/roboleon1295/ts.state --socket="$SOCK" \
  >/work/roboleon1295/openpi/logs/mem_serve_tsd_${SLURM_JOB_ID}.log 2>&1 &
sleep 5
"$TS/tailscale" --socket="$SOCK" up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" \
  --hostname=nchc-mem-serve --reset
echo "TS_UP_EXIT=$?"
for i in $(seq 1 20); do
  IP=$("$TS/tailscale" --socket="$SOCK" ip -4 2>/dev/null | head -1)
  [ -n "$IP" ] && { echo "TAILSCALE_IP=$IP host=nchc-mem-serve"; break; }
  sleep 3
done

# --- serve (foreground; holds the job) ---
echo "SERVING config=$CONFIG port=$PORT"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/serve_mem_session.py \
  --config "$CONFIG" --ckpt-dir "$CKPT" --port "$PORT" --host 127.0.0.1
echo "SERVE_EXIT=$?"
"$TS/tailscale" --socket="$SOCK" logout 2>/dev/null
```

- [ ] **Step 2: Syntax check**

Run: `ssh nano4 'bash -n /work/roboleon1295/openpi/scripts/nchc/serve_mem_session_nano4.sbatch && echo BASH_OK'`
Expected: `BASH_OK`. (This file is edited/committed in the openpi repo; the operator copies it to `/work` via the same surgical-checkout path used for sub-project B, or it is already present there.)

- [ ] **Step 3: Commit**

```bash
git add scripts/nchc/serve_mem_session_nano4.sbatch
git commit -m "feat(mem): NCHC serve sbatch — MEM session server behind userspace tailscale"
```

---

### Task 5: End-to-end integration smokes (controller/operator; both machines here)

**Files:** none (verification only).

**Interfaces:** Consumes Tasks 1–4. This is the spec's Definition of Done.

- [ ] **Step 1: Local same-node smoke (NCHC)** — verifies `MemSessionPolicy` drives `Pi0MEM.sample_actions` on the real checkpoint.

Submit a short serving job for K1, then from the **same node** hit it with a synthetic DROID envelope. On NCHC (`/work/roboleon1295/openpi` up to date with Tasks 1–2 code):

```bash
sbatch scripts/nchc/serve_mem_session_nano4.sbatch pi0_mem_droid_k1_verify \
  /work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k1_verify/pi0_mem_droid_k1_verify/9999 8000
# then, on the SAME allocated node (srun --jobid=<id> --pty bash, or a second step):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run --no-sync python - <<'PY'
import numpy as np, uuid
from openpi_client import websocket_client_policy as w
c = w.WebsocketClientPolicy("127.0.0.1", 8000)
req = {"endpoint":"infer","session_id":str(uuid.uuid4()),
       "observation/exterior_image_1_left": np.zeros((224,224,3),np.uint8),
       "observation/wrist_image_left": np.zeros((224,224,3),np.uint8),
       "observation/joint_position": np.zeros((7,),np.float32),
       "observation/gripper_position": np.zeros((1,),np.float32),
       "prompt":"pick up the cup"}
r = c.infer(req)
print("actions shape:", np.asarray(r["actions"]).shape, "finite:", np.isfinite(np.asarray(r["actions"])).all())
PY
```

Expected: `actions shape: (16, 8) finite: True`. If `MemSessionPolicy`→`Policy.infer`→`Pi0MEM.sample_actions` needs a `sample_kwargs` (e.g. `num_steps`), add it via `create_trained_policy(..., sample_kwargs={"num_steps": 10})` in `serve_mem_session.py` and re-run (this is the "small shim" risk).

- [ ] **Step 2: Tailscale reachability from the dev box**

With the K1 serving job running (tailscale up as `nchc-mem-serve`), from **this dev box**:

```bash
tailscale ping nchc-mem-serve
XLA_PYTHON_CLIENT_MEM_FRACTION=0.1 uv run python - <<'PY'
import numpy as np, uuid
from openpi_client import websocket_client_policy as w
c = w.WebsocketClientPolicy("nchc-mem-serve", 8000)
req = {"endpoint":"infer","session_id":str(uuid.uuid4()),
       "observation/exterior_image_1_left": np.zeros((224,224,3),np.uint8),
       "observation/wrist_image_left": np.zeros((224,224,3),np.uint8),
       "observation/joint_position": np.zeros((7,),np.float32),
       "observation/gripper_position": np.zeros((1,),np.float32),
       "prompt":"pick up the cup"}
print("actions:", np.asarray(c.infer(req)["actions"]).shape)
print("reset ack:", c.infer({"endpoint":"reset","session_ids":None}))
PY
```

Expected: `pong from nchc-mem-serve …`; `actions: (16, 8)`; `reset ack: {'ok': True, ...}`.

- [ ] **Step 3: End-to-end from RoboLab's `MemDroidJointposClient`**

From the dev box, drive a few synthetic steps through the actual RoboLab client (isolates the client wiring without Isaac Sim):

```bash
cd /home/chungyili/Codes/RoboLab
python - <<'PY'
import numpy as np, torch
from policies.pi0_family.client import MemDroidJointposClient
c = MemDroidJointposClient(remote_host="nchc-mem-serve", remote_port=8000, policy_variant="pi05")
raw = {"image_obs": {"over_shoulder_left_camera": torch.zeros(1,224,224,3),
                     "wrist_cam": torch.zeros(1,224,224,3)},
       "proprio_obs": {"arm_joint_pos": torch.zeros(1,7), "gripper_pos": torch.zeros(1,1)}}
for step in range(20):
    out = c.infer(raw, "pick up the cup", env_id=0)
    assert out["action"].shape[-1] == 8, out["action"].shape
c.reset(env_id=0)
print("OK: 20 steps + reset via MemDroidJointposClient")
PY
```

Expected: `OK: 20 steps + reset via MemDroidJointposClient` (the 21st-step chunk refresh after `open_loop_horizon=15` proves a fresh server query; `reset` evicts the session).

- [ ] **Step 4: Repeat Step 3 against the K6 checkpoint**

Stop the K1 job; `sbatch` the same serving job with `pi0_mem_droid_k6_verify` + its 9999 dir on port 8000; re-run Step 3. Expected: identical `OK:` line (proves the K=6 buffer path serves too).

**Done when:** Steps 1–4 all pass — a `MemDroidJointposClient` on the dev box gets finite `[16,8]` chunks and clean resets from **both** arms served on NCHC over tailscale.

---

## Self-Review

**Spec coverage:**
- Change 1 (request envelope) → Global Constraints + Task 1 (`infer` routing) + Task 3 (`_pack_request`/`reset`). ✓
- Change 2 (`MemSessionPolicy` + `serve_mem_session.py`) → Tasks 1–2. ✓ (LL-only via standard `Policy`; the spec's "per-session `MEMPolicy`" is the HL-ready form — documented as the swap-in extension point in `mem_session_policy.py`'s docstring, not built now since DROID HL is untrained. Deviation noted.)
- Change 3 (`MemDroidJointposClient`) → Task 3. ✓
- Change 4 (NCHC serving sbatch + tailscale) → Task 4. ✓
- Change 5 (dev-box tailnet + wiring) → Task 5 Steps 2–4. ✓
- DoD (both arms, finite [16,8], clean reset) → Task 5. ✓
- LRU/TTL session cap → Task 1 (`max_sessions`, OrderedDict LRU). ✓

**Placeholder scan:** No TBD/TODO; `$PORT`/`<id>` are operator values in the ops steps only. All code steps carry complete code.

**Type consistency:** `MemSessionPolicy(policy, num_video_frames, max_sessions=256)`, `.infer(dict)->dict`, `_sessions`, `_VIDEO_KEY_MAP` consistent across Tasks 1–2. `MemDroidJointposClient._env_session_id`, envelope keys (`endpoint`/`session_id`/`session_ids`) consistent Tasks 1/3/5. `create_trained_policy(train_config, ckpt_dir)` matches its real signature.

**Note on the plan-vs-spec deviation (Task 1):** the spec Change 2 says "per-session `MEMPolicy`"; the plan implements the LL-only equivalent (session buffer + shared standard `Policy`) because DROID has no trained HL and `MEMPolicy(hl_interval=∞, use_rtc=False).step` reduces to `sample_actions`. This is a YAGNI simplification, with the `MEMPolicy` swap documented as the HL-ready extension point. If you want the literal `MEMPolicy` wiring now, that's a one-task addition — flag it.
