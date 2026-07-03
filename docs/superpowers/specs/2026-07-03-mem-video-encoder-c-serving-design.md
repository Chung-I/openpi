# MEM Video-Encoder Eval — Sub-project C: MEM→RoboLab DROID serving + NCHC↔cml18 tunnel

**Date:** 2026-07-03
**Status:** Design (approved for spec)
**Part of:** "Verify MEM video-encoder effectiveness (pi0.5, DROID → RoboLab)" — the third of five
sub-projects (A gate → B training → **C serving/tunnel** → D RoboLab eval → E wandb comparison).
**Depends on:** Sub-project B (complete) — two trained checkpoints at step 9999:
`/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k{6,1}_verify/pi0_mem_droid_k{6,1}_verify/9999`
(each with `params` 6.6 GB + `assets` norm-stats + `train_state`).

## Goal

Make each trained arm's checkpoint answer RoboLab's `Pi0DroidJointposClient` over the network, so
sub-project D can run Isaac-Sim evals on **cml18** against a model served on an **NCHC H200** —
serving **one arm at a time** on the same endpoint.

## Context and key findings (drive the design)

### F1 — The serving contract is standard pi0-DROID

RoboLab's `Pi0DroidJointposClient` (`RoboLab/policies/pi0_family/client.py`) packs a request with
exactly the keys openpi's DROID path expects and reads `response["actions"]`:

```python
{ "observation/exterior_image_1_left": <224x224>, "observation/wrist_image_left": <224x224>,
  "observation/joint_position": <7>, "observation/gripper_position": <1>, "prompt": <instruction> }
```

The client (via `robolab.eval.base_client.InferenceClient`) handles chunk caching and the
**open-loop horizon** (pi05 → 15: it executes 15 actions of a returned chunk before re-querying),
and thresholds the gripper dim in `_postprocess_chunk`. Our `action_horizon=16 ≥ 15`, so the chunk
is long enough. This is the same contract `pi05_droid` serves.

### F2 — Serve LL-only (the HL head is untrained on DROID)

`DroidInputs` (`droid_policy.py`) produces **no** subtask/memory fields, and DROID RLDS carries no
HL labels — so the B checkpoints trained the **low-level action head only**; `Pi0MEM`'s HL
(`predict_subtask_and_memory`) is untrained and would emit garbage. Serving must run
`sample_actions` with **empty subtask/memory + the DROID `prompt` as goal + the K video frames** —
exactly the training regime. Consequence: we do **not** use `MEMPolicy`'s HL orchestration at all;
the standard openpi `Policy.infer` path (which calls `module_jit(model.sample_actions)`,
`policy.py:64,94`) already does the right thing once the obs carries the `video_*` keys.

### F3 — A rolling K-frame video buffer is the only serving-side statefulness

RoboLab sends one single-frame observation per query (no `video_*` keys). The K=6 arm's `video_img`
encoder needs the last 6 frames; the K=1 arm needs 1. The server must buffer frames and synthesize
the four `observation/video_*` keys (`video_exterior_image_1_left`, `video_wrist_image_left`,
`video_joint_position`, `video_gripper_position`) that `DroidInputs` consumes (`droid_policy.py:52-110`).
Because the client re-queries every open-loop horizon (~15 sim steps ≈ 1 s at 15 Hz), consecutive
server queries are ~15 steps apart, which naturally matches the training `video_stride_frames=15`.

## Design

### Change 1 — `FrameBufferPolicy` (the only new openpi runtime code)

A stateful wrapper around a standard trained `Policy`, in a new
`src/openpi/policies/frame_buffer_policy.py`:

- **Construction:** `FrameBufferPolicy(policy: Policy, num_video_frames: int)` where `policy` is a
  `create_trained_policy(config, ckpt_dir)` result. Holds rolling `collections.deque(maxlen=K)` for
  exterior frames, wrist frames, joint_position, gripper_position.
- **`infer(obs: dict) -> dict`** (matches `BasePolicy.infer`):
  1. Append the current query's `observation/exterior_image_1_left`, `observation/wrist_image_left`,
     `observation/joint_position`, `observation/gripper_position` to the deques.
  2. Build the four `observation/video_*` keys by stacking the buffered frames into `[K, ...]`; when
     fewer than K are buffered, **left-pad by repeating the oldest** so the shape is always `[K, ...]`.
  3. Inject those four keys into a copy of `obs`, then return `self._policy.infer(obs_with_video)`.
- **`reset()`** clears the deques (available for future episode-boundary handling; see Risks).
- **K=1:** the deque holds 1 frame → the `video_*` keys carry the single current frame, matching the
  K=1 training path (A-validated). **K=6:** last 6 queries.
- No RTC, no HL. All model/transform/norm-stats machinery is inherited from the wrapped `Policy`.

### Change 2 — `scripts/serve_mem_droid.py`

Mirrors `scripts/serve_policy.py`: tyro `Args(config: str, ckpt_dir: str, port: int = 8000)`;
`policy = create_trained_policy(get_config(config), ckpt_dir)`;
`served = FrameBufferPolicy(policy, num_video_frames=config.model.num_video_frames)`;
`websocket_policy_server.WebsocketPolicyServer(served, host="127.0.0.1", port=port).serve_forever()`.
`num_video_frames` is read from the config so K is never hard-coded.

### Change 3 — NCHC serving job + tunnel (`scripts/nchc/serve_mem_droid_nano4.sbatch`)

`sbatch serve_mem_droid_nano4.sbatch <config> <ckpt_dir>` — 4 GPUs, `--partition=8gpus`,
`--time=4:00:00`, `--account=MST114563`, the proven nano4 env (`.venv` via `uv run --no-sync`,
`HF_HOME=/work/roboleon1295/huggingface`, `TMPDIR`/`JAX_COMPILATION_CACHE_DIR` on `/work`,
`CURL_CA_BUNDLE`=certifi, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`). It:

1. echoes `SLURMD_NODENAME` (the compute node, for the relay fallback);
2. starts `serve_mem_droid.py` on `127.0.0.1:$PORT` (background); waits for the websocket
   `/healthz`-style readiness (poll the port);
3. **Primary tunnel:** `ssh -N -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes
   -R $PORT:127.0.0.1:$PORT cml18` (compute node → cml18). Restart it in a loop if it drops.
4. holds until the job's `--time` expires or it is cancelled.

The **tunnel tier is chosen by a gating pre-test** (Change 4). If the primary is not viable, the job
runs server-only (no `ssh -R`) and the **dev-box relay** (Change 5) is used instead.

### Change 4 — Tunnel gating pre-test (decides primary vs fallback)

A tiny `--partition=dev --gres=gpu:1 --time=00:05:00` job runs, **from a compute node**:
`ssh -o BatchMode=yes -o ConnectTimeout=10 cml18 hostname`. Success ⇒ the **primary direct reverse
tunnel** (Change 3 step 3) works. Failure (compute→NTU/22 blocked) ⇒ use the **dev-box relay**
(Change 5). Recorded once; the serving sbatch then either includes or omits its `ssh -R` step.

### Change 5 — Dev-box relay fallback (`NCHC ↔ this box ↔ cml18`)

The proven `transfer_checkpoint.sh` topology (this machine already reaches both NCHC via a
persistent nano4 ControlMaster and cml18 via SSH). Orchestrated from **this dev box**, which the
controller runs on, via two backgrounded SSH processes:

1. **Pull** the compute-node server here through an internal jump:
   `ssh -N -L $PORT:127.0.0.1:$PORT -J nano4 <compute-node>` (nano4→compute-node is internal cluster
   SSH, available while our job holds that node; far more reliable than compute→internet). If
   ProxyJump to the compute node is disallowed, the serving job instead opens an internal reverse
   tunnel `ssh -R $PORT:127.0.0.1:$PORT nano4` and this box forwards `nano4:$PORT` locally.
2. **Push** to cml18: `ssh -N -R $PORT:127.0.0.1:$PORT cml18`.

RoboLab's side is **identical to the primary path** — it connects to `localhost:$PORT` on cml18
either way.

### Change 6 — RoboLab client wiring (no fork)

On cml18, launch RoboLab's eval with
`Pi0DroidJointposClient(remote_host="localhost", remote_port=$PORT, policy_variant="pi05")`.
Config/launch only; RoboLab code is untouched. `open_loop_horizon` stays 15.

## Data flow (serving one arm)

```
RoboLab (cml18) --ws--> localhost:$PORT (cml18)
   [ primary: ssh -R from compute node  |  fallback: this box bridges via -J nano4 + ssh -R cml18 ]
      --> 127.0.0.1:$PORT (NCHC compute node) = websocket_policy_server
            --> FrameBufferPolicy.infer(obs)
                 → push frame to K-deque → build observation/video_* [K,...]
                 → Policy.infer → DroidInputs → module_jit(Pi0MEM.sample_actions)  (LL; prompt=goal, empty HL, K video)
                 → DroidOutputs → { "actions": [16, 8] }
```

## Testing / definition of done

### Committed unit test (CPU, dummy/tiny MEM), `frame_buffer_policy_test.py`

- **T-C1 — buffer shapes + delegation.** Feed a stream of synthetic single-frame DROID obs dicts
  through `FrameBufferPolicy` wrapping a stub policy that records the obs it received; assert the
  injected `observation/video_*` keys have leading dim **K** for both K=1 and K=6, that fewer-than-K
  history left-pads (repeats oldest), and that the wrapped policy is called once per `infer`. With a
  real dummy `Pi0MEM` Policy: one `infer` returns finite actions of shape `[16, 8]`.

### On-NCHC smokes (operator/controller steps)

- **Local same-node smoke:** serve the K1 checkpoint on `127.0.0.1:$PORT`; from the same node, an
  `openpi_client` websocket request with a synthetic DROID obs returns a well-formed `[16,8]` chunk.
  (Verifies the standard `Policy` path drives `Pi0MEM.sample_actions` — the "shim?" risk.)
- **Tunnel gating pre-test** (Change 4) result recorded.
- **Tunnel smoke from cml18:** a synthetic `Pi0DroidJointposClient` request through the chosen tunnel
  returns actions.

### Done when

A `Pi0DroidJointposClient` on cml18 receives valid `[16,8]` action chunks from **each** arm's
step-9999 checkpoint served on NCHC (K6 then K1 on the same `$PORT`).

## Non-goals (later sub-projects)

- RoboLab simple-tier subset selection and eval runs (**D**).
- Parsing RoboLab output → wandb comparison (**E**).
- Inference-time RTC (`sample_actions_rtc` exists but is out of scope here).
- Any RoboLab code change; overlay-network (tailscale) tunneling (evaluated, dropped for SSH).

## Risks / open items

- **compute→NTU/22 blocked** ⇒ primary direct tunnel unavailable → dev-box relay (gated by Change 4;
  no wasted GPU time — the pre-test is a 5-min `dev` job).
- **`Pi0MEM.sample_actions` via the standard `Policy`** may need a small adapter (e.g. a
  `sample_kwargs` such as `num_steps`) → caught by the local same-node smoke before any tunnel work.
- **K=6 episode-start staleness:** with no reset signal in the websocket contract, the first ~K
  queries of each RoboLab episode carry stale buffered frames from the prior episode (~K×15 sim
  steps). Accepted for a "get-a-feel" eval; K=1 is unaffected. `FrameBufferPolicy.reset()` exists if
  D later wires an episode boundary.
- **Tunnel drops during a long eval** → `ssh` with `ServerAliveInterval=30 -o ExitOnForwardFailure`
  in a restart loop (autossh-style).
- **ProxyJump to a compute node disallowed** (relay path) → serving job opens an internal reverse
  tunnel to nano4 instead; this box forwards from there.

## Relation to the umbrella experiment

A proved the K=1 baseline; B produced the two checkpoints; **C** (this spec) makes them reachable
from cml18. **D** runs the RoboLab simple-tier subset (~30 min/eval) per arm against this endpoint.
**E** parses RoboLab output into wandb for the K=1-vs-K=6 comparison. Each is its own
spec → plan → implementation cycle.
