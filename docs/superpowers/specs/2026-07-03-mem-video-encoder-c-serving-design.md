# MEM Video-Encoder Eval — Sub-project C: session-stateful MEM serving + tailscale (dev↔NCHC)

**Date:** 2026-07-03
**Status:** Design (approved for spec)
**Part of:** "Verify MEM video-encoder effectiveness (pi0.5, DROID → RoboLab)" — the third of five
sub-projects (A gate → B training → **C serving/tunnel** → D RoboLab eval → E wandb comparison).
**Depends on:** Sub-project B (complete) — two trained checkpoints at step 9999:
`/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k{6,1}_verify/pi0_mem_droid_k{6,1}_verify/9999`
(each `params` 6.6 GB + `assets` norm-stats + `train_state`).

## Goal

Serve each arm's checkpoint so RoboLab's Isaac-Sim eval (running **on this dev box's RTX 5090**)
can drive it, with the model on an **NCHC H200**. Serve **one arm at a time** on the same endpoint,
over a **tailscale** link (dev box ↔ NCHC compute node) — no cml18, no SSH relay.

## Context and key findings (drive the design)

### F1 — RoboLab is multi-env, per-env, and expects a *client subclass* (not core edits)

RoboLab eval runs **32–128 parallel envs** (Isaac throughput report). `robolab/eval/base_client.py`
`InferenceClient` is an explicit **subclass extension surface**: `infer(obs, instruction, *, env_id)`
is per-env, the base owns per-env chunk caches (`_chunks`/`_counters`) and the **open-loop horizon**
(pi05 → 15), and `reset(env_id)` is **already called per episode** (`episode.py:190`). Existing policy
clients live in `policies/<family>/client.py` (`pi0_family`, `cosmos3`, `gr00t`, `dreamzero`). So the
integration is a **new client subclass**, not a fork of RoboLab core.

Because requests from 32 envs interleave with **no env identity on the wire by default**, a stateful
video policy needs a **per-env session id**. `DreamZeroClient` (RoboLab's world-model client) is the
precedent: it assigns a **`session_id` per `env_id`** (`"parallel envs must not share one or their
histories get mixed"`), sends it in an **`endpoint`-tagged request envelope**, and on `reset` sends an
`{"endpoint": "reset", "session_ids": [...]}` message to evict exactly those sessions.

### F2 — Serve LL-only for DROID, but architect for HL+LL (stateful)

DROID RLDS has no HL labels, so the B checkpoints trained the **low-level action head only**;
`Pi0MEM`'s HL (`predict_subtask_and_memory`) is untrained → serve with **empty subtask/memory + the
DROID `prompt` as goal + K video frames**. But the *full* MEM policy is inherently **stateful**:
`MEMPolicy` carries per-episode `memory`, `subtask`, `step_count` (HL every `hl_interval_steps`) and
`prev_action_chunk` (RTC). That state belongs **with the model** (only the server runs HL), so the
server is stateful and **session-keyed**; DROID simply disables HL (`hl_interval_steps=∞`). The design
is HL-ready without re-architecting.

### F3 — Transport: tailscale, dev↔NCHC (measured constraints)

Measured on a compute node (`25a-hgpn010`): **outbound SSH/22 to NTU is blocked** (`Connection timed
out`), and the SSH relay would ride the nano4 login master (~1.7 MB/s, and saturating it reaps the
master → 2FA re-auth). But **443 egress is fast** (GCS pulled 11.6 GB at ~380 MB/s). Since RoboLab runs
on **this dev box** (not cml18), the link is **dev ↔ NCHC**, and **tailscale over 443** gives a direct,
fast, uncapped path with no login master and no cml18:

- **Compute node:** `tailscaled --tun=userspace-networking` (no root; **inbound is forwarded to
  `localhost`**), `tailscale up --authkey=… --hostname=nchc-mem-serve --ephemeral`; serve on
  `127.0.0.1:$PORT`.
- **This dev box:** already a tailnet peer; RoboLab connects to `nchc-mem-serve:$PORT` (MagicDNS) or
  the `100.x` tailscale IP. Nothing else in the path.

**Validated end-to-end (2026-07-03):** a compute node (`25a-hgpn027`) joined the tailnet in userspace
mode over 443 as `nchc-mem-serve`; from this dev box, `tailscale ping` returned pong via **DERP(hkg)
~48 ms** (direct UDP NAT-blocked → 443 relay, as expected) and `curl http://nchc-mem-serve:18080/`
(and the `100.x` IP) returned HTTP 200 against a `127.0.0.1:18080` server — confirming userspace
**inbound→localhost forwarding** and **MagicDNS** both work. ~50 ms RTT is ample for ~1 Hz serving.

## Design

### Change 1 — The request envelope (explicit `endpoint`, per-env `session_id`)

A structured msgpack dict carried over openpi's stock `infer(dict) -> dict` transport (no bespoke wire
protocol; the transport is schema-free):

- **Infer:** `{ "endpoint": "infer", "session_id": <uuid>, "observation/exterior_image_1_left":
  <224×224>, "observation/wrist_image_left": <224×224>, "observation/joint_position": <7>,
  "observation/gripper_position": <1>, "prompt": <str> }` → response `{ "actions": <[16,8]> }`.
- **Reset:** `{ "endpoint": "reset", "session_ids": [<uuid>, …] | null }` (null = all) → `{ "ok": true }`.

### Change 2 — `MemSessionPolicy` (new openpi server-side policy; stateful, session-keyed)

New `src/openpi/policies/mem_session_policy.py` implementing `openpi_client.base_policy.BasePolicy`:

- Holds one **shared** trained model + DROID input/output transforms + norm-stats (built via
  `create_trained_policy(get_config(config), ckpt_dir)`; the wrapped `Policy` is reused for its
  transform/`sample_actions` plumbing), and a `num_video_frames` (K) read from the config.
- **`sessions: OrderedDict[str, deque(maxlen=K)]`** of per-session K-frame buffers (LRU-capped).
- **`infer(request: dict) -> dict`** routes on `request["endpoint"]`:
  - `"reset"`: `for sid in request["session_ids"] or list(sessions): sessions.pop(sid, None)`;
    return `{"ok": True}`.
  - `"infer"` (default): `sid = request["session_id"]`; get-or-create the session buffer; append the
    current frame; build the `observation/video_*` keys (`[K,…]`, left-pad by repeating the oldest
    when < K); inject them into the obs and **delegate to the shared `Policy.infer`** (its DROID input
    transforms + `Pi0MEM.sample_actions` handle the rest); return `{"actions": <[16,8]>}`.
- **HL-ready (LL-only realized):** for DROID the HL head is untrained, and `MEMPolicy(hl_interval=∞,
  use_rtc=False).step` reduces to `sample_actions` — so the LL path is served via the shared standard
  `Policy` (proven in B). To enable HL later (HL-trained checkpoint), swap the shared `Policy` for a
  **per-session `MEMPolicy`** here; the session buffer + routing are unchanged. Documented as the
  extension point, not built now (YAGNI).
- Idle-session guard: an optional LRU/TTL cap on `sessions` to bound memory if a reset is ever missed
  (evict least-recently-used beyond N sessions).

Served by a thin `scripts/serve_mem_session.py` (tyro `Args(config, ckpt_dir, port=8000)`):
`MemSessionPolicy` handed to the **stock** `websocket_policy_server.WebsocketPolicyServer(policy,
host="127.0.0.1", port=port)` — **no server/protocol change**; all statefulness lives in the policy.

### Change 3 — `MemDroidJointposClient` (new RoboLab client subclass)

New class in `RoboLab/policies/pi0_family/client.py`, subclassing `Pi0DroidJointposClient` (reuses its
`openpi_client` websocket transport, connection/retry, and gripper `_postprocess_chunk`):

- `_env_session_id: dict[int, str]`; `_extract_observation(raw, env_id)` lazily assigns
  `uuid4()` per `env_id`.
- `_pack_request(extracted, instruction)` adds `"session_id"` and `"endpoint": "infer"` to the pi0
  request keys.
- `reset(env_id)`: build the session-id list (this env, or all), send
  `{"endpoint": "reset", "session_ids": …}` via the existing `self.client.infer(...)`, clear
  `_env_session_id`, then `super().reset(env_id=env_id)`.
- `open_loop_horizon` stays 15 (our `action_horizon=16 ≥ 15`).

RoboLab core is untouched — this is the sanctioned `policies/<family>/` extension, alongside the
existing clients.

### Change 4 — NCHC serving job + tailscale (`scripts/nchc/serve_mem_session_nano4.sbatch`)

`sbatch serve_mem_session_nano4.sbatch <config> <ckpt_dir>` — 4 GPUs, `--partition=8gpus`,
`--time=4:00:00`, `--account=MST114563`, the proven nano4 env (`.venv` via `uv run --no-sync`,
`HF_HOME=/work/roboleon1295/huggingface`, `TMPDIR`/`JAX_COMPILATION_CACHE_DIR` on `/work`,
`CURL_CA_BUNDLE`=certifi, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`). Steps:

1. Stage the static tailscale binaries once under `/work/roboleon1295/tailscale/` (no root;
   `pkgs.tailscale.com/stable/#static`).
2. `tailscaled --tun=userspace-networking --state=/work/roboleon1295/ts.state
   --socket=/work/roboleon1295/ts.sock &`
3. `tailscale --socket=… up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)"
   --hostname=nchc-mem-serve --reset` (note: **`--ephemeral` is NOT an `up` flag** — ephemerality is
   a property of the auth key, set in the console; verified 2026-07-03). Wait for `tailscale status`
   = connected; print the `100.x` IP.
4. `uv run --no-sync python scripts/serve_mem_session.py <config> <ckpt_dir> --port $PORT` on
   `127.0.0.1:$PORT`; hold until `--time` / cancellation.

### Change 5 — Dev-box tailnet + RoboLab wiring (no fork)

This dev box is a tailnet peer (root here → normal TUN). RoboLab's eval launches
`MemDroidJointposClient(remote_host="nchc-mem-serve", remote_port=$PORT, policy_variant="pi05")`
(or the `100.x` tailscale IP). Config/launch only.

### Tailscale account setup (user does; I list it in the plan)

1. Admin console → **Generate auth key** → **Reusable + Ephemeral** (+ pre-approved/tag if ACLs are
   on). Save to `/work/roboleon1295/.tailscale_authkey`, `chmod 600` (never in git).
2. Confirm this dev box is on the tailnet (`tailscale status`); enable **MagicDNS** so
   `nchc-mem-serve` resolves (else use the printed IP).

## Data flow (serving one arm)

```
RoboLab eval (dev box, 32 envs on RTX 5090)
  → MemDroidJointposClient.infer(obs, instr, env_id)     [per-env session_id, endpoint="infer"]
     → openpi_client websocket ── tailscale/443 ──▶ nchc-mem-serve:$PORT (NCHC compute node)
          websocket_policy_server → MemSessionPolicy.infer(request)
             route "infer": session[sid].frames.append(frame) → build observation/video_* [K,…]
                → DROID input transforms → MEMPolicy.step (LL; HL off) → DROID output transforms
             → { "actions": [16,8] }
  (episode end) client.reset(env_id) → {"endpoint":"reset","session_ids":[…]} → evict those sessions
```

## Testing / definition of done

### Committed openpi unit test — `mem_session_policy_test.py` (CPU, dummy/tiny MEM)

- **T-C1 — routing + session isolation.** Drive `MemSessionPolicy` with a stub/dummy model: two
  interleaved `session_id`s each get an independent K-frame buffer (frames never cross sessions);
  fewer-than-K history left-pads; a `{"endpoint":"reset","session_ids":[sid_a]}` evicts only `sid_a`;
  an `"infer"` returns finite actions of shape `[16, 8]` for K=1 and K=6.

### On-NCHC + dev-box smokes (controller steps — I can run both, RoboLab is here)

- **Local same-node smoke (NCHC):** serve K1; from the same node an `openpi_client` request with the
  envelope returns a well-formed `[16,8]` chunk (verifies the `MEMPolicy.step` / transform path).
- **Tailscale reachability:** from this dev box, `nc`/`openpi_client` hits `nchc-mem-serve:$PORT` and
  gets metadata + an inference response.
- **End-to-end from RoboLab:** a short `MemDroidJointposClient` run (few envs, few steps) against the
  NCHC-served K1 checkpoint returns valid action chunks and `reset()` cleanly evicts sessions.

### Done when

`MemDroidJointposClient` on the dev box gets valid `[16,8]` action chunks (and clean per-episode
resets) from **each** arm's step-9999 checkpoint served on NCHC over tailscale (K6 then K1, same port).

## Non-goals (later sub-projects)

- RoboLab simple-tier subset selection and eval runs (**D**).
- Parsing RoboLab output → wandb comparison (**E**).
- Training-time RTC; enabling HL (needs an HL-trained checkpoint) — architecture is ready, not wired on.
- Any RoboLab **core** change (only the sanctioned `policies/` client subclass is added).

## Risks / open items

- **Tailscale userspace inbound lag:** inbound TCP can trail `tailscale up` by a few seconds → the
  sbatch waits for `tailscale status` = connected and the serving job self-checks before use.
- **`MEMPolicy.step` on the serving path** may need a small adapter (rng handling, `num_steps`) → the
  local same-node smoke catches it before any tailscale work.
- **Session lifecycle:** a missed `reset` would leak a session → the LRU/TTL cap bounds memory; the
  per-episode `reset(env_id)` (already called by RoboLab) is the primary cleanup.
- **K=6 stride at inference:** the client re-queries every ~15 sim steps (open-loop 15) → the per-env
  buffer captures ~stride-15 history, matching training `video_stride_frames=15`.
- **Ephemeral-key secrecy:** auth key lives only in `/work/.../.tailscale_authkey` (chmod 600), never
  committed.

## Relation to the umbrella experiment

A proved the K=1 baseline; B produced the two checkpoints; **C** (this spec) serves them to RoboLab on
the dev box over tailscale. **D** runs the RoboLab simple-tier subset (~30 min/eval) per arm against
this endpoint. **E** parses RoboLab output into wandb for the K=1-vs-K=6 comparison. Each is its own
spec → plan → implementation cycle.
