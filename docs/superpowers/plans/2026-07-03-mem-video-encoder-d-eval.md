# MEM Video-Encoder Eval — Sub-project D (3-arm RoboLab eval) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Isaac-Sim evaluation of three arms — vanilla pi0.5, MEM K=1, MEM K=6 — on 5 simple-tier benchmark tasks (32 envs, adaptive-to-200/task) and record per-task success + credible intervals, so we can sanity-check MEM against vanilla pi0.5 and measure the video-encoder effect.

**Architecture:** Two small artifacts + running the experiment. **`run_mem.py`** (RoboLab) is `run.py` with `MemDroidJointposClient` swapped in. **`serve_pi05_droid_nano4.sbatch`** (openpi) reuses C's validated userspace-tailscale bring-up + the stock `serve_policy.py --env DROID`. The bulk is 3 serve→eval cycles on the same `nchc-mem-serve:8000` endpoint, driven from the dev-box RTX 5090, producing three `output/<ts>_<arm>/` dirs.

**Tech Stack:** RoboLab (Isaac Sim/Lab), openpi serving, tailscale (from C), SLURM (NCHC).

## Global Constraints

- **Two repos:** openpi `/home/chungyili/Codes/openpi`; RoboLab `/home/chungyili/Codes/RoboLab` (runs on the dev box; branch `investigate/isaac-throughput-knobs`, where C's `MemDroidJointposClient` lives).
- **3 arms, identical eval config each:** `--task-dirs benchmark --tag simple` restricted to the **same 5** simple tasks; `--num-envs 32 --num-episodes-adaptive 200 --ci-pp-width 0.14`; native resolution; **no `--enable-subtask`**; `--video-mode fail`; distinct `--output-folder-name <arm>`; `--policy pi05`; `--remote-host nchc-mem-serve --remote-port 8000`.
- **Serve one arm at a time** on `nchc-mem-serve:8000`: vanilla via `serve_policy.py --env DROID` + RoboLab `run.py`; MEM K1/K6 via `serve_mem_session.py <config> <ckpt>` + `run_mem.py`.
- **Checkpoints:** vanilla = `gs://openpi-assets/checkpoints/pi05_droid` (served by `--env DROID`); MEM = `/work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k{1,6}_verify/pi0_mem_droid_k{1,6}_verify/9999`.
- **Tailscale (validated in C):** binaries `/work/roboleon1295/tailscale/`; `tailscaled --tun=userspace-networking …`; `tailscale up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" --hostname=nchc-mem-serve --reset` (**no `--ephemeral`**); inbound→localhost.
- **NCHC env (proven):** `--account=MST114563 --partition=8gpus`, `uv run --no-sync`, `HF_HOME=/work/roboleon1295/huggingface`, `TMPDIR`/`JAX_COMPILATION_CACHE_DIR` on `/work`, `CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.
- **RoboLab run:** `cd /home/chungyili/Codes/RoboLab && uv run --no-sync python policies/pi0_family/<script>.py …` (isaac deps pre-synced).

---

### Task 1: `run_mem.py` — MEM eval entry (RoboLab repo)

**Files:**
- Create: `/home/chungyili/Codes/RoboLab/policies/pi0_family/run_mem.py`

**Interfaces:**
- Consumes: `MemDroidJointposClient` (from C, `policies/pi0_family/client.py`); `robolab.eval.runner.{add_common_eval_args, run_evaluation}`; `auto_register_droid_envs`.
- Produces: a CLI `uv run --no-sync python policies/pi0_family/run_mem.py --policy pi05 --remote-host nchc-mem-serve …` that evaluates the MEM policy.

- [ ] **Step 1: Write `run_mem.py`** (this is `run.py` with the client swapped — full file, no placeholders)

Create `/home/chungyili/Codes/RoboLab/policies/pi0_family/run_mem.py`:

```python
"""Evaluate the MEM video policy (pi0.5-MEM) on RoboLab.

Identical to policies/pi0_family/run.py but drives MemDroidJointposClient, which
adds a per-env session_id + endpoint envelope so the openpi MemSessionPolicy server
keeps per-env K-frame history. Serve the model with openpi scripts/serve_mem_session.py.
"""

import argparse
import sys  # noqa: F401
import traceback  # noqa: F401

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]

parser = argparse.ArgumentParser(description="Evaluate a pi0.5-MEM policy backend.")
parser.add_argument("--policy", choices=PI0_VARIANTS, default="pi05",
                    help="Which pi0-family variant to evaluate (default: pi05).")
parser.add_argument("--remote-host", "--remote_host", type=str, default="nchc-mem-serve",
                    help="Remote host for policy server (default: nchc-mem-serve).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Remote port for policy server (default: 8000).")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help="Full WebSocket URI for policy server. Overrides host/port when set.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help="Actions executed per predicted chunk before re-querying (default: per-variant).")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                    help="Debug output (default: False).")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                    help="Enable proprio image data recording (default: False).")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help="Sample a random non-default background per task at registration time.")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling.")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from policies.pi0_family.client import MemDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

auto_register_droid_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    randomize_background=args_cli.randomize_background,
    background_seed=args_cli.background_seed,
)


def make_client(args: argparse.Namespace) -> MemDroidJointposClient:
    kwargs = dict(
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        remote_uri=args.remote_uri,
        open_loop_horizon=args.open_loop_horizon,
        policy_variant=args.policy,
    )
    return MemDroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


def main() -> None:
    run_evaluation(args_cli, policy=args_cli.policy, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it compiles and differs from run.py only in the client**

Run: `cd /home/chungyili/Codes/RoboLab && python3 -m py_compile policies/pi0_family/run_mem.py && echo COMPILE_OK`
Expected: `COMPILE_OK`.

Run: `cd /home/chungyili/Codes/RoboLab && diff <(grep -oE "Client" policies/pi0_family/run.py | sort -u) <(grep -oE "MemDroidJointposClient|Pi0DroidJointposClient" policies/pi0_family/run_mem.py | sort -u); grep -c "MemDroidJointposClient" policies/pi0_family/run_mem.py`
Expected: `run_mem.py` references `MemDroidJointposClient` (count ≥ 2: import + make_client) and NOT `Pi0DroidJointposClient`.

(Full behavior is verified by the Task 3 1-env smoke — importing the module launches Isaac Sim, so it can't be unit-tested.)

- [ ] **Step 3: Commit (RoboLab repo)**

```bash
cd /home/chungyili/Codes/RoboLab
git add policies/pi0_family/run_mem.py
git commit -m "feat: run_mem.py — evaluate pi0.5-MEM via MemDroidJointposClient"
```

---

### Task 2: Vanilla-pi0.5 serve sbatch (openpi repo)

**Files:**
- Create: `scripts/nchc/serve_pi05_droid_nano4.sbatch`

**Interfaces:**
- Consumes: C's tailscale bring-up; the stock `scripts/serve_policy.py` (`--env DROID` → serves `pi05_droid`).
- Produces: a running vanilla-pi0.5 server on `127.0.0.1:$PORT`, reachable at `nchc-mem-serve:$PORT`.

- [ ] **Step 1: Write the sbatch**

Create `scripts/nchc/serve_pi05_droid_nano4.sbatch`:

```bash
#!/bin/bash
#SBATCH --account=MST114563
#SBATCH --partition=8gpus
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --job-name=pi05_serve
#SBATCH --output=/work/roboleon1295/openpi/logs/pi05_serve_%j.log
set -x
PORT="${1:-8000}"
TS=/work/roboleon1295/tailscale
SOCK=/work/roboleon1295/ts.sock
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/huggingface
export TMPDIR=/work/roboleon1295/.jaxtmp; mkdir -p "$TMPDIR"
export JAX_COMPILATION_CACHE_DIR=/work/roboleon1295/.jaxcache
export CURL_CA_BUNDLE=/work/roboleon1295/openpi/.venv/lib/python3.11/site-packages/certifi/cacert.pem
export SSL_CERT_FILE="$CURL_CA_BUNDLE"

echo "NODE=$(hostname)"
rm -f "$SOCK"
"$TS/tailscaled" --tun=userspace-networking --state=/work/roboleon1295/ts.state --socket="$SOCK" \
  >/work/roboleon1295/openpi/logs/pi05_serve_tsd_${SLURM_JOB_ID}.log 2>&1 &
sleep 5
"$TS/tailscale" --socket="$SOCK" up --authkey="$(cat /work/roboleon1295/.tailscale_authkey)" \
  --hostname=nchc-mem-serve --reset
echo "TS_UP_EXIT=$?"
for i in $(seq 1 20); do
  IP=$("$TS/tailscale" --socket="$SOCK" ip -4 2>/dev/null | head -1)
  [ -n "$IP" ] && { echo "TAILSCALE_IP=$IP host=nchc-mem-serve"; break; }
  sleep 3
done

echo "SERVING vanilla pi05_droid port=$PORT"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 uv run --no-sync python scripts/serve_policy.py --env DROID --port "$PORT"
echo "SERVE_EXIT=$?"
"$TS/tailscale" --socket="$SOCK" logout 2>/dev/null
```

- [ ] **Step 2: Syntax check + confirm serve path**

Run: `bash -n scripts/nchc/serve_pi05_droid_nano4.sbatch && echo BASH_OK`
Expected: `BASH_OK`.

Run: `grep -n "serve_policy.py --env DROID" scripts/nchc/serve_pi05_droid_nano4.sbatch && grep -n "hostname=nchc-mem-serve --reset" scripts/nchc/serve_pi05_droid_nano4.sbatch`
Expected: both present; the tailscale `up` line has no `--ephemeral`.

- [ ] **Step 3: Commit**

```bash
git add scripts/nchc/serve_pi05_droid_nano4.sbatch
git commit -m "feat(mem): NCHC serve sbatch — vanilla pi05_droid behind userspace tailscale"
```

---

### Task 3: Pre-checks + 1-env full-loop smokes (controller/operator)

**Files:** none (verification). Deliver: the 5 task names, the sim-rate confirmation, and a green smoke per serve path.

- [ ] **Step 1: Enumerate the 5 simple benchmark tasks**

On the dev box, list benchmark tasks with ≤2 subtasks (simple tier). Prefer RoboLab's own difficulty computation; a file-based fallback:
```bash
cd /home/chungyili/Codes/RoboLab
for f in robolab/tasks/benchmark/*.py; do
  n=$(python3 - "$f" <<'PY'
import ast,sys
t=ast.parse(open(sys.argv[1]).read())
c=0
for node in ast.walk(t):
    if isinstance(node,ast.Assign) and any(getattr(x,'id','')=='subtasks' for x in node.targets):
        if isinstance(node.value,(ast.List,ast.Tuple)): c=len(node.value.elts)
print(c)
PY
)
  [ "$n" -ge 1 ] && [ "$n" -le 2 ] && echo "$(basename "$f" .py)  subtasks=$n"
done | head -20
```
Pick **5** (single-object pick-and-place); record the task-env names (the registered form the runner expects). If fewer than 5 simple tasks parse, fall back to `--tag simple` at eval time (the runner filters) and let it select.

- [ ] **Step 2: Sim-rate / stride fidelity check (from C's review)**

Confirm the DROID-jointpos env control rate ≈ 15 Hz so the serving video-frame stride (`open_loop_horizon=15`) matches training (`video_stride_frames=15`):
```bash
cd /home/chungyili/Codes/RoboLab
grep -rnE "control_dt|decimation|sim.*dt|dt *=|control_freq|policy_dt|15" robolab/registrations/droid/*.py robolab/core/environments/*.py 2>/dev/null | grep -iE "dt|decimation|freq|hz" | head
```
Record the control rate. If it is not ≈15 Hz, note it as a caveat on the K=6 result (the temporal signal differs from training). This does not block the run.

- [ ] **Step 3: Get D's code onto the machines**

- openpi (NCHC `/work`): surgically checkout the new sbatch (Task 2) — `git fetch chungyi <branch>; git checkout FETCH_HEAD -- scripts/nchc/serve_pi05_droid_nano4.sbatch` (new file, no conflict), same as C's Task-5 flow. (C's `serve_mem_session_nano4.sbatch`, `serve_mem_session.py`, `mem_session_policy.py` are already on `/work` from C.)
- RoboLab (dev box): `run_mem.py` (Task 1) is already local; `MemDroidJointposClient` is already present (C).

- [ ] **Step 4: 1-env full-loop smoke per serve path**

**MEM path:** submit `serve_mem_session_nano4.sbatch pi0_mem_droid_k1_verify <k1_9999> 8000` on NCHC; wait for `SERVING`; then on the dev box:
```bash
cd /home/chungyili/Codes/RoboLab
timeout 900 uv run --no-sync python policies/pi0_family/run_mem.py \
  --remote-host nchc-mem-serve --remote-port 8000 --policy pi05 \
  --task-dirs benchmark --tag simple --num-envs 1 --num-runs 1 --video-mode none --headless 2>&1 | tail -20
```
Expected: one episode completes; `output/<ts>/` written; no client/connection errors. Cancel the serve job.

**Vanilla path:** submit `serve_pi05_droid_nano4.sbatch 8000`; wait for `SERVING`; then:
```bash
cd /home/chungyili/Codes/RoboLab
timeout 900 uv run --no-sync python policies/pi0_family/run.py \
  --remote-host nchc-mem-serve --remote-port 8000 --policy pi05 \
  --task-dirs benchmark --tag simple --num-envs 1 --num-runs 1 --video-mode none --headless 2>&1 | tail -20
```
Expected: one episode completes; confirms `serve_policy.py` is reachable via tailscale (`0.0.0.0` bind includes localhost). Cancel the serve job.

---

### Task 4: Run the 3-arm evaluation (controller/operator)

**Files:** none. Deliver: three `output/<ts>_<arm>/` dirs with per-task success + CI.

- [ ] **Step 1: Arm = vanilla pi0.5**

NCHC: `sbatch --gres=gpu:1 --cpus-per-task=12 --mem=200G scripts/nchc/serve_pi05_droid_nano4.sbatch 8000`; wait for `SERVING` + model loaded. Dev box:
```bash
cd /home/chungyili/Codes/RoboLab
uv run --no-sync python policies/pi0_family/run.py \
  --remote-host nchc-mem-serve --remote-port 8000 --policy pi05 \
  --task-dirs benchmark --tag simple --num-envs 32 \
  --num-episodes-adaptive 200 --ci-pp-width 0.14 \
  --video-mode fail --output-folder-name pi05_vanilla --headless
```
(If the concrete 5 tasks were pinned in Task 3 Step 1, pass `--task <5 names>` instead of `--tag simple`.) On completion, `scancel` the serve job; record the `output/` dir path.

- [ ] **Step 2: Arm = MEM K=1**

NCHC: `sbatch --gres=gpu:1 --cpus-per-task=12 --mem=200G scripts/nchc/serve_mem_session_nano4.sbatch pi0_mem_droid_k1_verify /work/roboleon1295/openpi/checkpoints/pi0_mem_droid_k1_verify/pi0_mem_droid_k1_verify/9999 8000`; wait for `SERVING`. Dev box:
```bash
cd /home/chungyili/Codes/RoboLab
uv run --no-sync python policies/pi0_family/run_mem.py \
  --remote-host nchc-mem-serve --remote-port 8000 --policy pi05 \
  --task-dirs benchmark --tag simple --num-envs 32 \
  --num-episodes-adaptive 200 --ci-pp-width 0.14 \
  --video-mode fail --output-folder-name mem_k1 --headless
```
`scancel` the serve job; record the `output/` dir.

- [ ] **Step 3: Arm = MEM K=6**

Same as Step 2 but `serve_mem_session_nano4.sbatch pi0_mem_droid_k6_verify <k6_9999> 8000` and `run_mem.py … --output-folder-name mem_k6`. `scancel`; record the `output/` dir.

- [ ] **Step 4: Confirm all three completed**

Each arm's `output/<ts>_<arm>/` contains per-task result files with `k/n` success + a 95% Beta CI, for the same 5 tasks. If a serve job timed out mid-eval, resubmit the serve and re-run that arm.

---

### Task 5: Aggregate + diagnostic read (controller/operator)

**Files:** optional scratch summary (not committed; E does the formal parse). Deliver: the recorded comparison.

- [ ] **Step 1: Per-arm aggregate success + per-task CIs**

From each `output/<ts>_<arm>/`, read RoboLab's summary (per-task `k/n` + CI). Compute the **aggregate success across the 5 tasks** per arm (pooled `Σk / Σn`, with a 95% CI). Tabulate: task × {pi05_vanilla, mem_k1, mem_k6} success, plus the aggregate row.

- [ ] **Step 2: Record the diagnostic read**

Write a short summary (scratchpad) capturing:
- **vanilla vs MEM K=1** — comparable? A large `K=1 ≪ vanilla` gap ⇒ a MEM impl/serving/training bug (or the stride/sim-rate caveat from Task 3 Step 2); flag it for debugging **before** trusting K=6.
- **MEM K=6 vs K=1** — the video-encoder effect (aggregate; per-task noted but noisy).
- The pinned 5 tasks, the control-rate finding, and per-arm episode counts (adaptive early-stops).

**Done when:** three `output/<ts>_<arm>/` dirs exist with per-task success + CI for the same 5 tasks; the aggregate-per-arm table and the vanilla/K1/K6 diagnostic read are recorded. (Formal parse → wandb is sub-project E.)

---

## Self-Review

**Spec coverage:**
- Change 1 (`run_mem.py`) → Task 1. ✓
- Change 2 (vanilla serve sbatch) → Task 2. ✓
- Change 3 (eval config: 5 tasks, 32 envs, adaptive-200, no subtask, video fail, per-arm output) → Task 4 (+ Global Constraints). ✓
- Change 4 (orchestration, 3 cycles) → Task 4. ✓
- Pre-checks 1–3 (enumerate tasks, sim-rate, 1-env smoke) → Task 3. ✓
- DoD (3 output dirs, aggregate, diagnostic read) → Tasks 4–5. ✓

**Placeholder scan:** No TBD/TODO. `$PORT`/`<ts>`/`<k1_9999>` are operator values in ops steps; `run_mem.py` and the sbatch carry complete code. The concrete 5 task names are enumerated in Task 3 Step 1 (a real step, not a placeholder).

**Type consistency:** `MemDroidJointposClient(**kwargs)` in `run_mem.py:make_client` matches C's subclass (`remote_host`, `remote_port`, `remote_uri`, `open_loop_horizon`, `policy_variant` → parent). `serve_policy.py --env DROID`, `serve_mem_session_nano4.sbatch <config> <ckpt> <port>`, and `serve_pi05_droid_nano4.sbatch <port>` match their real CLIs. `--num-episodes-adaptive/--ci-pp-width/--num-envs/--tag/--task-dirs/--video-mode/--output-folder-name` are the runner's real args.

**Note (RoboLab branch):** `run_mem.py` commits to the RoboLab repo on `investigate/isaac-throughput-knobs` (where C's client lives) unless you branch it first — flag for the finishing step, same as C.
