# VLASH-on-DROID — Task 0 verification notes

> Source of truth for constants consumed by later tasks in
> `docs/superpowers/plans/2026-08-01-vlash-droid.md`. Recorded 2026-08-01.

## Step 1 — Local branch state (`~/Codes/openpi`)

- Branch: `vlash-droid`, HEAD `20f7a01` ("docs: VLASH-on-DROID implementation plan (11 tasks)"),
  parent `f9dd49a` ("docs: VLASH-on-DROID design spec ...").
- Working tree was **not clean** at verification time: modified `.gitignore` (adds
  `.superpowers/`) plus untracked `docs/DATASET_STATS.md`, `scripts/transfer_checkpoint.sh`,
  `src/openpi/models/resnet.py`. These are pre-existing local changes unrelated to Task 0 and
  were left untouched (not committed, not reverted) — flagged as a **concern** below.
- Pushed: `git push -u chungyi vlash-droid` → new branch created at
  `git@github.com:Chung-I/openpi.git`, tracking `chungyi/vlash-droid` set up.

## Step 2 — nano4 openpi checkout (`/work/roboleon1295/openpi`)

- Remotes already present: `chungyi` → `git@github.com:Chung-I/openpi.git` (fetch+push),
  `origin` → `https://github.com/Physical-Intelligence/openpi` (fetch+push). No remote needed
  to be added.
- Working tree: **clean** (`git status --short` empty).
- Current branch: `mem-video-encoder-d-eval` @ `9ab4b61` — truncation/mem work, **left
  untouched**, not deleted, not checked out away from.
- `git fetch chungyi` succeeded; `vlash-droid` now known as `chungyi/vlash-droid`.
- **Worktree decision:** created a new worktree rather than switching the existing checkout,
  even though the checkout was clean — per explicit operating guidance to never disturb the
  truncation-work checkout regardless of dirty/clean state (switching branches in-place would
  still be disruptive if anything on nano4 assumes that path is on
  `mem-video-encoder-d-eval`).
  ```
  cd /work/roboleon1295/openpi
  git worktree add /work/roboleon1295/openpi-vlash chungyi/vlash-droid -b vlash-droid
  ```
  Result: `/work/roboleon1295/openpi-vlash` @ `20f7a01 [vlash-droid]`.
  **`OPENPI_ROOT=/work/roboleon1295/openpi-vlash` is the path all later sbatch scripts must
  use** (mirrors the existing `OPENPI_ROOT` override convention already used by
  `scripts/nchc/train_truncation.sbatch` on the `pi05-layer-truncation` branch).
- Existing worktrees on nano4 for reference: `/work/roboleon1295/openpi` (mem-video-encoder-d-eval),
  `/work/roboleon1295/openpi-clean` (detached HEAD), `/work/roboleon1295/openpi-hl`
  (mem-hl-training), `/work/roboleon1295/openpi-vlash` (vlash-droid, new).
- **Concern:** `/work/roboleon1295/openpi-vlash` has **no `.venv`** — `git worktree add` does
  not copy it. Later tasks (Task 3+, training) must either run `uv sync` inside the new
  worktree to build its own venv, or point `UV_PROJECT_ENVIRONMENT` at the shared
  `/work/roboleon1295/openpi/.venv` if dependencies are identical on both branches. Not
  resolved in Task 0 — verification only.

## Step 3 — DROID RLDS location

**`DROID_RLDS_DIR = gs://gresearch/robotics`** (dataset path within it:
`gs://gresearch/robotics/droid/1.0.1`). **This is a remote GCS path, not a local directory —
there is no on-disk DROID RLDS copy on nano4, and none is needed.**

Evidence trail:
- `grep -rn "rlds_data_dir\|data_dir" src/openpi/training/config.py` on the (clean, unmodified)
  `/work/roboleon1295/openpi` checkout surfaced several candidates; the one actually used by the
  runs that produced the existing checkpoints is inside `_truncation_arm(...)` on branch
  `chungyi/pi05-layer-truncation` (not the current `mem-video-encoder-d-eval` HEAD):
  ```python
  data=RLDSDroidDataConfig(
      repo_id="droid",
      rlds_data_dir="gs://gresearch/robotics",
      ...
      datasets=(RLDSDataset(name="droid", version="1.0.1", weight=1.0,
                             filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"),),
  ),
  ```
- No local `*droid*`/`rlds*`/`tensorflow_datasets` directory exists under `/work/roboleon1295`
  (checked via `ls` + `find -maxdepth 3`); the only hit was
  `/work/roboleon1295/.openpi_cache/openpi-assets/droid/droid_sample_ranges_v1_0_1.json` (28M —
  the *filter/sample-range* file referenced above, not the RLDS shards themselves).
  `/work/roboleon1295/openpi/data/droid_subset` (referenced by a separate, unused
  `pi0_mem_droid_local` smoke-test config) does **not exist** — irrelevant to production runs.
- **Confirmed working in production**: log
  `/work/roboleon1295/openpi/logs/trunc6_201730.log` (a completed training run for
  `pi05_droid_jointpos_trunc6`, the run whose checkpoint exists at
  `/work/roboleon1295/openpi/checkpoints/pi05_droid_jointpos_trunc6`) shows:
  ```
  Load dataset info from gs://gresearch/robotics/droid/1.0.1
  Creating a tf.data.Dataset reading 2048 files located in folders: gs://gresearch/robotics/droid/1.0.1.
  Step 0: grad_norm=0.4404, loss=1.6602, ...
  Step 1800: grad_norm=0.1187, loss=0.0386, ...
  ```
  A GCS auth-token warning appears first ("All attempts to get a Google authentication bearer
  token failed... NOT_FOUND") but is benign — `gs://gresearch/robotics` is a **public** bucket
  readable anonymously over HTTPS; training proceeds and loss decreases normally.
- **Operational requirements for any future training sbatch** (from
  `chungyi/pi05-layer-truncation`'s `scripts/nchc/train_truncation.sbatch`, not yet on
  `vlash-droid` — port these when writing Task-4+ sbatch scripts):
  - Must export `CURL_CA_BUNDLE` / `SSL_CERT_FILE` / `GCS_CA_BUNDLE` pointed at a `certifi`
    CA bundle (`uv run --no-sync python -c 'import certifi; print(certifi.where())'`) — nano4
    compute nodes lack `/etc/ssl/certs/ca-certificates.crt`, and TF's GCS client fails with
    `libcurl code 77 ... Problem with the SSL CA cert` without it.
  - Needs egress from the compute node to `storage.googleapis.com` — confirmed reachable (see
    log above).
  - `export HOME=/work/roboleon1295/jobhome` with `~/.netrc` already staged there (for wandb
    auth) — matches Global Constraints.
  - tf.data's shuffle buffer is large (~250k DROID timesteps per arm) — the truncation sbatch
    requests `--mem=1200G` for an 8-GPU job specifically because of this; size later jobs'
    `--mem` accordingly, don't assume the per-GPU 200G default in Global Constraints covers a
    multi-GPU DROID run with a full shuffle buffer.

**No BLOCKED status — DROID RLDS access is confirmed working**, just remote rather than staged
locally. Do not attempt to download/stage a local copy (1.8TB > quota; this is by design, GCS
streaming is how upstream openpi itself trains on DROID).

## Step 4 — Checkpoint + venv

- `JOINTPOS_CKPT=/work/roboleon1295/checkpoints/pi05_droid_jointpos` — confirmed present,
  contains both `assets/` and `params/` (orbax format: `array_metadatas`,
  `_CHECKPOINT_METADATA`, `d`, `manifest.ocdbt`, `_METADATA`, `ocdbt.process_0`, `_sharding`).
  Total size 12G.
- `/work/roboleon1295/openpi/.venv/bin/python -c "import openpi, jax; print(jax.__version__)"`
  → **`0.5.3`**, imports clean. This venv belongs to the *original* `/work/roboleon1295/openpi`
  checkout only (see Step 2 concern — the new `openpi-vlash` worktree has no venv yet).

## Step 5 — Local RoboLab sanity (`~/Codes/RoboLab`)

- `git status --short` — no modified tracked files; untracked only:
  `assets/objects/basic/ball.usd`, `assets/objects/basic/can.usd`,
  `policies/pi0_family/run_mem.py`. RoboLab was not modified by this task.
- `ls robolab/tasks/benchmark/ | grep ball` → both eval tasks required by the plan exist:
  `ball_in_bowl_common.py`, `rolling_ball_in_bowl_task.py`, `static_ball_in_bowl_task.py`.
- `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` → **empty** (no compute
  processes). `nvidia-smi --query-gpu=...` → RTX 5090, 114 MiB / 32607 MiB used.
  **The 5090 is currently free — vLLM is NOT occupying it right now.** This is a point-in-time
  reading only; re-check immediately before Task 9 since this is a shared local machine.

## Recorded constants (for later tasks)

| Constant | Value |
|---|---|
| `DROID_RLDS_DIR` | `gs://gresearch/robotics` (dataset: `gs://gresearch/robotics/droid/1.0.1`; remote GCS, not local) |
| DROID filter/sample-ranges | `gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json` (cached locally at `/work/roboleon1295/.openpi_cache/openpi-assets/droid/droid_sample_ranges_v1_0_1.json`, 28M) |
| `JOINTPOS_CKPT` | `/work/roboleon1295/checkpoints/pi05_droid_jointpos` (12G, has `params/` + `assets/`) |
| nano4 openpi repo (untouched) | `/work/roboleon1295/openpi` @ `mem-video-encoder-d-eval` (`9ab4b61`), clean |
| nano4 vlash-droid worktree | `/work/roboleon1295/openpi-vlash` @ `vlash-droid` (`20f7a01`) — **use this `OPENPI_ROOT` for all sbatch scripts**; no `.venv` yet |
| nano4 jax version (shared venv) | `0.5.3` (`/work/roboleon1295/openpi/.venv`) |
| Local RTX 5090 | free (0 compute processes, 114 MiB used) at verification time |
| `/work/roboleon1295` quota | 1.5T total, 1.3T used, **274G free (83% used)** — watch this on long training runs |

## Concerns / follow-ups for later tasks

1. **Local repo working tree is dirty** on `vlash-droid` (see Step 1) — files unrelated to
   Task 0 (`docs/DATASET_STATS.md`, `scripts/transfer_checkpoint.sh`,
   `src/openpi/models/resnet.py`, `.gitignore` diff). Not touched here; whoever owns those
   changes should commit or discard them before they collide with Task 1+ diffs.
2. **`openpi-vlash` worktree has no venv** — must be created (`uv sync`) before any training
   task runs there.
3. **GCS CA-bundle / egress requirements are not yet ported to `vlash-droid`** — the working
   recipe lives only in `scripts/nchc/train_truncation.sbatch` on `chungyi/pi05-layer-truncation`.
   Task 4+'s sbatch scripts must incorporate the `CURL_CA_BUNDLE`/`SSL_CERT_FILE`/`GCS_CA_BUNDLE`
   exports and the large `--mem` sizing, or DROID data loading will fail with an SSL CA error.
4. **`/work/roboleon1295` is at 83% capacity (274G free).** Multiple ~12-16G checkpoints per
   training arm plus JAX compilation caches could approach this; check `df -h
   /work/roboleon1295` before launching Task 4+ training, per Global Constraints.
5. **5090 occupancy is a live/shared-machine reading**, not a guarantee — re-verify immediately
   before Task 9.

## Task 2 — Temporal-offset transform + rollforward

- Implemented `src/openpi/transforms_vlash.py`:
  - `apply_offset(sample, *, delta, action_horizon)` — pure function. Slices
    `actions[delta : delta + action_horizon]`; when `delta > 0`, rolls `state` forward:
    `state[:7] += actions[:delta, :7].sum(axis=0)` (delta accumulation for the 7 joint-position
    dims) and `state[7] = actions[delta - 1, 7]` (dim 7 is the absolute gripper command, so it
    is replaced outright rather than accumulated). Records `sample["vlash_offset"] = delta`.
    `delta == 0` is an exact identity (no copy, no mutation of `state` beyond the slice).
  - `VlashTemporalOffset(delta_max, action_horizon, rng_key="vlash_offset")` — a
    `transforms.DataTransformFn`-compatible frozen dataclass wrapper (matches the pattern used
    by `DeltaActions`/`AbsoluteActions` etc. in `src/openpi/transforms.py`) that samples
    `delta ~ Uniform{0, ..., delta_max}` and calls `apply_offset`. `rng_key` lets a caller
    rename the output field from the default `"vlash_offset"` if needed for downstream
    logging; `apply_offset` itself always writes the literal key `"vlash_offset"` (this is
    part of its tested contract), and the wrapper renames post hoc only if `rng_key` differs.
  - Must be inserted **after** `DeltaActions` in the jointpos pipeline (Task 3), since it
    assumes `actions[..., :7]` are already delta-encoded and `actions[..., 7]` is still
    absolute.

- **Randomness decision** (brief asked to read the codebase convention and mirror it, or
  document a `np.random.default_rng`-based choice if none exists):
  - `grep -n "rng\|seed\|np.random\|random_state\|jax.random" src/openpi/transforms.py` →
    **zero hits**. None of the existing transforms (`Normalize`, `DeltaActions`,
    `ResizeImages`, `SubsampleActions`, ...) are randomized at all, so there is no
    RNG/seeding convention in that module to mirror.
  - A repo-wide `grep -rn "np.random\|default_rng"  src/openpi/` turned up only
    unseeded `np.random.rand`/`np.random.randint` calls used to build **fake/test
    observations** in `policies/*.py` and model tests — again, not a per-sample-deterministic
    convention, just ad hoc unseeded sampling for synthetic inputs.
  - **Decision**: `VlashTemporalOffset.__call__` draws a fresh, OS-entropy-seeded
    `np.random.default_rng()` on every call (no stored/counter-based state, no hash of a
    per-example key). Rationale: `vlash_offset` is a **training-time augmentation** —
    analogous to random image augmentation — and is explicitly meant to vary per epoch for
    the same underlying RLDS example, not to be a stable deterministic function of the
    example's identity. RLDS map-level determinism (same example → same transform output on
    every pass) is therefore *not* a requirement here, unlike e.g. normalization stats lookup
    which must be deterministic. `test_transform_samples_in_range` in
    `tests/test_vlash_offsets.py` encodes this: it calls the transform 200 times on the *same*
    input sample and asserts all three possible offsets `{0, 1, 2}` are observed, which would
    be impossible under per-sample-deterministic seeding.
  - Caveat worth flagging for Task 3+: `np.random.default_rng()` with no seed is **not
    reproducible across runs** (no global seed control, no way to replay an exact epoch's
    offsets). If exact reproducibility of the offset sequence is ever needed (e.g. for
    debugging a specific bad batch), a seeded variant would need a counter or worker-id-based
    seed threaded through the data loader; out of scope for Task 2 per the brief.

- **Tests**: `tests/test_vlash_offsets.py` (fixtures used verbatim from the Task 2 brief) —
  `test_offset_zero_identity`, `test_offset_two_rollforward`, `test_transform_samples_in_range`.
  All 3 pass: `.venv/bin/python -m pytest tests/test_vlash_offsets.py -x -q` → `3 passed`.
  Verified TDD red→green: prior to writing `transforms_vlash.py`, the same test file failed
  collection with `ModuleNotFoundError: No module named 'openpi.transforms_vlash'`.
- `ruff check` and `ruff format --check` pass clean on `src/openpi/transforms_vlash.py`.
  `tests/test_vlash_offsets.py` has two pre-existing ruff nits from the brief's verbatim
  fixture (unsorted single-line import, `N803` on the `H` parameter name) — left as-is since
  the brief specifies the fixture verbatim and this repo's `.pre-commit-config.yaml` is not
  installed as a git hook here, so it does not block the commit.
