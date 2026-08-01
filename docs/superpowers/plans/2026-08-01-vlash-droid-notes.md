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

## Task 3 — Training config, RLDS window extension, 12-step repro

**Status: PASSED.** Config `pi05_droid_jointpos_vlash` added, RLDS window mechanism found and
extended, 12-step repro (Slurm job **228203** on nano4) completed cleanly: `sacct` shows
`COMPLETED 0:0`, `TRAIN_EXIT=0`.

- **RLDS window mechanism** (how `action_horizon` reaches the RLDS loader): `data_loader.
  create_data_loader` hardcoded `action_horizon=config.model.action_horizon` for the RLDS
  branch, which flows into `create_rlds_data_loader` → `create_rlds_dataset` →
  `DroidRldsDataset(..., action_chunk_size=action_horizon)` (`droid_rlds_dataset.py`'s
  `chunk_actions` uses it to build the `[traj_len, action_chunk_size]` gather-index tensor).
  Added `DataConfig.rlds_action_horizon: int | None` (defaults `None` = old behavior) and
  `RLDSDroidDataConfig.vlash_delta_max: int | None` (the factory-level knob); when set, `.create()`
  computes `rlds_action_horizon = model_config.action_horizon + vlash_delta_max` (15+3=18) and
  inserts `transforms_vlash.VlashTemporalOffset(delta_max=3, action_horizon=15)` into
  `data_transforms` right after `DeltaActions`. `data_loader.create_data_loader` reads
  `data_config.rlds_action_horizon` (falling back to `config.model.action_horizon`) so only
  this config's window is extended; every other `RLDSDroidDataConfig` user
  (`pi0_fast_full_droid_finetune`, `pi05_full_droid_finetune`) is unaffected (`vlash_delta_max`
  defaults `None`).
- **Tokenizer wiring**: `ModelTransformFactory`'s `PI05` branch now passes
  `pi05_no_state=model_config.state_cond` into `_transforms.TokenizePrompt`, and `TokenizePrompt`
  forwards it to `tokenizer.tokenize(..., pi05_no_state=...)`. Confirmed in the real job's
  `data_config` log line: `TokenizePrompt(..., discrete_state_input=False, pi05_no_state=True)`.
- **Weight loader**: `CheckpointWeightLoader` gained a `missing_regex` field (was hardcoded
  `.*lora.*`); `pi05_droid_jointpos_vlash` uses
  `missing_regex=r"(state_proj|state_mlp_in|state_mlp_out)/.*"` against the local checkpoint
  `/work/roboleon1295/checkpoints/pi05_droid_jointpos/params`. Confirmed via the job's full
  param listing (`train.py:238`): `state_proj`/`state_mlp_in`/`state_mlp_out` present with
  freshly-initialized shapes (`state_proj.kernel: (32, 1024)`), everything else restored.
- **Pi0Config assertion** (Task 1 review follow-up): `__post_init__` now raises `ValueError` if
  `state_cond=True` and `pi05=False`. Covered by `config_test.py::test_state_cond_requires_pi05`.
- **`__post_init__` for the new config**: `pi05_droid_jointpos_vlash` explicitly sets
  `discrete_state_input=False` (needed so `TokenizePrompt` passes `state=None` into
  `tokenizer.tokenize`, which is required for the `pi05_no_state` branch to trigger — see
  `tokenizer.py`'s `if state is not None: ... elif pi05_no_state: ...` priority order).
- **Config test**: `src/openpi/training/config_test.py` (new file — did not exist on this
  branch before Task 3) — 11 tests, all pass locally and on nano4's venv. Per the brief's
  correction: the truncation branch's `pi05_droid_jointpos` is not a standalone `TrainConfig`
  (only exists inline inside `_truncation_arm`), so instead of comparing against it directly,
  `test_data_is_rlds_jointpos_matching_truncation_branch_recipe` documents exactly what's
  ported verbatim (`rlds_data_dir`, `action_space`, filter path, `num_workers=0`) vs.
  deliberately different (local nano4 `assets_dir`/`params_path` instead of GCS), and
  `test_upstream_pi05_droid_is_untouched` guards the pre-existing `pi05_droid` config.

### Debug loop — 3 root-caused bugs found and fixed on the real repro job

All were genuine bugs (or a genuine resource-sizing miss), not flukes — verified by rerunning
after each fix and observing progress past the previous failure point.

1. **Read-only array in `DeltaActions`** (job 228096). `IterableTransformedDataset.__iter__`
   split a batched RLDS sample into per-example views via bare numpy indexing (`x[i]`), which
   inherits the parent array's writeable flag. `EagerTensor.numpy()` batches coming out of the
   tf.data/dlimp pipeline can be read-only (TF avoids copying its internal buffer), and
   `DeltaActions` mutates `actions` in place (`actions[..., :dims] -= ...`), which raised
   `ValueError: output array is read-only`. **This is a general RLDS-pipeline bug, not
   vlash-specific** — it would affect any RLDS+jointpos config using `DeltaActions`, but nothing
   in this repo's existing RLDS configs had apparently been run against this exact TF version
   until now. Fixed in `data_loader.py`.
2. **Over-eager fix #1 broke prompt decoding** (job 228124). The first fix
   (`np.array(x[i])` unconditionally) also wrapped scalar bytes/str leaves (`"prompt"`) into a
   0-d ndarray. `droid_policy.DroidInputs` decodes prompt via `isinstance(data["prompt"], bytes)`,
   which a 0-d ndarray never satisfies (even though the raw `numpy.bytes_`/`bytes` it wrapped
   would have), so the decode step silently no-opped and `PaligemmaTokenizer.tokenize()` got a
   raw `bytes` object, crashing on `.replace("_", " ")` with `TypeError: a bytes-like object is
   required, not 'str'`. Fixed by only copying when the per-sample leaf is itself an `ndarray`
   (`v.copy() if isinstance(v, np.ndarray) else v`), which fixes the read-only "actions" case
   while passing scalar bytes/str leaves through untouched.
3. **OOM at the step-6 checkpoint save, initially misdiagnosed as an async-checkpointing
   hang** (jobs 228149 → 228165). Job 228149 appeared to hang indefinitely at
   "Transferring arrays to host memory" (zero bytes written for 10+ minutes, main thread parked
   on a futex) — looked like an orbax/JAX deadlock specific to a single-GPU allocation. Added
   `TrainConfig.enable_async_checkpointing` (CLI-overridable, default `True`) as a diagnostic +
   candidate fix. Job 228165 with `--no-enable-async-checkpointing` progressed further (wrote
   the `params` item, 12.5GB) but was then **`Slurm oom_kill`'d (exit 137)** while writing the
   larger `train_state` item (params + Adam `mu`/`nu` + EMA). This revealed the real root cause:
   `DroidRldsDataset`'s default `shuffle_buffer_size=250_000` timesteps holds ~75GB of raw
   images alone (two 224×224×3 images/timestep) — fine for real training's 8-GPU/`--mem=1200G`
   allocations (`train_truncation.sbatch`), but far too much for a single-GPU job's `--mem=190G`,
   on top of the checkpoint's own ~50-60GB transient footprint during save. **The original
   228149 "hang" was very likely the same OOM condition, just manifesting as a slow, silent
   thrash rather than an immediate kill** (memory pressure, not a real deadlock — confirmed via
   `ps -T -o wchan` showing an actively-running (`Rl`, not blocked) worker thread mid-save,
   ruling out a true deadlock hypothesis before finding the `oom_kill` line in the Slurm log).
   Fixed by adding `DataConfig.shuffle_buffer_size` / `RLDSDroidDataConfig.shuffle_buffer_size`
   (CLI-overridable via `--data.shuffle-buffer-size`, defaults `None` = unchanged 250_000 for
   real training), and the repro sbatch now passes `--data.shuffle-buffer-size=5000`. Kept
   `--no-enable-async-checkpointing` too, as cheap insurance against the async path's extra
   host-memory buffering on a memory-constrained single-GPU job — re-evaluate both flags
   (probably revert both) for the real multi-GPU training job, which has a much larger `--mem`
   budget and wants async checkpointing for throughput.

### venv decision

`/work/roboleon1295/openpi-vlash` (the Task 0 worktree) had no venv. Built a **fresh venv in
the worktree** rather than reusing `/work/roboleon1295/openpi/.venv` (the `mem-video-encoder-d-eval`
checkout's venv): `docs/eval/TRUNCATION.md` (chungyi/pi05-layer-truncation) already documents
why — openpi is installed editable, so a worktree sharing another checkout's venv would
silently import the OTHER branch's code.

```bash
cd /work/roboleon1295/openpi-vlash
export UV_CACHE_DIR=/work/roboleon1295/.cache/uv
uv venv --python 3.11 .venv        # python 3.11 required: tensorflow-cpu==2.15.0 only ships cp311 wheels
uv sync --group rlds               # NOT --no-sync here -- this is the one-time env build
```

No `GIT_LFS_SKIP_SMUDGE` needed — `uv sync --group rlds` completed without any LFS-related
errors (nothing in this dependency set pulls LFS-tracked files). Training invocations use
`uv run --no-sync python scripts/train.py ...` (matches `train_truncation.sbatch`'s
convention: sync once, `--no-sync` on every actual run to avoid a redundant resync).

### Gate evidence (Slurm job 228203)

```
$ sacct -j 228203 --format=JobID,State,ExitCode,Elapsed -n
228203        COMPLETED      0:0   00:04:28
```
- **Reaches step 12** (steps 0-11 logged): yes, e.g. `Step 11: grad_norm=20.1563, loss=1.9622,
  param_norm=1833.5470`.
- **Checkpoint saved**: yes, both the mid-run save (step 6, `save_interval=6`) and the final
  save (step 11, `step == num_train_steps - 1`) finalized cleanly (`Finished saving checkpoint
  (finalized tmp dir) to .../vlash_repro/6` and `.../vlash_repro/11`); `max_to_keep=1` pruned
  step 6's directory after step 11 saved (expected — `ls .../vlash_repro/` shows only `11`).
  Checkpoint contains `assets/`, `_CHECKPOINT_METADATA`, `params/`, `train_state/`.
- **No NaN loss**: yes, all 12 steps logged finite `grad_norm`/`loss` (loss ranged ~1.3-6.0,
  `param_norm` constant at `1833.5470` as expected for one training step's worth of movement).
- **Offset histogram shows spread over {0..3}**: yes —
  `[vlash] first 32 sampled vlash_offset values: {0: 4, 1: 8, 2: 13, 3: 7}` (sums to 32, all
  four values present).

### Concerns / follow-ups for later tasks

1. `enable_async_checkpointing=False` and `shuffle_buffer_size=5000` are repro-only overrides
   passed on the CLI in `scripts/nchc/vlash_droid_repro.sbatch` — the base
   `pi05_droid_jointpos_vlash` config itself still defaults to async checkpointing enabled and
   the full 250_000-timestep shuffle buffer, which is correct for real training but means
   nobody has yet exercised a real multi-GPU run's checkpoint-save path on this branch. Worth a
   deliberate check (not just this repro) before a long real training job, given how close the
   single-GPU repro came to OOMing even after the fix.
2. The `state_proj`/`state_mlp_in`/`state_mlp_out` fresh-module restore was verified via the
   full param listing in the job log, not via a dedicated automated check against the actual
   downloaded checkpoint's key set (the config test's `test_weight_loader_allows_the_three_fresh_state_modules_missing`
   only checks the regex against literal key strings, since the real checkpoint isn't available
   in the local dev environment). Low risk since the nano4 job's full listing confirms it works
   end-to-end, but flagging for completeness.
3. `vlash_shared_obs: bool = False` (mentioned in the Task 3 brief's "Interfaces" section as a
   forward-looking flag) and the LoRA variant via `freeze_filter` were explicitly out of scope
   for this task (deferred to Task 5 per the brief) and were not added.
