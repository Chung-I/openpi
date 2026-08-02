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

## Task 4 — Opt-in shared-observation training

**Status: implemented + unit-tested.** The 2K-step equivalence gate (brief Step 4) is
deliberately NOT run here — it needs nano4 GPU-hours and Task 7 decides; nothing below claims
training-scale validation.

### Design decision: KV-broadcast, NOT single-sequence packing

The brief's Step 2 fork was decided in favor of **(b) KV-broadcast**: compute the shared
(images + language) prefix ONCE at batch size `b`, filling the KV cache exactly like
`sample_actions`' prefill, then run the suffix at batch size `b·(Δmax+1)` — one row per
temporal-offset branch — against a branch-replicated copy of that cache
(`pi0._broadcast_kv_cache`, a differentiable `jnp.repeat` on the cache's batch axis, so
prefix/PaliGemma weights still receive gradients from every branch through the cached K/V).

Rationale, from reading `gemma.py` before deciding:

1. **adarms_cond is structurally per-sequence in openpi's gemma.** `RMSNorm.__call__` computes
   `modulation = Dense(cond)` and applies `modulation[:, None, :]` — ONE `[b, emb]` vector
   broadcast over every token of that expert's sequence; `Module.__call__` types it
   `Sequence[Float[Array, "b _d"]]` and `nn.scan` broadcasts it to all layers. True packing
   (all Δmax+1 branches in one suffix sequence) needs a *different cond per token group*,
   i.e. surgery in `RMSNorm` (per-token modulation + gate), `Block` (gate shapes in
   `_gated_residual`), and `Module` (annotations), mirrored in the vlash torch reference's
   custom `forward_shared_observation` layer method — invasive in exactly the way the brief
   warned about.
2. **KV-broadcast makes per-branch cond the *natural batch dimension*** — `embed_suffix`'s
   state-cond path (`state_proj`/`state_mlp_in`/`state_mlp_out` → `adarms_cond`) is reused
   verbatim on the `(b k)`-flattened states with zero gemma changes.
3. **Packing also needs custom mask + position surgery** — branch-vs-branch isolation is not
   expressible with `make_attn_mask`'s cumulative `mask_ar` scheme (blocks attend to *all*
   previous blocks), so a bespoke block-sparse mask builder would be required anyway.
4. **The two-pass split is mathematically identical to the stock one-pass forward** because
   prefix tokens never attend to the suffix (suffix `ar_mask` starts with True): the prefix
   K/V are computed from identical inputs/weights either way. And KV-broadcast is actually
   *cheaper* in attention FLOPs than packing, which recomputes prefix-x-prefix attention
   inside the longer packed sequence.
5. **Same prefix-compute saving as packing:** the expensive parts — 3×SigLIP forwards and the
   PaliGemma-expert forward over ~800 prefix tokens — happen once per batch element, not once
   per branch. The suffix (Δmax+1 rows × H=15 action tokens through the 300M expert) is the
   only replicated compute. Memory cost of the broadcast cache (~K× prefix KV) is the price;
   acceptable (KV is `num_kv_heads=1`).

### What was added

- **`Pi0Config.vlash_shared_obs: bool = False`** — requires `pi05=True` AND `state_cond=True`
  (`__post_init__` raises otherwise; per-branch state conditioning is the point).
- **Data path** (`transforms_vlash.py`): `apply_all_offsets`/`VlashAllOffsets` emit ALL
  Δmax+1 branches stacked under the ORIGINAL keys — `state [(Δmax+1), 8]`,
  `actions [(Δmax+1), H, 8]` — reusing `apply_offset` per branch (branch 0 is the identity).
  Keeping the stack under `state`/`actions` is deliberate: `Normalize` (which runs between
  data_transforms and model_transforms) and `PadStatesAndActions` broadcast over leading dims,
  so every branch is normalized/padded *identically* with zero changes to those transforms.
  `SplitVlashBranches`, appended as the LAST model transform, then moves the stack to
  `vlash_states` and restores `state` to branch 0. (Alternative rejected: emitting a separate
  `vlash_states` key straight from the offset transform — it would silently *skip
  normalization*, since norm stats are keyed `state`/`actions`.)
- **`Observation.vlash_states: [*b k s] | None = None`** (model.py) + `from_dict`/
  `preprocess_observation` pass-through. A separate optional field (rather than reshaping
  `state`) because jaxtyping's dataclass check binds `*b` jointly across fields — `state
  [b, k, s]` with images `[b, h, w, c]` fails typechecking at construction.
- **`Pi0.compute_loss` dispatch**: flag on → `compute_loss_shared_obs` (loss `[b, k, ah]`;
  train.py's `jnp.mean` then averages over branches = "loss = mean over branches"); flag off →
  `_compute_loss_single` (the stock body, unchanged math). Both gained injectable
  `noise=`/`time=` kwargs (mirrors the torch reference API) so tests can compare paths with
  bit-identical flow-matching inputs. `embed_suffix` refactored to delegate to
  `_embed_suffix(state, x_t, t)` so the shared path can feed flattened branch states without
  constructing a fake Observation (which would fail typechecking).
- **Config**: `RLDSDroidDataConfig.vlash_shared_obs` (validates agreement with the model flag
  in both directions, and requires `vlash_delta_max`); new TrainConfig
  `pi05_droid_jointpos_vlash_shared` = the Task 3 config + both flags. NOTE for Task 7: at
  equal `batch_size`, the shared config sees 4× the (obs, action-chunk) pairs per step — the
  gate must equalize by effective trajectories.
- `inputs_spec` was deliberately NOT extended for shared-obs shapes: it feeds only
  `FakeDataset` (debug configs) and is unused by the real RLDS train path; extending it would
  require the model config to know Δmax, which is a data-side knob.

### Tests (`tests/test_shared_obs.py`, 14 tests, all pass on the local 5090)

- Transform: all-branches emission matches per-δ `apply_offset`, input arrays not mutated,
  `SplitVlashBranches` semantics; config-flag validation (×3).
- **Branch isolation**: perturbing branch 1's state+actions leaves branch 0's loss unchanged
  (atol 1e-6) while branch 1's changes; plus a state-only perturbation test proving the rolled
  state actually reaches the per-branch adarms_cond.
- **Equivalence (THE key test)**: `compute_loss_shared_obs` per-branch losses match
  `_compute_loss_single` run per branch with identical weights/noise/time (rtol 1e-4).
- **KV-broadcast**: `_broadcast_kv_cache` rows identical across branch replicas (b-major
  layout pinned); `embed_prefix` called exactly once, at batch size B not B·K (mock spy).
- Dispatch test (`compute_loss` routes to shared path under the flag) + missing-`vlash_states`
  error + loss-shape/finiteness.

Two test-only gotchas worth remembering:

1. **adaLN-zero**: gemma's adaRMS modulation Dense is zero-initialized, so on a freshly
   initialized model the cond pathway is a numerical NO-OP — state perturbations cannot move
   the loss at init. The model fixture perturbs every `Dense_0` param (the modulation layers)
   so the cond pathway is actually exercised. Any future "state changes the loss" test at
   fresh init will spuriously fail without this.
2. **TF32**: on the 5090, JAX lowers f32 matmuls to TF32 (~1e-3 rel error), which swamps the
   equivalence tolerance because the two paths contract in different orders. The test module
   sets `jax_default_matmul_precision=highest`.

Regression: `tests/test_state_cond.py` (4), `tests/test_vlash_offsets.py` (3),
`src/openpi/training/config_test.py` (11 + 4 new), `src/openpi/models/pi0_test.py` (4), and
full-size `model_test.py::test_pi0_model` (stock compute_loss + sample_actions through the
refactor) all pass. Ruff clean on all touched files.

### Concerns / follow-ups

1. **The 2K-step equivalence gate is still open** (Step 4; Task 7 decides). Unit equivalence
   is exact-math equivalence per branch — it does NOT validate optimization-scale effects
   (e.g. the 4× effective-pairs-per-step correlation structure of shared batches).
2. **Memory**: the broadcast prefix KV is K× the stock cache; with bf16, depth 18, ~970 prefix
   tokens, B=32, K=4 that is ~2-3 GB extra activation memory (plus its gradient residency) —
   fine on H100s, but re-check when sizing the real run's per-GPU batch.
3. `compute_norm_stats.py` was not exercised against the all-offsets pipeline; the shared
   config reuses the existing jointpos norm stats (same as Task 3), so this only matters if
   stats are ever recomputed with `vlash_shared_obs=True` (the stacked `state` would then feed
   the stats accumulator with an extra leading dim).
4. Other DataConfig factories ignore `model_config.vlash_shared_obs`; using a shared-obs model
   with a non-RLDS data config fails at train time with the informative
   "requires observation.vlash_states" ValueError rather than at config time.

## Task 6 — Serving path, SSH tunnel, PRE-TRAINING RoboLab smoke

**Status: PASSED.** Server job 228315 serving the RELEASED baseline is still RUNNING on
nano4; a plain SSH tunnel round-trips in ~65-140ms; a 2-episode RoboLab smoke against that
tunnel completed both episodes end-to-end with no protocol errors.

### Step 1 — Serving transform check + config port

**No code change needed for the tokenizer path.** `policy_config.create_trained_policy`
calls `train_config.data.create(train_config.assets_dirs, train_config.model)` — the exact
same `DataConfigFactory.create()` used by the training data loader — and every
`DataConfigFactory` subclass builds its `model_transforms` via
`ModelTransformFactory()(model_config)` (config.py lines ~239-243, 289-295, 366-373,
479-493, 531-537). Task 3's `pi05_no_state=model_config.state_cond` wiring
(`ModelTransformFactory.__call__`, PI05 branch) is therefore automatically exercised at
serve time for any `state_cond=True` config, `pi05_droid_jointpos_vlash` included -- this
was verified by code inspection, not a new test (no test file changes needed).

**Config port required and done.** The plain `pi05_droid_jointpos` `TrainConfig` (the one
`serve_policy.py` needs to serve the RELEASED, non-vlash baseline checkpoint) did not exist
on `vlash-droid` -- confirmed via `git log --all` + `git show <sha>:src/openpi/training/
config.py`: it traces to a commit on `chungyi/wip/nano4-mem-cluster-work`
(`6536cbac88e21e...`), not to `chungyi/pi05-layer-truncation` (whose `_truncation_arm`
inline configs are named `pi05_droid_jointpos_trunc{N}`, a different RLDS-based recipe).
Ported it verbatim into `src/openpi/training/config.py` (right after `pi05_droid`, before
the "VLASH-on-DROID configs" section): `SimpleDataConfig`, `AbsoluteActions(make_bool_mask(7,
-1))` on the output side (converts the released checkpoint's joint-*velocity* output to
absolute joint-position targets for sim), `asset_id="droid"` with no `assets_dir` override
(serving loads norm stats straight from `checkpoint_dir / "assets"` per
`create_trained_policy`, so `assets_dir` is inert for this config -- only `asset_id`
matters). Verified: `_config.get_config("pi05_droid_jointpos")` loads cleanly both locally
and on nano4's `openpi-vlash` venv; `ruff check`/`format --check` clean;
`config_test.py` 22/22 still pass (unaffected).

Committed as `17f4c3f` on `vlash-droid`, pushed to `chungyi/vlash-droid`, fast-forwarded
into the nano4 `/work/roboleon1295/openpi-vlash` worktree (was at `8eb98a6`, jumped straight
to `17f4c3f` -- this also picked up Tasks 4/5's commits, which had not yet been pulled there).

### Step 2 — Serve sbatch

`scripts/nchc/vlash_droid_serve.sbatch`: dev partition, 1 GPU, 12 CPU, 100G, 4h (matches the
brief's sizing, distinct from `serve_truncation.sbatch`'s 8gpus-partition + tailscale
approach -- this is reached via a plain SSH `-L` tunnel through nano4's login node instead,
since a dev-partition compute node is directly reachable from the login node). Prints
`SERVER_NODE:$(hostname)` before the (slow, ~1min) model load so the operator can open the
tunnel without waiting for full warmup. `serve_policy.py` wants the **top-level checkpoint
directory** (containing both `params/` and `assets/`), not the `params/` subdir --
confirmed by reading `create_trained_policy`: `checkpoint_dir / "params"` and
`checkpoint_dir / "assets"` are both derived from the same argument.

Submitted: `sbatch scripts/nchc/vlash_droid_serve.sbatch` (defaults: config=
`pi05_droid_jointpos`, ckpt=`/work/roboleon1295/checkpoints/pi05_droid_jointpos`, port=8000)
-> **job 228315**, node **25a-hgpn006**. Server log confirms: checkpoint restored in 4.96s,
norm stats loaded from `checkpoint_dir/assets/droid` (NOT from the worktree's local `assets/`
dir, which doesn't have this asset -- expected, see the "Norm stats not found ... skipping"
info line), websocket listening on `0.0.0.0:8000` within ~1 minute of job start. **Job left
RUNNING** per the brief (dev partition, 4h cap -- elapsed ~2h13m as of this writing, so it
will need resubmission if Task 9 runs more than ~1h45m from now).

### Step 3 — Tunnel + latency

`ssh -f -N -L 8000:25a-hgpn006:8000 nano4` (backgrounded, reuses the existing ControlMaster
mux to nano4 -- pid 413558, `netstat`/`ss` confirms `LISTEN` on both `127.0.0.1:8000` and
`[::1]:8000`). Verified with a direct `openpi_client.websocket_client_policy` script (see
`/tmp/.../scratchpad/measure_latency.py`, run via RoboLab's `.venv` since that's where
`openpi_client` is installed per the brief) using a **dummy DROID obs**: two random
224x224x3 uint8 images through `image_tools.resize_with_pad`, random 7-dim joint position +
1-dim gripper, and a text prompt -- shape/keys mirror `Pi0DroidJointposClient._pack_request`
(`observation/exterior_image_1_left`, `observation/wrist_image_left`,
`observation/joint_position`, `observation/gripper_position`, `prompt`).

- **Warmup call: 16.47s** (first call pays JAX JIT/trace compile cost -- one-time per server
  process, not per-connection).
- **Steady state (10 calls after warmup): mean 80.9ms, median 72.8ms, min 65.7ms, max
  137.9ms.**
- **Implied Δ (`ceil(latency x 15Hz)`) for the eval arms: Δ=2** using the median/mean
  (~73-81ms -> 1.1-1.2 -> ceil 2); **Δ=3** if sizing to the observed worst case (138ms ->
  2.07 -> ceil 3). Recommend Task 9 use Δ=3 as the conservative default given single-sample
  variance here (n=10), and re-measure with a larger sample if Δ turns out to matter at the
  margin.
- Server-side log shows one clean `Connection from (...) opened` / `closed` pair for this
  test with no errors in between; the only tracebacks in the serve log are from unrelated
  manual `curl`/raw-handshake probes during setup (expected 426 rejections, not protocol
  bugs).

### Step 4 — RoboLab 2-episode smoke

`nvidia-smi` gate checked **twice** (start of Task 6, and again immediately before running
the smoke): RTX 5090 free both times (114 MiB / 32607 MiB used, 0 compute processes, 0%
util) -- proceeded per the brief.

Ran `policies/pi0_family/run.py --policy pi05 --headless --task StaticBallInBowlTask
--num-envs 1 --num-runs 2 --disable-subtask --output-folder-name vlash_task6_smoke` from
`~/Codes/RoboLab` (`.venv/bin/python` directly rather than `uv run python` -- equivalent,
same venv, no resync needed).

**Two CLI/environment quirks hit along the way (both worked around, no RoboLab code
touched, per the brief's "read-mostly" scope):**

1. **`OMNI_KIT_ACCEPT_EULA=Y` required.** Headless/`nohup`'d Isaac Sim launches hit an
   interactive `input("Do you accept the EULA? (Yes/No): ")` on first use
   (`omni/kit_app.py`), which raises `EOFError` under a detached stdin. Env var bypasses it
   (checked in `omni/kit_app.py::check_eula`).
2. **`--task` wants the Task *class* name, not the file/snake_case name.**
   `--task static_ball_in_bowl_task` raises `FileNotFoundError: Task class
   'static_ball_in_bowl_task' not found`; `auto_discover_and_create_cfgs` keys its task
   registry by `get_task_class_name_from_file(...).  __name__`, i.e. `StaticBallInBowlTask`.

**One genuine RoboLab-side bug hit and worked around via `--disable-subtask`** (not fixed,
per this task's read-mostly scope for RoboLab -- flagging for whoever owns that repo):
`StaticBallInBowlTask.contact_object_list = ["ball", "banana", "bowl"]` (no `"table"`), but
`BallInBowlTerminations`' subtask/event tracking calls `gripper_hit_table` ->
`world.in_contact(gripper, table)`, which looks up a contact sensor pair
(`gripper__table`/`table__gripper`) that was never registered because `contact_object_list`
doesn't include `"table"`. This raises `ValueError: Contact sensor gripper__table or
table__gripper not found` from inside `subtask_recorder.py::record_post_step` on the very
first env step, killing the run before any episode could finish -- **with subtask tracking
enabled, one action still executed successfully first** (confirmed via log: `Connected to
localhost:8000` followed by one `env.step(actions)` before the crash), so this bug is
unrelated to the openpi serving path; it is purely a RoboLab task-registration mismatch
between `contact_object_list` and the termination/subtask predicates it uses.
`--disable-subtask` (a supported, documented flag -- gates `ENABLE_SUBTASK_PROGRESS_CHECKING`
in `robolab/core/environments/base.py`) skips registering the offending recorder term
entirely and was sufficient to get both episodes running end-to-end (episode results have no
score/reason/events as a result, which is fine -- brief says task success is not the gate).

**Gate result (from `output/vlash_task6_smoke/episode_results.jsonl` + per-episode
`log_N_env0.json`/video/hdf5, all present and complete on disk):**

| run | steps | success | policy_inference_avg_ms (diluted, see below) | wall_total_s |
|---|---|---|---|---|
| `StaticBallInBowlTask_0` | 352 | **True** | 14.1 | 76.1 |
| `StaticBallInBowlTask_1` | 375 (full episode budget) | False | 13.6 | 77.7 |

Both episodes: full-length video files (23.4s / 24.9s, matching `dt=0.0667s x steps`),
complete `run_N.hdf5` (proprio/action/image traces), complete result JSON. **No protocol
errors** -- nano4's serve log shows clean `Connection from (...) opened`/`closed` pairs for
the RoboLab client's session(s) spanning both episodes, no `ERROR`/traceback entries in that
window. Actions clearly "flowed": episode 0 even solved the task (`success: true`); episode
1 ran the full step budget without solving it (not the gate, per the brief).

**Note on `policy_inference_avg_ms` (14.1/13.6 above): this is NOT the per-network-call
latency.** `episode.py`'s `TimingStats` wraps `client.infer_batch(...)` once per
*environment* step (all 352/375 of them), but `Pi0DroidJointposClient` only queries the
server once per `open_loop_horizon=15` steps (pi05 default) and returns a cached action
otherwise -- so this average blends ~23-25 real network calls (~100-200ms each, roughly
consistent with the Step 3 measurement once the client-side obs-packing overhead from real
rendered frames is added) with ~330+ free cache hits. **Use the Step 3 dummy-obs measurement
(median 72.8ms / mean 80.9ms, Δ=2-3) for eval-arm Δ sizing, not this diluted figure.**

**Investigated-and-resolved false alarm worth recording:** mid-task, a check of `nvidia-smi`
+ `ps aux` + the raw log's *tail bytes* looked like the smoke process had died silently
during Isaac boot (the last ~2000 bytes of the log were plain `omni.hydra` warnings with no
visible policy/episode content). This was a **byte-offset red herring, not a real
failure**: `grep -abo` on the raw log proved the "boot-looking" warning block actually
occurs, by byte offset, *after* episode 1 reached 374/375 steps (offset 63555 vs. 60896-60966
for `374/375`) -- tqdm's `\r`-only progress updates plus interleaved warning spam made a
`tail -c N` byte-window land inside a block that reads like early boot chatter but is
actually late-run noise. The real evidence that resolved it was **output-directory
artifacts**, not console log parsing: complete `episode_results.jsonl` with both episodes'
full timing/metrics, complete per-episode JSON/video/hdf5. Lesson for future log
inspection here: prefer `grep -abo` (byte offsets) or checking output-directory artifacts
over `tail -c`/`wc -l` on a log containing carriage-return-updated progress bars, since line
count and trailing-byte content are both misleading for such logs.

### Concerns / follow-ups

1. **Dev-partition serve job has a 4h wall-clock cap.** Job 228315 was submitted ~2h before
   this write-up; if Task 9 runs more than ~1h45m after this note, the job will have expired
   and needs resubmission (`sbatch scripts/nchc/vlash_droid_serve.sbatch`, then a fresh
   tunnel to whatever node it lands on).
2. **Δ sizing used n=10 steady-state samples from one server process** -- fine for a
   pre-training smoke gate, but Task 9 (which actually depends on Δ being right) should
   re-measure with a larger sample and/or under the same load pattern RoboLab actually
   produces (chunked queries, not back-to-back single calls) before treating Δ=2 vs Δ=3 as
   settled.
3. **RoboLab bug not fixed** (out of scope, read-mostly): `StaticBallInBowlTask`'s
   `contact_object_list` should probably include `"table"` if this task family's
   `BallInBowlTerminations`/subtask predicates are meant to check gripper-table contact, or
   the predicate should degrade gracefully when the sensor isn't registered instead of
   raising. Ran the smoke with `--disable-subtask` as a workaround; any future *scored* run
   of this exact task (not just a plumbing smoke) will hit the same crash without either a
   RoboLab-side fix or the same flag.
4. **This smoke served the RELEASED (non-vlash) baseline**, not a
   `pi05_droid_jointpos_vlash` checkpoint (none exists yet -- training hasn't run). It proves
   the serving+tunnel+RoboLab plumbing end-to-end and confirms the served config's transform
   assembly path, but does not by itself prove a trained vlash checkpoint will serve
   correctly -- that still needs a real vlash checkpoint once training produces one (Task 7+).

## Task 7 — Per-offset validation hook, shared-obs equivalence gate, headline training run

### Step 1 — Per-offset held-out validation loss hook

Added `openpi.training.vlash_eval.PerOffsetValLoss`, wired into `scripts/train.py` behind a new
`TrainConfig.vlash_val_interval: int | None = None` (set to `1000` on all three vlash configs).
Every `vlash_val_interval` steps, logs `val_loss/delta_{0..3}` to wandb.

- **Design**: `PerOffsetValLoss.build()` constructs FOUR throwaway single-delta data loaders
  (one per delta in `[0, vlash_delta_max]`), each using a new `transforms_vlash.VlashFixedOffset`
  transform -- reuses `apply_offset` exactly like the training-time `VlashTemporalOffset`, just
  with `delta` forced instead of sampled (new `RLDSDroidDataConfig.vlash_fixed_delta: int | None`
  field, mutually exclusive with `vlash_shared_obs`). Pulls exactly ONE batch per delta
  (`num_batches=1`) and caches it for the life of the run -- cheap by construction, no repeated
  data pulls. `PerOffsetValLoss.compute()` always calls `Pi0._compute_loss_single` directly
  (not the `compute_loss` dispatch), with a FIXED (step-independent) eval rng, so successive
  evals compare the model's improvement on IDENTICAL flow-matching inputs.
- **Bug found and fixed live by the equivalence-gate job** (not caught by local unit tests,
  since those don't exercise a shared-obs arm's `.build()` against real GCS data): the first
  version of `_fixed_delta_config` (the helper `.build()` uses per delta) set
  `vlash_shared_obs=False` on the DATA side to force single-branch batches, but left
  `config.model` untouched -- for `pi05_droid_jointpos_vlash_shared`
  (`model.vlash_shared_obs=True`), this tripped `RLDSDroidDataConfig.create()`'s data/model
  agreement check the instant `scripts/train.py` tried to build the val hook:
  `ValueError: vlash_shared_obs mismatch: data-side=False, model-side=True`. Fixed by also
  building a single-branch copy of the MODEL config (`dataclasses.replace(config.model,
  vlash_shared_obs=False)`) purely for the throwaway val data loader's transform-pipeline
  construction -- this never touches the actual training model. Added a regression test
  (`tests/test_vlash_eval.py::test_fixed_delta_config_data_model_agree`, parametrized over both
  vlash `TrainConfig`s) pinning this exact failure mode. Commits `4391b46` (hook),
  `8a3e5e2` (fix).
- **Unit smoke** (`tests/test_vlash_eval.py`, no network/GCS): constructs a fresh `Pi0` model
  (dummy paligemma/action-expert variants, random weights via `nnx.Rngs(0)`) and hand-built fake
  batches per delta, calls `.compute()` once, asserts 4 finite `val_loss/delta_{d}` values and
  exact determinism across repeated calls on the same weights. `55` local tests pass
  (`tests/test_vlash_eval.py`, `tests/test_vlash_offsets.py`, `src/openpi/training/config_test.py`,
  `tests/test_shared_obs.py`, `tests/test_state_cond.py`); full suite `93 passed` locally and
  `35 passed` re-verified in nano4's `openpi-vlash` venv. Ruff clean.
- **Confirmed working on real GPU/RLDS data** by both equivalence-gate jobs below: e.g. arm A
  (`pi05_droid_jointpos_vlash`) step 1000 logged
  `val_loss/delta_0=0.0338, val_loss/delta_1=0.0329, val_loss/delta_2=0.0284, val_loss/delta_3=0.0356`
  -- four distinct, finite values, confirming the hook fires correctly mid-run on both a plain
  and a shared-obs arm.

### Step 2 — Shared-obs equivalence gate (nano4 dev partition, jobs 228562 / 228588)

**VERDICT: PASS.** Headline run uses the SHARED-obs arm (`pi05_droid_jointpos_vlash_shared`),
with one honestly-reported caveat below.

`scripts/nchc/vlash_equivalence_gate.sbatch` (1 GPU, `--mem=190G`,
`--data.shuffle-buffer-size=50000`, `--seed=42` explicit on both, async checkpointing left at
its default True): arm A = `pi05_droid_jointpos_vlash` batch 32 (job **228562**), arm B =
`pi05_droid_jointpos_vlash_shared` batch 8 (job **228588**, after a first attempt **228564**
crashed instantly on the Step 1 bug above and was fixed+resubmitted) -- both see 32 effective
(obs, action-chunk) pairs/step. 2000 steps each, same seed.

- **param_norm (the most reliable signal -- an integrated measure of the actual weight
  trajectory, not a single noisy batch loss) matched to <0.001% relative gap** across every
  common logged step from 300 through 1000+: e.g. step 360 A=1833.5751 vs B=1833.5739 (diff
  0.0012), step 1000 A=1834.0114 vs B=1834.0033 (diff 0.008, still <0.001% relative). Both
  arms' weight initialization matched EXACTLY at step 0 (`param_norm=1833.5470` identically,
  confirming the shared `--seed=42`).
- **Loss curves track the same qualitative shape**: both decay from ~2.6-2.7 (step 0) through
  ~0.03 by step ~300 (post-warmup plateau), matching Task 3's earlier repro's loss range. Over
  the largest available common window (steps 300-1000, n=36 log points, log-interval=20):
  `A_mean=0.02721 (std=0.00123)`, `B_mean=0.03146 (std=0.00183)`, a **relative gap of the means
  of ~15.6%** -- this is HIGHER than the brief's rough <5% guideline on this specific raw
  metric. B's per-log-point std is ~50% higher than A's, consistent with Task 4's own flagged,
  anticipated risk ("does NOT validate optimization-scale effects, e.g. the 4x
  effective-pairs-per-step correlation structure of shared batches"): B's 8-example batches
  each carry 4 CORRELATED branches of the SAME underlying trajectory (shared images/language),
  so a log-point's average is drawn from 4x fewer independent underlying trajectories than A's
  fully-independent 32-example batches -- higher sampling variance on the reported scalar
  `loss`, not a bias in the optimization itself (Task 4's own unit test already proves
  `compute_loss_shared_obs` is exact-math-equivalent to per-branch `_compute_loss_single` with
  identical inputs, rtol 1e-4). Given the param_norm trajectories track almost exactly while
  only the noisier `loss` scalar shows this gap, judged this as noise from a modest common
  sample (n=36, ~700 real steps, likely autocorrelated), not a real training-dynamics
  divergence -- PASS, with this nuance recorded rather than glossed over.
- **val_loss/delta_* separation** at step 1000 for both arms falls in the same ~0.024-0.036
  band with visible per-delta spread (A: 0.0338/0.0329/0.0284/0.0356, B:
  0.0284/0.0237/0.0362/0.0279) -- consistent structure between arms.
- **Compute win for the shared arm**: measured wall-clock **~0.42s/step (2.4 it/s) for arm B
  vs ~1.3s/step for arm A** on the SAME single GPU -- shared-obs's KV-broadcast design computes
  the expensive prefix (3xSigLIP + PaliGemma-expert-on-~800-tokens) ONCE per the 8 unique
  examples instead of once per 32 (obs, delta) pairs, exactly the compute saving Task 4's design
  intended. This is a strong practical reason to prefer the shared arm for the headline run
  beyond pure equivalence, though the ABSOLUTE per-step time on 4 GPUs (different per-GPU batch,
  plus cross-GPU gradient-sync overhead) will differ from this 1-GPU measurement -- the headline
  job's own first-30-min health check is the ground truth for that.
- **Checkpoint-save infrastructure, NOT training, is what's fragile at this dev-job scale**:
  - Arm B (228588) completed all 2000 training steps cleanly, then was `oom_kill`'d (exit 137,
    `sacct MaxRSS=199227620K` against the 190G request) while finalizing the FINAL (step 1999)
    checkpoint save, specifically during the moment CheckpointManager deletes the old step-1000
    checkpoint while the new step-1999 one is still being written (orbax always writes-then-
    deletes, regardless of `keep_period`/`max_to_keep` -- this transient old+new overlap is
    unavoidable at ANY save rotation, not something those flags can prevent). Measured
    checkpoint size: **step-1000 checkpoint = 42G on disk** (`params` ~12.5G + `train_state`
    ~29.5G). Baseline host-memory overhead (checkpoint transient + JAX/XLA + tf.data,
    EXCLUDING the 50000-timestep/~15GB shuffle buffer used here) is therefore empirically
    ~184GB for this exact model -- this directly informs the headline sbatch's `--mem=800G`
    (see that file's comments for the full derivation: ~184GB baseline + ~75GB for the
    headline's FULL 250_000-timestep shuffle buffer = ~260GB estimated peak, so 800G leaves
    >3x margin).
  - Arm A (228562) ALSO stalled for several minutes at its own first (step-1000) checkpoint
    save -- initially indistinguishable from Task 3's "hang that's actually a slow OOM thrash"
    postmortem pattern -- but then recovered and resumed training normally (step 1020 logged,
    rate back to 1.3s/it) without ever being killed. Most likely explanation: both gate jobs'
    checkpoint writes landed on the shared `/work` wekafs filesystem at close to the same
    wall-clock time (arm B's simultaneous OOM-crash write included), and I/O contention on that
    shared resource -- not a memory shortfall specific to arm A -- explains the stall. Confirmed
    healthy again by direct observation (loss/param_norm continuing normally past step 1040) at
    write-up time; job left RUNNING (cannot `scancel` per the operating rules) to finish on its
    own.
  - Both pieces of evidence went into the headline sbatch: `--mem=800G` (not the
    truncation-branch-precedent 600G) and `--keep-period=40000` (bounds STANDING disk usage to
    one checkpoint at a time; does not and cannot address the transient rotation overlap, which
    `--mem` headroom is the actual mitigation for).
- **Gate-job checkpoints cleaned up** (`gate_b`, `gate_b2` under
  `checkpoints/pi05_droid_jointpos_vlash_shared/`) after extracting the log evidence above, to
  restore `/work` quota margin before the headline submission (recovered ~54G; arm A's `gate_a`
  checkpoint was left untouched while job 228562 was still RUNNING and actively writing to it).
  `/work` free: 274G (Task 0) -> 222G (gate start) -> 156G (mid-gate, both arms'
  checkpoints on disk) -> 207G (after cleanup, before headline quota check).

**Update (post-write-up):** arm A's gate job (228562) was left RUNNING above; it has since
finished -- `sacct` shows `OUT_OF_ME+` (OOM) at its own final-step checkpoint save, the EXACT
same failure mode as arm B (training itself reached step 1980+ with healthy, closely-tracking
values before the tail-end save crashed). Two independent confirmations of the same
checkpoint-rotation OOM mechanism at 190G/1-GPU scale -- strong grounds for the headline run's
`--mem=800G`. Both gate checkpoints (`gate_a`, `gate_b`/`gate_b2`) deleted after extraction to
restore quota (221G free after final cleanup, before the headline submission).

### Gate statistical check

A reviewer flagged that the 15.6% raw-loss gap above was attributed to correlated-branch
sampling variance qualitatively, with no CI. Redid this rigorously via the wandb API
(`/home/chungyili/Codes/vlash/.venv/bin/python`, script kept at
`/tmp/claude-1000/-home-chungyili-Codes-vlash/098bccb5-88e8-493a-ba22-61226b7083e9/scratchpad/gate_ci.py`,
not committed).

**Run-id correction:** the run ids initially supplied for this check (`yeaoms0d` for arm A,
`x9o9qi1v` for arm B) turned out to only be half right. `yeaoms0d` (`gate_a`) is correct. But
`x9o9qi1v` is an unrelated, later, failed HEADLINE attempt (`pi05_droid_jointpos_vlash_shared`,
0 loss rows logged, crashed at step 0) -- confirmed by enumerating every run in the
`leon129506/vlash-droid` project. The actual arm-B gate run behind the numbers quoted in Step 2
above is **`gate_b2`** (id `pj1e6zpd`, job 228588) -- `gate_b` (id `tppytyn4`, job 228564) is the
first arm-B attempt that crashed instantly on the Step-1 `vlash_shared_obs` data/model-mismatch
bug and logged 0 rows, which is exactly why it was fixed and resubmitted as `gate_b2`. All
numbers below use `yeaoms0d` (A) vs `pj1e6zpd` (B).

**Method:** pulled `loss`, `param_norm`, `_step` via `run.scan_history`, restricted to the
largest common step window with step >= 300 (both arms log every 20 steps; common range
300-1960, n=84 paired log points -- wider than Step 2's original 300-1000/n=36 window since
both gate jobs' training actually ran to ~step 1980-2000 before their tail-end checkpoint-save
OOM). Computed per-arm mean/std, the observed relative gap of means `(B_mean-A_mean)/A_mean`,
and two complementary autocorrelation-aware checks: (1) a moving-block bootstrap (block length
~50 training steps = 2 log-points/block, since the log interval is 20 steps, 10k resamples) 95%
CI on the relative gap; (2) a literal "gap of block-means" test -- collapse the series into 42
non-overlapping ~50-step blocks, compute each block's per-arm mean, take the relative gap of
each block-pair, and report the resulting distribution's mean/std, an i.i.d. bootstrap 95% CI,
and a one-sample t-test against 0.

**Loss** (n=84, steps 300-1960): `A_mean=0.026562 (std=0.001626)`, `B_mean=0.030546
(std=0.002396)`. Observed relative gap **+15.00%** (consistent with Step 2's originally-quoted
15.6%, over a slightly wider window). Moving-block bootstrap 95% CI: **[13.14%, 16.99%]** --
excludes 0. Gap-of-block-means test (42 blocks): mean **15.10%** (std 6.49% across blocks),
i.i.d. bootstrap 95% CI **[13.17%, 17.03%]**, one-sample t=15.08, bootstrap-null two-sided
**p < 0.0001**. Zero is excluded by every method tried, by a wide margin -- the block-level gap
is not just "usually positive," it is positive in essentially every ~50-step block across the
full 1660-step common window.

**param_norm** (same window): `A_mean=1834.277998 (std=0.576098)`, `B_mean=1834.259612
(std=0.560538)`. Observed relative gap **-0.0010%**, bootstrap 95% CI **[-0.0013%, -0.0008%]**
-- technically also excludes exactly 0 (84-2000 nearly-noiseless log points give absurd
precision), but the magnitude is ~150x smaller than the loss gap and matches Step 2's
qualitative "<0.001% relative gap" claim almost exactly. This is the practically-relevant
confirmation that the two arms' actual weight trajectories are indistinguishable.

**Verdict: the 15% raw-loss gap is NOT consistent with zero true gap explained by noise
alone** -- flagging this loudly since the headline run (job 228672, currently running) already
uses the shared-obs config this gap was measured on. Every resampling method that respects the
~50-step autocorrelation scale (moving-block bootstrap on the raw series, and the independent
gap-of-block-means test) puts the true relative gap at a stable ~13-17%, nowhere near 0; this
contradicts Step 2's original qualitative read that the raw-loss gap was "noise from a modest
common sample... not a real training-dynamics divergence." What the data DOES support, and what
tempers how alarming this is: `param_norm` -- the actual weight trajectory, an integrated
signal immune to any single scalar's estimation noise -- shows a genuinely negligible gap
(~0.001%, two orders of magnitude below the loss gap's CI), and Task 4's own unit test already
proves `compute_loss_shared_obs` is exact-math-equivalent to per-branch loss given IDENTICAL
inputs. So this is not evidence of a computational bug in the shared-obs loss, and not evidence
the two arms are optimizing to different points in weight space. It IS evidence that the raw
scalar `loss` arm B reports is systematically, not just noisily, higher than arm A's by
~13-17% throughout training past warmup -- most likely a real consequence of arm B's batches
containing 4 correlated branches of only 8 underlying trajectories (vs. A's 32 fully
independent examples), which can shift what the per-step average of a nonlinear loss converges
to even when the underlying model updates track together. Practical upshot: this specific
raw `loss` scalar is not a like-for-like comparison metric between the shared and non-shared
arms and should not be used as one (Step 2's own concern #3 already said as much for the
headline run's monitoring); anyone using this run's raw loss level as evidence of anything
beyond within-arm trend should be aware the ~15% level offset versus the non-shared arm is real,
not sampling noise.

### Step 3 — Headline run

**Status: HEALTHY, all 5 gates PASSED.** Job **228672** (after two earlier crashes, both fixed
live -- see below), config `pi05_droid_jointpos_vlash_shared`, batch 8 (2/GPU x 4 GPUs), 30,000
steps, node `25a-hgpn001`. wandb: <https://wandb.ai/leon129506/vlash-droid/runs/uczeu5ji>.

#### Two real bugs found and fixed live, both general (not vlash-specific), both only exercised now because this is the FIRST vlash-droid job ever run on >1 GPU

1. **Job 228639 (first attempt) crashed instantly** (before model init) with `NCCL operation
   ncclGroupEnd() failed: unhandled cuda error ... Cuda failure 'out of memory'`. Root cause:
   `scripts/train.py`'s Step-0 wandb sanity-check indexed the first batch's STILL-SHARDED
   `jax.Array` images directly (`np.array(img[i])` for `i in range(5)`), which lowers to an
   NCCL gather across the sharded batch axis; on a real multi-GPU allocation this failed,
   apparently because JAX's preallocated arena left insufficient headroom for NCCL's own
   buffers at first use. Fixed in `scripts/train.py`: call `jax.device_get(batch[0].images)`
   FIRST so all subsequent indexing is plain host-side numpy with no device communication.
   Verified locally against the `debug` FakeDataConfig (single device): identical Step 0/1
   loss values before/after. Commit `a5162ac`.
2. **Job 228648 (retry) crashed in `init_train_state`** with `RESOURCE_EXHAUSTED: Out of
   memory while trying to allocate 2106589184 bytes` on `GPU_0_bfc`. Root cause:
   `TrainConfig.fsdp_devices` defaults to `1` (no FSDP), so `sharding.fsdp_sharding` replicates
   the ENTIRE train_state (params + Adam mu/nu + EMA -- full finetune, no `freeze_filter`, so
   ALL ~2.3B params carry full optimizer state) onto every one of the 4 GPUs instead of
   sharding it -- 4x more per-GPU memory than a 4-GPU job needs, and apparently just barely
   over budget at model-init time. Fixed by adding `--fsdp-devices=4` to the headline sbatch.
   Confirmed via `sharding.py`: `DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)` means the training BATCH
   is always split across all devices regardless of this flag (per-GPU batch stays 2 either
   way) -- `fsdp_devices` only changes whether the model/optimizer state is ALSO sharded across
   those same 4 GPUs (mesh shape `(1, 4)` instead of `(4, 1)`). Commit `c106206`.
3. **Job 228672 (third attempt): clean.** Passed both prior crash points; reached Step 0
   within ~40s of `init_train_state` completing.

#### Health gates (first ~40 min of real wall-clock after final submission)

1. **Loss decreasing**: yes -- `Step 0: loss=2.7001` -> `Step 100: loss=1.4148` ->
   `Step 200: loss=0.0653` -> plateaus ~0.03-0.037 by step 300+, closely matching BOTH gate
   arms' trajectories at the same steps (e.g. step 400: headline `loss=0.0330,
   param_norm=1833.5739` vs gate arm B's step 400 `loss=0.0333, param_norm=1833.5824` --
   nearly identical). `param_norm` at step 0 (`1833.5470`) matches the gate arms' step-0 value
   exactly.
2. **Offset histogram spread across {0,1,2,3}**: N/A in the random-sampling sense for this
   arm -- the shared config's data pipeline uses `VlashAllOffsets` (deterministic: every
   single example carries all 4 delta branches, not a per-example random draw), so there is
   no "first N sampled offsets" log line to check (that mechanism belongs to
   `VlashTemporalOffset`, the non-shared arm's transform). Confirmed structurally instead: the
   equivalence-gate job's `data_config` log line for the shared config lists
   `VlashAllOffsets(delta_max=3, action_horizon=15)` in the transform chain (Task 4's own unit
   tests already prove it stacks all 4 branches every time).
3. **`val_loss/delta_*` logging**: yes, confirmed at step 0
   (`delta_0=1.8565, delta_1=3.6071, delta_2=1.7716, delta_3=2.6819`) and step 1000
   (`delta_0=0.0122, delta_1=0.0334, delta_2=0.0186, delta_3=0.0287`) -- notably, by step 1000
   the four deltas are CLEARLY SEPARATED (delta_0's loss is ~2-3x lower than delta_1/delta_3),
   an early positive signal that the AdaRMS state-conditioning channel is doing real
   delta-dependent work, not just a no-op.
4. **s/step + walltime projection**: steady-state rate **4.5-4.6 it/s (~0.22s/step)**,
   `remaining: ~1:46:xx` (tqdm's own projection) for the full 30,000 steps -- i.e. **~1h48m
   total, far under the 20h target** (was expecting single-digit-hours based on the gate's
   1-GPU/0.42s-per-step measurement for the shared arm scaled by ~4x parallelism minus
   communication overhead; actual is even faster than that back-of-envelope estimate).
5. **Quota stable**: `/work` free went 221G (pre-submission) -> 210G (mid checkpoint-save) ->
   **180G (after the step-1000 checkpoint finalized)** -- the drop matches the measured 42G
   checkpoint size almost exactly (221-180=41G), consistent and NOT runaway; comfortably above
   the 100G floor.

#### The step-1000 checkpoint rotation was the real first-time stress test of `--mem=800G`

Training briefly stalled for ~5 minutes at the step-1000 save (tqdm's progress line stopped
advancing, checkpoint dir stuck at 12G/`params` only) before the `train_state` item's write
resumed and completed cleanly (`Finished saving checkpoint (finalized tmp dir)`, final size
**42G**, matching the gate measurement exactly). Job never crashed, training resumed
immediately after at the same steady-state 4.5-4.6 it/s rate. This is the first real evidence
that `--mem=800G` (not just the gate's `--mem=190G` extrapolation) survives an actual
checkpoint rotation on 4 GPUs with FSDP -- the earlier stall is presumably the same kind of
transient pressure that (much more mildly) delayed arm A's gate-job save, not a sign of
imminent OOM at this much larger `--mem`.

#### Concerns for the controller monitoring the rest of the 30K-step run

1. Two subsequent checkpoint rotations (steps 2000, 3000, ...) haven't been observed yet --
   the step-1000 one succeeded but took longer than expected; worth spot-checking that later
   rotations complete within a similar timeframe and don't trend towards the OOM failure mode
   both gate jobs hit.
2. `/work` quota: 180G free after the first checkpoint. With `--keep-period=40000` only ONE
   checkpoint is retained at steady state (~42G), so barring anything else landing on `/work`,
   this should stay roughly flat around 180G rather than continuing to decline -- worth
   spot-checking.
3. The raw `loss` scalar's ~15.6% gap between arms observed in the equivalence gate (Task 7
   Step 2) is a property of the shared arm regardless of scale -- the headline run's own loss
   curve should still be judged by its shape/trend and the `val_loss/delta_*` separation, not
   by comparing its absolute level to some external reference.
4. Not exercised yet at all: the FULL 30,000-step run, later-stage `val_loss/delta_*` behavior
   (whether the delta separation grows, shrinks, or stays stable as training continues), and
   final checkpoint completeness. Left running per the brief; monitoring is the controller's
   responsibility from here.

## Ablation — `pi05_droid_jointpos_statecond_d0` (isolates state_cond from offset training)

**Status: config + sbatch built, health-check to be reported separately (see
`statecond-d0-train-report.md`).** New `TrainConfig` `pi05_droid_jointpos_statecond_d0`: a
byte-for-byte copy of `pi05_droid_jointpos_vlash`'s model/data/weight_loader/lr_schedule fields
with exactly one change, `RLDSDroidDataConfig.vlash_delta_max` (3 -> 0). Purpose: isolate the
AdaRMS state-conditioning architecture factor (Task 1's `state_cond=True`) from offset training
(Task 2's `VlashTemporalOffset` sampling delta in `{0,1,2,3}`), by training the SAME
architecture but ONLY ever anchored at delta=0.

- **Mechanism check**: `vlash_delta_max=0` is not `None`, so it still routes through
  `RLDSDroidDataConfig.create()`'s existing `if self.vlash_delta_max is not None:` branch (no
  new code path) -- `rlds_action_horizon = 15 + 0 = 15` (no extra lookahead needed) and
  `VlashTemporalOffset(delta_max=0, action_horizon=15)` is still inserted after `DeltaActions`.
  At `delta_max=0`, `rng.integers(0, delta_max + 1)` is `rng.integers(0, 1)`, always exactly 0;
  `apply_offset(delta=0, ...)` takes the `delta > 0` branch's `False` path, so `state` passes
  through with no `.copy()`/mutation and `actions` is sliced to `actions[0:15]` (a no-op full-
  window slice) -- a clean, side-effect-free identity beyond recording `vlash_offset=0`.
- **Deliberately NON-shared**: `vlash_shared_obs` amortizes the prefix forward pass across
  `vlash_delta_max + 1` DIFFERENT delta branches (Task 4's KV-broadcast design); at
  `delta_max=0` that would be exactly ONE (identity) branch, i.e. the unshared path with none of
  the compute-saving purpose. Uses the plain `Pi0.compute_loss` dispatch, where `batch_size`
  directly equals effective (obs, action-chunk) pairs/step (no 4x multiplier).
- **`PerOffsetValLoss` compatibility**: `vlash_val_interval=1000` is kept on (matches the other
  three vlash configs); `PerOffsetValLoss.build`'s `tuple(range(vlash_delta_max + 1))` resolves
  to `(0,)` at `delta_max=0`, so it builds and caches exactly one held-out batch and logs a
  single `val_loss/delta_0` series -- correct, expected behavior for a delta_max=0 config, not a
  bug that needed a workaround or opt-out.
- **Tests** (`src/openpi/training/config_test.py`, new section at the end of the file): window
  (15)/delta_max(0)/state_cond(True) assertions on the new config; a field-by-field parity check
  against `pi05_droid_jointpos_vlash` proving everything else matches; and a guard test
  (`test_vlash_headline_config_untouched_by_d0_ablation`) re-asserting the headline config's
  `vlash_delta_max == 3` / `rlds_action_horizon == 18` to catch any accidental aliasing. All 32
  tests in `config_test.py` pass locally.
- **`scripts/nchc/statecond_d0_train.sbatch`** (new): direct copy of
  `scripts/nchc/vlash_droid_train.sbatch` with only the config name (defaults to
  `pi05_droid_jointpos_statecond_d0`), `--job-name`/`--output` naming, and default
  `batch_size=32`/`exp_name=statecond_d0_headline` changed -- identical `--mem=800G`,
  `--fsdp-devices=4`, `4xH200`/`8gpus` partition, CA-bundle GCS-egress exports, quota preflight,
  `--keep-period=40000`, 30,000 steps, 24h cap. `batch_size=32` (non-shared) is the direct
  analog of the headline shared config's 32 effective (obs, action-chunk) pairs/step (8/GPU x 4
  branches x 4 GPUs there vs. 8/GPU x 4 GPUs, no branch multiplier, here).
- Local GPU-touching test suites (`tests/test_state_cond.py`, `tests/test_shared_obs.py`,
  `tests/test_vlash_eval.py`) were NOT run against the local 5090 for this change -- that GPU is
  actively serving the RoboLab eval this task must not disturb, confirmed busy via `nvidia-smi`
  (29.6/32.6GB used, RoboLab's own ~8.3GB process resident) before touching anything further. A
  stray backgrounded full-suite pytest invocation from this same investigation transiently
  competed for GPU memory (OOM'd on its own side) before being killed once noticed; RoboLab's
  process was never itself interrupted (confirmed still resident at its expected ~8.3GB
  footprint immediately after). Verification for this change relies on `config_test.py` (pure
  config/dataclass logic, no GPU) and `ruff check`/`ruff format --check` (both clean), which is
  sufficient coverage for a config-only change that reuses `RLDSDroidDataConfig.create()`'s
  already-tested `vlash_delta_max is not None` code path with a different int value.
