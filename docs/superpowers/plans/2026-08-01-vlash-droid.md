# VLASH-on-DROID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune `pi05_droid_jointpos` with VLASH temporal-offset training + AdaRMS state conditioning inside openpi/JAX, and demonstrate the async benefit in RoboLab: VLASH-async beats naive-async on `rolling_ball_in_bowl_task` while matching baseline on `static_ball_in_bowl_task`.

**Architecture:** Spec at `docs/superpowers/specs/2026-08-01-vlash-droid-design.md` (read it for rationale; this plan is self-contained for execution). Training+serving on nano4 (openpi JAX, branch `vlash-droid`), Isaac/RoboLab on the local RTX 5090, websocket over an SSH tunnel. Delay emulated in sim steps.

**Tech Stack:** openpi (JAX/nnx, RLDS), RoboLab (Isaac Lab), openpi websocket client/server, Slurm on nano4, wandb.

## Global Constraints

- **Code reaches nano4 only via GitHub**: push branch `vlash-droid` to fork `git@github.com:Chung-I/openpi.git`, pull on nano4. Never scp/rsync source.
- nano4: installs/caches/outputs under `/work/roboleon1295/`; GPU jobs `#SBATCH --account=MST114563`, ≤12 CPU + ≤200G per GPU; `export HOME=/work/roboleon1295/jobhome`; check `df -h /work/roboleon1295` before big runs (wekafs at full quota drops writes silently — truncated logs mean CHECK QUOTA FIRST).
- All experiments logged to **wandb project `vlash-droid`**.
- openpi on nano4: repo `/work/roboleon1295/openpi`, venv `/work/roboleon1295/openpi/.venv`. NOTE: that checkout currently tracks the truncation branch — Task 0 reconciles branches.
- Local RoboLab: `~/Codes/RoboLab`, venv `.venv` (Isaac Lab; needs the RTX 5090 free of the vLLM process — user prerequisite before Task 9).
- Key model facts (verified from source, cite before changing): openpi π0.5 = state discretized into prompt (`src/openpi/models/tokenizer.py:24-28`), suffix = action tokens only, AdaRMS cond = time only (`src/openpi/models/pi0.py:161-169`); π0.5 jointpos training converts absolute→delta actions via `DeltaActions(make_bool_mask(7,-1))`; vlash reference for state-cond: `~/Codes/vlash/vlash/policies/pi05/modeling_pi05.py:160-215` — `adarms_cond = time_emb + silu(state_mlp_out(silu(state_mlp_in(state_proj(state)))))`.
- Δmax = 3, action_horizon H = 15, control 15 Hz. Eval arms: sync d0 (released), naive-async d∈{1,2} (released, fully stale), vlash-async d∈{1,2} (finetune, stale obs + rolled-forward state). 50 trials/arm/task.

---

### Task 0: Branch/asset verification on both machines

**Files:** Create: `docs/superpowers/plans/2026-08-01-vlash-droid-notes.md` (recorded findings; later tasks read constants from here).

**Interfaces:** Produces verified values: `DROID_RLDS_DIR` (nano4 path), `JOINTPOS_CKPT=/work/roboleon1295/checkpoints/pi05_droid_jointpos` (confirm has `params/`), nano4 openpi branch state, local RoboLab venv sanity.

- [ ] **Step 1: Local branch state.** `cd ~/Codes/openpi && git status --short && git log --oneline -2` — expect branch `vlash-droid` with the spec commit. Push it: `git push -u chungyi vlash-droid` (remote `chungyi` = git@github.com:Chung-I/openpi.git, exists).
- [ ] **Step 2: nano4 openpi checkout.** `ssh nano4 'cd /work/roboleon1295/openpi && git remote -v && git status --short && git log --oneline -1'`. Add fork remote if missing (`git remote add chungyi https://github.com/Chung-I/openpi.git`), `git fetch chungyi`. Record the current branch (truncation work — do NOT delete it); create nano4-side worktree if the checkout is dirty: `git worktree add /work/roboleon1295/openpi-vlash vlash-droid` and use THAT path in all later sbatch scripts (record decision in notes file).
- [ ] **Step 3: Find DROID RLDS.** `ssh nano4 'grep -rn "rlds_data_dir\|data_dir" /work/roboleon1295/openpi/src/openpi/training/config.py | head; ls /work/roboleon1295/*droid* /work/roboleon1295/rlds* /work/roboleon1295/tensorflow_datasets 2>/dev/null; find /work/roboleon1295 -maxdepth 3 -name "droid" -type d 2>/dev/null | head'`. Record `DROID_RLDS_DIR` + `du -sh` of it. If absent, STOP and report BLOCKED (dataset staging is a user decision — full DROID RLDS is ~1.8TB > quota; the truncation runs prove *some* form exists).
- [ ] **Step 4: Checkpoint + venv.** `ssh nano4 'ls /work/roboleon1295/checkpoints/pi05_droid_jointpos; /work/roboleon1295/openpi/.venv/bin/python -c "import openpi, jax; print(jax.__version__)"'`.
- [ ] **Step 5: Local RoboLab sanity.** `cd ~/Codes/RoboLab && git status --short | head -3; ls robolab/tasks/benchmark/ | grep ball` and `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` (record whether vLLM still occupies the 5090 — informational, blocks Task 9 only).
- [ ] **Step 6: Write the notes file with all recorded values; commit both repos' branches as needed and push.**

---

### Task 1: π0.5 AdaRMS state conditioning (JAX model surgery)

**Files:**
- Modify: `src/openpi/models/pi0_config.py` (add field), `src/openpi/models/pi0.py` (modules + embed_suffix), `src/openpi/models/tokenizer.py` (no-state π0.5 prompt variant)
- Test: `tests/test_state_cond.py` (new)

**Interfaces:**
- Produces: `Pi0Config(state_cond: bool = False)`; when `pi05=True and state_cond=True`: fresh modules `state_proj` (action_dim→expert width), `state_mlp_in`, `state_mlp_out` (width→width); `adarms_cond = time_emb + state_emb`; tokenizer called with `state=None` but π0.5 prompt style via new arg `pi05_no_state=True` producing `f"Task: {cleaned_text};\nAction: "`.
- Consumed by Tasks 2/3/6.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_state_cond.py
"""State-cond pi05: fresh modules exist, adarms_cond depends on state, prompt has no State: section."""
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0, pi0_config
from openpi.models.tokenizer import PaligemmaTokenizer


def _make(state_cond):
    cfg = pi0_config.Pi0Config(pi05=True, state_cond=state_cond, action_horizon=15)
    return cfg, pi0.Pi0(cfg, rngs=jax.experimental.nnx.Rngs(0)) if False else (cfg, None)


def test_config_flag_default_false():
    cfg = pi0_config.Pi0Config(pi05=True)
    assert cfg.state_cond is False


def test_state_changes_adarms_cond():
    import flax.nnx as nnx
    cfg = pi0_config.Pi0Config(pi05=True, state_cond=True, action_horizon=15)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    x_t = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    t = jnp.array([0.5])
    _, _, _, cond_a = model.embed_suffix(obs, x_t, t)
    obs2 = obs.replace(state=obs.state + 1.0)
    _, _, _, cond_b = model.embed_suffix(obs2, x_t, t)
    assert cond_a is not None
    assert not jnp.allclose(cond_a, cond_b), "adarms_cond must depend on state when state_cond=True"


def test_state_cond_false_matches_stock():
    import flax.nnx as nnx
    cfg = pi0_config.Pi0Config(pi05=True, state_cond=False, action_horizon=15)
    model = pi0.Pi0(cfg, rngs=nnx.Rngs(0))
    obs = cfg.fake_obs()
    x_t = jnp.zeros((1, cfg.action_horizon, cfg.action_dim))
    t = jnp.array([0.5])
    _, _, _, cond_a = model.embed_suffix(obs, x_t, t)
    obs2 = obs.replace(state=obs.state + 1.0)
    _, _, _, cond_b = model.embed_suffix(obs2, x_t, t)
    assert jnp.allclose(cond_a, cond_b), "stock pi05 adarms_cond is time-only"


def test_tokenizer_pi05_no_state_prompt():
    tok = PaligemmaTokenizer(max_len=48)
    with_state, _ = tok.tokenize("pick the ball", state=np.zeros(8))
    no_state, _ = tok.tokenize("pick the ball", state=None, pi05_no_state=True)
    text_with = tok._tokenizer.decode(with_state.tolist())
    text_no = tok._tokenizer.decode(no_state.tolist())
    assert "State:" in text_with
    assert "State:" not in text_no
    assert "Task: pick the ball" in text_no and "Action:" in text_no
```

(Adjust the tokenizer-internal attribute name to whatever `PaligemmaTokenizer` actually holds — read `tokenizer.py` first; the assertion targets are the decoded strings.)

- [ ] **Step 2: Run to verify failure.** `cd ~/Codes/openpi && uv run pytest tests/test_state_cond.py -x -q` → fails (`state_cond` unknown field).
- [ ] **Step 3: Implement.**
  - `pi0_config.py`: add `state_cond: bool = False` alongside `pi05`.
  - `pi0.py` `__init__` (pi05 branch, after `time_mlp_out`):
```python
        if config.pi05 and config.state_cond:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.state_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.state_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
```
    store `self.state_cond = config.state_cond` next to the existing `self.pi05 = config.pi05` (find it; add if absent).
  - `embed_suffix` pi05 branch, after `adarms_cond = time_emb`:
```python
            if self.state_cond:
                state_emb = self.state_proj(obs.state)
                state_emb = nnx.swish(self.state_mlp_in(state_emb))
                state_emb = nnx.swish(self.state_mlp_out(state_emb))
                adarms_cond = adarms_cond + state_emb
```
    (Mirrors vlash modeling_pi05.py:199-204 exactly: proj → mlp_in → silu → mlp_out → silu → add. Note vlash applies silu after each MLP, not after proj.)
  - `tokenizer.py` `tokenize`: add kwarg `pi05_no_state: bool = False`; when set (and `state is None`), build `full_prompt = f"Task: {cleaned_text};\nAction: "` and follow the π0.5 tokenization path (same as the state branch minus the State section).
- [ ] **Step 4: Run tests to green.** Same command; all 4 pass.
- [ ] **Step 5: Commit** `feat(vlash): pi05 AdaRMS state conditioning + no-state prompt variant` (+ session trailer used repo-wide).

---

### Task 2: Temporal-offset transform + rollforward

**Files:**
- Create: `src/openpi/transforms_vlash.py`
- Test: `tests/test_vlash_offsets.py`

**Interfaces:**
- Produces: `VlashTemporalOffset(delta_max: int, action_horizon: int, rng_key: str = "vlash_offset")` — a data transform inserted AFTER `DeltaActions` in the jointpos pipeline. Input sample has `state` (8,), `actions` (H+Δmax, action_dim) in DELTA space for dims 0..6 and raw gripper at dim 7. Output: `actions` (H, action_dim) sliced `[δ:δ+H]`, `state` replaced by rollforward, and `vlash_offset` (int) recorded for logging.
- Rollforward: `state[:7] += sum(actions[:δ, :7])` (delta accumulation); `state[7] = command gripper at δ-1` i.e. `actions[δ-1, 7]` when δ>0 (gripper actions are absolute — the delta mask excludes dim 7); state unchanged when δ=0.
- δ sampling: numpy RNG seeded per-sample (RLDS map determinism: use a hash of a per-example key if available, else stateless `np.random.default_rng` created from the transform's counter — decide from how existing openpi transforms obtain randomness; document the choice in the notes file).

- [ ] **Step 1: Failing tests** (hand-computed fixture):

```python
# tests/test_vlash_offsets.py
import numpy as np
from openpi.transforms_vlash import VlashTemporalOffset, apply_offset


def _sample(H=4, dmax=2, adim=8):
    actions = np.zeros((H + dmax, adim), dtype=np.float32)
    actions[:, 0] = np.arange(H + dmax) + 1        # deltas 1,2,3,...
    actions[:, 7] = 0.1 * (np.arange(H + dmax) + 1)  # absolute gripper cmds
    state = np.zeros(adim, dtype=np.float32)
    state[0] = 10.0
    state[7] = 0.5
    return {"state": state, "actions": actions}


def test_offset_zero_identity():
    s = _sample()
    out = apply_offset(dict(s), delta=0, action_horizon=4)
    np.testing.assert_array_equal(out["actions"], s["actions"][:4])
    np.testing.assert_array_equal(out["state"], s["state"])


def test_offset_two_rollforward():
    s = _sample()
    out = apply_offset(dict(s), delta=2, action_horizon=4)
    np.testing.assert_array_equal(out["actions"], s["actions"][2:6])
    assert out["state"][0] == 10.0 + 1 + 2          # accumulated deltas
    assert abs(out["state"][7] - 0.2) < 1e-6         # gripper cmd at delta-1
    assert out["vlash_offset"] == 2


def test_transform_samples_in_range():
    tr = VlashTemporalOffset(delta_max=2, action_horizon=4)
    seen = set()
    for _ in range(200):
        out = tr(_sample())
        seen.add(int(out["vlash_offset"]))
        assert out["actions"].shape == (4, 8)
    assert seen == {0, 1, 2}
```

- [ ] **Step 2: Verify failure**, **Step 3: Implement** (`apply_offset(sample, delta, action_horizon)` pure function + the sampling wrapper class), **Step 4: green**, **Step 5: commit** `feat(vlash): temporal-offset transform with jointpos rollforward`.

---

### Task 3: Training config + window extension + 12-step repro

**Files:**
- Modify: `src/openpi/training/config.py` (new `TrainConfig pi05_droid_jointpos_vlash`), the DROID RLDS data-config path (window length H+Δmax — locate where `action_horizon` reaches the RLDS loader, likely the `RLDSDataset`/`action_chunk_size` plumbing; extend by `delta_max` for this config only)
- Create: `scripts/slurm/vlash_droid_repro.sbatch` (nano4)
- Test: extend `src/openpi/training/config_test.py` with a config-integrity test

**Interfaces:**
- Produces: config name `pi05_droid_jointpos_vlash` — jointpos base + `Pi0Config(pi05=True, state_cond=True, action_horizon=15)`, RLDS windows of 18, transform chain `... DeltaActions → VlashTemporalOffset(3, 15) → model`, tokenizer `pi05_no_state`, weight loader = released jointpos params with `missing_ok` for the three fresh state modules (find openpi's loader mechanism for partial restore — the LoRA/truncation configs already restore subsets; mirror it). Flags `vlash_shared_obs: bool = False`, and LoRA variant via openpi's existing `freeze_filter` mechanics (Task 5).
- Config test asserts: state_cond on, window=18, offset transform present, baseline `pi05_droid_jointpos` untouched.

- [ ] **Step 1: config test (failing) → Step 2: implement config + window plumbing → Step 3: green locally** (`uv run pytest src/openpi/training/config_test.py -q`).
- [ ] **Step 4: sbatch — 12-step repro on nano4** (dev partition, 1 GPU, PYTHONUNBUFFERED, batch 4, `--exp-name vlash_repro --overwrite`, wandb disabled). Push, pull on nano4, submit. Gate: reaches step 12, checkpoint save works, no NaN loss, log shows offsets sampled (add one log line printing the first batch's `vlash_offset` histogram). Debug until green (this is where RLDS plumbing surprises surface).
- [ ] **Step 5: commit** all + notes update (RLDS window mechanism found, loader decisions).

---

### Task 4: Opt-in shared observation

**Files:** Modify `src/openpi/models/pi0.py` (multi-branch suffix + block mask under flag), `src/openpi/transforms_vlash.py` (emit ALL offsets), config flag `vlash_shared_obs`. Test: `tests/test_shared_obs.py`.

**Interfaces:** when enabled, each batch element carries `states[(Δmax+1), 8]` and `actions[(Δmax+1), H, adim]`; suffix packs Δmax+1 branches (each = its action tokens, with per-branch adarms_cond from its rolled state); attention: every branch sees prefix + itself only. Loss = mean over branches.

- [ ] **Step 1: mask unit test** — build the combined attention mask for a tiny config (H=2, Δmax=1) and assert: prefix↔prefix full, branch_i→prefix allowed, branch_i→branch_j (i≠j) DISALLOWED, branch_i internal causal-by-first-token as in stock. Also per-branch cond isolation: changing branch-1's state must not change branch-0's output (run `compute_loss` twice on crafted inputs).
- [ ] **Step 2: implement** (this is the hardest JAX task — the per-branch adarms_cond requires the expert's adarms path to accept per-token-group conds; if openpi's `adarms_cond` is per-sequence (one vector for the whole suffix — it is, `[b, emb]`), implement branch packing as `(B·(Δmax+1))`-batch replication of the SHARED prefix KV instead of true single-sequence packing IF the mask surgery proves invasive: compute prefix once, broadcast its KV cache across branch batch — same compute saving, no mask surgery. Choose whichever is less invasive after reading `gemma.py`'s cache handling; document in notes. The equivalence gate (Step 4) validates either.)
- [ ] **Step 3: tests green.**
- [ ] **Step 4: equivalence gate** — two 2K-step nano4 runs (shared on/off, same seed, batch equalized by effective trajectories); loss curves must track within noise (overlay in wandb). Record verdict in notes: headline run uses shared-obs only if PASS.
- [ ] **Step 5: commit.**

---

### Task 5: Opt-in LoRA

**Files:** config variant `pi05_droid_jointpos_vlash_lora` reusing openpi's existing LoRA gemma variants (see truncation configs for the pattern); freeze_filter amended so `state_proj|state_mlp_in|state_mlp_out` stay trainable. Test: param-partition test asserting those params are in the trainable set and base attention weights are frozen.

- [ ] Steps: failing partition test → wire config → green → commit. (No training run here; it's an offered ablation.)

---

### Task 6: Serving path + tunnel + PRE-TRAINING smoke (de-risks eval before GPU-hours)

**Files:**
- Modify: `scripts/serve_policy.py` env/args if needed (it selects config+checkpoint; ensure a `--config pi05_droid_jointpos_vlash --checkpoint <dir>` path works and that the policy input pipeline uses `pi05_no_state` tokenization when the config has `state_cond=True` — this lives in the policy-serving transform assembly; locate via `create_trained_policy` equivalent in openpi serving code)
- Create: `scripts/slurm/vlash_droid_serve.sbatch` (1 GPU, runs websocket server on `$PORT`, prints node hostname), `docs/superpowers/plans/tunnel-notes.md` (bring-up commands)
- RoboLab side smoke: run `policies/pi0_family/run.py` (stock, sync) against the RELEASED checkpoint served from nano4, tasks `static_ball_in_bowl_task`, 2 episodes.

- [ ] **Step 1:** serve released `pi05_droid_jointpos` on a nano4 dev job; record node + port.
- [ ] **Step 2:** tunnel: `ssh -L 8000:<node>:8000 nano4` (background, ControlMaster). Verify from local: openpi client health/metadata call.
- [ ] **Step 3:** RoboLab 2-episode static smoke on the 5090 (needs GPU free). Gate: episodes run end-to-end, actions plausible, no protocol errors; record wall-clock per `/predict` (this measures Δ for the arms: steps = ceil(latency×15Hz)).
- [ ] **Step 4:** commit scripts + notes. (If the 5090 is still occupied, mark the RoboLab half deferred and continue — Tasks 7/8 don't need it; Task 9 does.)

---

### Task 7: Headline VLASH training run

**Files:** Create `scripts/slurm/vlash_droid_train.sbatch` (4×H200 `8gpus`, 24h, effective batch 32, 30K steps, wandb `vlash-droid`, per-offset val loss logged — add an eval hook computing loss at fixed δ∈{0..3} on a held-out RLDS slice every 1K steps).

- [ ] Submit (shared-obs per Task 4 verdict; else defaults). Health gates (first 30 min): loss decreasing; offset histogram uniform; quota >100G; s/step recorded + walltime projection <20h. Monitor to completion; verify final checkpoint dir contains params; per-offset losses SEPARATE (δ>0 ≠ δ=0 curves) — this is the AdaRMS-channel-in-use evidence. Commit sbatch + record run URL.

---

### Task 8: RoboLab delay-emulating runner (build while Task 7 trains)

**Files:**
- Create: `~/Codes/RoboLab/policies/pi0_family/vlash_executor.py` (port of the LIBERO `DelayedChunkExecutor` incl. `stale_state` mode — copy semantics + tests from `~/Codes/vlash/benchmarks/libero/executor.py` and `~/Codes/vlash/tests/test_delayed_chunk_executor.py`), `~/Codes/RoboLab/policies/pi0_family/run_vlash_arms.py` (arm runner: `--arm {sync,naive,vlash} --delay N --episodes N --task T`), tests `~/Codes/RoboLab/tests/test_vlash_executor.py`.
- RoboLab is NVIDIA's repo: create branch `vlash-eval` on the user's RoboLab fork (create fork if none: `gh repo fork NVLabs/RoboLab --clone=false`; RoboLab runs LOCALLY so GitHub sync is for provenance, not transfer).

**Interfaces:** arm semantics — sync: fresh obs+state each chunk; naive: obs AND `observation/joint_position` from step T−Δ; vlash: obs from T−Δ, `observation/joint_position` REPLACED by the last commanded action of the previous chunk (client-side rollforward — exact for jointpos), gripper likewise from last command. Chunk = server's `open_loop_horizon` actions; delay in SIM STEPS.

- [ ] Steps: executor tests (reuse LIBERO expectations incl. stale-state) → port → runner (episode loop calls RoboLab's task registration like `run.py` does; per-episode incremental JSON identical schema to the LIBERO harness) → green local tests → commit.

---

### Task 9: Eval smokes + gates (needs 5090 free)

- [ ] Released ckpt, sync arm, 10 episodes per task. Gate: static success ≥20% (else the tasks are out-of-distribution for the policy — STOP and report; reaction contrast has no power from floor). Record baseline latency → pin Δ set (expect {1,2}).
- [ ] VLASH ckpt served (state_cond path), sync-mode 2-episode smoke: verifies serve-side transform (the pi05_libero lesson: state path mismatches are silent — compare a dataset-frame prediction against the released model as a sanity probe first if anything looks off).

---

### Task 10: Full sweeps

- [ ] 5 arms × 2 tasks × 50 trials, resumable per-episode JSONs under `~/Codes/RoboLab/output/vlash_arms/`. Run arms sequentially (one server checkpoint at a time: released for sync+naive, finetune for vlash arms). Monitor + spot-check early success rates per arm.

---

### Task 11: Aggregate, wandb, report

- [ ] Aggregator (LIBERO `aggregate_results.py` pattern): table = arms × tasks, wandb run `vlash-droid/robolab-arms`; verify success criteria from the spec (static parity within 5 pts; rolling VLASH ≥ naive+15 pts); write summary to `docs/superpowers/plans/2026-08-01-vlash-droid-RESULTS.md`; commit; final report to user.

---

## Self-Review Notes

- Spec coverage: offsets+rollforward (T2/T3), AdaRMS-from-start (T1), opt-in shared-obs (T4) + LoRA (T5), venues + tunnel (T6), 5-arm eval (T8-T10), gates everywhere (LIBERO lessons: pre-training serve smoke T6, per-offset loss evidence T7, floor gate T9, quota checks).
- Honest unknowns are confined to Task 0 (RLDS path/branch state) and explicitly-marked implementation decision points (RLDS randomness source T2, window plumbing T3, packing-vs-KV-broadcast T4, serve transform locus T6) — each with a "document in notes" requirement rather than a silent guess.
- Type consistency: `state` 8-dim everywhere client-side; model sees action_dim-padded (32) state — padding happens in openpi's existing `DroidInputs`/normalization pipeline unchanged; rollforward operates on the 8-dim pre-padding sample in T2 and on the client's raw joints in T8.
