"""VLASH per-offset held-out validation loss (Task 7 of VLASH-on-DROID).

Every `TrainConfig.vlash_val_interval` steps, `scripts/train.py` calls `PerOffsetValLoss.compute`
to report flow-matching loss at each FIXED temporal offset delta in `[0, vlash_delta_max]`,
logging `val_loss/delta_{d}` to wandb. This isolates the AdaRMS state-conditioning channel's
per-offset behavior from the single averaged training `loss` scalar (which mixes whatever
deltas the training-time random `VlashTemporalOffset` / `VlashAllOffsets` happened to sample
that step) -- a curve where `val_loss/delta_0` and `val_loss/delta_{d>0}` visibly separate is
evidence the AdaRMS state-conditioning channel is actually in use.

Design (matches the Task 7 brief):
  - ONE small held-out batch is fetched and cached PER delta at hook-construction time
    (`PerOffsetValLoss.build`), via a throwaway data loader whose `RLDSDroidDataConfig` sets
    `vlash_fixed_delta` -- reusing the exact same `apply_offset` core the training-time offset
    transform uses (`transforms_vlash.VlashFixedOffset`), just forced instead of sampled. The
    batch is never refetched: cheap by construction, one forward pass per delta every
    `vlash_val_interval` steps, not a new data pull.
  - `compute()` always calls the SINGLE-branch loss body (`Pi0._compute_loss_single`)
    regardless of whether the model being evaluated is a shared-obs arm: each cached batch is
    a single-branch (state, action-chunk) pair at one delta, and the shared-obs dispatch
    inside `Pi0.compute_loss` requires `observation.vlash_states`, which a fixed-delta batch
    does not (and should not) carry.
  - A fixed (step-independent) rng is used for noise/time, so successive evals compare the
    model's improvement on IDENTICAL flow-matching inputs rather than mixing in fresh
    randomness -- isolates "did the model get better at this delta" from "did we resample a
    different noise/time".
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger("openpi")

# Step-independent base rng for noise/time -- deliberately NOT derived from the training step,
# so every call to `compute()` over the life of a run evaluates the SAME flow-matching inputs
# per delta and only the model's weights differ between calls.
_EVAL_RNG = jax.random.key(0xDEAD_5EED)


def _fixed_delta_config(config: _config.TrainConfig, *, delta: int, eval_batch_size: int) -> _config.TrainConfig:
    """Builds a throwaway single-delta TrainConfig for caching one held-out batch.

    Factored out of `PerOffsetValLoss.build` so it's unit-testable without a real RLDS/GCS data
    loader (`config.data.create(...)` alone touches no network) -- this exact construction
    caused a real bug once: for a shared-obs arm (`config.model.vlash_shared_obs=True`), forcing
    `vlash_shared_obs=False` on the DATA side only (without also overriding the MODEL side) trips
    `RLDSDroidDataConfig.create()`'s data/model agreement check
    (`ValueError: vlash_shared_obs mismatch`), caught by the Task 7 equivalence-gate job.
    """
    data_cfg = config.data
    if not isinstance(data_cfg, _config.RLDSDroidDataConfig) or data_cfg.vlash_delta_max is None:
        raise ValueError(
            "PerOffsetValLoss requires an RLDSDroidDataConfig with vlash_delta_max set "
            f"(got {type(data_cfg).__name__}, vlash_delta_max={getattr(data_cfg, 'vlash_delta_max', None)})."
        )
    # If `config.model` is a shared-obs arm, the fixed-delta batch is deliberately single-branch
    # (one delta, no `vlash_states`) -- build a matching single-branch copy of the MODEL config
    # too, purely for `RLDSDroidDataConfig.create()`'s data/model agreement check and so
    # `ModelTransformFactory` does not append `SplitVlashBranches`. This does NOT affect the
    # actual training model (only `config.model`, a separate object, is ever merged with
    # `train_state.params` in scripts/train.py).
    fixed_model_cfg = dataclasses.replace(config.model, vlash_shared_obs=False)
    fixed_data_cfg = dataclasses.replace(
        data_cfg,
        vlash_fixed_delta=delta,
        vlash_shared_obs=False,
        # We only ever pull ONE batch then cache it for the life of the run, so a small
        # dedicated buffer avoids paying the full 250k-timestep (~75GB) shuffle buffer cost
        # just to draw one held-out batch.
        shuffle_buffer_size=min(data_cfg.shuffle_buffer_size or 50_000, 2_000),
    )
    return dataclasses.replace(config, batch_size=eval_batch_size, model=fixed_model_cfg, data=fixed_data_cfg)


@dataclasses.dataclass
class PerOffsetValLoss:
    """Caches one held-out (observation, actions) batch per fixed temporal offset."""

    batches: dict[int, tuple[_model.Observation, _model.Actions]]

    @classmethod
    def build(
        cls,
        config: _config.TrainConfig,
        *,
        sharding: jax.sharding.Sharding | None = None,
        deltas: tuple[int, ...] | None = None,
        eval_batch_size: int = 8,
    ) -> "PerOffsetValLoss":
        """Builds one throwaway single-delta data loader per delta and caches its first batch.

        Requires `config.data` to be an `RLDSDroidDataConfig` with `vlash_delta_max` set (the
        RLDS window extension that gives `VlashFixedOffset` steps to slice from is threaded
        through `rlds_action_horizon`, computed inside `RLDSDroidDataConfig.create()`).
        """
        data_cfg = config.data
        if not isinstance(data_cfg, _config.RLDSDroidDataConfig) or data_cfg.vlash_delta_max is None:
            raise ValueError(
                "PerOffsetValLoss requires an RLDSDroidDataConfig with vlash_delta_max set "
                f"(got {type(data_cfg).__name__}, vlash_delta_max="
                f"{getattr(data_cfg, 'vlash_delta_max', None)})."
            )
        resolved_deltas = deltas if deltas is not None else tuple(range(data_cfg.vlash_delta_max + 1))

        batches: dict[int, tuple[_model.Observation, _model.Actions]] = {}
        for delta in resolved_deltas:
            fixed_config = _fixed_delta_config(config, delta=delta, eval_batch_size=eval_batch_size)
            loader = _data_loader.create_data_loader(fixed_config, sharding=sharding, shuffle=True, num_batches=1)
            (batches[delta],) = list(loader)
            logger.info(f"[vlash-val] cached held-out batch for delta={delta}")

        return cls(batches=batches)

    def compute(self, model: _pi0.Pi0) -> dict[str, float]:
        """Runs one no-grad forward pass per cached delta.

        Returns `{"val_loss/delta_{d}": float}` for every cached delta. Uses
        `Pi0._compute_loss_single` directly (not the `compute_loss` dispatch) so this works
        uniformly whether `model` is a plain or shared-obs VLASH arm -- see module docstring.
        """
        out: dict[str, float] = {}
        for delta, (observation, actions) in self.batches.items():
            delta_rng = jax.random.fold_in(_EVAL_RNG, delta)
            loss = model._compute_loss_single(delta_rng, observation, actions, train=False)  # noqa: SLF001
            out[f"val_loss/delta_{delta}"] = float(jnp.mean(loss))
        return out
