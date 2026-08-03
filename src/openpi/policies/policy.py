from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        norm_stats: dict | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        # --- Real-Time Chunking (arXiv 2506.07339) serving support ----------------
        # Active only when a request carries "rtc/*" keys AND the model implements
        # sample_actions_rtc. The server (not the client) caches each env's previous
        # chunk in MODEL space plus the raw joint state it was anchored on, because
        # re-expressing the previous chunk in the new request's delta frame needs the
        # action norm stats -- which live here, not on the client.
        self._norm_stats = norm_stats
        self._rtc_prev: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sample_actions_rtc = None
        if not is_pytorch and hasattr(model, "sample_actions_rtc"):
            self._sample_actions_rtc = nnx_utils.module_jit(model.sample_actions_rtc)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # RTC control fields ride alongside the observation; strip them before the
        # input transforms (which expect only observation keys). Also capture the raw
        # joint state -- the delta frame of this request -- before any transform runs.
        rtc = {k[4:]: inputs.pop(k) for k in list(inputs) if k.startswith("rtc/")}
        raw_joints = None
        if "observation/joint_position" in inputs:
            raw_joints = np.asarray(inputs["observation/joint_position"], dtype=np.float64).copy()
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        env_id = int(rtc.get("env_id", 0)) if rtc else 0
        use_rtc = bool(rtc) and self._sample_actions_rtc is not None and env_id in self._rtc_prev
        if use_rtc:
            prev_model, prev_joints = self._rtc_prev[env_id]
            horizon = prev_model.shape[0]
            d = int(rtc["inference_delay"])
            pah = int(rtc.get("prefix_attention_horizon", horizon - int(rtc["executed"])))
            shift = int(rtc["executed"])
            # Align the previous chunk to this request's frame: index 0 = this request's
            # observation time. Pad the tail by edge-repeat; weights are zero there.
            aligned = np.concatenate([prev_model[shift:], np.repeat(prev_model[-1:], shift, axis=0)], axis=0)
            # Re-anchor the joint deltas (dims 0..6) from the previous request's state to
            # this one's. Quantile normalization is affine, so the shift is exact:
            # y_norm += 2*(s_prev - s_new)/(q99 - q01 + eps).
            if raw_joints is not None and self._norm_stats is not None and prev_joints is not None:
                stats = self._norm_stats["actions"]
                scale = 2.0 / (np.asarray(stats.q99)[:7] - np.asarray(stats.q01)[:7] + 1e-6)
                aligned[:, :7] = aligned[:, :7] + (prev_joints - raw_joints) * scale
            weights = _rtc_prefix_weights(d, pah, horizon)
            actions = self._sample_actions_rtc(
                sample_rng_or_pytorch_device,
                observation,
                jnp.asarray(aligned)[np.newaxis, ...],
                jnp.asarray(weights),
                **{k: v for k, v in sample_kwargs.items() if k != "noise"},
            )
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if rtc:
            # Cache this chunk (model space) and its anchor state for the next request.
            self._rtc_prev[env_id] = (np.asarray(actions[0], dtype=np.float64), raw_joints)
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _rtc_prefix_weights(start: int, end: int, total: int, schedule: str = "exp") -> np.ndarray:
    """Numpy port of get_prefix_weights (real-time-chunking-kinetix/src/model.py:40).

    start=inference_delay (frozen region, weight 1), end=prefix_attention_horizon
    (weights 0 from here on), exponential decay in between. Computed host-side so the
    jitted RTC sampler takes only arrays.
    """
    start = min(start, end)
    idx = np.arange(total, dtype=np.float64)
    if schedule == "ones":
        w = np.ones(total)
    elif schedule == "zeros":
        w = (idx < start).astype(np.float64)
    elif schedule in ("linear", "exp"):
        w = np.clip((start - 1 - idx) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * np.expm1(w) / (np.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return np.where(idx >= end, 0.0, w)


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
