import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


# Every parameter under this prefix is produced by the shared nn.scan over transformer
# blocks and carries a leading depth axis. Verified against a real checkpoint: the
# embedder, final norms, SigLIP and the action projections sit outside it.
_SCANNED_LAYER_PREFIX = "PaliGemma/llm/layers/"


def _gather_scanned_layers(loaded_params: at.Params, keep_layers: tuple[int, ...]) -> at.Params:
    """Slices the transformer stack's scan axis down to `keep_layers`.

    Both the PaliGemma backbone and the action expert live in one scan, so this single
    gather truncates both towers. Non-scanned parameters pass through unchanged.
    """
    if not keep_layers:
        raise ValueError(f"keep_layers must be non-empty, got {keep_layers!r}.")

    flat = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    index = list(keep_layers)
    result = {}
    for k, v in flat.items():
        if k.startswith(_SCANNED_LAYER_PREFIX):
            depth = v.shape[0]
            for i in index:
                if not (0 <= i < depth):
                    raise ValueError(
                        f"keep_layers index {i} is out of range for '{k}', whose scan axis has depth {depth} "
                        f"(valid indices are 0..{depth - 1}). Full keep_layers={tuple(keep_layers)}."
                    )
            result[k] = v[index]
        else:
            result[k] = v
    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class LayerSubsetWeightLoader(WeightLoader):
    """Loads a checkpoint keeping only a subset of its transformer layers.

    `keep_layers` must match the `Pi0Config.keep_layers` of the model being loaded into —
    the model is built at `depth=len(keep_layers)`, and the gathered params must match
    those shapes.

    Example:
        LayerSubsetWeightLoader("gs://.../params", keep_layers=(0, 3, 7, 11, 14, 17))
    """

    params_path: str
    keep_layers: tuple[int, ...]

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = _gather_scanned_layers(loaded_params, self.keep_layers)
        # Add all missing LoRA weights. These come from the freshly built model, so they
        # already carry the truncated depth.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
