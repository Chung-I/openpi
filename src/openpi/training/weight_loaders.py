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
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


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


def _unstack_scanned_encoderblocks(flat_img: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert a scan=True SigLIP subtree to the scan=False per-layer layout.

    scan=True stores all transformer blocks as one stacked `encoderblock` param with a
    leading depth axis; scan=False (what VideoViTEncoder uses) expects per-layer
    `encoderblock_{i}` params. Keys are '/'-joined paths relative to the SigLIP root.
    Non-encoderblock keys (embedding, pos_embedding, Transformer/encoder_norm, head) are
    copied unchanged. A subtree already in scan=False layout is returned unchanged.
    """
    out: dict[str, np.ndarray] = {}
    for key, value in flat_img.items():
        parts = key.split("/")
        if "encoderblock" in parts:  # exact segment match (not the "encoderblock_{i}" prefix)
            idx = parts.index("encoderblock")
            depth = value.shape[0]
            for lyr in range(depth):
                new_key = "/".join(parts[:idx] + [f"encoderblock_{lyr}"] + parts[idx + 1 :])
                out[new_key] = value[lyr]
        else:
            out[key] = value
    return out


def _seed_video_img_from_img(loaded: at.Params) -> at.Params:
    """Return a copy of `loaded` with `PaliGemma/video_img` populated from the un-stacked
    `PaliGemma/img` SigLIP params. No-op if `PaliGemma/img` is absent."""
    pg = loaded.get("PaliGemma")
    if not isinstance(pg, dict) or "img" not in pg:
        return loaded
    flat_img = flax.traverse_util.flatten_dict(pg["img"], sep="/")
    flat_video = _unstack_scanned_encoderblocks(flat_img)
    new_pg = dict(pg)
    new_pg["video_img"] = flax.traverse_util.unflatten_dict(flat_video, sep="/")
    return {**loaded, "PaliGemma": new_pg}


@dataclasses.dataclass(frozen=True)
class SiglipToVideoImgWeightLoader(WeightLoader):
    """CheckpointWeightLoader that additionally seeds `PaliGemma/video_img` from the
    checkpoint's pretrained SigLIP (`PaliGemma/img`).

    Without this, `video_img` falls through `missing_regex` to a random init, and a short
    finetune would train the LL/video vision encoder from scratch.
    """

    params_path: str
    missing_regex: str = ".*(lora|state_proj|video_img).*"

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded = _seed_video_img_from_img(loaded)
        return _merge_params(loaded, params, missing_regex=self.missing_regex)
