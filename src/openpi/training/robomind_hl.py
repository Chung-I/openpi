"""Manifest -> HL training-example transform for Pi0MEM.

Turns the RoboMIND HL assembler's output (manifest.jsonl + frames/, produced by
`openpi.training.robomind.assemble`) into the inputs of `Pi0MEM.compute_loss_hl`:
an `Observation` (single frame o_t + goal g + input memory m_t) paired with the
target text (l_{t+1}, m_{t+1}) the HL policy must predict.

The target is formatted as ``<subtask>..</subtask><memory>..</memory>`` -- the exact
form `openpi.policies.mem_policy.MEMPolicy._parse_hl_output` parses at inference, so
training targets match generation. Tokenization uses the PaliGemma SentencePiece model.

Usage::

    import sentencepiece
    from openpi.training.robomind_hl import RobomindHLDataset, collate_hl, load_paligemma_sp
    ds = RobomindHLDataset("data/robomind_hl_fr3/manifest.jsonl", "data/robomind_hl_fr3",
                           tokenizer=load_paligemma_sp())
    obs_dict, target_tokens, target_mask = collate_hl([ds[i] for i in range(B)])
    obs = Observation.from_dict(obs_dict)
    loss = model.compute_loss_hl(rng, obs, target_tokens, target_mask)
"""

import json
import pathlib
from typing import Protocol

import numpy as np

# Camera key the HL branch reads (single-frame o_t). RoboMIND camera_top -> base_0_rgb.
HL_CAMERA = "base_0_rgb"
HL_TARGET_TEMPLATE = "Subtask: {subtask} Memory: {memory}"
_PADDING_TOKEN_ID = 0
_EOS_TOKEN_ID = 1  # PaliGemma / Gemma SentencePiece EOS


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


def load_paligemma_sp() -> _Tokenizer:
    """Load the PaliGemma SentencePiece tokenizer (same model MEMPolicy uses)."""
    import sentencepiece

    import openpi.shared.download as download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


def _encode(tokenizer: _Tokenizer, text: str, max_len: int, *, add_eos: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Encode `text` to (token_ids[max_len] int32, mask[max_len] bool), right-padded with 0.

    Empty text -> all-pad, all-False (the model ignores it). With add_eos, an EOS token is
    appended (so an HL target teaches the model to stop) before padding/truncation.
    """
    ids = list(tokenizer.encode(text)) if text else []
    if add_eos:
        ids = [*ids, _EOS_TOKEN_ID]
    ids = ids[:max_len]
    n = len(ids)
    tokens = np.full(max_len, _PADDING_TOKEN_ID, dtype=np.int32)
    tokens[:n] = ids
    mask = np.zeros(max_len, dtype=np.bool_)
    mask[:n] = True
    return tokens, mask


def build_hl_example(
    row: dict,
    frames_dir,
    tokenizer: _Tokenizer,
    *,
    state_dim: int = 32,
    max_prompt_tokens: int = 48,
    max_memory_tokens: int = 128,
    max_target_tokens: int = 200,
) -> dict:
    """Build one HL training example from a manifest row (dict) + its frame on disk."""
    import cv2

    frames_dir = pathlib.Path(frames_dir)
    img = cv2.imread(str(frames_dir / row["image"]))  # BGR uint8 HxWx3
    if img is None:
        raise FileNotFoundError(f"frame not found: {frames_dir / row['image']}")
    img = img[:, :, ::-1]  # BGR -> RGB (frames were written RGB via PIL; cv2 reads BGR)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    prompt_t, prompt_m = _encode(tokenizer, row.get("goal", ""), max_prompt_tokens)
    mem_t, mem_m = _encode(tokenizer, row.get("input_memory", ""), max_memory_tokens)
    target_text = HL_TARGET_TEMPLATE.format(subtask=row["target_subtask"], memory=row["target_memory"])
    target_t, target_m = _encode(tokenizer, target_text, max_target_tokens, add_eos=True)

    return {
        "image": img,  # uint8 [H,W,3]; from_dict normalizes to [-1,1]
        "image_mask": np.ones((), dtype=np.bool_),  # scalar True; collate -> [b]
        "state": np.zeros(state_dim, dtype=np.float32),  # HL policy ignores state; required by Observation
        "tokenized_prompt": prompt_t,
        "tokenized_prompt_mask": prompt_m,
        "tokenized_memory": mem_t,
        "tokenized_memory_mask": mem_m,
        "target_tokens": target_t,
        "target_mask": target_m,
    }


class RobomindHLDataset:
    """Map-style dataset over an assembler manifest.jsonl (one HL sample per line)."""

    def __init__(
        self,
        manifest_path,
        frames_dir,
        tokenizer: _Tokenizer,
        *,
        episode_ids: set[str] | None = None,
        state_dim: int = 32,
        max_prompt_tokens: int = 48,
        max_memory_tokens: int = 128,
        max_target_tokens: int = 200,
    ):
        self.frames_dir = pathlib.Path(frames_dir)
        self.tokenizer = tokenizer
        self._kw = {
            "state_dim": state_dim,
            "max_prompt_tokens": max_prompt_tokens,
            "max_memory_tokens": max_memory_tokens,
            "max_target_tokens": max_target_tokens,
        }
        rows = [json.loads(line) for line in pathlib.Path(manifest_path).read_text().splitlines() if line.strip()]
        if episode_ids is not None:
            rows = [r for r in rows if r["episode_id"] in episode_ids]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return build_hl_example(self.rows[i], self.frames_dir, self.tokenizer, **self._kw)


def collate_hl(examples: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Stack examples into (obs_dict for Observation.from_dict, target_tokens, target_mask)."""
    stack = lambda k: np.stack([e[k] for e in examples])  # noqa: E731
    obs_dict = {
        "image": {HL_CAMERA: stack("image")},
        "image_mask": {HL_CAMERA: stack("image_mask")},
        "state": stack("state"),
        "tokenized_prompt": stack("tokenized_prompt"),
        "tokenized_prompt_mask": stack("tokenized_prompt_mask"),
        "tokenized_memory": stack("tokenized_memory"),
        "tokenized_memory_mask": stack("tokenized_memory_mask"),
    }
    return obs_dict, stack("target_tokens"), stack("target_mask")
