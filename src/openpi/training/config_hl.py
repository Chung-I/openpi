"""Standalone HL-only training configs (decoupled from the RLDS TrainConfig registry).

Two arms differ only by init weights: pi05_base vs pi05_droid. Strict LoRA-only:
only the LLM LoRA adapters train; base PaliGemma, the SigLIP image encoder, and the
action expert are all frozen.
"""

import dataclasses
import difflib
import pathlib

import flax.nnx as nnx
import tyro

import openpi.models.pi0_mem_config as pi0_mem_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

# Root for HL data + splits on the training host (nano4). Overridable via CLI.
_DATA_ROOT = "data"


def _model() -> pi0_mem_config.Pi0MEMConfig:
    return pi0_mem_config.Pi0MEMConfig(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        num_video_frames=6,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        lora=True,
    )


@dataclasses.dataclass(frozen=True)
class HLTrainConfig:
    name: tyro.conf.Suppress[str]

    model: pi0_mem_config.Pi0MEMConfig = dataclasses.field(default_factory=_model)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        )
    )
    ema_decay: float | None = None
    grad_accum_steps: int = 1
    # B: also finetune the SigLIP image encoder (base LLM still frozen) for visual grounding.
    train_image_encoder: bool = False

    seed: int = 42
    batch_size: int = 32
    fsdp_devices: int = 1
    num_train_steps: int = 20_000
    log_interval: int = 50
    eval_interval: int = 1_000
    save_interval: int = 5_000
    keep_period: int | None = 5_000  # = save_interval: keep every saved checkpoint (runs overfit; best is early)

    # Data
    manifest_train: str = f"{_DATA_ROOT}/robomind_hl_fr3/manifest.jsonl"
    frames_train: str = f"{_DATA_ROOT}/robomind_hl_fr3"
    manifest_test: str = f"{_DATA_ROOT}/robomind_hl/manifest.jsonl"
    frames_test: str = f"{_DATA_ROOT}/robomind_hl"
    splits_path: str = "splits/fr3_splits.json"
    num_workers: int = 8
    # Oversample update=True (subtask/memory transition) examples to ~match update=False,
    # so the model can't shortcut by copying the input memory. Dataset itself is unchanged.
    upsample_update: bool = False

    # Tokenization / decode
    max_prompt_tokens: int = 48
    max_memory_tokens: int = 128
    max_target_tokens: int = 200
    max_new_tokens: int = 64
    eval_gen_examples: int = 64  # generation-metric subset size per dev slice
    eval_log_samples: int = 8  # generated-vs-target examples logged (stdout + wandb table) per slice

    # Bookkeeping
    checkpoint_base_dir: str = "./checkpoints"
    project_name: str = "mem-hl-training"
    exp_name: str = tyro.MISSING
    wandb_enabled: bool = True
    overwrite: bool = False
    resume: bool = False

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return pathlib.Path(self.checkpoint_base_dir).resolve() / self.project_name / self.exp_name

    @property
    def freeze_filter(self) -> nnx.filterlib.Filter:
        if self.train_image_encoder:
            # B: train LoRA + the SigLIP image encoder (path .../img/...); the base LLM and
            # video_img stay frozen. "/img/" matches SigLIP but NOT "video_img" (no "/img/").
            return nnx.Not(nnx_utils.PathRegex(".*(/img/|lora).*"))
        # A (strict LoRA-only): freeze every param whose path does NOT contain "lora"
        # (base LLM + SigLIP image encoder + action expert all frozen).
        return nnx.Not(nnx_utils.PathRegex(".*lora.*"))

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))


_CONFIGS = [
    HLTrainConfig(
        name="pi0_mem_hl_fr3_base",
        exp_name="pi0_mem_hl_fr3_base",
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    HLTrainConfig(
        name="pi0_mem_hl_fr3_droid",
        exp_name="pi0_mem_hl_fr3_droid",
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    # Upsample arm: frozen SigLIP (matches the A baseline) but oversample update=True examples.
    HLTrainConfig(
        name="pi0_mem_hl_fr3_base_upsample",
        exp_name="pi0_mem_hl_fr3_base_upsample",
        upsample_update=True,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    # B + upsample: unfrozen SigLIP AND update-upsampling (both levers together).
    HLTrainConfig(
        name="pi0_mem_hl_fr3_base_vis_upsample",
        exp_name="pi0_mem_hl_fr3_base_vis_upsample",
        train_image_encoder=True,
        upsample_update=True,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    # B arms: same as above but with the SigLIP image encoder unfrozen (visual grounding).
    HLTrainConfig(
        name="pi0_mem_hl_fr3_base_vis",
        exp_name="pi0_mem_hl_fr3_base_vis",
        train_image_encoder=True,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
    HLTrainConfig(
        name="pi0_mem_hl_fr3_droid_vis",
        exp_name="pi0_mem_hl_fr3_droid_vis",
        train_image_encoder=True,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_droid/params",
            missing_regex=".*(lora|state_proj|video_img).*",
        ),
    ),
]
_CONFIGS_DICT = {c.name: c for c in _CONFIGS}


def get_config(config_name: str) -> HLTrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        hint = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"HL config '{config_name}' not found.{hint}")
    return _CONFIGS_DICT[config_name]


def cli() -> HLTrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})
