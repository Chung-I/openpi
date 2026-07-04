import flax.nnx as nnx
import jax

from openpi.training import config_hl


def test_two_arms_registered():
    base = config_hl.get_config("pi0_mem_hl_fr3_base")
    droid = config_hl.get_config("pi0_mem_hl_fr3_droid")
    assert base.model.lora and droid.model.lora
    assert "pi05_base" in base.weight_loader.params_path
    assert "pi05_droid" in droid.weight_loader.params_path
    assert base.project_name == "mem-hl-training"


def test_freeze_filter_trains_only_lora():
    # Use nnx.eval_shape so we DON'T allocate a real ~2B gemma_2b model on CPU —
    # the freeze/trainable filters are structural (path-based) and work on the
    # abstract (ShapeDtypeStruct) module.
    cfg = config_hl.get_config("pi0_mem_hl_fr3_base")
    abstract = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    trainable = nnx.state(abstract, cfg.trainable_filter)
    leaves = jax.tree_util.tree_leaves_with_path(trainable)
    assert leaves, "expected some trainable lora params"
    for path, _ in leaves:
        assert "lora" in jax.tree_util.keystr(path), f"non-lora trainable param: {jax.tree_util.keystr(path)}"


def test_vis_arm_trains_lora_plus_siglip():
    cfg = config_hl.get_config("pi0_mem_hl_fr3_base_vis")
    assert cfg.train_image_encoder
    abstract = nnx.eval_shape(lambda: cfg.model.create(jax.random.key(0)))
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(nnx.state(abstract, cfg.trainable_filter))
    ]
    assert paths
    # every trainable leaf is a LoRA param or a SigLIP image-encoder param (/img/, not video_img)
    for p in paths:
        assert ("lora" in p) or ("img" in p and "video_img" not in p), p
    # base LLM (non-lora) stays frozen; SigLIP img is actually included
    assert not any(("llm" in p and "lora" not in p) for p in paths)
    assert any(("img" in p and "video_img" not in p) for p in paths)
