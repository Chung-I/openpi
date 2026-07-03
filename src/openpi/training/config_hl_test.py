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
