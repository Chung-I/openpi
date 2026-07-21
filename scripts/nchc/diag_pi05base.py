"""Test pi05_base FAST head on DROID joint-position (standard tokenization, jointpos norm-stats)."""
import dataclasses, jax, jax.numpy as jnp
import flax.nnx as nnx
from openpi.training import config as _config
from openpi.training import data_loader as _dl, sharding as _sharding
from openpi.training import weight_loaders as _wl
from openpi.models import model as _model

cfg = _config.get_config("pi0_mem_droid_k1_verify")  # DROID jointpos data + std FAST tokenization
cfg = dataclasses.replace(cfg, batch_size=8)
mesh = _sharding.make_mesh(cfg.fsdp_devices)
ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
print("loading DROID jointpos batch...", flush=True)
obs, act = next(iter(_dl.create_data_loader(cfg, sharding=ds, shuffle=False)))
print("building model with pi05_base params...", flush=True)
model = cfg.model.create(jax.random.key(0))
partial = _wl.SiglipToVideoImgWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params").load(nnx.state(model).to_pure_dict())
gd, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd, st); model.eval()
key = jax.random.key(0)
o = _model.preprocess_observation(key, obs, train=False)
fast = float(jnp.mean(model.compute_loss_fast(o, *model.embed_prefix_ll(o))))
print(f"=== pi05_base FAST loss on DROID jointpos: {fast:.4f} ===", flush=True)
print("  (ref: pi0_fast_droid_jointpos=1.20 | pi05_droid(flow-only)~15 | random~12.5)", flush=True)
