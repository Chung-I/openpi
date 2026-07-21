"""Compute base pi0.5-DROID (step-0/untrained MEM K=1) loss components on a real DROID batch."""
import dataclasses, jax, jax.numpy as jnp
import flax.nnx as nnx
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding as _sharding
from openpi.models import model as _model

cfg = _config.get_config("pi0_mem_droid_k1_verify")
# ll=1, fast=1, hl=0 so total = ll + fast; then split via compute_loss_fast
cfg = dataclasses.replace(
    cfg,
    batch_size=8,
    model=dataclasses.replace(cfg.model, ll_loss_weight=1.0, fast_loss_weight=1.0, hl_loss_weight=0.0),
)
print("building data loader...", flush=True)
mesh = _sharding.make_mesh(cfg.fsdp_devices)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
loader = _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
obs, actions = next(iter(loader))
print("got batch; building step-0 model (weight-loader init == base)...", flush=True)

model = cfg.model.create(jax.random.key(0))
partial = cfg.weight_loader.load(nnx.state(model).to_pure_dict())
gd, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd, st)
model.eval()

key = jax.random.key(0)
# fast loss (raw CE) directly
obs_ll = _model.preprocess_observation(key, obs, train=False)
prefix = model.embed_prefix_ll(obs_ll)
fast = jnp.mean(model.compute_loss_fast(obs_ll, *prefix))
# total = ll + fast  (ll_weight=fast_weight=1, hl=0)
total = jnp.mean(model.compute_loss(key, obs, actions))
ll = total - fast
print(f"=== BASE pi0.5-DROID loss on a real DROID batch (untrained) ===", flush=True)
print(f"  FAST loss (cross-entropy over action tokens): {float(fast):.4f}", flush=True)
print(f"  FLOW ll  (flow-matching MSE):                 {float(ll):.4f}", flush=True)
print(f"  (combined total): {float(total):.4f}", flush=True)
