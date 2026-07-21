"""Decisive: on the REAL model + REAL DROID batch, does the flow loss gradient reach the
video encoder / backbone? (insulate_flow_prefix=False, fast=0, ll=1)."""
import dataclasses, jax, jax.numpy as jnp, flax.nnx as nnx, collections
from openpi.training import config as _config
from openpi.training import data_loader as _dl, sharding as _sharding
from openpi.models import model as _model

cfg = _config.get_config("pi0_mem_droid_k6_verify")  # already: insulate=False, ll=1, fast=0
cfg = dataclasses.replace(cfg, batch_size=2)
print("insulate_flow_prefix:", cfg.model.insulate_flow_prefix, "ll:", cfg.model.ll_loss_weight, "fast:", cfg.model.fast_loss_weight, flush=True)
mesh=_sharding.make_mesh(1); ds=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
obs, act = next(iter(_dl.create_data_loader(cfg, sharding=ds, shuffle=False)))
# check video mask is real (not all False)
vm = obs.video_image_masks
print("video_image_masks present:", vm is not None, "| any True:", None if vm is None else {k: bool(jnp.any(v)) for k,v in vm.items()}, flush=True)

model = cfg.model.create(jax.random.key(0))
partial = cfg.weight_loader.load(nnx.state(model).to_pure_dict())
gd0, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd0, st)
gd, params = nnx.split(model, nnx.Param)
grads = jax.grad(lambda p: nnx.merge(gd,p).compute_loss(jax.random.key(0), obs, act).mean())(params)
g=collections.defaultdict(float)
for path,leaf in jax.tree_util.tree_leaves_with_path(grads):
    s="/".join(str(getattr(x,'key',getattr(x,'idx',x))) for x in path)
    k='video_img' if 'video_img' in s else ('img' if '/img/' in s else ('expert1' if '_1' in s else ('backbone' if 'llm' in s else 'other')))
    g[k]+=float(jnp.sum(jnp.abs(leaf)))
print("REAL FLOW GRAD:", {k:round(v,4) for k,v in g.items()}, flush=True)
print("video_img trained by flow?", g['video_img']>1e-6, flush=True)
