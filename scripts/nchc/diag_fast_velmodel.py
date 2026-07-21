"""Velocity hypothesis, done right: load the VELOCITY pi05_droid and score its FAST head
on velocity actions vs joint-position actions (same DROID frames, each native setup)."""
import dataclasses, jax, jax.numpy as jnp, numpy as np
import flax.nnx as nnx
from openpi.training import config as _config
from openpi.training import data_loader as _dl
from openpi.training import sharding as _sharding
from openpi.training import weight_loaders as _wl
from openpi.training import droid_rlds_dataset as _droid
from openpi.models import model as _model

base = _config.get_config("pi0_mem_droid_k1_verify")
base = dataclasses.replace(base, batch_size=8)

# jointpos data config (JOINT_POSITION + jointpos norm-stats == k1)
cfg_jp = base
# velocity data config (JOINT_VELOCITY + velocity norm-stats)
data_vel = dataclasses.replace(base.data,
    action_space=_droid.DroidActionSpace.JOINT_VELOCITY,
    assets=_config.AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets", asset_id="droid"))
cfg_vel = dataclasses.replace(base, data=data_vel)

mesh = _sharding.make_mesh(base.fsdp_devices)
ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
print("loading batches (same frames, shuffle buffer=1)...", flush=True)
obs_vel, _ = next(iter(_dl.create_data_loader(cfg_vel, sharding=ds, shuffle=False)))
obs_jp, _  = next(iter(_dl.create_data_loader(cfg_jp,  sharding=ds, shuffle=False)))
print(f"state |vel-jp| mean = {float(np.mean(np.abs(np.asarray(obs_vel.state)-np.asarray(obs_jp.state)))):.4f}", flush=True)

# build model with VELOCITY pi05_droid params
model = base.model.create(jax.random.key(0))
vel_loader = _wl.SiglipToVideoImgWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params")
partial = vel_loader.load(nnx.state(model).to_pure_dict())
gd, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd, st); model.eval()

key = jax.random.key(0)
def fast(obs):
    o = _model.preprocess_observation(key, obs, train=False)
    return float(jnp.mean(model.compute_loss_fast(o, *model.embed_prefix_ll(o))))

print("=== VELOCITY pi05_droid model FAST loss (native setups) ===", flush=True)
print(f"  on VELOCITY targets:       {fast(obs_vel):.4f}", flush=True)
print(f"  on JOINT-POSITION targets: {fast(obs_jp):.4f}", flush=True)
print(f"  (random ~ log(vocab) ~ 12.5)", flush=True)
