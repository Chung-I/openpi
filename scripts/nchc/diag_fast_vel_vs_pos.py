"""Confirm pi05_droid_jointpos's FAST head is velocity-trained:
same DROID frames, FAST loss on VELOCITY action targets vs JOINT-POSITION targets,
evaluated against the SAME prefix (only tokenized_action differs)."""
import dataclasses, jax, jax.numpy as jnp
import flax.nnx as nnx
from openpi.training import config as _config
from openpi.training import data_loader as _dl
from openpi.training import sharding as _sharding
from openpi.training import droid_rlds_dataset as _droid
from openpi.models import model as _model

cfg_jp = _config.get_config("pi0_mem_droid_k1_verify")
cfg_jp = dataclasses.replace(cfg_jp, batch_size=8)
# velocity variant: same everything, but JOINT_VELOCITY action space + velocity norm-stats
data_vel = dataclasses.replace(
    cfg_jp.data,
    action_space=_droid.DroidActionSpace.JOINT_VELOCITY,
    assets=_config.AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets", asset_id="droid"),
)
cfg_vel = dataclasses.replace(cfg_jp, data=data_vel)

mesh = _sharding.make_mesh(cfg_jp.fsdp_devices)
ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
print("loading jointpos batch (shuffle=False)...", flush=True)
obs_jp, act_jp = next(iter(_dl.create_data_loader(cfg_jp, sharding=ds, shuffle=False)))
print("loading velocity batch (shuffle=False, same frames)...", flush=True)
obs_vel, act_vel = next(iter(_dl.create_data_loader(cfg_vel, sharding=ds, shuffle=False)))

# sanity: same frames? compare state
import numpy as np
sdiff = float(np.mean(np.abs(np.asarray(obs_jp.state) - np.asarray(obs_vel.state))))
print(f"state |jp - vel| mean = {sdiff:.4f}  (should be ~0 if same frames)", flush=True)

# base model (untrained K1 == pi05_droid_jointpos)
model = cfg_jp.model.create(jax.random.key(0))
partial = cfg_jp.weight_loader.load(nnx.state(model).to_pure_dict())
gd, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd, st); model.eval()

key = jax.random.key(0)
obs_jp_ll = _model.preprocess_observation(key, obs_jp, train=False)
prefix = model.embed_prefix_ll(obs_jp_ll)  # SAME prefix for both

# jointpos FAST loss
fast_jp = float(jnp.mean(model.compute_loss_fast(obs_jp_ll, *prefix)))
# velocity FAST loss: swap ONLY the tokenized_action fields (same prefix)
obs_vel_swapped = dataclasses.replace(
    obs_jp_ll,
    tokenized_action=obs_vel.tokenized_action,
    tokenized_action_mask=obs_vel.tokenized_action_mask,
    tokenized_action_loss_mask=obs_vel.tokenized_action_loss_mask,
)
fast_vel = float(jnp.mean(model.compute_loss_fast(obs_vel_swapped, *prefix)))

print("=== pi05_droid_jointpos base FAST loss (same prefix, same frames) ===", flush=True)
print(f"  FAST loss on VELOCITY action targets:       {fast_vel:.4f}", flush=True)
print(f"  FAST loss on JOINT-POSITION action targets: {fast_jp:.4f}", flush=True)
print(f"  (random baseline over vocab ~ log(vocab); >~12.5 = worse than random)", flush=True)
