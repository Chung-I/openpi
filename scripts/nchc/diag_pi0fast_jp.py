"""Definitive tokenization rule-out: score pi0_fast_droid_jointpos (a model FAST-TRAINED
on DROID joint-position) on DROID jointpos with the standard FAST tokenization.
Low loss => tokenization is fine AND this is the right base for the FAST/KI experiment."""
import dataclasses, jax, jax.numpy as jnp
import flax.nnx as nnx
from openpi.training import config as _config
from openpi.training.config import RLDSDroidDataConfig, AssetsConfig
from openpi.training import data_loader as _dl, sharding as _sharding
from openpi.training import weight_loaders as _wl
from openpi.training import droid_rlds_dataset as _droid

cfg = _config.get_config("pi0_fast_droid")  # Pi0FASTConfig(action_dim=8, action_horizon=10)
data = RLDSDroidDataConfig(
    repo_id="droid", rlds_data_dir="gs://gresearch/robotics",
    action_space=_droid.DroidActionSpace.JOINT_POSITION,
    assets=AssetsConfig(assets_dir="gs://openpi-assets-simeval/pi0_fast_droid_jointpos/assets", asset_id="droid"),
)
cfg = dataclasses.replace(cfg, data=data, batch_size=8,
    weight_loader=_wl.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_fast_droid_jointpos/params"))

mesh = _sharding.make_mesh(1)
ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(_sharding.DATA_AXIS))
print("loading DROID jointpos batch...", flush=True)
obs, act = next(iter(_dl.create_data_loader(cfg, sharding=ds, shuffle=False)))
print("building pi0_fast_droid_jointpos model...", flush=True)
model = cfg.model.create(jax.random.key(0))
partial = cfg.weight_loader.load(nnx.state(model).to_pure_dict())
gd, st = nnx.split(model); st.replace_by_pure_dict(partial); model = nnx.merge(gd, st); model.eval()
loss = float(jnp.mean(model.compute_loss(jax.random.key(0), obs, act)))
print(f"=== pi0_fast_droid_jointpos FAST loss on DROID jointpos: {loss:.4f} ===", flush=True)
print("  (LOW ~1-3 => tokenization OK + real FAST head; ~12.5 = random)", flush=True)
