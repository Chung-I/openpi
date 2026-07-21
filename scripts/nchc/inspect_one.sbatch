#!/bin/bash
#SBATCH --job-name=h5_inspect
#SBATCH --account=MST114563
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/work/roboleon1295/openpi/logs/h5_inspect_%j.log
set -x
cd /work/roboleon1295/openpi
export HF_HOME=/work/roboleon1295/hf_cache HOME=/work/roboleon1295 XDG_CACHE_HOME=/work/roboleon1295/.cache
SNAP=/work/roboleon1295/hf_cache/hub/datasets--x-humanoid-robomind--RoboMIND/snapshots/e7ffe31d1fe983a42c3d7b79d192f554fc05b86e/benchmark1_0_compressed/h5_franka_1rgb
OUT=/work/roboleon1295/tmp/one_h5; rm -rf "$OUT"; mkdir -p "$OUT"
# stream all cached parts, extract only the FIRST trajectory.hdf5, stop early
cat "$SNAP"/bread_in_basket.tar.gz.part-* | tar -xz --wildcards --occurrence=1 -C "$OUT" "*/data/trajectory.hdf5"
echo "TAR_EXIT=$?"
find "$OUT" -name trajectory.hdf5
uv run --no-sync python - <<PY
import h5py, glob
p=glob.glob("$OUT/**/trajectory.hdf5",recursive=True)[0]
print("FILE:",p)
def show(name,obj):
    t="DS " if isinstance(obj,h5py.Dataset) else "GRP"
    extra=" shape=%s dtype=%s"%(obj.shape,obj.dtype) if isinstance(obj,h5py.Dataset) else ""
    print("  %s %s%s"%(t,name,extra))
with h5py.File(p,"r") as f:
    print("ROOT_ATTRS:",dict(f.attrs))
    f.visititems(show)
PY
