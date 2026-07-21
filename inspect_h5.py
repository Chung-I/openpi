import h5py, glob, numpy as np
p=glob.glob("data/robomind_cache_bread/bread_in_basket/**/1016_161244/data/trajectory.hdf5",recursive=True)[0]
print("FILE:",p)
def show(name,obj):
    if isinstance(obj,h5py.Dataset):
        print("  DS  %-45s shape=%s dtype=%s"%(name,obj.shape,obj.dtype))
    else:
        print("  GRP %s"%name)
with h5py.File(p,"r") as f:
    print("ATTRS:",dict(f.attrs))
    f.visititems(show)
