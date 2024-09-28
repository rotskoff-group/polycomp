import cupy as cp
import numpy as np

#Script to merge all of the individual numpy fields files into one big file
#Also shuffles the order of all the configurations so that the test train split is fully random
files = ["AB_fields.npy", "AC_fields.npy", "BC_fields.npy", "ABC_fields.npy", "CAABBC_fields.npy", "ABCCBA_fields.npy"]

fields = []

cp_fields = None
for f in files:
    print(f)
    if not isinstance(cp_fields, cp.ndarray):
        cp_fields = cp.load("../" + f)
        continue
    cp_fields = cp.concatenate((cp_fields, cp.load("../" + f)))
    print(cp.get_default_memory_pool().used_bytes())
    print(cp_fields.shape)
cp.random.shuffle(cp_fields)
mask = ~cp.all(cp.isclose(cp_fields[:,:3], 0, atol=1e-4), axis=(1,2,3))
print(cp_fields.shape[0] - cp.sum(mask))
cp_fields = cp_fields[mask]
cp.save("big_field", cp_fields)
