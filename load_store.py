import numpy as np
import os


# dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
path = "checkpoints/neulab/gpt2-finetuned-wikitext103"
vals_filename = "dstore_gpt2_116988150_768_vals.npy"
dstore_size = 19254850
values = np.memmap(os.path.join(path, vals_filename), dtype=np.int32, mode="r", shape=(dstore_size, 1))

print(dstore_vals.shape)
print(dstore_vals[0])
