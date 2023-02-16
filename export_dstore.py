"""Export the datastore in numpy format to a space-separated text file."""

import numpy as np
from tqdm import tqdm

IN_PATH = "checkpoints/neulab/gpt2-finetuned-wikitext103/dstore_gpt2_116988150_768_vals.npy"
OUT_PATH = "checkpoints/neulab/gpt2-finetuned-wikitext103/dstore_gpt2_116988150_768_vals.txt"

dstore_size = 116988150
vals = np.memmap(IN_PATH, dtype=np.int32, mode='r', shape=(dstore_size, 1))
print("Converting the memmap to a list.")
vals = vals.tolist()

with open(OUT_PATH, "w") as out_file:
    for idx, token in enumerate(tqdm(vals)):
        out_file.write(str(token))
        if idx != len(vals) - 1:
            out_file.write(" ")

print("Completed!")
