import os
import numpy as np
import hashlib

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def calc_dataset_stats(prompts):
    lens = [len(prompt) for prompt in prompts]
    np_vec = np.array(lens)
    return {
        "mean": np_vec.mean().item(),
        "min": np_vec.min().item(),
        "max": np_vec.max().item(),
        "median": np.median(np_vec).item()
}

def hash_string_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()