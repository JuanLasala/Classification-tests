import json
import os
import random
import numpy as np
import torch
import glob

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log(msg):
    print(f"[INFO] {msg}")

def load_image_paths(directory):
    """Loads all image paths recursively from a folder."""
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    return image_paths
