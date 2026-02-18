
import os
import sys
import numpy as np
import scipy.io as sio
import scipy.signal as sps

print("Starting extraction test...")
sys.stdout.flush()

BASE_DIR = r"d:\Mendeley-Sound-DS"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MF = "FADHD.mat"
FPATH = os.path.join(DATASET_DIR, MF)

print(f"Loading {FPATH}...")
if not os.path.exists(FPATH):
    print("File not found!")
    sys.exit(1)

try:
    d = sio.loadmat(FPATH)
    print("Loaded mat file.")
    d = {k: v for k, v in d.items() if not k.startswith('__')}
    key = next(iter(d.keys()))
    cell = d[key]
    print(f"Key: {key}, Shape: {cell.shape}")
except Exception as e:
    print(f"Error loading: {e}")
    sys.exit(1)

print("Test complete.")
