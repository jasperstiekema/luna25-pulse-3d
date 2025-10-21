import numpy as np
import pickle
import os
from pathlib import PosixPath, WindowsPath

# The file path that failed to load
path = r"d:\PULSE\results\auc check\lr_1e-4_auc_check-3D-20251009\config.npy"

class WindowsSafeUnpickler(pickle.Unpickler):
    """
    Custom unpickler to safely load PosixPath objects on Windows by
    mapping them back to the appropriate WindowsPath or simply a string.
    """
    def find_class(self, module, name):
        # The key change: if it tries to load 'pathlib.PosixPath',
        # we tell it to use 'pathlib.WindowsPath' or a safe alternative.
        if module == "pathlib" and name == "PosixPath":
            # Map PosixPath to WindowsPath. WindowsPath is generally capable
            # of handling the path string internally.
            return WindowsPath 
        
        # Another common issue is torch's default device/dtype
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return torch.load
        
        # Fallback to default behavior for all other classes
        return super().find_class(module, name)

def load_posix_safe_npy(path):
    """Loads a numpy file potentially containing PosixPath objects."""
    # .npy files are structured as a header followed by data.
    # The allow_pickle=True loading path involves opening the file and
    # passing the file handle to pickle.
    with open(path, 'rb') as f:
        # 1. Read the numpy file header and skip past it
        magic = f.read(6)
        version = f.read(2)
        header_len_bytes = f.read(2)
        header_len = np.frombuffer(header_len_bytes, dtype='<u2')[0]
        header = f.read(header_len)

        # 2. Use the custom unpickler on the remaining data (the pickled object)
        unpickler = WindowsSafeUnpickler(f)
        return unpickler.load()

try:
    data = load_posix_safe_npy(path)
    
    print("Contents of the .npy file:")
    # Check if the loaded object is a dictionary or a list
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    elif isinstance(data, (list, tuple)):
         print("--- Loaded as a List/Tuple ---")
         for i, item in enumerate(data):
             print(f"[{i}]: {item}")
    else:
        print(f"--- Loaded as a single object (Type: {type(data)}) ---")
        print(data)

except Exception as e:
    print(f"\n--- Final Loading Error ---")
    print(f"An error occurred while loading or printing: {e}")