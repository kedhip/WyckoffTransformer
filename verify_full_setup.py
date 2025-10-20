import os
import sys
import torch
import wandb
from pathlib import Path

def check_environment():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    
    print("\nEnvironment variables:")
    print("WANDB_DIR:", os.getenv('WANDB_DIR'))
    print("OMP_NUM_THREADS:", os.getenv('OMP_NUM_THREADS'))
    print("MKL_NUM_THREADS:", os.getenv('MKL_NUM_THREADS'))

def check_directories():
    base_dir = Path(__file__).parent
    required_dirs = [
        "cache/mp_20",
        "cache/mp_20/tokenisers",
        "data/mp_20",
        "yamls/models/NextToken/v6",
    ]
    
    print("\nChecking directories:")
    for d in required_dirs:
        path = base_dir / d
        exists = path.exists()
        print(f"{d}: {'✓' if exists else '✗'}")

def check_dependencies():
    try:
        import ase
        print("\nASE version:", ase.__version__)
    except ImportError:
        print("\nASE: Not installed")
    
    try:
        import pyxtal
        print("Pyxtal version:", pyxtal.__version__)
    except ImportError:
        print("Pyxtal: Not installed")
    
    try:
        from chgnet.model import CHGNet
        print("CHGNet: Installed")
    except ImportError:
        print("CHGNet: Not installed")

if __name__ == "__main__":
    print("=== WyFormer and DiffCSP++ Setup Verification ===\n")
    check_environment()
    check_directories()
    check_dependencies()
