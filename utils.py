"""Utility functions."""
import os
import json
from pathlib import Path


def ensure_dir(directory: str) -> Path:
    """Ensure directory exists."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: str):
    """Save dict to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Load dict from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def setup_wandb_offline_mode():
    """Setup W&B offline mode if no API key."""
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
        print("W&B offline mode enabled (no API key found)")


def get_cuda_info() -> dict:
    """Get CUDA/GPU information."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(",")
                    gpus.append({
                        "name": parts[0].strip(),
                        "memory_gb": int(parts[1].strip().split()[0]) / 1024,
                    })
            return {"gpus": gpus, "available": len(gpus) > 0}
    except Exception as e:
        pass
    
    return {"gpus": [], "available": False}
