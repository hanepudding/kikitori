import platform

import torch


def best() -> str:
    """Where to load models.

    Apple Silicon reports no CUDA, and transformers' "auto" placement then falls
    back to the CPU instead of the GPU.
    """
    if platform.system() == "Darwin":
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def free() -> None:
    """Release cached blocks between the two model loads."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif platform.system() == "Darwin":
        torch.mps.empty_cache()
