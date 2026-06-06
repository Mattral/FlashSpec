"""Device detection helpers and reproducibility utilities.

All code that needs to detect or configure the compute device, or that
needs to set random seeds for reproducibility, should use this module.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = [
    "get_device",
    "set_seed",
    "is_cuda_available",
    "device_memory_gb",
]


def get_device(device_str: str) -> torch.device:
    """Parse and validate a device string, returning a ``torch.device``.

    Parameters
    ----------
    device_str : str
        Device string such as ``"cuda:0"``, ``"cuda"``, or ``"cpu"``.

    Returns
    -------
    torch.device
        Validated device object.

    Raises
    ------
    ValueError
        If ``device_str`` is empty.
    RuntimeError
        If a CUDA device is requested but CUDA is not available.

    Examples
    --------
    >>> device = get_device("cuda:0")
    >>> device
    device(type='cuda', index=0)
    """
    if not device_str.strip():
        raise ValueError(f"device_str must be non-empty, got: {device_str!r}")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device_str!r}) but CUDA is not available. "
            "Set device='cpu' or ensure a CUDA-capable GPU is present."
        )
    return device


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    Sets seeds in ``torch``, ``numpy``, and the stdlib ``random`` module,
    and enables CuDNN determinism when CUDA is available.

    Parameters
    ----------
    seed : int
        Integer seed value.  All test fixtures must call this function
        rather than setting seeds individually.

    Returns
    -------
    None

    Examples
    --------
    >>> set_seed(42)
    >>> torch.randint(0, 100, (1,))
    tensor([...])
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_cuda_available() -> bool:
    """Return whether at least one CUDA device is visible.

    Returns
    -------
    bool
        ``True`` if ``torch.cuda.is_available()`` is ``True``.

    Examples
    --------
    >>> is_cuda_available()
    False
    """
    return torch.cuda.is_available()


def device_memory_gb(device: torch.device) -> float:
    """Return total GPU memory in GiB for a CUDA device, or 0.0 for CPU.

    Parameters
    ----------
    device : torch.device
        Target device.

    Returns
    -------
    float
        Total GPU memory in gibibytes (GiB), or ``0.0`` for CPU devices.

    Examples
    --------
    >>> device_memory_gb(torch.device("cpu"))
    0.0
    """
    if device.type != "cuda":
        return 0.0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024 ** 3)
