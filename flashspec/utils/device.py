"""Device detection helpers and reproducibility utilities.

All code that needs to detect or configure the compute device, or that
needs to set random seeds for reproducibility, should use this module.
"""

from __future__ import annotations

import importlib.util
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

__all__ = [
    "get_device",
    "set_seed",
    "is_cuda_available",
    "device_memory_gb",
    "CompileDecision",
    "resolve_compile",
    "triton_version",
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
        If ``device_str`` is empty or whitespace-only.
    RuntimeError
        If a CUDA device is requested but CUDA is not available.

    Notes
    -----
    This is the single authoritative entry-point for device creation in
    FlashSpec.  Direct calls to ``torch.device()`` in library code are
    avoided; always use this function instead.  The CUDA availability
    check prevents silent CPU fallback on misconfigured systems.

    Examples
    --------
    >>> device = get_device("cuda:0")
    >>> device
    device(type='cuda', index=0)
    """
    if not device_str.strip():
        raise ValueError(f"device_str must be a non-empty string; got: {device_str!r}")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({device_str!r}) but CUDA is not available. "
            "Set device='cpu' or ensure a CUDA-capable GPU is present."
        )
    return device


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for full reproducibility.

    Sets seeds in ``torch``, ``numpy``, and the stdlib ``random`` module,
    and enables CuDNN determinism when CUDA is available.

    Parameters
    ----------
    seed : int
        Integer seed value.  All test fixtures and benchmark runs must call
        this function rather than setting seeds individually.

    Returns
    -------
    None

    Notes
    -----
    Sets four seed sources atomically:

    1. ``torch.manual_seed(seed)`` — covers CPU and GPU PyTorch ops.
    2. ``torch.cuda.manual_seed_all(seed)`` — covers all CUDA devices.
    3. ``numpy.random.seed(seed)`` — covers NumPy-based bandit samplers.
    4. ``random.seed(seed)`` — covers stdlib random (used by hypothesis).

    CuDNN determinism (``torch.backends.cudnn.deterministic = True``,
    ``benchmark = False``) is enabled when CUDA is available.  This may
    reduce throughput on the first run but ensures bit-exact reproducibility
    across runs.

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
        ``True`` if ``torch.cuda.is_available()`` returns ``True``.

    Notes
    -----
    This is a thin wrapper that exists so tests and library code never
    import ``torch`` just to call ``torch.cuda.is_available()``.  It also
    makes the check mockable in unit tests without patching ``torch``
    directly.

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
        Total GPU memory in gibibytes (GiB = 2³⁰ bytes), or ``0.0`` for
        CPU devices.

    Notes
    -----
    Reads ``torch.cuda.get_device_properties(device).total_memory`` and
    converts to GiB by dividing by 1024³.  For CPU devices the return
    value is always ``0.0``; callers should check before using it as a
    denominator.

    Examples
    --------
    >>> device_memory_gb(torch.device("cpu"))
    0.0
    """
    if device.type != "cuda":
        return 0.0
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024**3)


def triton_version() -> str | None:
    """Return the installed Triton version, or ``None`` if Triton is not importable.

    Only the package metadata is read; Triton itself is not imported.

    Returns
    -------
    str or None
        Version string such as ``"3.1.0"``, ``"unknown"`` when the package
        exists without metadata, or ``None`` when it is not installed.
    """
    if importlib.util.find_spec("triton") is None:
        return None
    from importlib import metadata

    for dist in ("triton", "pytorch-triton", "triton-windows"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


@dataclass(frozen=True)
class CompileDecision:
    """Whether the static-cache + CUDA-graph path is used, and why.

    Attributes
    ----------
    requested : bool or "auto"
        The ``FlashSpecConfig.compile`` value.
    enabled : bool
        Whether the engine should switch to the compiled path.
    reason : str
        One human-readable sentence explaining ``enabled``.
    """

    requested: bool | Literal["auto"]
    enabled: bool
    reason: str


def resolve_compile(
    requested: bool | Literal["auto"],
    device: torch.device,
    *,
    triton: str | None = None,
    check_triton: bool = True,
) -> CompileDecision:
    """Decide whether ``FlashSpecConfig.compile`` turns the compiled path on.

    ``"auto"`` is on when ``device`` is a CUDA device and Triton (the
    backend ``torch.compile`` uses for CUDA) is importable, and off
    otherwise. ``True`` is on for any CUDA device; without Triton
    ``torch.compile`` is expected to fail on the first call, and the
    static-cache session then falls back to eager and records why.
    ``False`` is always off.

    Parameters
    ----------
    requested : bool or "auto"
        The configured value.
    device : torch.device
        The engine device.
    triton : str or None
        Triton version to assume. Only used when ``check_triton`` is
        ``False`` (tests); otherwise :func:`triton_version` is called.
    check_triton : bool
        Look Triton up with :func:`triton_version`.

    Returns
    -------
    CompileDecision
        The decision and its reason.

    Examples
    --------
    >>> resolve_compile("auto", torch.device("cpu")).enabled
    False
    """
    if requested is False:
        return CompileDecision(requested, False, "compile=False")
    label = "compile=auto" if requested == "auto" else "compile=True"
    if device.type != "cuda":
        return CompileDecision(
            requested,
            False,
            f"{label} needs a CUDA device for CUDA graphs; the device is {device.type}",
        )
    version = triton_version() if check_triton else triton
    if version is None:
        if requested == "auto":
            return CompileDecision(
                requested,
                False,
                f"{label}: Triton is not installed, and torch.compile needs it on CUDA",
            )
        return CompileDecision(
            requested,
            True,
            f"{label} on {device}: Triton is not installed, so torch.compile may fail "
            "and fall back to eager",
        )
    return CompileDecision(
        requested, True, f"{label}: CUDA device {device} and Triton {version} are available"
    )
