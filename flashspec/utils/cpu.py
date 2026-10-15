"""Host CPU feature detection.

This module only answers whether the CPU advertises AVX2. It does not
load ``flashspec-asm-kernel``. Kernel selection lives in
``flashspec.sampling.rejection`` and falls back to pure PyTorch when the
shared library is missing, the ABI does not match, or AVX2 is absent.
"""

from __future__ import annotations

import sys

__all__ = ["cpu_has_avx2"]

# WinNT.h PROCESSOR_FEATURE: PF_AVX2_INSTRUCTIONS_AVAILABLE
_PF_AVX2_INSTRUCTIONS_AVAILABLE: int = 40


def cpu_has_avx2() -> bool:
    """Return whether the host CPU advertises AVX2.

    Windows uses ``IsProcessorFeaturePresent``. Linux reads the ``flags``
    line of ``/proc/cpuinfo``. Any other platform returns ``False``.

    A ``False`` result only forces the pure-PyTorch residual path. The
    assembly kernel itself is a Linux x86-64 ELF library and is never
    selected from this check alone.

    Returns
    -------
    bool
        ``True`` when the operating system reports AVX2.
    """
    if sys.platform == "win32":
        return _windows_has_avx2()
    if sys.platform.startswith("linux"):
        return _linux_has_avx2()
    return False


def _windows_has_avx2() -> bool:
    """Query the Windows kernel for the AVX2 processor feature.

    Returns
    -------
    bool
        ``True`` when ``IsProcessorFeaturePresent`` reports AVX2.
    """
    import ctypes

    try:
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent(
            _PF_AVX2_INSTRUCTIONS_AVAILABLE
        )
    except (AttributeError, OSError):
        return False
    return bool(present)


def _linux_has_avx2() -> bool:
    """Read ``/proc/cpuinfo`` for an ``avx2`` flag.

    Returns
    -------
    bool
        ``True`` when any CPU flags line contains ``avx2``.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return False
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        if key.strip().lower() not in {"flags", "features"}:
            continue
        if "avx2" in value.split():
            return True
    return False
