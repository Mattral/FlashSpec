"""Linux AVX2 detection reads /proc/cpuinfo and does not load the kernel."""

from __future__ import annotations

import io
import sys

import pytest

from flashspec.utils.cpu import cpu_has_avx2


def test_linux_flags_with_avx2(monkeypatch: pytest.MonkeyPatch) -> None:
    """An avx2 flag on the flags line returns True."""
    monkeypatch.setattr(sys, "platform", "linux")
    text = "processor : 0\nflags : fpu sse avx avx2\n"

    def _open(*args: object, **kwargs: object) -> io.StringIO:
        return io.StringIO(text)

    monkeypatch.setattr("builtins.open", _open)
    assert cpu_has_avx2() is True


def test_linux_features_without_avx2(monkeypatch: pytest.MonkeyPatch) -> None:
    """A features line that lacks avx2 returns False."""
    monkeypatch.setattr(sys, "platform", "linux")
    text = "features : neon asimd\nnoise without a colon\n"

    def _open(*args: object, **kwargs: object) -> io.StringIO:
        return io.StringIO(text)

    monkeypatch.setattr("builtins.open", _open)
    assert cpu_has_avx2() is False


def test_linux_unreadable_cpuinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing cpuinfo file returns False."""
    monkeypatch.setattr(sys, "platform", "linux")

    def _open(*args: object, **kwargs: object) -> io.StringIO:
        raise OSError("absent")

    monkeypatch.setattr("builtins.open", _open)
    assert cpu_has_avx2() is False


def test_other_platforms_are_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Platforms other than Windows and Linux do not claim AVX2."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cpu_has_avx2() is False
