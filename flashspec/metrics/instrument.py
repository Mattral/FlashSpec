"""Test and benchmark probes: forward-call lengths and host syncs.

:class:`ForwardRecorder` hooks ``module.forward`` and records the token
length of every call, which is how the forward-count tests tell a
one-forward-per-step loop from one that re-forwards the residual token.

:func:`count_host_syncs` counts device-to-host reads. On CUDA it uses
``torch.cuda.set_sync_debug_mode("warn")`` and counts the warnings. On
CPU nothing actually synchronises, so it counts the Python calls that
would force a sync on a GPU: ``Tensor.item``, ``Tensor.tolist``, and
``bool``/``int``/``float`` of a tensor. The CPU count is a proxy for
the GPU count, not a measurement of it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = ["ForwardRecorder", "SyncCounter", "count_host_syncs"]

_SYNC_METHODS: tuple[str, ...] = ("item", "tolist", "__bool__", "__int__", "__float__")


class ForwardRecorder:
    """Record the ``input_ids`` length of each ``forward`` call.

    Parameters
    ----------
    module : nn.Module
        Model to hook. The hook reads ``input_ids`` from the keyword
        arguments, or the first positional argument.

    Examples
    --------
    >>> with ForwardRecorder(model) as recorder:
    ...     model(torch.zeros(1, 3, dtype=torch.long))
    >>> recorder.lengths
    [3]
    """

    def __init__(self, module: nn.Module) -> None:
        self._module = module
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self.lengths: list[int] = []

    def _hook(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        token_ids = kwargs.get("input_ids")
        if token_ids is None and args:
            token_ids = args[0]
        if isinstance(token_ids, torch.Tensor) and token_ids.ndim == 2:
            self.lengths.append(int(token_ids.shape[1]))

    def attach(self) -> ForwardRecorder:
        """Register the hook. Calling it twice keeps one hook."""
        if self._handle is None:
            self._handle = self._module.register_forward_pre_hook(self._hook, with_kwargs=True)
        return self

    def detach(self) -> None:
        """Remove the hook. Recorded lengths are kept."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        """Forget recorded lengths."""
        self.lengths = []

    @property
    def calls(self) -> int:
        """Number of recorded forward calls."""
        return len(self.lengths)

    def __enter__(self) -> ForwardRecorder:
        return self.attach()

    def __exit__(self, *_exc: object) -> None:
        self.detach()


@dataclass(slots=True)
class SyncCounter:
    """Running count from :func:`count_host_syncs`.

    Parameters
    ----------
    count : int
        Device-to-host reads seen so far.
    by_kind : dict[str, int]
        The same count split by method name (CPU) or ``"cuda"`` (GPU).
    """

    count: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str) -> None:
        """Count one sync of ``kind``."""
        self.count += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1


def _counting(original: Callable[..., Any], name: str, counter: SyncCounter) -> Callable[..., Any]:
    def _wrapped(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        counter.add(name)
        return original(self, *args, **kwargs)

    return _wrapped


@contextmanager
def count_host_syncs(device: torch.device | str = "cpu") -> Iterator[SyncCounter]:
    """Count device-to-host reads inside the ``with`` block.

    Parameters
    ----------
    device : torch.device or str
        CUDA uses the sync debug mode. Every other device patches the
        tensor methods listed in the module docstring.

    Yields
    ------
    SyncCounter
        On CPU it is updated live while the block runs. On CUDA the
        warnings are counted when the block exits.

    Notes
    -----
    The CPU patch is process-wide while the block runs. Do not use it
    from more than one thread.
    """
    counter = SyncCounter()
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():  # pragma: no cover - needs CUDA
        previous_mode = torch.cuda.get_sync_debug_mode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                yield counter
            finally:
                torch.cuda.set_sync_debug_mode(previous_mode)
                for item in caught:
                    if "synchroniz" in str(item.message):
                        counter.add("cuda")
        return
    originals = {name: getattr(torch.Tensor, name) for name in _SYNC_METHODS}
    # Most of these are inherited from the C base class. Restore by
    # deleting the patch so the class dict looks the way it did before.
    owned = {name for name in _SYNC_METHODS if name in vars(torch.Tensor)}
    try:
        for name, original in originals.items():
            setattr(torch.Tensor, name, _counting(original, name, counter))
        yield counter
    finally:
        for name, original in originals.items():
            if name in owned:
                setattr(torch.Tensor, name, original)
            else:
                delattr(torch.Tensor, name)
