"""DraftModel protocol and entry-point registry.

New draft models are registered via the ``@register`` decorator or via
Python entry points under ``flashspec.drafters``, so external packages can
extend the registry without modifying this module.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Protocol, runtime_checkable

import torch

from flashspec.utils.logging import get_logger

__all__ = ["DraftModel", "register", "get_drafter", "list_drafters"]

logger = get_logger(__name__)

# ── Internal registry ─────────────────────────────────────────────────────────
_REGISTRY: dict[str, type["DraftModel"]] = {}


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class DraftModel(Protocol):
    """Protocol that every draft model must satisfy.

    A draft model proposes ``gamma`` candidate tokens autoregressively and
    returns their log-probabilities under itself.

    Notes
    -----
    Implementations must be registered via :func:`register` or via the
    ``flashspec.drafters`` entry point group to be discoverable by the engine.

    Examples
    --------
    >>> @register("my-draft")
    ... class MyDraft:
    ...     def generate_draft(self, input_ids, gamma):
    ...         ...
    ...     def compute_logprobs(self, input_ids, draft_ids):
    ...         ...
    """

    def generate_draft(
        self,
        input_ids: torch.Tensor,
        gamma: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate ``gamma`` draft tokens.

        Parameters
        ----------
        input_ids : torch.Tensor
            Current context.  Shape: ``(batch_size, seq_len)``, dtype int64.
        gamma : int
            Number of draft tokens to generate.

        Returns
        -------
        draft_token_ids : torch.Tensor
            Shape: ``(batch_size, gamma)``, dtype int64.
        draft_logprobs : torch.Tensor
            Log-probabilities at each draft step.
            Shape: ``(batch_size, gamma, vocab_size)``, dtype float32.
        """
        ...

    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log-probs of ``draft_token_ids`` given ``input_ids``.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape: ``(batch_size, seq_len)``, dtype int64.
        draft_token_ids : torch.Tensor
            Shape: ``(batch_size, gamma)``, dtype int64.

        Returns
        -------
        torch.Tensor
            Log-probabilities.  Shape: ``(batch_size, gamma, vocab_size)``.
        """
        ...


# ── Decorator ─────────────────────────────────────────────────────────────────

def register(name: str) -> Any:
    """Class decorator that registers a draft model under ``name``.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"llama3-1b"``).  Must be unique.

    Returns
    -------
    Callable
        The decorator function; returns the class unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.

    Examples
    --------
    >>> @register("llama3-1b")
    ... class Llama3_1B_Drafter:
    ...     ...
    """
    def _decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(
                f"Drafter '{name}' is already registered. "
                "Use a unique name or unregister the existing entry first."
            )
        _REGISTRY[name] = cls
        logger.debug("Drafter registered", extra={"name": name, "class": cls.__name__})
        return cls
    return _decorator


# ── Discovery helpers ─────────────────────────────────────────────────────────

def _load_entry_points() -> None:
    """Load external drafters registered via Python entry points."""
    try:
        eps = importlib.metadata.entry_points(group="flashspec.drafters")
    except Exception as exc:  # importlib.metadata may raise on broken env
        logger.debug(
            "Could not read flashspec.drafters entry points",
            extra={"error": str(exc)},
        )
        return
    for ep in eps:
        try:
            ep.load()
        except Exception as exc:
            logger.debug(
                "Failed to load drafter entry point",
                extra={"entry_point": ep.name, "error": str(exc)},
            )


def get_drafter(name: str) -> type[DraftModel]:
    """Look up a registered draft model class by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    type[DraftModel]
        The registered class.

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry.

    Examples
    --------
    >>> DrClass = get_drafter("llama3-1b")
    >>> drafter = DrClass(device="cuda:0")
    """
    _load_entry_points()
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"No drafter registered under '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def list_drafters() -> list[str]:
    """Return all registered drafter names.

    Returns
    -------
    list[str]
        Sorted list of registry keys.

    Examples
    --------
    >>> list_drafters()
    ['llama3-1b', 'llama3-68m']
    """
    _load_entry_points()
    return sorted(_REGISTRY.keys())
