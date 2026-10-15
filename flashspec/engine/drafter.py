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

__all__ = [
    "DraftModel",
    "get_drafter",
    "list_drafters",
    "load_drafters",
    "register",
    "register_hf",
    "unregister",
]

logger = get_logger(__name__)

# ── Internal registry ─────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[DraftModel]] = {}


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
            Log-probabilities at each draft step after temperature scaling.
            Shape: ``(batch_size, gamma, vocab_size)``, dtype float32.

        Notes
        -----
        Temperature scaling must be applied to the raw logits **before**
        ``log_softmax`` (§7 of AGENTS.md).  The returned ``draft_logprobs``
        are already temperature-scaled log-probs, not raw logits.

        Examples
        --------
        >>> draft_ids, draft_lp = drafter.generate_draft(input_ids, gamma=4)
        >>> draft_ids.shape
        torch.Size([1, 4])
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
            Temperature-scaled log-probabilities.
            Shape: ``(batch_size, gamma, vocab_size)``, dtype float32.

        Notes
        -----
        Used when the drafter needs to re-score previously generated tokens,
        for example after the context has been updated.  Temperature must be
        applied before ``log_softmax`` (§7 of AGENTS.md).

        Examples
        --------
        >>> lp = drafter.compute_logprobs(input_ids, draft_ids)
        >>> lp.shape
        torch.Size([1, 4, 32000])
        """
        ...


# ── Decorator ─────────────────────────────────────────────────────────────────


def register(name: str, *, replace: bool = False) -> Any:  # noqa: ANN401
    """Class decorator that registers a draft model under ``name``.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"llama3-1b"``).  Must be unique across all
        registered drafters unless ``replace`` is true.
    replace : bool
        When ``True``, an existing entry with the same name is overwritten.
        The default matches the historical contract and raises instead.

    Returns
    -------
    Callable
        The decorator function; returns the decorated class unchanged so
        the class can still be used normally after decoration.

    Raises
    ------
    ValueError
        If ``name`` is already registered and ``replace`` is false.

    Notes
    -----
    The registry is a module-level dict ``_REGISTRY``.  Names are
    case-sensitive.  External packages can also register drafters via Python
    entry points under the ``flashspec.drafters`` group without modifying
    this module (see §4.4 of AGENTS.md).

    A bandit with one registered drafter cannot adapt. Register two or
    more realistically different drafters before expecting a gain over a
    fixed choice. See :func:`load_drafters`.

    Examples
    --------
    >>> @register("llama3-1b")
    ... class Llama3_1B_Drafter:
    ...     ...
    """

    def _decorator(cls: type) -> type:
        if name in _REGISTRY and not replace:
            raise ValueError(
                f"Drafter '{name}' is already registered. "
                "Use a unique name or unregister the existing entry first."
            )
        _REGISTRY[name] = cls
        logger.debug("Drafter registered", extra={"name": name, "class": cls.__name__})
        return cls

    return _decorator


def unregister(name: str) -> None:
    """Remove ``name`` from the in-process drafter registry.

    Parameters
    ----------
    name : str
        Registry key. Missing names are ignored so tests can clean up
        unconditionally.

    Returns
    -------
    None
    """
    _REGISTRY.pop(name, None)


def register_hf(
    name: str,
    model_name_or_path: str,
    *,
    replace: bool = False,
) -> type:
    """Register a HuggingFace causal LM under ``name``.

    The registered class accepts ``device``, ``dtype``, ``temperature``,
    and an optional ``model`` module. Two calls with different hub ids
    are how a process exposes more than one real drafter to the bandit.

    Parameters
    ----------
    name : str
        Registry key used by :func:`get_drafter` and :func:`load_drafters`.
    model_name_or_path : str
        HuggingFace hub id or local directory passed to
        ``AutoModelForCausalLM.from_pretrained`` when no module is injected.
    replace : bool
        Overwrite an existing registry entry when ``True``.

    Returns
    -------
    type
        The registered drafter class (a :class:`HuggingFaceDrafter` subclass).

    Examples
    --------
    >>> register_hf("draft-a", "org/small-draft")
    >>> register_hf("draft-b", "org/other-draft")
    >>> drafters = load_drafters(["draft-a", "draft-b"], device="cpu")
    """
    # Local import: hf_drafter does not import this function, but keeping
    # the import inside the function avoids any future cycle at module load.
    from flashspec.engine.hf_drafter import HuggingFaceDrafter

    class _RegisteredHuggingFaceDrafter(HuggingFaceDrafter):
        """Drafter class bound to one hub id."""

        def __init__(
            self,
            device: torch.device | str,
            dtype: torch.dtype = torch.float32,
            temperature: float = 1.0,
            model: Any | None = None,  # noqa: ANN401
            model_name_or_path: str = model_name_or_path,
        ) -> None:
            super().__init__(
                model_name_or_path=model_name_or_path,
                device=device,
                dtype=dtype,
                temperature=temperature,
                model=model,
            )

    class_name = "HuggingFaceDrafter_" + "".join(ch if ch.isalnum() else "_" for ch in name)
    _RegisteredHuggingFaceDrafter.__name__ = class_name
    _RegisteredHuggingFaceDrafter.__qualname__ = class_name
    register(name, replace=replace)(_RegisteredHuggingFaceDrafter)
    return _RegisteredHuggingFaceDrafter


def load_drafters(
    names: list[str] | tuple[str, ...],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    temperature: float = 1.0,
    models: dict[str, Any] | None = None,
) -> list[DraftModel]:
    """Instantiate drafters in arm order.

    A name that is already registered is constructed as
    ``cls(device=..., dtype=..., temperature=...)``. Any other name is
    treated as a HuggingFace id, registered under that same string, and
    constructed the same way. Pass ``models`` to inject modules and skip
    the hub download.

    Practical note: with a single name the returned list is valid, and
    the bandit has nothing to adapt. Pass two or more drafters that
    actually disagree on some prompts before reading anything into a
    bandit-versus-fixed comparison.

    Parameters
    ----------
    names : sequence of str
        Registry keys or HuggingFace ids, one per bandit arm.
    device : torch.device or str
        Device forwarded to each constructor.
    dtype : torch.dtype
        Compute dtype forwarded to each constructor.
    temperature : float
        Sampling temperature forwarded to each constructor.
    models : dict or None
        Optional map of name to an already-built module. Entries are
        forwarded as the ``model`` keyword.

    Returns
    -------
    list[DraftModel]
        One instance per name, in the same order.

    Raises
    ------
    ValueError
        If ``names`` is empty or ``temperature`` is not positive.
    TypeError
        If a registered class does not accept the constructor keywords.
    """
    if not names:
        raise ValueError(
            "names must contain at least one drafter. A bandit comparison needs two or more."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}.")
    if len(names) < 2:
        logger.warning(
            "Loading a single drafter. The bandit cannot adapt and will "
            "not beat a fixed drafter, because there is only one arm.",
            extra={"names": list(names)},
        )
    loaded: list[DraftModel] = []
    for name in names:
        if name not in _REGISTRY:
            # Hub ids are registered on first use so list_drafters() shows them.
            _load_entry_points()
        if name not in _REGISTRY:
            register_hf(name, name)
        drafter_cls = _REGISTRY[name]
        kwargs: dict[str, Any] = {
            "device": device,
            "dtype": dtype,
            "temperature": temperature,
        }
        if models is not None and name in models:
            kwargs["model"] = models[name]
        loaded.append(drafter_cls(**kwargs))
    return loaded


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
        Registry key, as used in :func:`register`.

    Returns
    -------
    type[DraftModel]
        The registered class (not an instance; caller must instantiate it).

    Raises
    ------
    KeyError
        If ``name`` is not found in the registry after loading all entry points.

    Notes
    -----
    Triggers :func:`_load_entry_points` on every call so that external packages
    installed after the interpreter started are discovered automatically.
    The lookup is O(1) dict access after entry-point loading.

    Examples
    --------
    >>> DrClass = get_drafter("llama3-1b")
    >>> drafter = DrClass(device="cuda:0")
    """
    _load_entry_points()
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(f"No drafter registered under '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_drafters() -> list[str]:
    """Return all registered drafter names in alphabetical order.

    Returns
    -------
    list[str]
        Sorted list of registry keys.

    Notes
    -----
    Triggers :func:`_load_entry_points` to discover any externally registered
    drafters before returning the list.  The list reflects the state of the
    registry at call time; it is not a live view.

    Examples
    --------
    >>> list_drafters()
    ['llama3-1b', 'llama3-68m']
    """
    _load_entry_points()
    return sorted(_REGISTRY.keys())
