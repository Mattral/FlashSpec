"""Batch-1 KV cache for a causal language model.

A session has two ways to stay aligned with the sequence.

*Incremental* (:meth:`KvSession.start`, :meth:`KvSession.advance`,
:meth:`KvSession.crop`, :meth:`KvSession.push_pending`). The caller
says exactly what happened with integers: how many tokens to drop and
which tokens were committed. Committed tokens that have not been
forwarded yet are held as *pending* and are prepended to the next
forward, so committing never runs the model. The speculative engine
uses this mode: one target forward per step.

*Sync* (:meth:`KvSession.sync`). The caller passes the whole accepted
context; the session compares it with the token ids it cached, crops
to the shared prefix and forwards the suffix. The context-based
wrapper methods (``generate_draft``, ``score_draft``,
``compute_logprobs``) use this mode, as do batches the engine does not
run incrementally.

Transformers 5 crops with a negative count: ``crop(-n)`` drops ``n``
tokens. A positive count is the deprecated absolute-length form and is
not used here.

Set ``FLASHSPEC_DEBUG_CACHE=1`` to keep the cached token ids in the
incremental mode too, so :meth:`KvSession.check` can assert that the
integer bookkeeping matches the real sequence. It is off by default
because it concatenates ids on every forward.

Batch sizes other than 1, and models that do not return a cache, stay
on the caller's full-sequence forward.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch

from flashspec.utils.logging import get_logger

__all__ = [
    "CacheUnavailableError",
    "KvSession",
    "debug_cache_enabled",
    "model_accepts_past_key_values",
]

logger = get_logger(__name__)

_DEBUG_ENV = "FLASHSPEC_DEBUG_CACHE"


def debug_cache_enabled() -> bool:
    """Return whether ``FLASHSPEC_DEBUG_CACHE=1`` is set."""
    return os.environ.get(_DEBUG_ENV) == "1"


class CacheUnavailableError(RuntimeError):
    """The model cannot continue this call from ``past_key_values``."""


def model_accepts_past_key_values(model: Any) -> bool:  # noqa: ANN401
    """Return whether ``model.forward`` can take a KV cache.

    Parameters
    ----------
    model : Any
        Module whose ``forward`` may accept ``past_key_values``.

    Returns
    -------
    bool
        True when the signature names ``past_key_values`` or accepts
        arbitrary keywords. A true result does not promise that a call
        returns a cache; :class:`KvSession` disables itself if it does not.
    """
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if "past_key_values" in parameters:
        return True
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def _logits_to_keep_name(model: Any) -> str | None:  # noqa: ANN401
    """Return the keyword that limits returned logit rows, if ``forward`` names one.

    Transformers 4.50 and later call it ``logits_to_keep``; 4.45 to 4.49
    call it ``num_logits_to_keep``. A ``**kwargs`` forward is not trusted
    with either name.
    """
    forward = getattr(model, "forward", None)
    if forward is None:
        return None
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            return name
    return None


def _seq_length(past: Any) -> int | None:  # noqa: ANN401
    """Return the cached token count, or None when the object has none."""
    getter = getattr(past, "get_seq_length", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except (TypeError, ValueError):
        return None


class KvSession:
    """Incremental cache for one sequence.

    Parameters
    ----------
    model : Any
        Causal module. ``forward`` is called under ``torch.no_grad``.
    """

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        self._model = model
        self.enabled = model_accepts_past_key_values(model)
        self._keep_name = _logits_to_keep_name(model)
        self._past: Any = None  # noqa: ANN401
        self._cached_ids: torch.Tensor | None = None
        self._next_logits: torch.Tensor | None = None
        # Incremental mode: tokens in the cache, and committed tokens
        # waiting to be forwarded.
        self._length = 0
        self._pending: torch.Tensor | None = None
        self._debug = False
        self._debug_ids: torch.Tensor | None = None

    def supports(self, token_ids: torch.Tensor) -> bool:
        """Return whether this call can use the cache.

        Parameters
        ----------
        token_ids : torch.Tensor
            Token ids shaped ``(batch, seq)``.

        Returns
        -------
        bool
            True only for a still-enabled session and a batch of one.
        """
        return self.enabled and token_ids.ndim == 2 and token_ids.shape[0] == 1

    def reset(self) -> None:
        """Drop cached tokens without disabling the session."""
        self._past = None
        self._cached_ids = None
        self._next_logits = None
        self._length = 0
        self._pending = None
        self._debug_ids = None

    def disable(self) -> None:
        """Stop using the cache and drop any stored state."""
        self.enabled = False
        self.reset()
        logger.debug("KV cache disabled; later calls forward the full sequence")

    # ── Incremental mode ────────────────────────────────────────────────

    @property
    def pending_length(self) -> int:
        """Committed tokens that the next :meth:`advance` will forward first."""
        return 0 if self._pending is None else int(self._pending.shape[1])

    @property
    def cached_length(self) -> int:
        """Tokens currently in the cache, from the integer bookkeeping."""
        return self._length

    def start(self, prompt: torch.Tensor) -> None:
        """Begin incremental mode on ``prompt``.

        Every prompt token except the last is forwarded now, keeping a
        single logit row (``logits_to_keep=1`` when the model supports
        it). The last token becomes pending, so the first
        :meth:`advance` returns the logits that predict the first new
        token. A one-token prompt is not forwarded at all.

        Parameters
        ----------
        prompt : torch.Tensor
            Shape ``(1, seq)`` with ``seq >= 1``, int64, on the model device.

        Raises
        ------
        CacheUnavailableError
            If the session is disabled, the batch is not one, or the
            model does not return a usable cache.
        """
        if not self.supports(prompt) or int(prompt.shape[1]) < 1:
            raise CacheUnavailableError("start is only used for one non-empty cached row")
        self.reset()
        self._debug = debug_cache_enabled()
        ids = prompt.to(dtype=torch.long)
        if int(ids.shape[1]) > 1:
            head = ids[:, :-1]
            self._past, _ = self._call(head, None, keep=1)
            self._length = int(head.shape[1])
            if self._debug:
                self._debug_ids = head.clone()
        self._pending = ids[:, -1:]

    def advance(self, new_ids: torch.Tensor | None, keep: int) -> torch.Tensor:
        """Forward the pending tokens plus ``new_ids`` in one call.

        Parameters
        ----------
        new_ids : torch.Tensor or None
            Tokens after the pending ones. Shape ``(1, m)``. ``None``
            forwards only the pending tokens.
        keep : int
            Logit rows to return, counted back from the last input
            position. Must be at least 1 and at most the input length.

        Returns
        -------
        torch.Tensor
            Raw logits for the last ``keep`` input positions. Shape
            ``(1, keep, vocab)``, float32. The last row predicts the
            token after the last input token.

        Raises
        ------
        CacheUnavailableError
            If there is nothing to forward, ``keep`` is out of range, or
            the model does not extend the cache.
        """
        pieces = [piece for piece in (self._pending, new_ids) if piece is not None]
        if not pieces:
            raise CacheUnavailableError("advance with no pending or new tokens")
        token_ids = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise CacheUnavailableError("advance expects one row")
        added = int(token_ids.shape[1])
        if not 1 <= keep <= added:
            raise CacheUnavailableError(f"keep={keep} must be in [1, {added}]")
        self._past, logits = self._call(token_ids, self._past, keep=keep)
        self._length += added
        self._pending = None
        if self._debug:
            fresh = token_ids.to(dtype=torch.long)
            self._debug_ids = (
                fresh.clone() if self._debug_ids is None else torch.cat([self._debug_ids, fresh], 1)
            )
        return logits

    def crop(self, n_tokens: int) -> None:
        """Drop the last ``n_tokens`` cached tokens. Never forwards.

        Parameters
        ----------
        n_tokens : int
            Tokens to drop. ``0`` is a no-op.

        Raises
        ------
        CacheUnavailableError
            If ``n_tokens`` is negative or larger than the cache, or the
            cache object cannot crop. The session is left disabled.
        """
        if n_tokens == 0:
            return
        new_len = self._length - n_tokens
        if n_tokens < 0 or new_len < 0 or not self._crop_to(new_len):
            self.disable()
            raise CacheUnavailableError(f"cannot crop {n_tokens} of {self._length} cached tokens")
        self._length = new_len
        if self._debug and self._debug_ids is not None:
            self._debug_ids = self._debug_ids[:, :new_len]

    def push_pending(self, token_ids: torch.Tensor) -> None:
        """Commit ``token_ids`` after the current pending tokens. Never forwards.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(1, n)`` with ``n >= 1``.
        """
        ids = token_ids.to(dtype=torch.long)
        self._pending = ids if self._pending is None else torch.cat([self._pending, ids], dim=1)

    def check(self, context: torch.Tensor) -> None:
        """Assert that cache plus pending tokens equal ``context``.

        Only meaningful when ``FLASHSPEC_DEBUG_CACHE=1`` was set before
        :meth:`start`; otherwise it returns without checking.

        Parameters
        ----------
        context : torch.Tensor
            Accepted sequence. Shape ``(1, seq)``.

        Raises
        ------
        AssertionError
            If the cache length or the token ids disagree.
        """
        if not self._debug:
            return
        seq = _seq_length(self._past) if self._past is not None else 0
        if seq != self._length:
            raise AssertionError(f"cache holds {seq} tokens, bookkeeping says {self._length}")
        pieces = [piece for piece in (self._debug_ids, self._pending) if piece is not None]
        held = torch.cat(pieces, dim=1) if pieces else context.new_empty((1, 0))
        if not torch.equal(held.to(context.device), context.to(dtype=torch.long)):
            raise AssertionError(
                f"cache + pending ids {held.tolist()} do not match context {context.tolist()}"
            )

    # ── Sync mode ───────────────────────────────────────────────────────

    def sync(self, context: torch.Tensor) -> torch.Tensor:
        """Make the cache match ``context`` and return next-token logits.

        Parameters
        ----------
        context : torch.Tensor
            Accepted prefix. Shape ``(1, seq)``, dtype int64, already on
            the model device.

        Returns
        -------
        torch.Tensor
            Raw next-token logits. Shape ``(1, vocab)``, dtype float32.

        Raises
        ------
        CacheUnavailableError
            If the model does not return a usable cache. The session is
            left disabled.
        """
        if not self.supports(context):
            raise CacheUnavailableError("sync is only used for one cached row")
        if not self._aligned(context):
            return self._prefill(context)
        assert self._cached_ids is not None
        pos = self._common_prefix_len(context)
        if pos == 0:
            return self._prefill(context)
        if int(self._cached_ids.shape[1]) > pos:
            if not self._crop_to(pos):
                return self._prefill(context)
            self._cached_ids = self._cached_ids[:, :pos].contiguous()
            self._next_logits = None
        if int(context.shape[1]) > int(self._cached_ids.shape[1]):
            logits = self.append(context[:, int(self._cached_ids.shape[1]) :])
            return logits[:, -1, :]
        if self._next_logits is None:
            return self._replay_last(context)
        return self._next_logits

    def append(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward ``token_ids`` onto the cache.

        Parameters
        ----------
        token_ids : torch.Tensor
            New tokens only. Shape ``(1, n)`` with ``n >= 1``.

        Returns
        -------
        torch.Tensor
            Logits at the new positions. Shape ``(1, n, vocab)``, float32.
            Position 0 predicts the token after the first new token, which
            is the same offset a full forward uses.

        Raises
        ------
        CacheUnavailableError
            If there is no cache yet, or the model does not extend it.
        """
        if self._past is None or self._cached_ids is None:
            raise CacheUnavailableError("append without a cache")
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
            raise CacheUnavailableError("append expects one non-empty row")
        past, logits = self._call(token_ids, self._past)
        self._remember(token_ids, past, logits[:, -1, :])
        return logits

    def _aligned(self, context: torch.Tensor) -> bool:
        if self._past is None or self._cached_ids is None:
            return False
        ids = self._cached_ids
        if ids.shape[0] != context.shape[0] or ids.device != context.device:
            return False
        if ids.dtype != context.dtype:
            return False
        seq = _seq_length(self._past)
        return seq is not None and seq == int(ids.shape[1])

    def _common_prefix_len(self, context: torch.Tensor) -> int:
        assert self._cached_ids is not None
        limit = min(int(self._cached_ids.shape[1]), int(context.shape[1]))
        if limit == 0:
            return 0
        mismatch = self._cached_ids[0, :limit].ne(context[0, :limit])
        if not bool(mismatch.any()):
            return limit
        return int(mismatch.nonzero()[0].item())

    def _prefill(self, context: torch.Tensor) -> torch.Tensor:
        self.reset()
        past, logits = self._call(context, None)
        self._remember(context, past, logits[:, -1, :])
        assert self._next_logits is not None
        return self._next_logits

    def _replay_last(self, context: torch.Tensor) -> torch.Tensor:
        # After a crop, the logit that predicts the next token was the
        # logit at the last kept position. Replaying that one token
        # restores it. A one-token context is cheaper to prefill.
        if int(context.shape[1]) < 2:
            return self._prefill(context)
        keep = int(context.shape[1]) - 1
        if not self._crop_to(keep):
            return self._prefill(context)
        self._cached_ids = context[:, :keep].to(dtype=torch.long).contiguous().clone()
        try:
            logits = self.append(context[:, -1:])
        except CacheUnavailableError:
            return self._prefill(context)
        return logits[:, -1, :]

    def _crop_to(self, new_len: int) -> bool:
        if self._past is None:
            return False
        current = _seq_length(self._past)
        if current is None or new_len < 0 or new_len > current:
            return False
        extra = current - new_len
        if extra == 0:
            return True
        crop = getattr(self._past, "crop", None)
        if not callable(crop):
            return False
        crop(-extra)
        return _seq_length(self._past) == new_len

    def _remember(self, token_ids: torch.Tensor, past: Any, next_logits: torch.Tensor) -> None:  # noqa: ANN401
        fresh = token_ids.detach().to(dtype=torch.long)
        if self._cached_ids is None:
            self._cached_ids = fresh.clone()
        else:
            self._cached_ids = torch.cat([self._cached_ids, fresh], dim=1)
        self._past = past
        self._next_logits = next_logits.float()

    def _call(
        self,
        token_ids: torch.Tensor,
        past: Any,  # noqa: ANN401
        keep: int | None = None,
    ) -> tuple[Any, torch.Tensor]:  # noqa: ANN401
        kwargs: dict[str, Any] = {"use_cache": True}
        if past is not None:
            kwargs["past_key_values"] = past
        if keep is not None and self._keep_name is not None:
            kwargs[self._keep_name] = keep
        previous = 0 if past is None else _seq_length(past)
        try:
            with torch.no_grad():
                output = self._model(token_ids, **kwargs)
        except TypeError as exc:
            text = str(exc)
            if "past_key_values" in text or "use_cache" in text or "unexpected keyword" in text:
                self.disable()
                raise CacheUnavailableError(text) from exc
            raise
        past_out = getattr(output, "past_key_values", None)
        logits = getattr(output, "logits", None)
        if past_out is None or logits is None:
            self.disable()
            raise CacheUnavailableError("forward did not return past_key_values and logits")
        added = int(token_ids.shape[1])
        if keep is not None and int(logits.shape[1]) != keep:
            # The model ignored logits_to_keep or has no such argument.
            if int(logits.shape[1]) != added:
                self.disable()
                raise CacheUnavailableError("logits do not cover the new tokens")
            logits = logits[:, -keep:, :]
        elif keep is None and int(logits.shape[1]) != added:
            self.disable()
            raise CacheUnavailableError("logits do not cover the new tokens")
        logits = logits.float()
        expected = added if previous is None else int(previous) + added
        if _seq_length(past_out) != expected:
            self.disable()
            raise CacheUnavailableError("cache length did not grow by the new tokens")
        return past_out, logits
