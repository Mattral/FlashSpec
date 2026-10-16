"""Batch-1 KV cache for a causal language model.

The session stores ``past_key_values`` and the token ids that produced
them. A later call crops the cache back to the shared prefix of the
accepted tokens, then forwards only the suffix. Transformers 5 crops
with a negative count: ``crop(-n)`` drops ``n`` tokens. A positive
count is the deprecated absolute-length form and is not used here.

Batch sizes other than 1, and models that do not return a cache, stay
on the caller's full-sequence forward.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from flashspec.utils.logging import get_logger

__all__ = ["CacheUnavailableError", "KvSession", "model_accepts_past_key_values"]

logger = get_logger(__name__)


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
        self._past: Any = None  # noqa: ANN401
        self._cached_ids: torch.Tensor | None = None
        self._next_logits: torch.Tensor | None = None

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

    def disable(self) -> None:
        """Stop using the cache and drop any stored state."""
        self.enabled = False
        self.reset()
        logger.debug("KV cache disabled; later calls forward the full sequence")

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

    def _call(self, token_ids: torch.Tensor, past: Any) -> tuple[Any, torch.Tensor]:  # noqa: ANN401
        kwargs: dict[str, Any] = {"use_cache": True}
        if past is not None:
            kwargs["past_key_values"] = past
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
        logits = logits.float()
        if int(logits.shape[1]) != int(token_ids.shape[1]):
            self.disable()
            raise CacheUnavailableError("logits do not cover the new tokens")
        added = int(token_ids.shape[1])
        expected = added if previous is None else int(previous) + added
        if _seq_length(past_out) != expected:
            self.disable()
            raise CacheUnavailableError("cache length did not grow by the new tokens")
        return past_out, logits
