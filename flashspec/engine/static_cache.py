"""Batch-1 static KV cache with optional CUDA-graph capture (``FlashSpecConfig.compile``).

:class:`StaticKvSession` has the same incremental interface as
:class:`flashspec.engine.kv_cache.KvSession` (``start``, ``advance``,
``crop``, ``push_pending``, ``check``), on a HuggingFace ``StaticCache``:

- The cache is allocated once per model, at a fixed length, so its
  tensors keep their addresses across steps and across ``generate``
  calls. That is what lets a CUDA graph replay against it.
- ``crop`` moves each layer's write position back (an in-place device
  op). ``StaticLayer`` has no ``crop``, and none is needed: entries past
  the write position are masked by the causal mask and overwritten by
  the next forward.
- ``cache_position`` is computed on the device from that write position,
  so no step reads it back to the host.
- The prompt prefill runs eagerly. Fixed-shape decode and verify calls
  (at most ``max_graph_tokens`` tokens) go through
  ``torch.compile(model.__call__, mode="reduce-overhead")``, the same
  compile settings HuggingFace ``generate`` uses for a static cache
  (``CompileConfig()``), with its own compiled callable. Longer calls (an
  idle bandit arm catching up) run eagerly on the same cache.
- With ``pad_widths``, every call of at most ``max(pad_widths)`` tokens is
  padded at the end to the next listed width and asks for all of that
  width's logit rows, so only one shape per width is ever captured. The
  real rows are sliced out afterwards. Causal attention keeps the padded
  positions out of the real rows, and the write position is moved back
  over them right after the forward, so they never stay in the cache that
  is kept. Longer calls run eagerly, unpadded, in one call.

On CPU, or with ``compile_forward=False``, every call is eager; the
bookkeeping is identical, which is how the tests check it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from flashspec.engine.kv_cache import (
    CacheUnavailableError,
    _logits_to_keep_name,
    debug_cache_enabled,
)
from flashspec.utils.logging import get_logger

__all__ = ["StaticKvSession"]

logger = get_logger(__name__)


class StaticKvSession:
    """Incremental batch-1 cache on a preallocated ``StaticCache``.

    Parameters
    ----------
    model : Any
        HuggingFace causal LM whose ``forward`` takes ``past_key_values``
        and ``cache_position``.
    max_cache_len : int
        Cache capacity in tokens. :meth:`reserve` grows it when a call
        needs more (a new allocation, so graphs are recaptured once).
    compile_forward : bool
        Capture fixed-shape calls with ``torch.compile(mode="reduce-overhead")``.
    max_graph_tokens : int
        Calls with more input tokens than this run eagerly. Ignored when
        ``pad_widths`` is set (its largest width is the limit).
    pad_widths : sequence of int or None
        Fixed input widths for calls of up to ``max(pad_widths)`` tokens.
        ``None`` forwards every call at its own length.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        max_cache_len: int,
        *,
        compile_forward: bool = False,
        max_graph_tokens: int = 16,
        pad_widths: Sequence[int] | None = None,
    ) -> None:
        self._model = model
        self.enabled = True
        self._max_len = int(max_cache_len)
        self._keep_name = _logits_to_keep_name(model)
        self._compile_forward = compile_forward
        self.compile_requested = compile_forward
        self._max_graph_tokens = max_graph_tokens
        self._cache: Any = None  # noqa: ANN401
        self._compiled: Any = None  # noqa: ANN401
        self._length = 0
        self._pending: torch.Tensor | None = None
        self._debug = False
        self._debug_ids: torch.Tensor | None = None
        self._arange: torch.Tensor | None = None
        self.compile_fallback_reason: str | None = None
        self._pad_widths: tuple[int, ...] = ()
        self.set_pad_widths(pad_widths)
        # Calls that take (or, with compile off, would take) the captured
        # path, by (input width, logit rows): one CUDA graph per key.
        self.graph_call_shapes: dict[tuple[int, int], int] = {}

    def set_pad_widths(self, widths: Sequence[int] | None) -> None:
        """Set (or clear, with ``None``) the fixed widths calls are padded to."""
        if widths is None:
            self._pad_widths = ()
            return
        cleaned = tuple(sorted({int(width) for width in widths}))
        if not cleaned or cleaned[0] < 1:
            raise ValueError(f"pad widths must be positive integers; got {widths!r}.")
        self._pad_widths = cleaned

    @property
    def pad_widths(self) -> tuple[int, ...]:
        """The fixed call widths, smallest first; empty when calls are unpadded."""
        return self._pad_widths

    def _padded_width(self, n_tokens: int) -> int | None:
        """Return the width a call of ``n_tokens`` is padded to, or ``None`` (unpadded)."""
        for width in self._pad_widths:
            if width >= n_tokens:
                return width
        return None

    # ── Setup ────────────────────────────────────────────────────────────

    def _allocate(self) -> None:
        from transformers import StaticCache

        self._cache = StaticCache(config=self._model.config, max_cache_len=self._max_len)
        if self._compile_forward and self._compiled is None:
            from transformers.generation.configuration_utils import CompileConfig

            # The drafter and target usually share one forward code object
            # (same model class), so their shapes share dynamo's per-code
            # recompile budget; the default of 8 is too small for both, and
            # several engines (each with its own static cache) add entries.
            dynamo_config: Any = torch._dynamo.config
            for knob in ("recompile_limit", "cache_size_limit"):
                if hasattr(dynamo_config, knob):
                    setattr(dynamo_config, knob, max(int(getattr(dynamo_config, knob)), 128))

            # The settings HuggingFace generate uses for a static cache.
            self._compiled = torch.compile(self._model.__call__, **CompileConfig().to_dict())

    def reserve(self, n_tokens: int) -> None:
        """Make sure the cache can hold ``n_tokens``; reallocate if not.

        With padding, room for the widest padded call is added: padded
        positions are written before they are moved back over.
        """
        if self._pad_widths:
            n_tokens += self._pad_widths[-1]
        if self._cache is None or n_tokens > self._max_len:
            if self._cache is not None:
                logger.info(
                    "Growing the static cache; CUDA graphs are captured again",
                    extra={"from": self._max_len, "to": n_tokens},
                )
            self._max_len = max(self._max_len, int(n_tokens))
            self._allocate()

    @property
    def compiled(self) -> bool:
        """Whether fixed-shape calls are captured."""
        return self._compiled is not None

    def supports(self, token_ids: torch.Tensor) -> bool:
        """Return True for an enabled session and a batch of one."""
        return self.enabled and token_ids.ndim == 2 and token_ids.shape[0] == 1

    def disable(self) -> None:
        """Stop using this session."""
        self.enabled = False

    def reset(self) -> None:
        """Rewind to an empty sequence without freeing or zeroing the cache."""
        if self._cache is not None:
            for layer in self._cache.layers:
                layer.cumulative_length.zero_()
        self._length = 0
        self._pending = None
        self._debug_ids = None

    # ── Incremental interface (same as KvSession) ────────────────────────

    @property
    def pending_length(self) -> int:
        """Committed tokens the next :meth:`advance` forwards first."""
        return 0 if self._pending is None else int(self._pending.shape[1])

    @property
    def cached_length(self) -> int:
        """Tokens in the cache, from the integer bookkeeping."""
        return self._length

    def _write_position(self) -> torch.Tensor:
        return self._cache.layers[0].cumulative_length  # type: ignore[no-any-return]

    def _forward(self, token_ids: torch.Tensor, keep: int, eager: bool) -> torch.Tensor:
        real = int(token_ids.shape[1])
        width = None if eager else self._padded_width(real)
        if width is not None and width > real:
            # Pad at the end with a copy of the last real token (a valid id).
            filler = token_ids[:, -1:].expand(1, width - real)
            token_ids = torch.cat([token_ids, filler], dim=1)
        n = int(token_ids.shape[1])
        if width is not None:
            # Every row of the padded call: one int for this width, so one
            # captured shape. The real rows are sliced out below.
            keep_rows = width
        else:
            keep_rows = keep
        if self._length + n > self._max_len:
            raise CacheUnavailableError(
                f"static cache full: {self._length} + {n} > {self._max_len} tokens"
            )
        device = token_ids.device
        longest = max(n, self._max_graph_tokens, self._pad_widths[-1] if self._pad_widths else 0)
        if self._arange is None or self._arange.device != device or self._arange.numel() < n:
            self._arange = torch.arange(longest, device=device)
        if self._length == 0:
            cache_position = (
                self._arange[:n] if n <= self._arange.numel() else torch.arange(n, device=device)
            )
        else:
            # Device-side: write position + 0..n-1. No host read.
            cache_position = self._write_position() + self._arange[:n]
        kwargs: dict[str, Any] = {
            "past_key_values": self._cache,
            "cache_position": cache_position,
            "use_cache": True,
        }
        if self._keep_name is not None:
            kwargs[self._keep_name] = keep_rows
        if self._pad_widths:
            fits = width is not None
        else:
            fits = n <= self._max_graph_tokens
        if not eager and fits:
            shape = (n, keep_rows)
            self.graph_call_shapes[shape] = self.graph_call_shapes.get(shape, 0) + 1
        use_graph = self._compiled is not None and not eager and fits
        with torch.no_grad():
            output = None
            if use_graph:
                try:
                    mark_step: Any = torch.compiler.cudagraph_mark_step_begin
                    mark_step()
                    output = self._compiled(token_ids, **kwargs)
                except Exception as exc:  # noqa: BLE001 - fall back to eager, recorded
                    self.compile_fallback_reason = f"{type(exc).__name__}: {exc}"[:500]
                    self._compiled = None
                    logger.warning(
                        "Compiled forward failed; using the eager forward from now on",
                        extra={"error": self.compile_fallback_reason},
                    )
                    # Undo a partial cache write: the bookkeeping is the truth.
                    for layer in self._cache.layers:
                        layer.cumulative_length.fill_(self._length)
                    kwargs["cache_position"] = cache_position
            if output is None:
                output = self._model(token_ids, **kwargs)
        logits: torch.Tensor = output.logits
        if int(logits.shape[1]) != keep_rows:
            logits = logits[:, -keep_rows:, :]
        if n > real:
            # Drop the padded positions from the cache (device-side, no read).
            for layer in self._cache.layers:
                layer.cumulative_length.sub_(n - real)
        if width is not None:
            # The rows the caller asked for: [real - keep, real) of the input.
            logits = logits[:, real - keep : real, :]
        # .float() copies out of the graph's output buffer before the next replay.
        return logits.float()

    def start(self, prompt: torch.Tensor, *, hold_last: bool = True) -> torch.Tensor | None:
        """Prefill ``prompt`` eagerly. See :meth:`KvSession.start`."""
        if not self.supports(prompt) or int(prompt.shape[1]) < 1:
            raise CacheUnavailableError("start is only used for one non-empty row")
        self.reserve(int(prompt.shape[1]) + 1)
        self.reset()
        self._debug = debug_cache_enabled()
        ids = prompt.to(dtype=torch.long)
        head = ids if not hold_last else ids[:, :-1]
        logits: torch.Tensor | None = None
        if int(head.shape[1]) > 0:
            logits = self._forward(head, keep=1, eager=True)
            self._length = int(head.shape[1])
            if self._debug:
                self._debug_ids = head.clone()
        if not hold_last:
            assert logits is not None
            return logits[:, -1, :]
        self._pending = ids[:, -1:]
        return None

    def advance(self, new_ids: torch.Tensor | None, keep: int) -> torch.Tensor:
        """Forward pending tokens plus ``new_ids`` in one call. See :meth:`KvSession.advance`."""
        pieces = [piece for piece in (self._pending, new_ids) if piece is not None]
        if not pieces:
            raise CacheUnavailableError("advance with no pending or new tokens")
        token_ids = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        added = int(token_ids.shape[1])
        if not 1 <= keep <= added:
            raise CacheUnavailableError(f"keep={keep} must be in [1, {added}]")
        logits = self._forward(token_ids.to(dtype=torch.long), keep=keep, eager=False)
        self._length += added
        self._pending = None
        if self._debug:
            fresh = token_ids.to(dtype=torch.long)
            self._debug_ids = (
                fresh.clone() if self._debug_ids is None else torch.cat([self._debug_ids, fresh], 1)
            )
        return logits

    def crop(self, n_tokens: int) -> None:
        """Drop the last ``n_tokens`` by moving the write position back. Never forwards."""
        if n_tokens == 0:
            return
        if n_tokens < 0 or n_tokens > self._length:
            self.disable()
            raise CacheUnavailableError(f"cannot crop {n_tokens} of {self._length} cached tokens")
        for layer in self._cache.layers:
            layer.cumulative_length.sub_(n_tokens)
        self._length -= n_tokens
        if self._debug and self._debug_ids is not None:
            self._debug_ids = self._debug_ids[:, : self._length]

    def push_pending(self, token_ids: torch.Tensor) -> None:
        """Queue committed tokens for the next forward. Never forwards."""
        ids = token_ids.to(dtype=torch.long)
        self._pending = ids if self._pending is None else torch.cat([self._pending, ids], dim=1)

    def check(self, context: torch.Tensor) -> None:
        """With ``FLASHSPEC_DEBUG_CACHE=1``: assert write position and ids match ``context``."""
        if not self._debug:
            return
        position = int(self._write_position()) if self._cache is not None else 0
        if position != self._length:
            raise AssertionError(f"write position {position} != bookkeeping {self._length}")
        pieces = [piece for piece in (self._debug_ids, self._pending) if piece is not None]
        held = torch.cat(pieces, dim=1) if pieces else context.new_empty((1, 0))
        if not torch.equal(held.to(context.device), context.to(dtype=torch.long)):
            raise AssertionError(
                f"cache + pending ids {held.tolist()} do not match context {context.tolist()}"
            )
