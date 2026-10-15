"""Single-worker request queue with fixed-size batching.

The worker is the only thread that calls ``engine.generate``. Requests
whose prompt lengths and ``max_new_tokens`` match are stacked into one
batch. Different lengths are not padded: the engine has no attention
mask, so padding would change the scores. Those requests run in a
later batch.

This is not continuous batching and it does not share a KV cache.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import torch

from flashspec.engine.speculative import GenerationResult, SpeculativeEngine

__all__ = ["QueueFullError", "RequestBatcher"]


class QueueFullError(RuntimeError):
    """Raised when the server queue is at ``max_queue``."""


@dataclass(slots=True)
class _WorkItem:
    """One queued generation request."""

    token_ids: list[int]
    max_new_tokens: int
    future: Future[GenerationResult]


class RequestBatcher:
    """Queue generations and run them on one engine thread.

    Parameters
    ----------
    engine : SpeculativeEngine
        Engine used for every batch. It is not re-entrant.
    batch_size : int
        Maximum number of equal-length requests per ``generate`` call.
        ``1`` disables stacking and still serialises the queue.
    max_wait_ms : float
        After the first request of a batch arrives, wait up to this long
        for more requests of the same shape.
    max_queue : int
        Maximum queued requests. Further :meth:`submit` calls raise
        :class:`QueueFullError`.

    Raises
    ------
    ValueError
        If ``batch_size`` or ``max_queue`` is less than 1, or
        ``max_wait_ms`` is negative.
    """

    def __init__(
        self,
        engine: SpeculativeEngine,
        *,
        batch_size: int = 1,
        max_wait_ms: float = 2.0,
        max_queue: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1; got {max_queue}.")
        if max_wait_ms < 0.0:
            raise ValueError(f"max_wait_ms must be >= 0; got {max_wait_ms}.")
        self._engine = engine
        self._batch_size = batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: queue.Queue[_WorkItem | None] = queue.Queue(maxsize=max_queue)
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name="flashspec-batcher",
            daemon=True,
        )
        self.generate_calls = 0

    @property
    def batch_size(self) -> int:
        """Maximum number of requests stacked into one ``generate`` call.

        Returns
        -------
        int
            The configured batch size.
        """
        return self._batch_size

    def start(self) -> None:
        """Start the worker thread. A second call is a no-op."""
        if not self._thread.is_alive():
            self._stop = False
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Ask the worker to finish the current batch and exit.

        Parameters
        ----------
        timeout_s : float
            How long to wait for the worker thread.
        """
        self._stop = True
        try:
            self._queue.put(None, timeout=timeout_s)
        except queue.Full:
            return
        self._thread.join(timeout=timeout_s)

    def submit(self, token_ids: list[int], max_new_tokens: int) -> Future[GenerationResult]:
        """Enqueue one prompt and return a future of its :class:`GenerationResult`.

        Parameters
        ----------
        token_ids : list[int]
            Prompt token ids. Must be non-empty.
        max_new_tokens : int
            Maximum new tokens. Must be ``>= 1``.

        Returns
        -------
        concurrent.futures.Future
            Completes with one row of the engine result.

        Raises
        ------
        ValueError
            If the prompt is empty or ``max_new_tokens < 1``.
        QueueFullError
            If the queue is already at capacity.
        """
        if not token_ids:
            raise ValueError("token_ids must be a non-empty list.")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}.")
        future: Future[GenerationResult] = Future()
        item = _WorkItem(token_ids=list(token_ids), max_new_tokens=max_new_tokens, future=future)
        try:
            self._queue.put_nowait(item)
        except queue.Full as exc:
            raise QueueFullError(
                f"Request queue is full (max_queue={self._queue.maxsize})."
            ) from exc
        return future

    def _run(self) -> None:
        """Pull work until a ``None`` sentinel arrives."""
        while not self._stop:
            item = self._queue.get()
            if item is None:
                return
            batch = [item]
            deadline = time.perf_counter() + self._max_wait_s
            while len(batch) < self._batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is None:
                    self._execute(batch)
                    return
                same_shape = (
                    len(nxt.token_ids) == len(batch[0].token_ids)
                    and nxt.max_new_tokens == batch[0].max_new_tokens
                )
                if not same_shape:
                    self._execute(batch)
                    batch = [nxt]
                    deadline = time.perf_counter() + self._max_wait_s
                    continue
                batch.append(nxt)
            self._execute(batch)

    def _execute(self, batch: list[_WorkItem]) -> None:
        """Run one rectangular batch and complete each request's future.

        Parameters
        ----------
        batch : list[_WorkItem]
            Requests that share a prompt length and ``max_new_tokens``.
        """
        if not batch:
            return
        try:
            input_ids = torch.tensor(
                [item.token_ids for item in batch],
                dtype=torch.long,
            )
            self.generate_calls += 1
            result = self._engine.generate(
                input_ids,
                max_new_tokens=batch[0].max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 — deliver the failure to waiters
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for index, item in enumerate(batch):
            if item.future.done():
                continue
            row = result.output_ids[index : index + 1]
            n_tokens = int((row != -1).sum().item())
            item.future.set_result(
                GenerationResult(
                    output_ids=row,
                    n_tokens_generated=n_tokens,
                    acceptance_rate=result.acceptance_rate,
                    tokens_per_second=result.tokens_per_second,
                )
            )
