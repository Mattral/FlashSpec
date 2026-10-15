"""Minimal Prometheus text exposition for the FlashSpec server.

The exposition format follows the Prometheus text format 0.0.4 well
enough for a scrape of the counters this process actually updates.
It does not depend on ``prometheus_client``.
"""

from __future__ import annotations

import threading

__all__ = ["ServeMetrics"]

# Latency histogram buckets, milliseconds. +Inf is rendered separately.
_LATENCY_BUCKETS_MS: tuple[float, ...] = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 10000)


class ServeMetrics:
    """Process-local counters for one FlashSpec server.

    The object is safe to update from the request tasks and the batch
    worker. Scrapes call :meth:`render` and receive a consistent snapshot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_active = 0
        self.requests_total = 0
        self.requests_rejected = 0
        self.tokens_generated = 0
        self._acceptance_sum = 0.0
        self._acceptance_count = 0
        self._latency_sum_ms = 0.0
        self._latency_count = 0
        self._latency_buckets = [0 for _ in _LATENCY_BUCKETS_MS]
        self._latency_inf = 0
        self.last_tokens_per_second = 0.0

    def inc_active(self) -> None:
        """Increment the active-request gauge."""
        with self._lock:
            self.requests_active += 1

    def dec_active(self) -> None:
        """Decrement the active-request gauge, floored at zero."""
        with self._lock:
            self.requests_active = max(0, self.requests_active - 1)

    def mark_rejected(self) -> None:
        """Count a request that was refused before generation."""
        with self._lock:
            self.requests_rejected += 1
            self.requests_total += 1

    def observe_generation(
        self,
        *,
        tokens: int,
        acceptance_rate: float,
        latency_ms: float,
        tokens_per_second: float,
    ) -> None:
        """Record one finished generation.

        Parameters
        ----------
        tokens : int
            New tokens returned to the caller.
        acceptance_rate : float
            Draft-token acceptance rate for that generation.
        latency_ms : float
            Wall time spent waiting for the generation, in milliseconds.
        tokens_per_second : float
            Throughput reported by the engine for that call.
        """
        with self._lock:
            self.requests_total += 1
            self.tokens_generated += max(0, tokens)
            self._acceptance_sum += acceptance_rate
            self._acceptance_count += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            self.last_tokens_per_second = tokens_per_second
            placed = False
            for index, edge in enumerate(_LATENCY_BUCKETS_MS):
                if latency_ms <= edge:
                    self._latency_buckets[index] += 1
                    placed = True
                    break
            if not placed:
                self._latency_inf += 1

    def render(self) -> str:
        """Render the current snapshot in Prometheus text format.

        Returns
        -------
        str
            A complete exposition document, ending in a newline.
        """
        with self._lock:
            acceptance = (
                self._acceptance_sum / self._acceptance_count if self._acceptance_count else 0.0
            )
            lines = [
                "# HELP flashspec_requests_active Requests currently waiting on generation.",
                "# TYPE flashspec_requests_active gauge",
                f"flashspec_requests_active {self.requests_active}",
                "# HELP flashspec_requests_total Generation requests accepted or rejected.",
                "# TYPE flashspec_requests_total counter",
                f"flashspec_requests_total {self.requests_total}",
                "# HELP flashspec_requests_rejected_total Requests refused before generation.",
                "# TYPE flashspec_requests_rejected_total counter",
                f"flashspec_requests_rejected_total {self.requests_rejected}",
                "# HELP flashspec_tokens_generated_total Tokens returned to clients.",
                "# TYPE flashspec_tokens_generated_total counter",
                f"flashspec_tokens_generated_total {self.tokens_generated}",
                "# HELP flashspec_acceptance_rate Mean draft acceptance of finished requests.",
                "# TYPE flashspec_acceptance_rate gauge",
                f"flashspec_acceptance_rate {acceptance:.6f}",
                "# HELP flashspec_last_tokens_per_second Throughput of the last finished call.",
                "# TYPE flashspec_last_tokens_per_second gauge",
                f"flashspec_last_tokens_per_second {self.last_tokens_per_second:.6f}",
                "# HELP flashspec_request_latency_ms End-to-end generation latency.",
                "# TYPE flashspec_request_latency_ms histogram",
            ]
            cumulative = 0
            for edge, count in zip(_LATENCY_BUCKETS_MS, self._latency_buckets, strict=True):
                cumulative += count
                lines.append(f'flashspec_request_latency_ms_bucket{{le="{edge:g}"}} {cumulative}')
            cumulative += self._latency_inf
            lines.append(f'flashspec_request_latency_ms_bucket{{le="+Inf"}} {cumulative}')
            lines.append(f"flashspec_request_latency_ms_sum {self._latency_sum_ms:.6f}")
            lines.append(f"flashspec_request_latency_ms_count {self._latency_count}")
            lines.append("")
            return "\n".join(lines)
