"""Baseline comparison utilities for FlashSpec benchmarks.

Provides vanilla autoregressive, Medusa, and EAGLE baseline stubs so
benchmark tables can be populated with consistent methodology.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

__all__ = ["BaselineResult", "run_autoregressive_baseline"]


@dataclass(frozen=True)
class BaselineResult:
    """Recorded metrics for one baseline method.

    Parameters
    ----------
    method : str
        Human-readable method name.
    tokens_per_second : float
        Throughput in tokens / second.
    p50_ms : float
        Median step latency in milliseconds.
    p99_ms : float
        99th-percentile step latency in milliseconds.
    n_tokens : int
        Total tokens generated.
    """

    method: str
    tokens_per_second: float
    p50_ms: float
    p99_ms: float
    n_tokens: int


def run_autoregressive_baseline(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
) -> BaselineResult:
    """Run vanilla autoregressive generation and return metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Causal language model with a ``forward`` method.
    input_ids : torch.Tensor
        Prompt token IDs.  Shape: ``(batch_size, seq_len)``.
    max_new_tokens : int
        Number of tokens to generate.
    device : torch.device
        Target device.

    Returns
    -------
    BaselineResult
        Throughput and latency metrics.

    Notes
    -----
    This is a stub implementation.  Real models should use
    ``model.generate()`` with ``do_sample=False`` for reproducibility.

    Examples
    --------
    >>> result = run_autoregressive_baseline(model, input_ids, 128, device)
    >>> result.tokens_per_second
    45.2
    """
    model.eval()
    ctx = input_ids.to(device)
    latencies: list[float] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(ctx)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ctx = torch.cat([ctx, next_id], dim=-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    total_wall_s = sum(latencies) / 1000.0
    tps = max_new_tokens / total_wall_s if total_wall_s > 0 else 0.0

    return BaselineResult(
        method="vanilla_ar",
        tokens_per_second=tps,
        p50_ms=sorted_lat[n // 2],
        p99_ms=sorted_lat[int(0.99 * n)],
        n_tokens=max_new_tokens,
    )
