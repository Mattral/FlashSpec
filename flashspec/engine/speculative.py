"""SpeculativeEngine — main orchestrator for FlashSpec.

Ties together the drafter, verifier, bandit selector, sampling algorithm,
and metrics trackers into a single generation loop.
"""

from __future__ import annotations

import torch

from flashspec.bandit.base import DraftSelector
from flashspec.engine.drafter import DraftModel
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.acceptance import AcceptanceTracker
from flashspec.metrics.latency import LatencyTracker
from flashspec.metrics.throughput import ThroughputTracker
from flashspec.sampling.rejection import rejection_sample
from flashspec.sampling.typical import typical_sample
from flashspec.utils.config import FlashSpecConfig
from flashspec.utils.device import get_device
from flashspec.utils.logging import get_logger

__all__ = ["SpeculativeEngine", "GenerationResult"]

logger = get_logger(__name__)

# ── Result dataclass ──────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class GenerationResult:
    """Result of a single :meth:`SpeculativeEngine.generate` call.

    Parameters
    ----------
    output_ids : torch.Tensor
        Generated token IDs.  Shape: ``(batch_size, n_new_tokens)``.
    n_tokens_generated : int
        Total number of new tokens in ``output_ids``.
    acceptance_rate : float
        Mean draft-token acceptance rate (alpha) for this call.
    tokens_per_second : float
        Wall-clock tokens per second for this call.
    """

    output_ids: torch.Tensor
    n_tokens_generated: int
    acceptance_rate: float
    tokens_per_second: float


# ── Engine ────────────────────────────────────────────────────────────────────


class SpeculativeEngine:
    """Adaptive speculative-decoding inference engine.

    Orchestrates the full speculative decoding loop:
    1. Bandit selects a draft model arm.
    2. Draft model proposes ``gamma`` tokens autoregressively.
    3. Target model scores all ``gamma`` positions in one forward pass.
    4. Sampling algorithm accepts/rejects draft tokens and samples residual.
    5. Bandit is updated with the acceptance outcome.
    6. Metrics are updated.

    Parameters
    ----------
    config : FlashSpecConfig
        Full engine configuration.
    drafters : list[DraftModel]
        List of draft model instances, one per bandit arm.
    target : TargetModel
        The target (verifier) model instance.
    bandit : DraftSelector
        Online bandit selector for adaptive arm selection.

    Raises
    ------
    ValueError
        If ``len(drafters)`` does not match ``config.bandit.n_arms``.

    Notes
    -----
    Output distribution correctness (Algorithm 1, Leviathan et al. 2023) is
    verified by ``tests/integration/test_e2e_sampling.py`` (KS test, α=0.01,
    N=10,000).  This is a CI hard gate.

    Examples
    --------
    >>> result = engine.generate(input_ids, max_new_tokens=200)
    >>> result.acceptance_rate
    0.73
    """

    def __init__(
        self,
        config: FlashSpecConfig,
        drafters: list[DraftModel],
        target: TargetModel,
        bandit: DraftSelector,
    ) -> None:
        if len(drafters) != config.bandit.n_arms:
            raise ValueError(
                f"len(drafters)={len(drafters)} must match "
                f"config.bandit.n_arms={config.bandit.n_arms}."
            )
        self._config = config
        self._drafters = drafters
        self._target = target
        self._bandit = bandit
        self._device = get_device(config.device)

        self._acceptance_tracker = AcceptanceTracker(gamma=config.sampling.gamma)
        self._throughput_tracker = ThroughputTracker()
        self._latency_tracker = LatencyTracker(
            window=config.metrics.latency_window,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Run the speculative decoding generation loop.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs.  Shape: ``(batch_size, seq_len)``, dtype int64.
        max_new_tokens : int or None
            Maximum tokens to generate.  Defaults to ``config.max_new_tokens``.

        Returns
        -------
        GenerationResult
            Contains output token IDs and generation metrics.

        Raises
        ------
        ValueError
            If ``input_ids`` is not 2-D or has batch size 0.

        Examples
        --------
        >>> result = engine.generate(input_ids, max_new_tokens=128)
        >>> result.n_tokens_generated
        128
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch_size, seq_len); got shape {input_ids.shape}."
            )
        if input_ids.shape[0] == 0:
            raise ValueError("input_ids batch size must be >= 1.")

        max_tokens = max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
        gamma = self._config.sampling.gamma
        config_sampling = self._config.sampling

        context = input_ids.to(self._device)
        generated_ids: list[torch.Tensor] = []

        self._throughput_tracker.start()

        while sum(t.shape[1] for t in generated_ids) < max_tokens:
            self._latency_tracker.start()

            # 1. Bandit selects a draft arm.
            arm = self._bandit.select()
            drafter = self._drafters[arm]

            # 2. Draft model proposes gamma tokens.
            draft_token_ids, draft_logprobs = drafter.generate_draft(context, gamma)

            # 3. Target model scores draft positions.
            target_logprobs = self._target.score_draft(context, draft_token_ids, gamma)

            # 4. Accept/reject sampling.
            if config_sampling.variant == "rejection":
                accepted_ids, first_rejection, alpha = rejection_sample(
                    input_ids=context,
                    draft_logprobs=draft_logprobs,
                    target_logprobs=target_logprobs,
                    draft_token_ids=draft_token_ids,
                    gamma=gamma,
                    temperature=config_sampling.temperature,
                )
            else:
                accepted_ids, first_rejection, alpha = typical_sample(
                    draft_logprobs=draft_logprobs,
                    target_logprobs=target_logprobs,
                    draft_token_ids=draft_token_ids,
                    gamma=gamma,
                    typical_p=config_sampling.top_p,
                )

            # 5. Update bandit with acceptance count.
            n_accepted = int(first_rejection.float().mean().item())
            self._bandit.update(arm, accepted=n_accepted)

            # 6. Update metrics.
            self._acceptance_tracker.record(n_accepted)

            # Collect non-padding accepted tokens.
            new_tokens = accepted_ids[accepted_ids != -1].view(input_ids.shape[0], -1)
            if new_tokens.shape[1] > 0:
                generated_ids.append(new_tokens)
                context = torch.cat([context, new_tokens], dim=1)

            self._latency_tracker.stop()

        total_new = sum(t.shape[1] for t in generated_ids)
        output_ids = (
            torch.cat(generated_ids, dim=1)[:, :max_tokens]
            if generated_ids
            else input_ids[:, :0]
        )
        tokens_per_second = self._throughput_tracker.stop(n_tokens=total_new)

        logger.debug(
            "Generation complete",
            extra={
                "n_tokens": total_new,
                "tokens_per_second": tokens_per_second,
                "alpha": self._acceptance_tracker.mean_acceptance_rate,
            },
        )

        return GenerationResult(
            output_ids=output_ids,
            n_tokens_generated=total_new,
            acceptance_rate=self._acceptance_tracker.mean_acceptance_rate,
            tokens_per_second=tokens_per_second,
        )
