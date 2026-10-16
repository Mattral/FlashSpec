"""SpeculativeEngine — main orchestrator for FlashSpec.

Ties together the drafter, verifier, bandit selector, sampling algorithm,
and metrics trackers into a single generation loop.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    checked by a chi-squared test in
    ``tests/integration/test_output_distribution.py``. That test uses
    ``p != q``, gamma 1 and 4, N=30,000, vocabulary 6, and the chi-square
    1% point for df=5. The gamma=1, ``p == q`` test in
    ``tests/integration/test_e2e_sampling.py`` only checks that every
    draft token is accepted.

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
        if config.bandit.n_arms < 2:
            logger.warning(
                "Bandit has a single arm and cannot adapt. Register two or "
                "more drafters before expecting a throughput or acceptance "
                "gain over a fixed drafter.",
                extra={"n_arms": config.bandit.n_arms},
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
            Contains output token IDs and generation metrics (acceptance rate,
            tokens per second).

        Raises
        ------
        ValueError
            If ``input_ids`` is not 2-D or has batch size 0.

        Notes
        -----
        On finite probabilities the output distribution follows autoregressive
        sampling from the target model (Leviathan et al., 2023, Theorem 1).
        The check is the chi-squared test in
        ``tests/integration/test_output_distribution.py`` (``p != q``,
        gamma 1 and 4, plus one context-dependent toy target).

        The generation loop follows these steps on each iteration:

        1. Bandit selects a draft model arm.
        2. Draft model proposes ``gamma`` tokens autoregressively.
        3. Target model scores all ``gamma`` positions in one forward pass
           with temperature applied before log-softmax (§7).
        4. Rejection sampling accepts/rejects tokens (Algorithm 1).
        5. Bandit is updated with the acceptance outcome.
        6. Metrics are updated.
        7. Accepted tokens are appended to the context.

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
        batch_size = int(input_ids.shape[0])

        # One context per row. Rows whose contexts share a length are scored
        # together. Rows that accept a different number of tokens no longer
        # have to be rectangular for torch.Tensor.view to succeed.
        contexts = [input_ids[i : i + 1].to(self._device) for i in range(batch_size)]
        generated_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        def _row_count(index: int) -> int:
            return sum(piece.shape[1] for piece in generated_rows[index])

        self._throughput_tracker.start()

        while any(_row_count(i) < max_tokens for i in range(batch_size)):
            self._latency_tracker.start()
            active = [i for i in range(batch_size) if _row_count(i) < max_tokens]
            groups: dict[int, list[int]] = {}
            for index in active:
                groups.setdefault(int(contexts[index].shape[1]), []).append(index)
            # One bandit decision per loop. The largest equal-length group
            # runs now; other lengths run on the next iteration.
            idxs = max(groups.values(), key=len)
            context = torch.cat([contexts[i] for i in idxs], dim=0)

            # 1. Bandit selects a draft arm.
            arm = self._bandit.select()
            drafter = self._drafters[arm]

            # 2. Draft model proposes gamma tokens.
            draft_token_ids, draft_logprobs = drafter.generate_draft(context, gamma)

            # 3. Target model scores draft positions (temperature applied inside).
            target_logprobs = self._target.score_draft(
                context,
                draft_token_ids,
                gamma,
                temperature=config_sampling.temperature,
            )

            # 4. Accept/reject sampling.
            if config_sampling.variant == "rejection":
                accepted_ids, first_rejection, _alpha = rejection_sample(
                    input_ids=context,
                    draft_logprobs=draft_logprobs,
                    target_logprobs=target_logprobs,
                    draft_token_ids=draft_token_ids,
                    gamma=gamma,
                )
            else:
                accepted_ids, first_rejection, _alpha = typical_sample(
                    draft_logprobs=draft_logprobs,
                    target_logprobs=target_logprobs,
                    draft_token_ids=draft_token_ids,
                    gamma=gamma,
                    typical_p=config_sampling.top_p,
                )

            # 5. Update bandit with the mean accepted-draft count.
            # first_rejection is in [0, gamma]; the tracker requires that range.
            n_accepted = int(first_rejection.float().mean().item())
            n_accepted = min(gamma, max(0, n_accepted))
            self._bandit.update(arm, accepted=n_accepted)

            # 6. Update metrics.
            self._acceptance_tracker.record(n_accepted)

            progressed = False
            for local, index in enumerate(idxs):
                row = accepted_ids[local]
                valid = row[row != -1]
                room = max_tokens - _row_count(index)
                if room <= 0 or valid.numel() == 0:
                    continue
                piece = valid[:room].view(1, -1)
                generated_rows[index].append(piece)
                contexts[index] = torch.cat([contexts[index], piece], dim=1)
                progressed = True

            self._latency_tracker.stop()
            if not progressed:
                logger.warning(
                    "Speculative step accepted no tokens; stopping early.",
                    extra={"batch_size": batch_size},
                )
                break

        output_rows: list[torch.Tensor] = []
        for index in range(batch_size):
            if generated_rows[index]:
                row_ids = torch.cat(generated_rows[index], dim=1)
            else:
                row_ids = input_ids.new_empty((1, 0), device=self._device)
            if row_ids.shape[1] < max_tokens:
                pad = torch.full(
                    (1, max_tokens - row_ids.shape[1]),
                    -1,
                    dtype=row_ids.dtype,
                    device=row_ids.device,
                )
                row_ids = torch.cat([row_ids, pad], dim=1)
            output_rows.append(row_ids[:, :max_tokens])
        output_ids = torch.cat(output_rows, dim=0)
        total_new = int((output_ids != -1).sum().item())
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
