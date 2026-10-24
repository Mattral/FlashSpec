"""SpeculativeEngine — main orchestrator for FlashSpec.

Ties together the drafter, verifier, bandit selector, sampling algorithm,
and metrics trackers into a single generation loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from flashspec.bandit.base import DraftSelector
from flashspec.engine.drafter import DraftModel
from flashspec.engine.kv_cache import CacheUnavailableError, debug_cache_enabled
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.acceptance import AcceptanceTracker
from flashspec.metrics.latency import LatencyTracker
from flashspec.metrics.step_profile import StepProfiler, StepRecord
from flashspec.metrics.throughput import ThroughputTracker
from flashspec.sampling.greedy import greedy_verify
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
    step_profile : tuple[StepRecord, ...] or None
        Per-step time breakdown for this call when
        ``config.metrics.profile_steps`` is true, otherwise ``None``.
    """

    output_ids: torch.Tensor
    n_tokens_generated: int
    acceptance_rate: float
    tokens_per_second: float
    step_profile: tuple[StepRecord, ...] | None = None


# ── Batch-1 incremental cache bookkeeping ────────────────────────────────────

_TARGET_API: tuple[str, ...] = ("begin", "verify", "commit")
_DRAFTER_API: tuple[str, ...] = ("begin", "draft", "commit", "push_pending")


def _offers(owner: object, names: tuple[str, ...]) -> bool:
    return all(callable(getattr(owner, name, None)) for name in names)


@dataclass(slots=True)
class _Incremental:
    """Which batch-1 components run on the incremental cache this call.

    A component is *capable* when its wrapper offers the incremental API
    and *live* once :meth:`begin` has succeeded on the current sequence.
    A capable component that is not live starts lazily from the accepted
    context the next time it is used. A component whose cache fails is no
    longer capable and uses its context-based method for the rest of the
    call.
    """

    target_capable: bool
    drafter_capable: list[bool]
    target_live: bool = False
    drafter_live: list[bool] = field(default_factory=list)
    debug: bool = False

    @classmethod
    def for_call(cls, target: object, drafters: list[Any]) -> _Incremental:
        capable = [_offers(drafter, _DRAFTER_API) for drafter in drafters]
        return cls(
            target_capable=_offers(target, _TARGET_API),
            drafter_capable=capable,
            drafter_live=[False] * len(drafters),
            debug=debug_cache_enabled(),
        )

    def draft(
        self,
        arm: int,
        drafter: Any,  # noqa: ANN401
        context: torch.Tensor,
        gamma: int,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.drafter_capable[arm] and not self.drafter_live[arm]:
            self.drafter_live[arm] = bool(drafter.begin(context))
            self.drafter_capable[arm] = self.drafter_live[arm]
        if self.drafter_live[arm]:
            try:
                ids, logprobs = drafter.draft(gamma, greedy=greedy)
            except CacheUnavailableError:
                self.drafter_live[arm] = False
                self.drafter_capable[arm] = False
            else:
                return ids, logprobs
        ids, logprobs = drafter.generate_draft(context, gamma)
        return ids, logprobs

    def verify(
        self,
        target: Any,  # noqa: ANN401
        context: torch.Tensor,
        draft_ids: torch.Tensor,
        gamma: int,
        temperature: float,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Target log-probs, or target argmax ids when ``greedy``."""
        if self.target_capable and not self.target_live:
            self.target_live = bool(target.begin(context))
            self.target_capable = self.target_live
        if self.target_live:
            try:
                if greedy and callable(getattr(target, "verify_greedy", None)):
                    scored: torch.Tensor = target.verify_greedy(draft_ids)
                elif greedy:
                    scored = target.verify(draft_ids, temperature=1.0).argmax(dim=-1)
                else:
                    scored = target.verify(draft_ids, temperature=temperature)
            except CacheUnavailableError:
                self.target_live = False
                self.target_capable = False
            else:
                return scored
        return _score_with_context(target, context, draft_ids, gamma, temperature, greedy)

    def commit(
        self,
        arm: int,
        drafters: list[Any],
        target: Any,  # noqa: ANN401
        n_accepted: int,
        piece: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        """Record one step. Nothing here runs a model forward.

        ``piece`` is what the step appended: the accepted draft tokens
        and then the residual or bonus token. A shorter piece (the last
        step, cut by ``max_new_tokens``) leaves no token to queue, so the
        affected caches are restarted from ``context`` if they are used
        again.
        """
        complete = int(piece.shape[1]) == n_accepted + 1
        next_token = piece[:, -1:]
        if self.target_live:
            if complete:
                target.commit(n_accepted, next_token)
            else:
                self.target_live = False
        for index, drafter in enumerate(drafters):
            if not self.drafter_live[index]:
                continue
            if index != arm:
                drafter.push_pending(piece)
            elif complete:
                drafter.commit(n_accepted, next_token)
            else:
                self.drafter_live[index] = False
        if self.debug:
            if self.target_live:
                target.check_cache(context)
            for index, drafter in enumerate(drafters):
                if self.drafter_live[index]:
                    drafter.check_cache(context)

    def uses_cache(self, owner: object, arm: int, drafter: object) -> bool:
        """Whether ``owner`` is kept in step by :meth:`commit` instead of ``roll_cache_to``."""
        if owner is drafter:
            return self.drafter_capable[arm]
        return self.target_capable


def _score_with_context(
    target: Any,  # noqa: ANN401
    context: torch.Tensor,
    draft_ids: torch.Tensor,
    gamma: int,
    temperature: float,
    greedy: bool,
) -> torch.Tensor:
    """Context-based scoring: log-probs, or argmax ids when ``greedy``."""
    if greedy:
        # The argmax does not depend on temperature; score at 1.
        scored: torch.Tensor = target.score_draft(
            context, draft_ids, gamma, temperature=1.0, include_bonus=True
        )
        return scored.argmax(dim=-1)
    legacy: torch.Tensor = target.score_draft(
        context, draft_ids, gamma, temperature=temperature, include_bonus=True
    )
    return legacy


# ── Engine ────────────────────────────────────────────────────────────────────


class SpeculativeEngine:
    """Adaptive speculative-decoding inference engine.

    Orchestrates the full speculative decoding loop:
    1. Bandit selects a draft model arm.
    2. Draft model proposes ``gamma`` tokens autoregressively.
    3. Target model scores all ``gamma`` positions in one forward pass.
       At batch 1, with wrappers that offer the incremental API
       (``begin``/``verify``/``commit`` on the target and
       ``begin``/``draft``/``commit``/``push_pending`` on the drafter),
       that is the only target forward of the step and the drafter runs
       exactly ``gamma`` forwards.
    4. Sampling algorithm accepts/rejects draft tokens and samples residual.
    5. Bandit is updated with accepted tokens per second of step time.
       The raw accept count is still recorded. The bonus token is not
       part of that count.
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
        self._profiler: StepProfiler | None = (
            StepProfiler(self._device) if config.metrics.profile_steps else None
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
        5. Accepted tokens are appended to the context. For a batch of one,
           the drafter and verifier crop their KV caches by the number of
           rejected tokens and queue the residual or bonus token for their
           next forward. No model runs during this step.
        6. The bandit records accepted draft tokens per second of the step.
           The raw accept count is still recorded. The bonus token is not
           part of that count.
        7. Metrics are updated.

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

        incremental = (
            _Incremental.for_call(self._target, self._drafters) if batch_size == 1 else None
        )

        profiler = self._profiler
        if profiler is not None:
            profiler.reset()
        self._throughput_tracker.start()

        while any(_row_count(i) < max_tokens for i in range(batch_size)):
            self._latency_tracker.start()
            if profiler is not None:
                profiler.start_step()
            step_started = time.perf_counter()
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

            greedy_bonus: torch.Tensor | None = None
            if config_sampling.greedy:
                # Greedy: argmax drafts, one argmax verify, accept up to the
                # first mismatch. No softmax and no random draws.
                if incremental is not None:
                    draft_token_ids, _ = incremental.draft(
                        arm, drafter, context, gamma, greedy=True
                    )
                else:
                    draft_token_ids, _ = drafter.generate_draft(context, gamma)
                if profiler is not None:
                    profiler.mark("draft")
                if incremental is not None:
                    target_argmax = incremental.verify(
                        self._target, context, draft_token_ids, gamma, 1.0, greedy=True
                    )
                else:
                    target_argmax = _score_with_context(
                        self._target, context, draft_token_ids, gamma, 1.0, greedy=True
                    )
                if profiler is not None:
                    profiler.mark("verify")
                accepted_ids, first_rejection, greedy_bonus = greedy_verify(
                    draft_token_ids, target_argmax
                )
                bonus_logprobs = None
            else:
                # 2. Draft model proposes gamma tokens.
                if incremental is not None:
                    draft_token_ids, maybe_logprobs = incremental.draft(
                        arm, drafter, context, gamma
                    )
                    if maybe_logprobs is None:
                        raise RuntimeError("Drafter returned no log-probs for a sampling step.")
                    draft_logprobs = maybe_logprobs
                else:
                    draft_token_ids, draft_logprobs = drafter.generate_draft(context, gamma)
                if profiler is not None:
                    profiler.mark("draft")

                # 3. Target model scores draft positions (temperature applied inside).
                # include_bonus asks for the extra row that predicts the token
                # after a fully accepted draft. Rejection sampling still sees
                # only the gamma draft positions.
                if incremental is not None:
                    target_logprobs = incremental.verify(
                        self._target,
                        context,
                        draft_token_ids,
                        gamma,
                        config_sampling.temperature,
                    )
                else:
                    target_logprobs = self._target.score_draft(
                        context,
                        draft_token_ids,
                        gamma,
                        temperature=config_sampling.temperature,
                        include_bonus=True,
                    )
                if profiler is not None:
                    profiler.mark("verify")
                bonus_logprobs = None
                if target_logprobs.shape[1] == gamma + 1:
                    bonus_logprobs = target_logprobs[:, gamma, :]
                    target_logprobs = target_logprobs[:, :gamma, :]

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

            # The raw count stays the acceptance metric. The bonus token
            # is appended below and is not included. first_rejection is in
            # [0, gamma]; the tracker requires that range.
            n_accepted = int(first_rejection.float().mean().item())
            n_accepted = min(gamma, max(0, n_accepted))

            progressed = False
            emitted = 0
            last_piece: torch.Tensor | None = None
            for local, index in enumerate(idxs):
                row = accepted_ids[local]
                valid = row[row != -1]
                room = max_tokens - _row_count(index)
                if room <= 0 or valid.numel() == 0:
                    continue
                take = valid[:room]
                # Bonus only when every draft token was accepted and one
                # more token still fits under max_new_tokens. A rejection
                # already wrote its residual into ``valid``.
                full_accept = int(first_rejection[local].item()) == gamma
                if (
                    (bonus_logprobs is not None or greedy_bonus is not None)
                    and full_accept
                    and take.numel() == gamma
                    and room > gamma
                ):
                    if greedy_bonus is not None:
                        bonus_id = greedy_bonus[local].view(1).to(take.dtype)
                    else:
                        assert bonus_logprobs is not None
                        probs = bonus_logprobs[local].exp()
                        probs = probs / probs.sum().clamp(min=1e-9)
                        bonus_id = torch.multinomial(probs, num_samples=1).to(take.dtype)
                    take = torch.cat([take, bonus_id.view(-1)])
                piece = take.view(1, -1)
                generated_rows[index].append(piece)
                contexts[index] = torch.cat([contexts[index], piece], dim=1)
                emitted += int(piece.shape[1])
                last_piece = piece
                progressed = True
            # The residual and bonus draws happen inside the loop above,
            # so its time counts as sampling.
            if profiler is not None:
                profiler.mark("sample")

            # One sequence can keep a KV cache. Incremental caches drop
            # the rejected draft positions and queue the residual or bonus
            # token without a forward. Any other wrapper is rolled to the
            # accepted prefix, which may forward the new tokens.
            if len(idxs) == 1:
                accepted_prefix = contexts[idxs[0]]
                if incremental is not None and last_piece is not None:
                    incremental.commit(
                        arm,
                        self._drafters,
                        self._target,
                        int(first_rejection[0]),
                        last_piece,
                        accepted_prefix,
                    )
                for owner in (drafter, self._target):
                    if incremental is not None and incremental.uses_cache(owner, arm, drafter):
                        continue
                    roll = getattr(owner, "roll_cache_to", None)
                    if callable(roll):
                        roll(accepted_prefix)

            elapsed = time.perf_counter() - step_started
            reward = (n_accepted / elapsed) if elapsed > 0.0 else 0.0
            self._bandit.update(arm, accepted=n_accepted, reward=reward)
            self._acceptance_tracker.record(n_accepted)
            self._latency_tracker.stop()
            if profiler is not None:
                profiler.mark("bookkeeping")
                profiler.end_step(emitted)
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
            step_profile=tuple(profiler.records()) if profiler is not None else None,
        )
