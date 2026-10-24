"""SpeculativeEngine — main orchestrator for FlashSpec.

Ties together the drafter, verifier, bandit selector, sampling algorithm,
and metrics trackers into a single generation loop.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Sequence
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
from flashspec.sampling.greedy import greedy_step, greedy_verify
from flashspec.sampling.rejection import rejection_sample, rejection_step
from flashspec.sampling.typical import typical_sample
from flashspec.utils.config import FlashSpecConfig
from flashspec.utils.device import CompileDecision, get_device, resolve_compile
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


@functools.lru_cache(maxsize=128)
def _takes_generator(method: Any) -> bool:  # noqa: ANN401
    """Whether ``method`` accepts a ``generator`` keyword."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return "generator" in parameters


def _draft_with_context(
    drafter: Any,  # noqa: ANN401
    context: torch.Tensor,
    gamma: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``generate_draft``, passing the seeded generator when the drafter takes one."""
    method = drafter.generate_draft
    if generator is not None and _takes_generator(method):
        ids, logprobs = method(context, gamma, generator=generator)
    else:
        ids, logprobs = method(context, gamma)
    return ids, logprobs


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
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.drafter_capable[arm] and not self.drafter_live[arm]:
            self.drafter_live[arm] = bool(drafter.begin(context))
            self.drafter_capable[arm] = self.drafter_live[arm]
        if self.drafter_live[arm]:
            try:
                if generator is not None and _takes_generator(drafter.draft):
                    ids, logprobs = drafter.draft(gamma, greedy=greedy, generator=generator)
                else:
                    ids, logprobs = drafter.draft(gamma, greedy=greedy)
            except CacheUnavailableError:
                self.drafter_live[arm] = False
                self.drafter_capable[arm] = False
            else:
                return ids, logprobs
        return _draft_with_context(drafter, context, gamma, generator)

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
    5. Bandit is updated with emitted tokens (accepted drafts plus the
       residual or bonus token) per second of step time. The raw accept
       count is still recorded separately.
    6. Metrics are updated.

    Batch 1 with ``variant`` ``"rejection"`` or ``"greedy"`` runs a
    dedicated loop that reads the device once per step: one small
    ``tolist()`` of ``[n_accepted, next_token, *draft_ids]``. Everything
    else (sampling, cache crops, the output buffer) stays on the device.
    Other batches and ``variant="typical"`` use the context-based loop.

    ``config.sampling.seed`` seeds a ``torch.Generator`` per
    :meth:`generate` call, so the global RNG is not touched and one seed
    reproduces one output. ``seed=None`` uses the global RNG.

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
        decision = resolve_compile(config.compile, self._device)
        if decision.enabled and not self._enable_compile():
            decision = CompileDecision(
                decision.requested,
                False,
                decision.reason + ", but no model wrapper supports a static cache",
            )
        self._compile_decision = decision
        self._compiled = decision.enabled
        # One line, always: which path runs and why.
        forced_but_degraded = decision.requested is True and (
            not decision.enabled or "may fail" in decision.reason
        )
        log = logger.warning if forced_but_degraded else logger.info
        log(
            "Compiled fast path %s (%s).",
            "on" if decision.enabled else "off",
            decision.reason,
            extra={"compile_requested": decision.requested, "compile_enabled": decision.enabled},
        )

    def _enable_compile(self) -> bool:
        """Move capable wrappers onto a static cache with CUDA-graph capture."""
        switched = False
        for owner in (self._target, *self._drafters):
            use_static = getattr(owner, "use_static_cache", None)
            if callable(use_static):
                use_static(self._config.compile_max_cache_len, compile_forward=True)
                switched = True
        return switched

    @property
    def compiled(self) -> bool:
        """Whether the batch-1 path runs on static caches with CUDA graphs."""
        return self._compiled

    @property
    def compile_decision(self) -> CompileDecision:
        """The resolved ``FlashSpecConfig.compile`` decision and its reason."""
        return self._compile_decision

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        eos_token_ids: Sequence[int] | None = None,
    ) -> GenerationResult:
        """Run the speculative decoding generation loop.

        Parameters
        ----------
        input_ids : torch.Tensor
            Prompt token IDs.  Shape: ``(batch_size, seq_len)``, dtype int64.
        max_new_tokens : int or None
            Maximum tokens to generate.  Defaults to ``config.max_new_tokens``.
        eos_token_ids : sequence of int or None
            Stop tokens for this call. ``None`` uses ``config.eos_token_ids``.
            A row stops right after it emits one; the stop token is kept and
            the rest of the row is ``-1``.

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
        6. The bandit records emitted tokens per second of the step. The
           raw accept count is still recorded separately.
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
        stop_ids = self._config.eos_token_ids if eos_token_ids is None else eos_token_ids
        eos = frozenset(int(token) for token in stop_ids)
        generator = self._make_generator()
        fast = self._config.sampling.variant in ("rejection", "greedy")
        if int(input_ids.shape[0]) == 1 and fast:
            return self._generate_single(input_ids, max_tokens, eos, generator)
        return self._generate_generic(input_ids, max_tokens, eos, generator)

    def _make_generator(self) -> torch.Generator | None:
        """Return a generator seeded with ``config.sampling.seed``, or ``None``."""
        seed = self._config.sampling.seed
        if seed is None:
            return None
        return torch.Generator(device=self._device).manual_seed(int(seed))

    def _generate_generic(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        eos: frozenset[int],
        generator: torch.Generator | None,
    ) -> GenerationResult:
        """Context-based loop: batches > 1 and ``variant="typical"``.

        Rows that share a context length are scored together. This loop
        reads the device several times per step; the batch-1 fast path is
        :meth:`_generate_single`.
        """
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

        finished: set[int] = set()

        def _open(index: int) -> bool:
            return index not in finished and _row_count(index) < max_tokens

        incremental = (
            _Incremental.for_call(self._target, self._drafters) if batch_size == 1 else None
        )

        profiler = self._profiler
        if profiler is not None:
            profiler.reset()
        self._throughput_tracker.start()

        while any(_open(i) for i in range(batch_size)):
            self._latency_tracker.start()
            if profiler is not None:
                profiler.start_step()
            step_started = time.perf_counter()
            active = [i for i in range(batch_size) if _open(i)]
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
                        arm, drafter, context, gamma, greedy=True, generator=generator
                    )
                else:
                    draft_token_ids, _ = _draft_with_context(drafter, context, gamma, generator)
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
                        arm, drafter, context, gamma, generator=generator
                    )
                    if maybe_logprobs is None:
                        raise RuntimeError("Drafter returned no log-probs for a sampling step.")
                    draft_logprobs = maybe_logprobs
                else:
                    draft_token_ids, draft_logprobs = _draft_with_context(
                        drafter, context, gamma, generator
                    )
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
                        generator=generator,
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
                        bonus_id = torch.multinomial(probs, num_samples=1, generator=generator).to(
                            take.dtype
                        )
                    take = torch.cat([take, bonus_id.view(-1)])
                if eos:
                    stop_at = _first_stop(take.tolist(), eos)
                    if stop_at is not None:
                        take = take[: stop_at + 1]
                        finished.add(index)
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
            # Emitted tokens: accepted drafts plus the residual or bonus.
            reward = (emitted / elapsed) if elapsed > 0.0 else 0.0
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

    def _generate_single(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        eos: frozenset[int],
        generator: torch.Generator | None,
    ) -> GenerationResult:
        """Batch-1 loop with one device-to-host read per step.

        Per step: draft, verify, then :func:`rejection_step` or
        :func:`greedy_step` give ``[n_accepted, next_token]`` on the
        device. One ``tolist()`` of that pair plus the draft ids is the
        only read. The accepted tokens are written into a preallocated
        sequence buffer by index, and the caches commit by count.
        """
        gamma = self._config.sampling.gamma
        sampling = self._config.sampling
        greedy = sampling.greedy
        device = self._device
        prompt = input_ids.to(device=device, dtype=torch.long)
        prompt_len = int(prompt.shape[1])
        capacity = prompt_len + max_tokens + gamma + 1
        buffer = torch.empty((1, capacity), dtype=torch.long, device=device)
        buffer[:, :prompt_len] = prompt
        length = prompt_len
        produced: list[int] = []
        incremental = _Incremental.for_call(self._target, self._drafters)
        for owner in (self._target, *self._drafters):
            reserve = getattr(owner, "reserve_cache", None)
            if callable(reserve):
                reserve(capacity + 1)

        profiler = self._profiler
        if profiler is not None:
            profiler.reset()
        self._throughput_tracker.start()
        # The step clock is read after the step's single device read, which
        # already waits for the device, so no extra synchronize is needed.
        step_clock = time.perf_counter()

        while len(produced) < max_tokens:
            if profiler is not None:
                profiler.start_step()
            context = buffer[:, :length]
            arm = self._bandit.select()
            drafter = self._drafters[arm]

            if greedy:
                draft_ids, _ = incremental.draft(
                    arm, drafter, context, gamma, greedy=True, generator=generator
                )
                if profiler is not None:
                    profiler.mark("draft")
                target_argmax = incremental.verify(
                    self._target, context, draft_ids, gamma, 1.0, greedy=True
                )
                if profiler is not None:
                    profiler.mark("verify")
                step = greedy_step(draft_ids, target_argmax)
            else:
                draft_ids, draft_logprobs = incremental.draft(
                    arm, drafter, context, gamma, generator=generator
                )
                if draft_logprobs is None:
                    raise RuntimeError("Drafter returned no log-probs for a sampling step.")
                if profiler is not None:
                    profiler.mark("draft")
                target_logprobs = incremental.verify(
                    self._target, context, draft_ids, gamma, sampling.temperature
                )
                if profiler is not None:
                    profiler.mark("verify")
                step = rejection_step(
                    draft_ids, draft_logprobs, target_logprobs, generator=generator
                )

            # The one device-to-host read of the step.
            host = torch.cat([step.view(-1), draft_ids.view(-1).to(step.device)]).tolist()
            n_accepted, next_token = int(host[0]), int(host[1])
            new_tokens = host[2 : 2 + n_accepted]
            if next_token >= 0:
                new_tokens.append(next_token)
            take = new_tokens[: max_tokens - len(produced)]
            stopped = False
            if eos:
                stop_at = _first_stop(take, eos)
                if stop_at is not None:
                    take = take[: stop_at + 1]
                    stopped = True
            if profiler is not None:
                profiler.mark("sample")

            emitted = len(take)
            if next_token >= 0:
                full = torch.cat([draft_ids[:, :n_accepted], step[:, 1:2]], dim=1)
            else:
                full = draft_ids[:, :n_accepted]
            piece = full[:, :emitted]
            buffer[:, length : length + emitted] = piece
            length += emitted
            produced.extend(take)
            accepted_prefix = buffer[:, :length]
            if emitted:
                incremental.commit(
                    arm, self._drafters, self._target, n_accepted, piece, accepted_prefix
                )
            for owner in (drafter, self._target):
                if incremental.uses_cache(owner, arm, drafter):
                    continue
                roll = getattr(owner, "roll_cache_to", None)
                if callable(roll):
                    roll(accepted_prefix)

            now = time.perf_counter()
            elapsed = now - step_clock
            step_clock = now
            # Emitted tokens: accepted drafts plus the residual or bonus.
            reward = (emitted / elapsed) if elapsed > 0.0 else 0.0
            self._bandit.update(arm, accepted=n_accepted, reward=reward)
            self._acceptance_tracker.record(n_accepted)
            self._latency_tracker.record(elapsed * 1000.0)
            if profiler is not None:
                profiler.mark("bookkeeping")
                profiler.end_step(emitted)
            if stopped:
                break
            if emitted == 0:
                logger.warning("Speculative step accepted no tokens; stopping early.")
                break

        n_new = len(produced)
        padded = produced + [-1] * (max_tokens - n_new)
        output_ids = torch.tensor([padded], dtype=input_ids.dtype, device=device)
        tokens_per_second = self._throughput_tracker.stop(n_tokens=n_new)
        logger.debug(
            "Generation complete",
            extra={
                "n_tokens": n_new,
                "tokens_per_second": tokens_per_second,
                "alpha": self._acceptance_tracker.mean_acceptance_rate,
            },
        )
        return GenerationResult(
            output_ids=output_ids,
            n_tokens_generated=n_new,
            acceptance_rate=self._acceptance_tracker.mean_acceptance_rate,
            tokens_per_second=tokens_per_second,
            step_profile=tuple(profiler.records()) if profiler is not None else None,
        )


def _first_stop(tokens: list[int], eos: frozenset[int]) -> int | None:
    """Index of the first stop token in ``tokens``, or ``None``."""
    for index, token in enumerate(tokens):
        if token in eos:
            return index
    return None
