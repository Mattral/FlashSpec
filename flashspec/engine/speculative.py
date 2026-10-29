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
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import (
    COMPOSITE_STRATEGIES,
    ArmSpec,
    ArmStep,
    default_arms,
    strategy_arms,
    validate_arms,
)
from flashspec.engine.drafter import DraftModel
from flashspec.engine.kv_cache import CacheUnavailableError, debug_cache_enabled
from flashspec.engine.ngram import NgramDrafter, is_lookup_drafter
from flashspec.engine.verifier import TargetModel
from flashspec.metrics.acceptance import AcceptanceTracker
from flashspec.metrics.latency import LatencyTracker
from flashspec.metrics.step_profile import StepProfiler, StepRecord
from flashspec.metrics.throughput import ThroughputTracker
from flashspec.sampling.greedy import greedy_step, greedy_verify
from flashspec.sampling.rejection import rejection_sample, rejection_step, rejection_step_onehot
from flashspec.sampling.typical import typical_sample
from flashspec.utils.config import FlashSpecConfig
from flashspec.utils.device import CompileDecision, get_device, resolve_compile
from flashspec.utils.logging import get_logger

__all__ = ["AUTO_STRATEGY", "DRAFTER_PAD_WIDTHS", "SpeculativeEngine", "GenerationResult"]

logger = get_logger(__name__)

# Fixed input widths for a model drafter's compiled calls: 1 (a draft
# step), 2 (pending residual + bonus) and buckets for an idle drafter's
# catch-up. A call is padded at the end to the next width, so these are the
# only shapes captured. Longer catch-ups run eagerly in one call.
DRAFTER_PAD_WIDTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: What ``drafter_strategy="auto"`` picks when it can: the winner of the
#: pre-registered composite comparison (bundle ``fdd3be9``, Tesla T4).
AUTO_STRATEGY = "composite-g6-f2"


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
        Draft-token acceptance rate (alpha) for this call: accepted draft
        tokens over proposed draft tokens, summed over this call's steps.
        Earlier calls on the same engine do not count.
    tokens_per_second : float
        Wall-clock tokens per second for this call.
    step_profile : tuple[StepRecord, ...] or None
        Per-step time breakdown for this call when
        ``config.metrics.profile_steps`` is true, otherwise ``None``.
    arm_steps : tuple[ArmStep, ...]
        One record per step: the arm, the drafter that proposed, proposed,
        accepted and emitted counts, step seconds and the reward.
    """

    output_ids: torch.Tensor
    n_tokens_generated: int
    acceptance_rate: float
    tokens_per_second: float
    step_profile: tuple[StepRecord, ...] | None = None
    arm_steps: tuple[ArmStep, ...] = ()


# ── Batch-1 incremental cache bookkeeping ────────────────────────────────────

_TARGET_API: tuple[str, ...] = ("begin", "verify", "commit")
_DRAFTER_API: tuple[str, ...] = ("begin", "draft", "commit", "push_pending")


def _offers(owner: object, names: tuple[str, ...]) -> bool:
    return all(callable(getattr(owner, name, None)) for name in names)


@functools.lru_cache(maxsize=128)
def _takes_generator(method: Any) -> bool:  # noqa: ANN401
    """Whether ``method`` accepts a ``generator`` keyword."""
    return _takes_keyword(method, "generator")


def _takes_keyword(method: Any, name: str) -> bool:  # noqa: ANN401
    """Whether ``method`` names the keyword ``name``."""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


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

    State is per drafter (``slot`` is a drafter index, not an arm index):
    arms that share a drafter share its cache.

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
        slot: int,
        drafter: Any,  # noqa: ANN401
        context: torch.Tensor,
        gamma: int,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.drafter_capable[slot] and not self.drafter_live[slot]:
            self.drafter_live[slot] = bool(drafter.begin(context))
            self.drafter_capable[slot] = self.drafter_live[slot]
        if self.drafter_live[slot]:
            try:
                if generator is not None and _takes_generator(drafter.draft):
                    ids, logprobs = drafter.draft(gamma, greedy=greedy, generator=generator)
                else:
                    ids, logprobs = drafter.draft(gamma, greedy=greedy)
            except CacheUnavailableError:
                self.drafter_live[slot] = False
                self.drafter_capable[slot] = False
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
        slot: int,
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
            if index != slot:
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

    def uses_cache(self, owner: object, slot: int, drafter: object) -> bool:
        """Whether ``owner`` is kept in step by :meth:`commit` instead of ``roll_cache_to``."""
        if owner is drafter:
            return self.drafter_capable[slot]
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
        Draft model instances. Without ``arms``, one per bandit arm. Lookup
        drafters (:class:`flashspec.engine.ngram.NgramDrafter`) may be mixed
        with model drafters.
    target : TargetModel
        The target (verifier) model instance.
    bandit : DraftSelector or None
        Selector that picks an arm each step. Optional when the resolved
        strategy has one arm (the default ``drafter_strategy="auto"`` with one
        model drafter, or any single arm): the engine then uses a one-arm
        :class:`flashspec.bandit.UCB1Selector`, which always picks that arm.
        With several arms it is required (``UCB1Selector(n_arms=k)`` or
        ``ThompsonSelector(n_arms=k)``), and omitting it raises
        ``ValueError``.
    arms : sequence of ArmSpec or None
        The bandit's arms: a drafter index and a gamma each, so arms can
        share a drafter (one model, one cache) at different gammas.
        ``None`` builds them from ``config.drafter_strategy``
        (:func:`flashspec.engine.arms.strategy_arms`): for ``"bandit"``,
        one arm per drafter at ``config.sampling.gamma``; for a composite
        strategy, one composite arm over the first lookup drafter and the
        first model drafter, with an :class:`NgramDrafter` added when none
        is given. ``"auto"`` (the default) resolves to
        :data:`AUTO_STRATEGY` when the drafters are one model drafter and at
        most one lookup drafter, and to ``"bandit"`` otherwise; a call that
        the composite cannot run (batch > 1 or ``variant="typical"``) then
        uses the model drafter at ``config.sampling.gamma``. The choice and
        its reason are logged once and kept in :attr:`drafter_strategy` and
        :attr:`drafter_strategy_reason`.

    Raises
    ------
    ValueError
        If the number of arms does not match ``config.bandit.n_arms`` (not
        checked for ``"auto"`` when ``config.bandit`` was left at its
        default), a composite strategy has no model drafter, or an arm is
        invalid.

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
        bandit: DraftSelector | None = None,
        arms: Sequence[ArmSpec] | None = None,
    ) -> None:
        drafters = list(drafters)
        lookup = [is_lookup_drafter(drafter) for drafter in drafters]
        gamma = config.sampling.gamma
        requested = config.drafter_strategy
        # Under "auto" with the bandit config left at its default, the arm
        # count follows the drafters instead of config.bandit.n_arms.
        check_n_arms = requested != "auto" or "bandit" in config.model_fields_set
        # Arms a call that the auto-chosen composite cannot run uses instead.
        self._auto_fallback: list[ArmSpec] | None = None
        if arms is not None:
            strategy, reason = "arms", "arms= given"
        else:
            models = [index for index, flag in enumerate(lookup) if not flag]
            n_lookup = len(drafters) - len(models)
            if requested != "auto":
                strategy, reason = requested, "set explicitly"
            elif len(models) == 1 and n_lookup <= 1:
                strategy = AUTO_STRATEGY
                reason = "auto: one model drafter; batch-1 calls only, others use the model drafter"
            else:
                strategy = "bandit"
                reason = (
                    f"auto: {len(models)} model and {n_lookup} lookup drafters given; "
                    f"{AUTO_STRATEGY} needs one model drafter"
                )
            if strategy in COMPOSITE_STRATEGIES and n_lookup == 0 and models:
                drafters.append(NgramDrafter())  # type: ignore[arg-type]
                lookup.append(True)
                reason += "; prompt lookup (NgramDrafter) added"
            if strategy == "bandit":
                if check_n_arms and len(drafters) != config.bandit.n_arms:
                    raise ValueError(
                        f"len(drafters)={len(drafters)} must match "
                        f"config.bandit.n_arms={config.bandit.n_arms}."
                    )
                arms = default_arms(len(drafters), gamma)
            else:
                arms = strategy_arms(strategy, lookup, gamma)
                if requested == "auto":
                    self._auto_fallback = [ArmSpec(models[0], gamma, label="model")]
            logger.info(
                "Drafter strategy %s (%s).",
                strategy,
                reason,
                extra={"drafter_strategy": strategy, "requested": requested},
            )
        validate_arms(arms, len(drafters), lookup)
        if check_n_arms and len(arms) != config.bandit.n_arms:
            raise ValueError(
                f"len(arms)={len(arms)} must match config.bandit.n_arms={config.bandit.n_arms}."
            )
        if bandit is None:
            if len(arms) != 1:
                raise ValueError(
                    f"{len(arms)} arms need a bandit selector to choose between them "
                    f"(drafter strategy {strategy!r}). Pass one, for example "
                    f"flashspec.bandit.UCB1Selector(n_arms={len(arms)}) or "
                    f"flashspec.bandit.ThompsonSelector(n_arms={len(arms)}). It is optional "
                    "only when there is one arm."
                )
            # One arm: nothing to choose. A one-arm UCB1 always selects arm 0.
            bandit = UCB1Selector(n_arms=1, window_size=config.bandit.window_size)
        self._strategy = strategy
        self._strategy_reason = reason
        self._fallback_logged: set[str] = set()
        if len(arms) < 2 and strategy == "bandit":
            logger.warning(
                "Bandit has a single arm and cannot adapt. Register two or "
                "more drafters before expecting a throughput or acceptance "
                "gain over a fixed drafter.",
                extra={"n_arms": config.bandit.n_arms},
            )
        if config.sampling.variant == "typical":
            logger.warning(
                "sampling.variant='typical' is lossy: the output distribution differs "
                "from the target model's. Use 'rejection' (or temperature 0) for "
                "lossless speculative decoding."
            )
        self._config = config
        self._drafters = drafters
        # Lookup drafters (prompt lookup) run no model and propose a
        # variable number of tokens; they need the batch-1 fast path.
        self._lookup = lookup
        self._arms: list[ArmSpec] = list(arms)
        self._gamma_max = max(arm.max_gamma for arm in (*self._arms, *(self._auto_fallback or ())))
        self._target = target
        self._bandit = bandit
        self._device = get_device(config.device)

        self._acceptance_tracker = AcceptanceTracker(gamma=self._gamma_max)
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
        """Move capable wrappers onto a static cache with CUDA-graph capture.

        With ``config.compile_fixed_shapes`` the target's verify calls are
        padded to one width, ``gamma_max + 1`` (pending token plus the
        longest draft), and a drafter's calls to :data:`DRAFTER_PAD_WIDTHS`,
        so per-arm gammas, short lookup proposals and catch-ups replay a few
        captured graphs instead of recording one per length.
        """
        fixed = self._config.compile_fixed_shapes
        switched = False
        for owner in (self._target, *self._drafters):
            use_static = getattr(owner, "use_static_cache", None)
            if not callable(use_static):
                continue
            kwargs: dict[str, Any] = {"compile_forward": True}
            if fixed and _takes_keyword(use_static, "pad_widths"):
                widths = (self._gamma_max + 1,) if owner is self._target else DRAFTER_PAD_WIDTHS
                kwargs["pad_widths"] = widths
            use_static(self._config.compile_max_cache_len, **kwargs)
            switched = True
        return switched

    @property
    def arms(self) -> tuple[ArmSpec, ...]:
        """The bandit's arms, in bandit index order."""
        return tuple(self._arms)

    @property
    def drafter_strategy(self) -> str:
        """The strategy the engine runs: ``"bandit"``, a composite name, or ``"arms"``.

        ``"arms"`` means explicit ``arms=`` were given. Under
        ``drafter_strategy="auto"`` this is what ``"auto"`` resolved to.
        """
        return self._strategy

    @property
    def drafter_strategy_reason(self) -> str:
        """Why :attr:`drafter_strategy` was chosen (also logged once at INFO)."""
        return self._strategy_reason

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
           with temperature applied before log-softmax.
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
        batch = int(input_ids.shape[0])
        if self._auto_fallback is not None and (batch != 1 or not fast):
            # drafter_strategy="auto" chose a composite this call cannot run.
            why = f"batch size {batch}" if batch != 1 else "variant 'typical'"
            if why not in self._fallback_logged:
                self._fallback_logged.add(why)
                logger.info(
                    "Drafter strategy bandit for this call (auto: %s; %s needs batch 1 and "
                    "variant 'rejection' or 'greedy'), with the model drafter at gamma %d.",
                    why,
                    self._strategy,
                    self._config.sampling.gamma,
                    extra={"drafter_strategy": "bandit", "requested": "auto"},
                )
            chosen = self._arms
            self._arms = self._auto_fallback
            try:
                return self._generate_generic(input_ids, max_tokens, eos, generator)
            finally:
                self._arms = chosen
        if batch == 1 and fast:
            return self._generate_single(input_ids, max_tokens, eos, generator)
        if any(self._lookup[spec.drafter] or spec.fallback is not None for spec in self._arms):
            raise ValueError(
                "A lookup (n-gram) drafter runs only on the batch-1 fast path: batch size 1 "
                "and sampling.variant 'rejection' or 'greedy'. "
                f"drafter_strategy={self._strategy!r} uses one; pass "
                "FlashSpecConfig(drafter_strategy='bandit') (or 'auto') for this batch size "
                "or variant."
            )
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
        arm_steps: list[ArmStep] = []

        def _open(index: int) -> bool:
            return index not in finished and _row_count(index) < max_tokens

        incremental = (
            _Incremental.for_call(self._target, self._drafters) if batch_size == 1 else None
        )

        profiler = self._profiler
        if profiler is not None:
            profiler.reset()
        # acceptance_rate covers this call's steps only.
        self._acceptance_tracker.reset()
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

            # 1. Bandit selects a draft arm: a drafter and its gamma.
            arm = self._bandit.select()
            spec = self._arms[arm]
            gamma = spec.gamma
            slot = spec.drafter
            drafter = self._drafters[slot]

            greedy_bonus: torch.Tensor | None = None
            if config_sampling.greedy:
                # Greedy: argmax drafts, one argmax verify, accept up to the
                # first mismatch. No softmax and no random draws.
                if incremental is not None:
                    draft_token_ids, _ = incremental.draft(
                        slot, drafter, context, gamma, greedy=True, generator=generator
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
                        slot, drafter, context, gamma, generator=generator
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
                        generator=generator,
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
                        slot,
                        self._drafters,
                        self._target,
                        int(first_rejection[0]),
                        last_piece,
                        accepted_prefix,
                    )
                for owner in (drafter, self._target):
                    if incremental is not None and incremental.uses_cache(owner, slot, drafter):
                        continue
                    roll = getattr(owner, "roll_cache_to", None)
                    if callable(roll):
                        roll(accepted_prefix)

            elapsed = time.perf_counter() - step_started
            # Emitted tokens: accepted drafts plus the residual or bonus.
            reward = (emitted / elapsed) if elapsed > 0.0 else 0.0
            self._bandit.update(arm, accepted=n_accepted, reward=reward)
            self._acceptance_tracker.record(n_accepted, gamma)
            self._latency_tracker.stop()
            arm_steps.append(
                ArmStep(
                    arm=arm,
                    drafter=slot,
                    proposed=gamma,
                    accepted=n_accepted,
                    emitted=emitted,
                    seconds=elapsed,
                    reward=reward,
                )
            )
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
            arm_steps=tuple(arm_steps),
        )

    def _generate_single(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        eos: frozenset[int],
        generator: torch.Generator | None,
    ) -> GenerationResult:
        """Batch-1 loop with one device-to-host read per step.

        Per step: draft, verify, then :func:`rejection_step`,
        :func:`rejection_step_onehot` or :func:`greedy_step` give
        ``[n_accepted, next_token]`` on the device. One ``tolist()`` is the
        only read. The accepted tokens are written into a preallocated
        sequence buffer by index, and the caches commit by count.

        With a lookup drafter, its proposal for the next step is computed
        at the end of this step, before that read: the step's tokens are
        written into the buffer on the device and the lookup runs on the
        device-side new length. The read then also carries the next
        proposal's real length, which is what a composite arm needs to
        choose between lookup and its model fallback without a second read.
        When a composite arm exists, the first step's choice costs one read
        before the loop.
        """
        sampling = self._config.sampling
        greedy = sampling.greedy
        device = self._device
        prompt = input_ids.to(device=device, dtype=torch.long)
        prompt_len = int(prompt.shape[1])
        capacity = prompt_len + max_tokens + self._gamma_max + 1
        # Zeros, not empty: a lookup drafter reads the buffer with fixed
        # shapes, so stale positions must hold valid token ids.
        buffer = torch.zeros((1, capacity), dtype=torch.long, device=device)
        buffer[:, :prompt_len] = prompt
        length = prompt_len
        produced: list[int] = []
        arm_steps: list[ArmStep] = []
        incremental = _Incremental.for_call(self._target, self._drafters)
        for owner in (self._target, *self._drafters):
            reserve = getattr(owner, "reserve_cache", None)
            if callable(reserve):
                reserve(capacity + 1)

        # Lookup drafters: proposal width per drafter (its widest arm), and
        # the current proposal (ids, n_valid on the device, n_valid on the
        # host when known).
        lookup_width: dict[int, int] = {}
        for spec in self._arms:
            if self._lookup[spec.drafter]:
                lookup_width[spec.drafter] = max(lookup_width.get(spec.drafter, 0), spec.gamma)
        lookup_slots = sorted(lookup_width)
        composite = any(spec.fallback is not None for spec in self._arms)
        proposals: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        proposal_len: dict[int, int] = {}
        for slot in lookup_slots:
            proposals[slot] = self._drafters[slot].propose(  # type: ignore[attr-defined]
                buffer, length, lookup_width[slot]
            )
        if composite and lookup_slots:
            counts = torch.cat([proposals[slot][1].view(-1) for slot in lookup_slots]).tolist()
            proposal_len = {
                slot: int(count) for slot, count in zip(lookup_slots, counts, strict=True)
            }

        profiler = self._profiler
        if profiler is not None:
            profiler.reset()
        # acceptance_rate covers this call's steps only.
        self._acceptance_tracker.reset()
        self._throughput_tracker.start()
        # The step clock is read after the step's single device read, which
        # already waits for the device, so no extra synchronize is needed.
        step_clock = time.perf_counter()

        while len(produced) < max_tokens:
            if profiler is not None:
                profiler.start_step()
            context = buffer[:, :length]
            arm = self._bandit.select()
            spec = self._arms[arm]
            gamma = spec.gamma
            slot = spec.drafter
            if spec.fallback is not None and proposal_len.get(slot, 0) == 0:
                # Composite arm, no lookup proposal this step: model drafter,
                # at the fallback's gamma.
                slot = spec.fallback
                gamma = spec.step_gamma(fallback=True)
            drafter = self._drafters[slot]
            n_valid: torch.Tensor | None = None
            pending_before = 0

            if self._lookup[slot]:
                # Prompt lookup (computed at the end of the previous step):
                # gamma proposals of which n_valid are real. No model
                # forward and no device read.
                full_ids, full_valid = proposals[slot]
                draft_ids = full_ids[:, :gamma]
                n_valid = full_valid.clamp(max=gamma)
                if profiler is not None:
                    profiler.mark("draft")
                if greedy:
                    target_argmax = incremental.verify(
                        self._target, context, draft_ids, gamma, 1.0, greedy=True
                    )
                    if profiler is not None:
                        profiler.mark("verify")
                    step = greedy_step(draft_ids, target_argmax, n_valid)
                else:
                    target_logprobs = incremental.verify(
                        self._target, context, draft_ids, gamma, sampling.temperature
                    )
                    if profiler is not None:
                        profiler.mark("verify")
                    # q is one-hot on each proposal: exact Algorithm 1 in
                    # probability space.
                    step = rejection_step_onehot(
                        draft_ids, target_logprobs, n_valid, generator=generator
                    )
            else:
                if incremental.drafter_live[slot]:
                    pending_before = int(getattr(drafter, "pending_length", 0))
                if greedy:
                    draft_ids, _ = incremental.draft(
                        slot, drafter, context, gamma, greedy=True, generator=generator
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
                        slot, drafter, context, gamma, generator=generator
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

            if lookup_slots:
                # Write this step's tokens on the device (accepted drafts,
                # then the next token at n_accepted) and look up the next
                # proposal on the device-side new length, before the read.
                count = step[:, 0:1]
                nxt = step[:, 1:2]
                written = torch.cat([draft_ids, nxt.clamp(min=0)], dim=1).scatter(
                    1, count, nxt.clamp(min=0)
                )
                buffer[:, length : length + gamma + 1] = written
                new_length = (length + count + (nxt >= 0).long()).view(())
                for lookup_slot in lookup_slots:
                    proposals[lookup_slot] = self._drafters[lookup_slot].propose(  # type: ignore[attr-defined]
                        buffer, new_length, lookup_width[lookup_slot]
                    )

            # The one device-to-host read of the step: [n_accepted,
            # next_token, (n_valid for a lookup step), *draft_ids,
            # *next n_valid per lookup drafter].
            pieces = [step.view(-1)]
            if n_valid is not None:
                pieces.append(n_valid.view(-1).to(step.device))
            pieces.append(draft_ids.view(-1).to(step.device))
            for lookup_slot in lookup_slots:
                pieces.append(proposals[lookup_slot][1].view(-1).to(step.device))
            host = torch.cat(pieces).tolist()
            n_accepted, next_token = int(host[0]), int(host[1])
            head = 2
            n_proposed = gamma
            if n_valid is not None:
                n_proposed = int(host[2])
                head = 3
            new_tokens = host[head : head + n_accepted]
            tail = host[head + gamma :]
            proposal_len = {
                lookup_slot: int(value)
                for lookup_slot, value in zip(lookup_slots, tail, strict=True)
            }
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
            if not lookup_slots:
                buffer[:, length : length + emitted] = piece
            length += emitted
            produced.extend(take)
            accepted_prefix = buffer[:, :length]
            if emitted:
                incremental.commit(
                    slot, self._drafters, self._target, n_accepted, piece, accepted_prefix
                )
            for owner in (drafter, self._target):
                if incremental.uses_cache(owner, slot, drafter):
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
            self._acceptance_tracker.record(n_accepted, n_proposed)
            self._latency_tracker.record(elapsed * 1000.0)
            arm_steps.append(
                ArmStep(
                    arm=arm,
                    drafter=slot,
                    proposed=n_proposed,
                    accepted=n_accepted,
                    emitted=emitted,
                    seconds=elapsed,
                    reward=reward,
                    drafter_pending=pending_before,
                )
            )
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
            arm_steps=tuple(arm_steps),
        )


def _first_stop(tokens: list[int], eos: frozenset[int]) -> int | None:
    """Index of the first stop token in ``tokens``, or ``None``."""
    for index, token in enumerate(tokens):
        if token in eos:
            return index
    return None
