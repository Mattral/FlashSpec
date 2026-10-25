"""Chi-squared checks for prompt lookup: Algorithm 1 with a one-hot ``q``.

A lookup drafter proposes each token with probability one, so its ``q``
is one-hot. :func:`rejection_step_onehot` accepts a proposal ``x`` with
probability ``min(1, p(x))`` and, on a rejection, samples
``max(0, p - onehot(x))`` renormalised. Given ``x``, the emitted token is
then distributed as ``p``, whatever rule chose ``x``. These tests check
that with the same pairs, sizes, seeds and critical value as
``tests/integration/test_output_distribution.py``:

- the step function on its own, with random proposal lengths in
  ``[0, gamma]`` (``0`` is a plain decode step), both residual backends,
  with and without a seeded generator;
- the batch-1 engine with an :class:`NgramDrafter` on the
  context-dependent toy target;
- that the checks bite: two deliberately wrong residuals are swapped in
  and every pair must then fail.

The context-based engine loop (batch > 1) has no lookup path, so there is
no batch-10,000 engine case here; the batch-1 loop is the only path a
lookup drafter runs on.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import flashspec.sampling.rejection as rejection
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.arms import ArmSpec
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig
from tests.integration.test_output_distribution import (
    _CHI2_CRITICAL_DF5,
    _N,
    _PAIRS,
    _chi2,
    _ContextTable,
    _probs,
    _rolled_table,
    _select_backend,
)

_ENGINE_SINGLE_N = 4_000


def _onehot_marginal(
    p: torch.Tensor,
    proposal: torch.Tensor,
    gamma: int,
    n: int,
    seed: int,
    use_generator: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First and second emitted tokens of ``rejection_step_onehot`` over ``n`` rows.

    Proposals are drawn from ``proposal`` and real lengths uniformly from
    ``[0, gamma]`` up front, with their own seed; the step sees them as
    fixed. The second token exists for rows that accepted at least one
    proposal.
    """
    setup = torch.Generator().manual_seed(seed + 500)
    draft_ids = torch.multinomial(
        proposal.expand(n * gamma, -1), num_samples=1, generator=setup
    ).view(n, gamma)
    draft_len = torch.randint(0, gamma + 1, (n,), generator=setup)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed) if use_generator else None
    target_lp = p.log().view(1, 1, -1).expand(n, gamma + 1, -1).contiguous()
    step = rejection.rejection_step_onehot(draft_ids, target_lp, draft_len, generator=generator)
    n_accepted, next_token = step[:, 0], step[:, 1]
    assert bool((n_accepted <= draft_len).all())
    emitted = torch.cat([draft_ids, torch.full((n, 1), -1, dtype=torch.long)], dim=1)
    emitted.scatter_(1, n_accepted.unsqueeze(1), next_token.unsqueeze(1))
    first = emitted[:, 0]
    return first, emitted[n_accepted >= 1, 1]


class TestOnehotStepMarginal:
    """Fixed ``p``, proposals from a different distribution: emitted tokens follow ``p``."""

    @pytest.mark.parametrize("use_generator", [False, True])
    @pytest.mark.parametrize("backend", ["pytorch", "asm"])
    @pytest.mark.parametrize("gamma", [1, 4])
    @pytest.mark.parametrize(("name", "p_values", "q_values"), _PAIRS)
    def test_first_and_second_token_match_p(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_generator: bool,
        backend: str,
        gamma: int,
        name: str,
        p_values: tuple[float, ...],
        q_values: tuple[float, ...],
    ) -> None:
        """Chi-square the first emitted token and the one after an accepted proposal."""
        calls = _select_backend(monkeypatch, backend)
        p = _probs(p_values)
        q = _probs(q_values)
        first, second = _onehot_marginal(p, q, gamma, _N, 1000 + gamma, use_generator)
        assert bool((first >= 0).all())
        first_stat = _chi2(first, p)
        assert first_stat < _CHI2_CRITICAL_DF5, (
            f"{name} gamma={gamma} {backend} generator={use_generator} "
            f"first-token chi-square {first_stat:.3f} >= {_CHI2_CRITICAL_DF5}"
        )
        # The residual row is built for every row, so the kernel runs once per row.
        assert calls["n"] == (_N if backend == "asm" else 0)
        # Expected cell counts stay >= 50 at this floor (smallest p is 0.05).
        assert int(second.numel()) >= 1_000
        assert bool((second >= 0).all())
        second_stat = _chi2(second, p)
        assert second_stat < _CHI2_CRITICAL_DF5, (
            f"{name} gamma={gamma} {backend} generator={use_generator} "
            f"second-token chi-square {second_stat:.3f} >= {_CHI2_CRITICAL_DF5}"
        )


def _ignore_q(p_at_rej: torch.Tensor, q_at_rej: torch.Tensor) -> torch.Tensor:
    """Wrong residual: resample from ``p`` without removing the rejected token."""
    del q_at_rej
    return p_at_rej / p_at_rej.sum(dim=-1, keepdim=True)


def _reversed(p_at_rej: torch.Tensor, q_at_rej: torch.Tensor) -> torch.Tensor:
    """Wrong residual: ``max(0, q - p)`` (the operands swapped)."""
    residual = torch.clamp(q_at_rej - p_at_rej, min=0.0)
    return residual / residual.sum(dim=-1, keepdim=True).clamp(min=1e-9)


class TestWrongResidualIsCaught:
    """With a wrong residual swapped in, every pair fails the same check."""

    @pytest.mark.parametrize("wrong", [_ignore_q, _reversed], ids=["ignore_q", "q_minus_p"])
    @pytest.mark.parametrize(("name", "p_values", "q_values"), _PAIRS)
    def test_first_token_chi_square_explodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        wrong: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        name: str,
        p_values: tuple[float, ...],
        q_values: tuple[float, ...],
    ) -> None:
        """Same seed and size as the passing test; the statistic must exceed the 1% point."""
        _select_backend(monkeypatch, "pytorch")
        monkeypatch.setattr(rejection, "_residual_probs", wrong)
        p = _probs(p_values)
        first, _second = _onehot_marginal(p, _probs(q_values), 4, _N, 1004, False)
        stat = _chi2(first, p)
        assert stat > _CHI2_CRITICAL_DF5, f"{name} {wrong.__name__}: chi-square {stat:.3f}"


class TestSingleRowEngineNgram:
    """The batch-1 engine with prompt lookup on the context-dependent toy target."""

    @pytest.mark.parametrize("gamma", [1, 4])
    def test_matches_target_chain(self, gamma: int) -> None:
        """First token follows the target row; second follows the one-step mixture.

        The prompt ``[0, 2, 3, 2]`` makes the lookup propose ``3`` (and then
        ``2``) after the last ``2``, so the first step verifies a real
        proposal whose probability under the target is well below one.
        """
        target_table = _rolled_table((0.40, 0.20, 0.15, 0.12, 0.08, 0.05))
        config = FlashSpecConfig(
            device="cpu",
            dtype="float32",
            max_new_tokens=2,
            bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
            sampling=SamplingConfig(gamma=gamma, temperature=1.0),
        )
        engine = SpeculativeEngine(
            config,
            [NgramDrafter()],  # type: ignore[list-item]
            _ContextTable(target_table),  # type: ignore[arg-type]
            UCB1Selector(n_arms=1, window_size=0),
        )
        prompt = torch.tensor([[0, 2, 3, 2]], dtype=torch.long)
        proposed, n_valid = NgramDrafter().propose(prompt, 4, gamma)
        assert int(n_valid) >= 1 and int(proposed[0, 0]) == 3
        torch.manual_seed(31 + gamma)
        outputs = torch.cat(
            [engine.generate(prompt).output_ids for _ in range(_ENGINE_SINGLE_N)], dim=0
        )
        assert outputs.shape == (_ENGINE_SINGLE_N, 2)
        assert bool((outputs >= 0).all())
        p_first = target_table[2]
        first_stat = _chi2(outputs[:, 0], p_first)
        mixture = (p_first[:, None] * target_table).sum(dim=0)
        second_stat = _chi2(outputs[:, 1], mixture)
        assert first_stat < _CHI2_CRITICAL_DF5, f"first-token chi-square {first_stat:.3f}"
        assert second_stat < _CHI2_CRITICAL_DF5, f"second-token chi-square {second_stat:.3f}"


class TestCompositeArmEngine:
    """Composite arm: lookup where the context has a match, the model drafter elsewhere."""

    def test_matches_target_chain(self) -> None:
        """Steps alternate between one-hot and model ``q``; the chain still follows the target.

        The prompt ``[0, 1, 2]`` has no repeat, so the first step drafts with
        the model; the second step usually finds a lookup match.
        """
        target_table = _rolled_table((0.40, 0.20, 0.15, 0.12, 0.08, 0.05))
        draft_table = _rolled_table((0.05, 0.08, 0.12, 0.15, 0.25, 0.35))
        config = FlashSpecConfig(
            device="cpu",
            dtype="float32",
            max_new_tokens=2,
            bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
            sampling=SamplingConfig(gamma=1, temperature=1.0),
        )
        engine = SpeculativeEngine(
            config,
            [_ContextTable(draft_table), NgramDrafter()],  # type: ignore[list-item]
            _ContextTable(target_table),  # type: ignore[arg-type]
            UCB1Selector(n_arms=1, window_size=0),
            arms=[ArmSpec(1, 1, fallback=0)],
        )
        prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
        torch.manual_seed(77)
        results = [engine.generate(prompt) for _ in range(_ENGINE_SINGLE_N)]
        used = {step.drafter for result in results for step in result.arm_steps}
        assert used == {0, 1}
        outputs = torch.cat([result.output_ids for result in results], dim=0)
        assert bool((outputs >= 0).all())
        p_first = target_table[2]
        first_stat = _chi2(outputs[:, 0], p_first)
        mixture = (p_first[:, None] * target_table).sum(dim=0)
        second_stat = _chi2(outputs[:, 1], mixture)
        assert first_stat < _CHI2_CRITICAL_DF5, f"first-token chi-square {first_stat:.3f}"
        assert second_stat < _CHI2_CRITICAL_DF5, f"second-token chi-square {second_stat:.3f}"
