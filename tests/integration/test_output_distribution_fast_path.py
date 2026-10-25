"""Chi-squared checks for the batch-1 fast path (``rejection_step``).

``tests/integration/test_output_distribution.py`` checks
``rejection_sample`` and the context-based engine loop at a batch of
10,000 rows. The batch-1 loop uses :func:`rejection_step` instead: the
same acceptance test and residual formula, one ``multinomial`` draw that
is either the residual or the bonus token, and no host sync. This file
repeats those checks against it with the same pairs, sizes, seeds and
critical value, so a wrong residual or a wrong bonus would fail here.

The second emitted token covers the bonus: at gamma 1 a full accept
emits the draft token and then the bonus, drawn from the target row.
"""

from __future__ import annotations

import pytest
import torch

import flashspec.sampling.rejection as rejection
from flashspec.bandit.ucb import UCB1Selector
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

# Batch-1 generations are a Python loop; this keeps every expected count
# well above 5 at vocab 6.
_ENGINE_SINGLE_N = 4_000


def _step_marginal(
    p: torch.Tensor,
    q: torch.Tensor,
    gamma: int,
    n: int,
    seed: int,
    use_generator: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """First and second emitted tokens from ``rejection_step`` over ``n`` rows.

    The second token exists for rows that accepted at least one draft
    token: it is the next draft token, the residual, or (after a full
    accept) the bonus.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed) if use_generator else None
    draft_lp = q.log().view(1, 1, -1).expand(n, gamma, -1).contiguous()
    target_lp = p.log().view(1, 1, -1).expand(n, gamma + 1, -1).contiguous()
    draft_ids = torch.multinomial(q.expand(n * gamma, -1), num_samples=1).view(n, gamma)
    step = rejection.rejection_step(draft_ids, draft_lp, target_lp, generator=generator)
    n_accepted, next_token = step[:, 0], step[:, 1]
    emitted = torch.cat([draft_ids, torch.full((n, 1), -1, dtype=torch.long)], dim=1)
    emitted.scatter_(1, n_accepted.unsqueeze(1), next_token.unsqueeze(1))
    first = emitted[:, 0]
    reached = n_accepted >= 1
    return first, emitted[reached, 1]


class TestRejectionStepMarginal:
    """Fixed ``(p, q)`` pairs. The first and second emitted tokens follow ``p``."""

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
        """Chi-square the first emitted token and the second one (draft, residual or bonus)."""
        calls = _select_backend(monkeypatch, backend)
        p = _probs(p_values)
        q = _probs(q_values)
        first, second = _step_marginal(p, q, gamma, _N, 1000 + gamma, use_generator)
        assert bool((first >= 0).all())
        first_stat = _chi2(first, p)
        assert first_stat < _CHI2_CRITICAL_DF5, (
            f"{name} gamma={gamma} {backend} generator={use_generator} "
            f"first-token chi-square {first_stat:.3f} >= {_CHI2_CRITICAL_DF5}"
        )
        # The residual row is computed for every row, so the kernel runs once per row.
        assert calls["n"] == (_N if backend == "asm" else 0)
        assert int(second.numel()) >= 5_000
        assert bool((second >= 0).all())
        second_stat = _chi2(second, p)
        assert second_stat < _CHI2_CRITICAL_DF5, (
            f"{name} gamma={gamma} {backend} generator={use_generator} "
            f"second-token chi-square {second_stat:.3f} >= {_CHI2_CRITICAL_DF5}"
        )


class TestSingleRowEngineMarginal:
    """The batch-1 engine loop, one generation at a time, on the context-dependent toy."""

    @pytest.mark.parametrize("gamma", [1, 4])
    def test_matches_target_chain(self, gamma: int) -> None:
        """First token follows the target row; second follows the one-step mixture."""
        target_table = _rolled_table((0.40, 0.20, 0.15, 0.12, 0.08, 0.05))
        draft_table = _rolled_table((0.05, 0.08, 0.12, 0.15, 0.25, 0.35))
        config = FlashSpecConfig(
            drafter_strategy="bandit",
            device="cpu",
            dtype="float32",
            max_new_tokens=2,
            bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
            sampling=SamplingConfig(gamma=gamma, temperature=1.0),
        )
        engine = SpeculativeEngine(
            config,
            [_ContextTable(draft_table)],  # type: ignore[list-item]
            _ContextTable(target_table),  # type: ignore[arg-type]
            UCB1Selector(n_arms=1, window_size=0),
        )
        torch.manual_seed(11 + gamma)
        prompt = torch.full((1, 4), 2, dtype=torch.long)
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
