"""``scripts/phase6_analysis.py`` on the committed Phase 6 bundle 3e38251: internal consistency."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.phase6_analysis import analyse, render

_RESULTS = Path(__file__).resolve().parents[2] / "benchmarks/results/gpu/3e38251/results"


@pytest.mark.skipif(not _RESULTS.is_dir(), reason="bundle 3e38251 is not in this checkout")
def test_decomposition_is_consistent() -> None:
    """Shares add to one, ratios follow from their parts, oracles bound the best single arm."""
    result = analyse(_RESULTS)
    assert set(result) == {"greedy", "t0.7"}
    for blocks in result.values():
        assert set(blocks) == {"humaneval", "humaneval_edit", "mt_bench", "mixed"}
        for dataset, entry in blocks.items():
            for bandit in ("bandit_ucb1", "bandit_thompson"):
                row = entry[bandit]
                arms = row["arms"]
                assert sum(arm["step_share"] for arm in arms.values()) == pytest.approx(1.0)
                assert sum(arm["seconds_share"] for arm in arms.values()) == pytest.approx(1.0)
                assert 0.0 <= row["off_best_step_share"] <= 1.0
                assert row["counterfactual_over_measured"] == pytest.approx(
                    row["step_tps_counterfactual_own_width"] / row["step_tps_measured"]
                )
            if dataset == "mixed":
                oracle = entry["prompt_oracles"]
                assert len(oracle["best_arm_per_prompt"]) == 30
                assert oracle["oracle_per_prompt_over_best_single"] >= 1.0
                assert (
                    oracle["oracle_per_prompt_over_best_single"]
                    >= oracle["oracle_per_set_over_best_single"]
                    >= 1.0
                )
    assert "best arm changes between consecutive prompts" in render(result)
