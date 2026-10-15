"""The mixed-workload demo reports a measured comparison, not a guarantee."""

from __future__ import annotations

from scripts.demo_bandit_mixed import run_demo


def test_demo_compares_ucb_with_fixed_drafters() -> None:
    """UCB, both fixed drafters, and the one-arm control all produce numbers."""
    report = run_demo(max_new_tokens=8, gamma=2, vocab=16, seed=0, window_size=2)
    acceptance = report["metrics"]["acceptance_rate"]
    assert set(acceptance) == {"fixed_a", "fixed_b", "ucb", "single_drafter_control"}
    assert report["single_drafter_matches_fixed_a"] is True
    assert report["measurement_kind"] == "controlled_mixed_workload_simulation"
    assert "speedup_vs_vanilla_ar" in report["unmeasured"]
    assert isinstance(report["improved_acceptance_vs_best_fixed"], bool)
    assert isinstance(report["improved_throughput_vs_best_fixed"], bool)
    # The one-arm control cannot see the second peak. Its acceptance is the
    # fixed-A number, which is the practical statement the demo exists to make.
    assert acceptance["single_drafter_control"] == acceptance["fixed_a"]
