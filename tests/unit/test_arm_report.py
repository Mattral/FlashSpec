"""``benchmarks.arm_report``: per-arm totals, selection over time, cumulative reward."""

from __future__ import annotations

import pytest

from benchmarks.arm_report import arm_report
from flashspec.engine.arms import ArmSpec, ArmStep


def _step(
    arm: int,
    drafter: int,
    proposed: int,
    accepted: int,
    emitted: int,
    seconds: float,
    pending: int = 0,
) -> ArmStep:
    return ArmStep(
        arm=arm,
        drafter=drafter,
        proposed=proposed,
        accepted=accepted,
        emitted=emitted,
        seconds=seconds,
        reward=emitted / seconds,
        drafter_pending=pending,
    )


def test_report_fields() -> None:
    """Totals, ratios, the composite split, windows, cumulative values and catch-up."""
    arms = [ArmSpec(0, 4, label="model"), ArmSpec(1, 4, fallback=0)]
    lookup = [False, True]
    runs = [
        [_step(0, 0, 4, 2, 3, 0.5, pending=1), _step(1, 1, 3, 3, 4, 0.25)],
        [_step(1, 0, 4, 0, 1, 0.5, pending=6), _step(0, 0, 4, 4, 5, 0.5, pending=1)],
    ]
    report = arm_report(runs, arms, lookup, window=3)
    assert report["labels"] == ["model", "arm1"]
    assert report["n_steps"] == 4
    model, composite = report["per_arm"]
    assert model["pulls"] == 2 and model["share"] == 0.5
    assert model["acceptance"] == 6 / 8
    assert model["tokens_per_step"] == 4.0
    assert model["tokens_per_second"] == 8 / 1.0
    assert model["mean_reward"] == pytest.approx((6.0 + 10.0) / 2)
    assert composite["steps_by_drafter"] == {"lookup": 1, "fallback": 1}
    assert composite["acceptance"] == 3 / 7
    assert "steps_by_drafter" not in model
    over_time = report["selection_over_time"]
    assert over_time["per_prompt"] == [[1, 1], [1, 1]]
    assert over_time["per_window"] == [[1, 2], [1, 0]]
    cumulative = report["cumulative"]
    assert cumulative[0]["steps"] == 2 and cumulative[1]["steps"] == 4
    assert cumulative[0]["tokens_per_second"] == 7 / 0.75
    assert cumulative[1]["reward_sum"] == pytest.approx(6.0 + 16.0 + 2.0 + 10.0)
    assert report["catch_up"]["steps"] == 1 and report["catch_up"]["tokens"] == 6


def test_empty_arm_has_no_ratios() -> None:
    """An arm never pulled reports counts of zero and ratios of None."""
    report = arm_report([[_step(0, 0, 2, 1, 2, 1.0)]], [ArmSpec(0, 2), ArmSpec(0, 4)], [False])
    unused = report["per_arm"][1]
    assert unused["pulls"] == 0
    assert unused["acceptance"] is None and unused["tokens_per_second"] is None
