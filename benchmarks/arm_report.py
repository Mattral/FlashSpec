"""Per-run bandit report built from ``GenerationResult.arm_steps``.

Input: one list of :class:`flashspec.engine.arms.ArmStep` per prompt, in run
order, from one engine (so the bandit's state carries across prompts).
Output: a JSON-ready dict with

- ``per_arm``: pulls, share of steps, proposed / accepted / emitted
  tokens, acceptance (accepted / proposed), tokens per step, mean reward,
  time, and tokens per second as total emitted / total seconds; for a
  composite arm, how many steps used the lookup and how many the fallback;
- ``selection_over_time``: arm pull counts per prompt, and per window of
  ``window`` consecutive steps;
- ``cumulative``: after each prompt, the running sum of rewards and the
  running tokens per second (emitted / seconds so far);
- ``catch_up``: model-drafter steps whose first call carried more than the
  drafter's own leftover (2 tokens at most), and the tokens they carried.

Every value is computed from the recorded steps; nothing is estimated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flashspec.engine.arms import ArmSpec, ArmStep, arm_label

__all__ = ["arm_report"]


def arm_report(
    runs: Sequence[Sequence[ArmStep]],
    arms: Sequence[ArmSpec],
    lookup: Sequence[bool],
    *,
    window: int = 50,
) -> dict[str, Any]:
    """Summarise per-step arm records of one run.

    Parameters
    ----------
    runs : sequence of sequence of ArmStep
        One entry per prompt, in run order.
    arms : sequence of ArmSpec
        The engine's arms, in bandit index order.
    lookup : sequence of bool
        Per drafter, whether it is a lookup drafter.
    window : int
        Steps per window in ``selection_over_time.per_window``.

    Returns
    -------
    dict
        See the module docstring.
    """
    n_arms = len(arms)
    labels = [arm_label(arms, index) for index in range(n_arms)]
    steps = [step for run in runs for step in run]
    total_steps = len(steps)
    per_arm: list[dict[str, Any]] = []
    for index in range(n_arms):
        mine = [step for step in steps if step.arm == index]
        pulls = len(mine)
        proposed = sum(step.proposed for step in mine)
        accepted = sum(step.accepted for step in mine)
        emitted = sum(step.emitted for step in mine)
        seconds = sum(step.seconds for step in mine)
        entry: dict[str, Any] = {
            "arm": index,
            "label": labels[index],
            "drafter": arms[index].drafter,
            "gamma": arms[index].gamma,
            "fallback": arms[index].fallback,
            "pulls": pulls,
            "share": pulls / total_steps if total_steps else 0.0,
            "proposed": proposed,
            "accepted": accepted,
            "emitted": emitted,
            "acceptance": accepted / proposed if proposed else None,
            "tokens_per_step": emitted / pulls if pulls else None,
            "mean_reward": sum(step.reward for step in mine) / pulls if pulls else None,
            "seconds": seconds,
            "tokens_per_second": emitted / seconds if seconds > 0.0 else None,
        }
        if arms[index].fallback is not None:
            entry["steps_by_drafter"] = {
                "lookup": sum(1 for step in mine if lookup[step.drafter]),
                "fallback": sum(1 for step in mine if not lookup[step.drafter]),
            }
        per_arm.append(entry)

    per_prompt = [[sum(1 for step in run if step.arm == k) for k in range(n_arms)] for run in runs]
    per_window = [
        [sum(1 for step in steps[start : start + window] if step.arm == k) for k in range(n_arms)]
        for start in range(0, total_steps, window)
    ]
    reward_sum = 0.0
    emitted_sum = 0
    seconds_sum = 0.0
    cumulative: list[dict[str, Any]] = []
    for prompt_index, run in enumerate(runs):
        for step in run:
            reward_sum += step.reward
            emitted_sum += step.emitted
            seconds_sum += step.seconds
        cumulative.append(
            {
                "prompt_index": prompt_index,
                "steps": sum(len(item) for item in runs[: prompt_index + 1]),
                "reward_sum": reward_sum,
                "tokens_per_second": emitted_sum / seconds_sum if seconds_sum > 0.0 else None,
            }
        )
    catch_up = [step for step in steps if not lookup[step.drafter] and step.drafter_pending > 2]
    return {
        "labels": labels,
        "n_steps": total_steps,
        "per_arm": per_arm,
        "selection_over_time": {
            "per_prompt": per_prompt,
            "window": window,
            "per_window": per_window,
        },
        "cumulative": cumulative,
        "catch_up": {
            "steps": len(catch_up),
            "tokens": sum(step.drafter_pending for step in catch_up),
            "definition": "model-drafter steps whose first draft call forwarded more than "
            "2 pending tokens (its own leftover is at most 2)",
        },
        "reward_definition": "emitted tokens / step seconds, per step (what the bandit sees); "
        "tokens_per_second is total emitted / total seconds",
    }
