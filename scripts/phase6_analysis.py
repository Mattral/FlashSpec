"""Why the Phase 6 bandit lost: decompose it from the arm reports of one bundle.

    python scripts/phase6_analysis.py benchmarks/results/gpu/3e38251/results [--json out.json]

Every number is copied from the result JSONs or computed from them. Rows
marked *counterfactual* are arithmetic on measured values under a stated
assumption, not measurements.

1. **Exploration.** Share of the bandit's steps and step seconds on each
   arm, and on arms other than the block's best fixed arm.
2. **Verify-width overhead.** The bandit verifies every arm at width 7.
   Per arm, its mean step time inside the bandit and alone (fixed arm, own
   width, same block). *Counterfactual:* the bandit's emitted tokens divided
   by its pulls costed at each arm's stand-alone step time. That removes the
   width overhead and also the bandit's other per-step costs (catch-up of the
   idle model drafter, the lookup computed every step); the gamma-6 arms,
   which verify at width 7 either way, show the non-width part.
3. **Reward vs throughput.** Per arm, the bandit's mean per-step reward
   (mean of emitted / step seconds, what the selector maximises) next to its
   tokens per second (emitted / seconds), and the arm each one ranks first.
4. **Best arm by prompt.** On the mixed block, each fixed arm's step
   seconds per prompt are recovered from its cumulative report (every prompt
   emits ``max_new_tokens``; no EOS stop). The best arm per prompt, how
   often it changes along the run order, which arm wins per prompt set, and
   *counterfactual* oracles: best arm per prompt, best arm per prompt set,
   against the best single arm (step seconds only; prefill excluded).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["analyse", "main", "render"]

BANDITS = ("bandit_ucb1", "bandit_thompson")


def _mode(doc: dict[str, Any]) -> str:
    return "greedy" if (doc.get("decoding") or {}).get("mode") == "greedy" else "t0.7"


def _blocks(directory: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    blocks: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "phase6" in doc:
            blocks.setdefault((_mode(doc), str(doc["dataset"])), {})[str(doc["method"])] = doc
    return blocks


def _fixed(docs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fixed-arm documents by arm label."""
    return {name.removeprefix("arm_"): doc for name, doc in docs.items() if name.startswith("arm_")}


def _per_arm(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm["label"]: arm for arm in doc["arm_report"]["per_arm"]}


def _step_tps(doc: dict[str, Any]) -> float:
    arms = doc["arm_report"]["per_arm"]
    seconds = sum(arm["seconds"] for arm in arms)
    return sum(arm["emitted"] for arm in arms) / seconds if seconds else 0.0


def _bandit_block(docs: dict[str, dict[str, Any]], bandit: str) -> dict[str, Any]:
    fixed = _fixed(docs)
    best_label = max(fixed, key=lambda label: fixed[label]["metrics"]["workload_tokens_per_second"])
    arms = _per_arm(docs[bandit])
    total_steps = sum(arm["pulls"] for arm in arms.values())
    total_seconds = sum(arm["seconds"] for arm in arms.values())
    rows: dict[str, Any] = {}
    counterfactual_seconds = 0.0
    for label, arm in arms.items():
        alone = _per_arm(fixed[label])[label]
        step_bandit = arm["seconds"] / arm["pulls"] if arm["pulls"] else None
        step_alone = alone["seconds"] / alone["pulls"] if alone["pulls"] else None
        if step_alone is not None:
            counterfactual_seconds += arm["pulls"] * step_alone
        rows[label] = {
            "step_share": arm["pulls"] / total_steps if total_steps else None,
            "seconds_share": arm["seconds"] / total_seconds if total_seconds else None,
            "step_ms_in_bandit": None if step_bandit is None else step_bandit * 1000.0,
            "step_ms_alone": None if step_alone is None else step_alone * 1000.0,
            "step_cost_ratio": step_bandit / step_alone if step_bandit and step_alone else None,
            "tokens_per_step_in_bandit": arm["tokens_per_step"],
            "tokens_per_step_alone": alone["tokens_per_step"],
            "mean_reward": arm["mean_reward"],
            "tokens_per_second": arm["tokens_per_second"],
            "reward_over_tps": (
                arm["mean_reward"] / arm["tokens_per_second"]
                if arm["mean_reward"] is not None and arm["tokens_per_second"]
                else None
            ),
        }
    emitted = sum(arm["emitted"] for arm in arms.values())
    measured = _step_tps(docs[bandit])
    counterfactual = emitted / counterfactual_seconds if counterfactual_seconds else None
    by_reward = max(
        (label for label in arms if arms[label]["mean_reward"] is not None),
        key=lambda label: arms[label]["mean_reward"],
    )
    by_tps = max(
        (label for label in arms if arms[label]["tokens_per_second"] is not None),
        key=lambda label: arms[label]["tokens_per_second"],
    )
    best_fixed_tps = fixed[best_label]["metrics"]["workload_tokens_per_second"]
    workload = docs[bandit]["metrics"]["workload_tokens_per_second"]
    return {
        "best_fixed_arm": best_label,
        "bandit_over_best_fixed_workload": workload / best_fixed_tps,
        "off_best_step_share": 1.0 - rows[best_label]["step_share"],
        "off_best_seconds_share": 1.0 - rows[best_label]["seconds_share"],
        "most_pulled_arm": max(arms, key=lambda label: arms[label]["pulls"]),
        "top_arm_by_mean_reward": by_reward,
        "top_arm_by_tokens_per_second": by_tps,
        "step_tps_measured": measured,
        "step_tps_counterfactual_own_width": counterfactual,
        "counterfactual_over_measured": counterfactual / measured if counterfactual else None,
        "counterfactual_over_best_fixed_step_tps": (
            counterfactual / _step_tps(fixed[best_label]) if counterfactual else None
        ),
        "catch_up": docs[bandit]["arm_report"]["catch_up"],
        "arms": rows,
    }


def _per_prompt_seconds(doc: dict[str, Any], n_tokens: list[int]) -> list[float]:
    """Step seconds per prompt from the cumulative tokens/s (emitted per prompt is known)."""
    seconds: list[float] = []
    emitted = 0
    previous = 0.0
    for entry, tokens in zip(doc["arm_report"]["cumulative"], n_tokens, strict=True):
        emitted += tokens
        total = emitted / entry["tokens_per_second"]
        seconds.append(total - previous)
        previous = total
    return seconds


def _prompt_oracles(docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fixed = _fixed(docs)
    first = next(iter(fixed.values()))
    ids = list(first["prompt_ids"])
    n_tokens = [entry["n_tokens"] for entry in first["outputs"]["prompts"]]
    seconds = {label: _per_prompt_seconds(doc, n_tokens) for label, doc in fixed.items()}
    best = [min(seconds, key=lambda label, i=i: seconds[label][i]) for i in range(len(ids))]
    changes = sum(1 for a, b in zip(best, best[1:], strict=False) if a != b)
    sets = sorted({pid.split("/", 1)[0] for pid in ids})
    by_set: dict[str, dict[str, int]] = {}
    for pid, label in zip(ids, best, strict=True):
        counts = by_set.setdefault(pid.split("/", 1)[0], {})
        counts[label] = counts.get(label, 0) + 1
    total_tokens = sum(n_tokens)
    single = {label: total_tokens / sum(values) for label, values in seconds.items()}
    best_single = max(single, key=lambda label: single[label])
    per_prompt_oracle = total_tokens / sum(seconds[label][i] for i, label in enumerate(best))
    set_seconds = 0.0
    set_best: dict[str, str] = {}
    for name in sets:
        index = [i for i, pid in enumerate(ids) if pid.split("/", 1)[0] == name]
        label = min(seconds, key=lambda lab: sum(seconds[lab][i] for i in index))
        set_best[name] = label
        set_seconds += sum(seconds[label][i] for i in index)
    per_set_oracle = total_tokens / set_seconds
    return {
        "order": ids,
        "best_arm_per_prompt": best,
        "best_arm_changes_between_consecutive_prompts": changes,
        "consecutive_pairs": len(best) - 1,
        "best_arm_counts_by_set": by_set,
        "best_arm_by_set": set_best,
        "single_arm_step_tps": single,
        "best_single_arm": best_single,
        "oracle_per_prompt_step_tps": per_prompt_oracle,
        "oracle_per_set_step_tps": per_set_oracle,
        "oracle_per_prompt_over_best_single": per_prompt_oracle / single[best_single],
        "oracle_per_set_over_best_single": per_set_oracle / single[best_single],
    }


def analyse(directory: Path) -> dict[str, Any]:
    """Compute the four decompositions for every mode and block that has the inputs."""
    blocks = _blocks(directory)
    out: dict[str, Any] = {}
    for (mode, dataset), docs in sorted(blocks.items()):
        if not _fixed(docs):
            continue
        entry: dict[str, Any] = {}
        for bandit in BANDITS:
            if bandit in docs:
                entry[bandit] = _bandit_block(docs, bandit)
        if dataset == "mixed":
            entry["prompt_oracles"] = _prompt_oracles(docs)
        out.setdefault(mode, {})[dataset] = entry
    return out


def render(result: dict[str, Any]) -> str:
    """Markdown tables of :func:`analyse`'s output, values as computed (not rounded)."""
    lines: list[str] = []
    for mode, blocks in result.items():
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(
            "| block | bandit | / best fixed (workload) | best fixed arm | off-best step share "
            "| off-best seconds share | step tok/s measured | counterfactual at own widths "
            "| counterfactual / best fixed (step) | top arm by mean reward | top arm by tok/s "
            "| most pulled |"
        )
        lines.append("|" + "---|" * 12)
        for dataset, entry in blocks.items():
            for bandit in BANDITS:
                if bandit not in entry:
                    continue
                row = entry[bandit]
                lines.append(
                    f"| {dataset} | {bandit} | {row['bandit_over_best_fixed_workload']} "
                    f"| {row['best_fixed_arm']} | {row['off_best_step_share']} "
                    f"| {row['off_best_seconds_share']} | {row['step_tps_measured']} "
                    f"| {row['step_tps_counterfactual_own_width']} "
                    f"| {row['counterfactual_over_best_fixed_step_tps']} "
                    f"| {row['top_arm_by_mean_reward']} | {row['top_arm_by_tokens_per_second']} "
                    f"| {row['most_pulled_arm']} |"
                )
        lines.append("")
        mixed = blocks.get("mixed", {})
        for bandit in BANDITS:
            if bandit not in mixed:
                continue
            lines.append(f"### mixed, {bandit}: per arm")
            lines.append("")
            lines.append(
                "| arm | step share | seconds share | step ms in bandit | step ms alone "
                "| ratio | tokens/step in bandit | alone | mean reward | tok/s | reward / tok/s |"
            )
            lines.append("|" + "---|" * 11)
            for label, arm in mixed[bandit]["arms"].items():
                lines.append(
                    f"| {label} | {arm['step_share']} | {arm['seconds_share']} "
                    f"| {arm['step_ms_in_bandit']} | {arm['step_ms_alone']} "
                    f"| {arm['step_cost_ratio']} | {arm['tokens_per_step_in_bandit']} "
                    f"| {arm['tokens_per_step_alone']} | {arm['mean_reward']} "
                    f"| {arm['tokens_per_second']} | {arm['reward_over_tps']} |"
                )
            lines.append(f"catch-up: {mixed[bandit]['catch_up']}")
            lines.append("")
        oracle = mixed.get("prompt_oracles")
        if oracle:
            lines.append("### mixed: best arm per prompt (fixed arms, step seconds)")
            lines.append("")
            lines.append(
                f"- best arm changes between consecutive prompts: "
                f"{oracle['best_arm_changes_between_consecutive_prompts']} of "
                f"{oracle['consecutive_pairs']}"
            )
            lines.append(f"- best arm counts by set: {oracle['best_arm_counts_by_set']}")
            lines.append(f"- best arm by set (summed seconds): {oracle['best_arm_by_set']}")
            lines.append(
                f"- best single arm: {oracle['best_single_arm']} "
                f"{oracle['single_arm_step_tps'][oracle['best_single_arm']]} tok/s (step)"
            )
            lines.append(
                "- counterfactual oracle per prompt: "
                f"{oracle['oracle_per_prompt_step_tps']} tok/s, "
                f"x{oracle['oracle_per_prompt_over_best_single']} the best single arm"
            )
            lines.append(
                "- counterfactual oracle per prompt set: "
                f"{oracle['oracle_per_set_step_tps']} tok/s, "
                f"x{oracle['oracle_per_set_over_best_single']} the best single arm"
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the analysis; with ``--json``, also write it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = analyse(args.results)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
