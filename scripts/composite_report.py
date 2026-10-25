"""Default drafter strategy for 0.3.0: apply the pre-registered rule to the result JSONs.

    python scripts/composite_report.py benchmarks/results/gpu/<commit>/results [--json out.json]

The rule is fixed in ``benchmarks/configs/qwen_t4_composite.yaml`` before
the run. Throughput is ``metrics.workload_tokens_per_second`` (block total
tokens / block total seconds).

1. Pick the variant with the higher mixed-block greedy throughput.
2. The pick must beat compiled HuggingFace on every single set in both
   modes (``hf_greedy_compiled`` greedy, ``hf_sample_compiled`` sampled).
3. The pick's greedy output must match ``hf_greedy_compiled`` on every
   prompt of every greedy block, except divergences that the document's
   ``greedy_tie_check`` classifies as fp16 exact ties.

The pick becomes the default only if 2 and 3 pass; otherwise neither
variant does. Every number printed is copied from a JSON or is a ratio of
two copied numbers; nothing is rounded. Missing inputs make a check
``n/a``, never a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["VARIANTS", "evaluate", "load_documents", "main", "render"]

VARIANTS = ("arm_composite-g4", "arm_composite-g6-f2")
SINGLE_SETS = ("humaneval", "mt_bench", "humaneval_edit")
GREEDY_BLOCKS = (*SINGLE_SETS, "mixed")
MODES = {"greedy": "hf_greedy_compiled", "t0.7": "hf_sample_compiled"}


def load_documents(directory: Path) -> list[dict[str, Any]]:
    """Load the runner's documents (``"phase6"`` key) in ``directory``."""
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
    return [doc for doc in docs if "phase6" in doc]


def _mode(doc: dict[str, Any]) -> str:
    decoding = doc.get("decoding") or {}
    if decoding.get("mode") == "greedy":
        return "greedy"
    return f"t{decoding.get('temperature')}"


def _tps(doc: dict[str, Any] | None) -> float | None:
    if doc is None:
        return None
    value = (doc.get("metrics") or {}).get("workload_tokens_per_second")
    return float(value) if isinstance(value, (int, float)) else None


def _verdict(ok: bool | None) -> str:
    return "n/a" if ok is None else ("pass" if ok else "FAIL")


def _greedy_exact(doc: dict[str, Any] | None) -> dict[str, Any]:
    """Return matches vs compiled HF and whether every divergence is an allowed exact tie."""
    if doc is None:
        return {"ok": None, "why": "no document"}
    match = doc.get("greedy_token_match_compiled")
    if not isinstance(match, dict):
        return {"ok": None, "why": "no greedy_token_match_compiled"}
    matched, total = match.get("prompts_matched"), match.get("n_prompts")
    out: dict[str, Any] = {"matched": f"{matched}/{total}", "first_divergence": None}
    out["first_divergence"] = match.get("first_divergence")
    if matched == total:
        return {**out, "ok": True, "exact_ties": 0}
    check = doc.get("greedy_tie_check")
    if not isinstance(check, dict) or not check.get("checked"):
        return {**out, "ok": None, "why": "divergences but no greedy_tie_check"}
    divergences = check.get("divergences") or []
    if len(divergences) != int(total) - int(matched):
        return {**out, "ok": None, "why": "greedy_tie_check does not cover every divergence"}
    ties = [entry for entry in divergences if entry.get("allowed_exact_tie") is True]
    out["exact_ties"] = len(ties)
    out["not_allowed"] = [
        {"prompt_id": entry.get("prompt_id"), "position": entry.get("position")}
        for entry in divergences
        if entry.get("allowed_exact_tie") is not True
    ]
    out["divergences"] = [
        {
            key: entry.get(key)
            for key in (
                "prompt_id",
                "position",
                "allowed_exact_tie",
                "replays_reproduce_recorded",
                "same_prefix",
                "reference_top2",
                "path_top2",
                "max_abs_logit_difference",
            )
        }
        for entry in divergences
    ]
    return {**out, "ok": len(ties) == len(divergences)}


def evaluate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Per variant: throughputs, ratios to compiled HF, greedy exactness; then the decision."""
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for doc in documents:
        by_key[(_mode(doc), str(doc.get("dataset")), str(doc.get("method")))] = doc
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        entry: dict[str, Any] = {"tps": {}, "vs_hf": {}, "beats_hf": {}, "greedy": {}}
        for mode, hf in MODES.items():
            for dataset in (*SINGLE_SETS, "mixed"):
                value = _tps(by_key.get((mode, dataset, variant)))
                entry["tps"][f"{mode}/{dataset}"] = value
                if dataset == "mixed":
                    continue
                base = _tps(by_key.get((mode, dataset, hf)))
                ratio = value / base if value is not None and base else None
                entry["vs_hf"][f"{mode}/{dataset}"] = ratio
                entry["beats_hf"][f"{mode}/{dataset}"] = None if ratio is None else ratio > 1.0
        for dataset in GREEDY_BLOCKS:
            entry["greedy"][dataset] = _greedy_exact(by_key.get(("greedy", dataset, variant)))
        beats = list(entry["beats_hf"].values())
        exact = [item["ok"] for item in entry["greedy"].values()]
        entry["check_2_beats_compiled_hf_every_set_both_modes"] = _verdict(
            None if any(v is None for v in beats) else all(beats)
        )
        entry["check_3_greedy_matches_compiled_hf_or_exact_tie"] = _verdict(
            None if any(v is None for v in exact) else all(exact)
        )
        variants[variant] = entry

    mixed = {variant: variants[variant]["tps"]["greedy/mixed"] for variant in VARIANTS}
    decision: dict[str, Any] = {"mixed_greedy_tps": mixed}
    if any(value is None for value in mixed.values()):
        decision.update({"pick": None, "default": None, "verdict": "n/a"})
    else:
        pick = max(VARIANTS, key=lambda name: float(mixed[name]))  # type: ignore[arg-type]
        others = [name for name in VARIANTS if name != pick]
        decision["pick"] = pick
        decision["pick_over_other_mixed_greedy"] = {
            other: float(mixed[pick]) / float(mixed[other])  # type: ignore[arg-type]
            for other in others
            if mixed[other]
        }
        checks = (
            variants[pick]["check_2_beats_compiled_hf_every_set_both_modes"],
            variants[pick]["check_3_greedy_matches_compiled_hf_or_exact_tie"],
        )
        if all(check == "pass" for check in checks):
            decision.update({"verdict": "pass", "default": pick.removeprefix("arm_")})
        else:
            decision.update({"verdict": "n/a" if "n/a" in checks else "FAIL", "default": None})
    commits = sorted({str(doc.get("git_commit")) for doc in documents})
    return {"commits": commits, "variants": variants, "decision": decision}


def render(result: dict[str, Any]) -> str:
    """Markdown view of :func:`evaluate`'s output."""
    decision = result["decision"]
    lines = [f"Composite default report, commits {', '.join(result['commits'])}", ""]
    lines.append(f"Mixed greedy tok/s: {decision['mixed_greedy_tps']}")
    lines.append(f"Pick (higher mixed greedy throughput): {decision.get('pick')}")
    if "pick_over_other_mixed_greedy" in decision:
        lines.append(f"Pick / other, mixed greedy: {decision['pick_over_other_mixed_greedy']}")
    lines.append(f"Verdict: {decision['verdict']}; default: {decision.get('default')}")
    lines.append("")
    for variant, entry in result["variants"].items():
        lines.append(f"## {variant}")
        lines.append("")
        lines.append(
            "| check | verdict |\n|---|---|\n"
            f"| 2. beats compiled HF, every set, both modes "
            f"| {entry['check_2_beats_compiled_hf_every_set_both_modes']} |\n"
            f"| 3. greedy matches compiled HF (fp16 exact ties allowed) "
            f"| {entry['check_3_greedy_matches_compiled_hf_or_exact_tie']} |"
        )
        lines.append("")
        lines.append("| mode/set | tok/s | / compiled HF |")
        lines.append("|---|---|---|")
        for key, value in entry["tps"].items():
            lines.append(f"| {key} | {value} | {entry['vs_hf'].get(key)} |")
        lines.append("")
        lines.append("| greedy block | matched | exact ties | ok |")
        lines.append("|---|---|---|---|")
        for dataset, item in entry["greedy"].items():
            lines.append(
                f"| {dataset} | {item.get('matched')} | {item.get('exact_ties')} "
                f"| {_verdict(item.get('ok'))} |"
            )
        for dataset, item in entry["greedy"].items():
            for div in item.get("divergences", []):
                lines.append(f"{dataset}: {div}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the report; with ``--json``, also write the evaluated dict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    documents = load_documents(args.results)
    if not documents:
        print(f"No runner result JSON in {args.results}.")
        return 1
    result = evaluate(documents)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
