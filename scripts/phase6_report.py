"""Phase 6 exit criteria, computed from the result JSONs of ``benchmarks/phase6.py``.

    python scripts/phase6_report.py benchmarks/results/gpu/<commit>/results [--json out.json]

Criteria (PLAN Phase 6, fixed in ``benchmarks/configs/qwen_t4_phase6.yaml``
before the run), per decoding mode. "Throughput" is a block's workload
throughput, ``metrics.workload_tokens_per_second`` (total tokens / total
seconds over the block's prompts). The primary bandit is ``bandit_ucb1``
(the engine's default selector); ``bandit_thompson`` is reported next to it
and does not decide a criterion.

1. mixed block: the bandit's throughput > the best fixed arm's.
2. each single-set block: the bandit's throughput >= 0.95 x that set's best
   fixed arm.
3. each single-set block: the bandit's throughput > compiled HuggingFace
   (``hf_greedy_compiled`` greedy, ``hf_sample_compiled`` sampled).
4. correctness: reported here as the greedy token matches against compiled
   HuggingFace; the test suite is the other half (CI).

Every number printed is copied from a JSON or is a ratio of two copied
numbers; nothing is rounded. Missing inputs make a criterion ``n/a``,
never a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["evaluate", "load_documents", "main", "render"]

PRIMARY = "bandit_ucb1"
SECONDARY = "bandit_thompson"
TOLERANCE = 0.95
SINGLE_SETS = ("humaneval", "mt_bench", "humaneval_edit")


def load_documents(directory: Path) -> list[dict[str, Any]]:
    """Load the Phase 6 documents (``"phase6"`` key) in ``directory``."""
    paths = sorted(directory.glob("*.json"))
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
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


def _best_fixed(docs: dict[str, dict[str, Any]]) -> tuple[str | None, float | None]:
    fixed = [(name, _tps(doc)) for name, doc in docs.items() if name.startswith("arm_")]
    fixed = [(name, value) for name, value in fixed if value is not None]
    if not fixed:
        return None, None
    return max(fixed, key=lambda item: item[1])


def _verdict(ok: bool | None) -> str:
    return "n/a" if ok is None else ("pass" if ok else "FAIL")


def evaluate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Criteria, per mode, with the inputs they were computed from."""
    blocks: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for doc in documents:
        key = (_mode(doc), str(doc.get("dataset")))
        blocks.setdefault(key, {})[str(doc.get("method"))] = doc
    modes = sorted({mode for mode, _dataset in blocks})
    out: dict[str, Any] = {"commits": sorted({str(d.get("git_commit")) for d in documents})}
    for mode in modes:
        hf = "hf_greedy_compiled" if mode == "greedy" else "hf_sample_compiled"
        section: dict[str, Any] = {"blocks": {}, "criteria": {}}
        for (block_mode, dataset), docs in sorted(blocks.items()):
            if block_mode != mode:
                continue
            best_name, best_tps = _best_fixed(docs)
            row: dict[str, Any] = {
                "best_fixed_arm": best_name,
                "best_fixed_tps": best_tps,
                "fixed_arms": {
                    name: _tps(doc) for name, doc in sorted(docs.items()) if name.startswith("arm_")
                },
            }
            for bandit in (PRIMARY, SECONDARY):
                value = _tps(docs.get(bandit))
                row[bandit] = {
                    "tps": value,
                    "vs_best_fixed": value / best_tps if value is not None and best_tps else None,
                    "vs_" + hf: (
                        value / _tps(docs.get(hf))  # type: ignore[operator]
                        if value is not None and _tps(docs.get(hf))
                        else None
                    ),
                }
            row[hf] = _tps(docs.get(hf))
            row["greedy_token_match_compiled"] = {
                name: _fraction(doc.get("greedy_token_match_compiled"))
                for name, doc in sorted(docs.items())
                if "arms" in doc
            }
            report = (docs.get(PRIMARY) or {}).get("arm_report") or {}
            row["primary_bandit_arms"] = [
                {
                    key: arm.get(key)
                    for key in (
                        "label",
                        "pulls",
                        "share",
                        "acceptance",
                        "tokens_per_step",
                        "tokens_per_second",
                        "mean_reward",
                    )
                }
                for arm in report.get("per_arm", [])
            ]
            section["blocks"][dataset] = row
        rows = section["blocks"]
        mixed = rows.get("mixed", {}).get(PRIMARY, {})
        ratio = mixed.get("vs_best_fixed")
        section["criteria"]["1_mixed_beats_best_fixed"] = _verdict(
            None if ratio is None else ratio > 1.0
        )
        within: list[bool | None] = []
        faster: list[bool | None] = []
        for dataset in SINGLE_SETS:
            primary = rows.get(dataset, {}).get(PRIMARY, {})
            r_fixed = primary.get("vs_best_fixed")
            r_hf = primary.get("vs_" + hf)
            within.append(None if r_fixed is None else r_fixed >= TOLERANCE)
            faster.append(None if r_hf is None else r_hf > 1.0)
        section["criteria"]["2_each_set_within_5pct_of_best_fixed"] = _all(within)
        section["criteria"]["3_each_set_faster_than_" + hf] = _all(faster)
        matches = [
            value
            for row in rows.values()
            for value in row["greedy_token_match_compiled"].values()
            if mode == "greedy"
        ]
        section["criteria"]["4_greedy_matches_compiled_hf"] = (
            "n/a"
            if not matches or "n/a" in matches
            else _verdict(all(a == b for a, b in (m.split("/") for m in matches)))
        )
        out[mode] = section
    out["cuda_graphs"] = _graphs(documents)
    return out


def _all(values: list[bool | None]) -> str:
    if any(value is None for value in values):
        return "n/a"
    return _verdict(all(bool(value) for value in values))


def _fraction(match: object) -> str:
    if not isinstance(match, dict):
        return "n/a"
    return f"{match.get('prompts_matched')}/{match.get('n_prompts')}"


def _graphs(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect CUDA-graph captures per wrapper group, summed over sessions.

    ``group_captures_in_process`` counts one process. A bundle assembled
    across sessions has one process per session, so each session's largest
    count is taken and the sessions are added.
    """
    groups: dict[str, dict[str, Any]] = {}
    for doc in documents:
        graphs = doc.get("cuda_graphs")
        if not isinstance(graphs, dict) or "group" not in graphs:
            continue
        entry = groups.setdefault(
            str(graphs["group"]),
            {
                "methods": graphs.get("group_methods"),
                "fixed_shapes": (doc.get("compile") or {}).get("fixed_shapes"),
                "available": graphs.get("available"),
                "captures": 0,
                "by_session": {},
                "call_shapes": None,
            },
        )
        session = str((doc.get("phase6") or {}).get("session"))
        total = graphs.get("group_captures_in_process")
        if isinstance(total, int) and total >= entry["by_session"].get(session, -1):
            entry["by_session"][session] = total
            entry["call_shapes"] = graphs.get("call_shapes")
    for entry in groups.values():
        entry["captures"] = sum(entry["by_session"].values())
    return groups


def render(result: dict[str, Any]) -> str:
    """Markdown view of :func:`evaluate`'s output."""
    lines = [f"Phase 6 report, commits {', '.join(result['commits'])}", ""]
    for mode, section in result.items():
        if mode in ("commits", "cuda_graphs"):
            continue
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("| criterion | verdict |")
        lines.append("|---|---|")
        for name, verdict in section["criteria"].items():
            lines.append(f"| {name} | {verdict} |")
        lines.append("")
        hf = "hf_greedy_compiled" if mode == "greedy" else "hf_sample_compiled"
        lines.append(
            f"| block | best fixed arm | its tok/s | {PRIMARY} tok/s | / best fixed "
            f"| / {hf} | {SECONDARY} tok/s | / best fixed | {hf} tok/s |"
        )
        lines.append("|" + "---|" * 9)
        for dataset, row in section["blocks"].items():
            p, s = row[PRIMARY], row[SECONDARY]
            lines.append(
                f"| {dataset} | {row['best_fixed_arm']} | {row['best_fixed_tps']} | {p['tps']} "
                f"| {p['vs_best_fixed']} | {p['vs_' + hf]} | {s['tps']} | {s['vs_best_fixed']} "
                f"| {row[hf]} |"
            )
        lines.append("")
        for dataset, row in section["blocks"].items():
            lines.append(f"{dataset}: fixed arms {row['fixed_arms']}")
            lines.append(f"{dataset}: {PRIMARY} arms {row['primary_bandit_arms']}")
            matches = row["greedy_token_match_compiled"]
            lines.append(f"{dataset}: greedy match vs compiled HF {matches}")
        lines.append("")
    lines.append("## CUDA-graph captures per wrapper group")
    lines.append("")
    lines.append("| group | fixed shapes | captures | methods | call shapes |")
    lines.append("|---|---|---|---|---|")
    for name, entry in sorted(result["cuda_graphs"].items()):
        lines.append(
            f"| {name} | {entry['fixed_shapes']} | {entry['captures']} "
            f"(counter available: {entry['available']}) | {entry['methods']} "
            f"| {entry['call_shapes']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Print the report; with ``--json``, also write the evaluated dict."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    documents = load_documents(args.results)
    if not documents:
        print(f"No phase6 result JSON in {args.results}.")
        return 1
    result = evaluate(documents)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
