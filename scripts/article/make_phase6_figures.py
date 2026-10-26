"""Draw the Phase 6 figures of docs/lessons.md (5-8) from FlashSpec's result JSONs.

    python scripts/article/make_phase6_figures.py [--out DIR]

Writes to docs/images/lessons/ by default. Reads only committed result files
in this repository:
- benchmarks/results/gpu/3e38251/results/*.json        (Phase 6 bundle, Tesla T4)
- benchmarks/results/diagnostics/width_gpu_t4_125d1a1.json (width-7 diagnostic, Tesla T4)
and computes the analysis with the repo's own scripts/phase6_analysis.py.
Visible labels are rounded for display; each mark's tooltip (<title>) has the
stored value. Colours are the dataviz reference palette's categorical slots
1-4 on its light surface, the same as figures 1-4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "images" / "lessons"
sys.path.insert(0, str(REPO))

from scripts.phase6_analysis import analyse  # noqa: E402

RESULTS = REPO / "benchmarks/results/gpu/3e38251/results"
WIDTH = REPO / "benchmarks/results/diagnostics/width_gpu_t4_125d1a1.json"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def text(
    x: float,
    y: float,
    s: str,
    size: float = 12,
    fill: str = INK2,
    anchor: str = "start",
    weight: int = 400,
) -> str:
    """Return an SVG <text> element."""
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' "
        f"text-anchor='{anchor}' font-weight='{weight}'>{escape(s)}</text>"
    )


def svg(width: int, height: int, title: str, body: list[str]) -> str:
    """Wrap SVG elements in a titled figure on the chart surface."""
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' "
        f"height='{height}' font-family='{FONT}'><title>{escape(title)}</title>"
        f"<rect width='{width}' height='{height}' fill='{SURFACE}' rx='8'/>"
        + "".join(body)
        + "</svg>"
    )


def legend(items: list[tuple[str, str]], x: float, y: float) -> list[str]:
    """Return a row of colour swatches with labels."""
    out = []
    for color, label in items:
        out.append(f"<rect x='{x:.1f}' y='{y}' width='12' height='12' rx='3' fill='{color}'/>")
        out.append(text(x + 18, y + 10, label))
        x += 18 + 7.0 * len(label) + 22
    return out


def load_docs() -> dict[tuple[str, str], dict[str, dict]]:
    """Load the Phase 6 documents by (mode, prompt set) and method."""
    blocks: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        doc = json.loads(path.read_text())
        if "phase6" not in doc:
            continue
        mode = "greedy" if doc["decoding"]["mode"] == "greedy" else "t0.7"
        blocks.setdefault((mode, doc["dataset"]), {})[doc["method"]] = doc
    return blocks


def tps(doc: dict) -> float:
    """Return a document's workload tokens per second."""
    return float(doc["metrics"]["workload_tokens_per_second"])


# ── Figure 5: bandit vs best fixed arm, per block ───────────────────────────────


def fig5(blocks: dict) -> str:
    """Figure 5: the bandit against each set's best fixed arm."""
    width, height = 760, 380
    left, right, top, bottom = 64, 744, 96, 318
    sets = [
        ("mixed", "Mixed (30 prompts)"),
        ("humaneval", "HumanEval"),
        ("mt_bench", "MT-Bench"),
        ("humaneval_edit", "Code edits"),
    ]
    series = [
        ("bandit_ucb1", S1, "UCB1 bandit"),
        ("bandit_thompson", S2, "Thompson"),
        ("hf_greedy_compiled", S3, "compiled HF"),
    ]
    ymax = 1.1

    def y(v: float) -> float:
        return bottom - (bottom - top) * v / ymax

    body = [
        text(left, 28, "The bandit never beat the best single arm", 16, INK, weight=600),
        text(
            left,
            48,
            "Throughput relative to that set's best fixed arm (= 1.0), greedy, "
            "Qwen2.5-1.5B + 0.5B on a T4, bundle 3e38251",
        ),
    ]
    body += legend([(c, name) for _k, c, name in series], left, 63)
    body.append(
        f"<line x1='{right - 250}' x2='{right - 226}' y1='69' y2='69' stroke='{S4}' "
        "stroke-dasharray='2 3' stroke-width='2'/>"
    )
    body.append(text(right - 220, 73, "0.95: the per-set bar (criterion 2)", 12))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(
            f"<line x1='{left}' x2='{right}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>"
        )
        body.append(text(left - 8, y(v) + 4, f"{v:g}", 11, anchor="end"))
    body.append(
        f"<line x1='{left}' x2='{right}' y1='{y(1.0):.1f}' y2='{y(1.0):.1f}' stroke='{INK2}' "
        "stroke-dasharray='4 4'/>"
    )
    body.append(
        f"<line x1='{left}' x2='{right}' y1='{y(0.95):.1f}' y2='{y(0.95):.1f}' stroke='{S4}' "
        "stroke-dasharray='2 3'/>"
    )
    group = (right - left) / len(sets)
    bar = 30
    for i, (key, label) in enumerate(sets):
        docs = blocks[("greedy", key)]
        best_name = max((n for n in docs if n.startswith("arm_")), key=lambda n: tps(docs[n]))
        best = tps(docs[best_name])
        present = [(m, c, name) for m, c, name in series if m in docs]
        x0 = left + i * group + (group - len(present) * (bar + 4)) / 2
        for j, (method, color, name) in enumerate(present):
            v = tps(docs[method]) / best
            x = x0 + j * (bar + 4)
            body.append(
                f"<rect x='{x:.1f}' y='{y(v):.1f}' width='{bar}' height='{bottom - y(v):.1f}' "
                f"rx='3' fill='{color}'><title>{escape(label)}, {name} / "
                f"{best_name.removeprefix('arm_')}: {v!r}</title></rect>"
            )
            body.append(text(x + bar / 2, y(v) - 5, f"{v:.2f}", 10.5, INK, "middle", 600))
        body.append(text(left + i * group + group / 2, bottom + 20, label, 12, INK, "middle"))
        body.append(
            text(
                left + i * group + group / 2,
                bottom + 36,
                f"best: {best_name.removeprefix('arm_')}",
                11,
                INK2,
                "middle",
            )
        )
    return svg(width, height, "Phase 6 bandit vs the best fixed arm per prompt set", body)


# ── Figure 6: why the bandit lost (mixed, greedy, step throughput) ──────────────


def fig6(analysis: dict) -> str:
    """Figure 6: why the bandit lost (counterfactuals and oracles)."""
    mixed = analysis["greedy"]["mixed"]
    oracle = mixed["prompt_oracles"]
    best_single = oracle["single_arm_step_tps"][oracle["best_single_arm"]]
    rows = [
        ("UCB1, as measured", mixed["bandit_ucb1"]["step_tps_measured"] / best_single, S1),
        (
            "UCB1, every arm at its own width*",
            mixed["bandit_ucb1"]["step_tps_counterfactual_own_width"] / best_single,
            S2,
        ),
        ("Thompson, as measured", mixed["bandit_thompson"]["step_tps_measured"] / best_single, S1),
        (
            "Thompson, every arm at its own width*",
            mixed["bandit_thompson"]["step_tps_counterfactual_own_width"] / best_single,
            S2,
        ),
        ("Best arm per prompt set*", oracle["oracle_per_set_over_best_single"], S3),
        ("Best arm per prompt*", oracle["oracle_per_prompt_over_best_single"], S3),
    ]
    width, height = 760, 318
    left, right, top = 270, 720, 92
    row_h = 30
    lo, hi = 0.85, 1.1

    def x(v: float) -> float:
        return left + (right - left) * (v - lo) / (hi - lo)

    body = [
        text(
            24,
            28,
            "Width was not the problem; exploration and missing context were",
            16,
            INK,
            weight=600,
        ),
        text(
            24,
            48,
            "Mixed workload, greedy: step throughput relative to the best single arm "
            f"({oracle['best_single_arm']}). * = arithmetic on measured values",
        ),
    ]
    body += legend(
        [
            (S1, "measured"),
            (S2, "counterfactual: no width-7 or catch-up cost"),
            (S3, "oracle ceiling (perfect choice)"),
        ],
        24,
        62,
    )
    bottom = top + row_h * len(rows)
    for v in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1):
        body.append(
            f"<line x1='{x(v):.1f}' x2='{x(v):.1f}' y1='{top - 6}' y2='{bottom}' stroke='{GRID}'/>"
        )
        body.append(text(x(v), bottom + 16, f"{v:g}", 11, anchor="middle"))
    body.append(
        f"<line x1='{x(1.0):.1f}' x2='{x(1.0):.1f}' y1='{top - 6}' y2='{bottom}' stroke='{INK2}' "
        "stroke-dasharray='4 4'/>"
    )
    for i, (label, v, color) in enumerate(rows):
        yy = top + i * row_h
        body.append(text(left - 10, yy + 15, label, 12, INK, "end"))
        x_end = x(v)
        body.append(
            f"<rect x='{x(lo):.1f}' y='{yy + 3}' width='{x_end - x(lo):.1f}' height='18' rx='3' "
            f"fill='{color}'><title>{escape(label)}: {v!r}</title></rect>"
        )
        body.append(text(x_end + 6, yy + 16, f"{v:.3f}", 11, INK, "start", 600))
    body.append(
        text(
            24,
            bottom + 40,
            "Axis starts at 0.85 to show the gaps; every bar is a ratio to "
            "the same best single arm.",
            11,
        )
    )
    return svg(width, height, "Decomposition of the Phase 6 bandit loss", body)


# ── Figure 7: fp16 near-ties at the width-7 divergences ─────────────────────────


def fig7() -> str:
    """Figure 7: top-2 margins at each fp16 divergence."""
    report = json.loads(WIDTH.read_text())
    cases = []
    for case in report["cases"]:
        if case["dtype"] != "float16":
            continue
        path = case["paths"].get("flashspec_g6_padded_w7", {})
        at = path.get("at_divergence")
        if at:
            cases.append(
                (
                    f"{case['dataset']}/{case['prompt_id']}".replace("humaneval/", ""),
                    at["position"],
                    at["hf_static"]["margin_top1_top2"],
                    at["path"]["margin_top1_top2"],
                    at["max_abs_logit_difference"],
                )
            )
    width, height = 760, 372
    left, right, top, bottom = 64, 744, 96, 296
    ymax = 0.04
    series = [
        (S1, "margin, compiled HF"),
        (S2, "margin, width-7 verify"),
        (S3, "largest logit difference"),
    ]

    def y(v: float) -> float:
        return bottom - (bottom - top) * v / ymax

    body = [
        text(left, 28, "Every fp16 mismatch was a tie, not a bug", 16, INK, weight=600),
        text(
            left,
            48,
            "At each divergence (Qwen2.5-1.5B, T4, fp16) one side's top two logits were "
            "exactly equal. fp32 matched on every path.",
        ),
    ]
    body += legend(series, left, 63)
    for v in (0, 0.01, 0.02, 0.03, 0.04):
        body.append(
            f"<line x1='{left}' x2='{right}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>"
        )
        body.append(text(left - 8, y(v) + 4, f"{v:g}", 11, anchor="end"))
    group = (right - left) / len(cases)
    bar = 34
    for i, (label, position, ref, path, diff) in enumerate(cases):
        x0 = left + i * group + (group - 3 * (bar + 4)) / 2
        for j, (v, (color, name)) in enumerate(zip((ref, path, diff), series, strict=True)):
            xx = x0 + j * (bar + 4)
            if v > 0:
                body.append(
                    f"<rect x='{xx:.1f}' y='{y(v):.1f}' width='{bar}' height='{bottom - y(v):.1f}' "
                    f"rx='3' fill='{color}'><title>{escape(label)} token {position}, {name}: {v!r}"
                    "</title></rect>"
                )
                body.append(text(xx + bar / 2, y(v) - 5, repr(v), 10.5, INK, "middle", 600))
            else:
                body.append(
                    f"<rect x='{xx:.1f}' y='{bottom - 2:.1f}' width='{bar}' height='2' "
                    f"fill='{color}'><title>{escape(label)} token {position}, {name}: {v!r} "
                    "(exact tie)</title></rect>"
                )
                body.append(text(xx + bar / 2, bottom - 8, "0 (tie)", 10.5, INK, "middle", 600))
        body.append(
            text(
                left + i * group + group / 2,
                bottom + 20,
                f"{label}, token {position}",
                12,
                INK,
                "middle",
            )
        )
    body.append(
        text(
            left,
            bottom + 46,
            "Margin = top-1 minus top-2 logit. Unpadded width 7 diverged at "
            "the same tokens; width 5 and unpadded gamma 2 did not.",
            11,
        )
    )
    body.append(
        text(
            left,
            bottom + 62,
            "Pad content never changed a real row (bit-identical, fp32 and fp16).",
            11,
        )
    )
    return svg(width, height, "fp16 top-2 margins at the width-7 greedy divergences", body)


# ── Figure 8: step cost of each arm alone ───────────────────────────────────────


def fig8(analysis: dict) -> str:
    """Figure 8: step cost of each arm alone."""
    arms = analysis["greedy"]["mixed"]["bandit_ucb1"]["arms"]
    order = ["qwen-g2", "qwen-g4", "qwen-g6", "composite-g4", "ngram-g2", "ngram-g4", "ngram-g6"]
    width, height = 760, 316
    left, right, top = 150, 700, 70
    row_h = 30
    vmax = 90.0

    def x(v: float) -> float:
        return left + (right - left) * v / vmax

    body = [
        text(24, 28, "A drafter with no model costs only the verify", 16, INK, weight=600),
        text(
            24,
            48,
            "Mean step time of each arm run alone, mixed workload, greedy (ms); "
            "label: tokens emitted per step",
        ),
    ]
    bottom = top + row_h * len(order)
    for v in (0, 20, 40, 60, 80):
        body.append(
            f"<line x1='{x(v):.1f}' x2='{x(v):.1f}' y1='{top - 6}' y2='{bottom}' stroke='{GRID}'/>"
        )
        body.append(text(x(v), bottom + 16, f"{v} ms", 11, anchor="middle"))
    for i, label in enumerate(order):
        arm = arms[label]
        v = arm["step_ms_alone"]
        yy = top + i * row_h
        color = S1 if label.startswith("qwen") else (S3 if label.startswith("ngram") else S2)
        body.append(text(left - 10, yy + 15, label, 12, INK, "end"))
        body.append(
            f"<rect x='{left}' y='{yy + 3}' width='{x(v) - left:.1f}' height='18' rx='3' "
            f"fill='{color}'><title>{label}: {v!r} ms per step, "
            f"{arm['tokens_per_step_alone']!r} tokens per step</title></rect>"
        )
        body.append(
            text(
                x(v) + 6,
                yy + 16,
                f"{v:.1f} ms · {arm['tokens_per_step_alone']:.2f} tok/step",
                11,
                INK,
                "start",
                600,
            )
        )
    body += legend(
        [(S1, "Qwen2.5-0.5B drafter"), (S2, "composite"), (S3, "n-gram lookup")], 24, bottom + 32
    )
    return svg(width, height, "Step cost of each Phase 6 arm alone", body)


def figures() -> dict[str, str]:
    """Return figure file name -> SVG text for figures 5-8."""
    blocks = load_docs()
    analysis = analyse(RESULTS)
    return {
        "fig5-bandit-vs-arms.svg": fig5(blocks),
        "fig6-why-bandit-lost.svg": fig6(analysis),
        "fig7-fp16-ties.svg": fig7(),
        "fig8-arm-step-cost.svg": fig8(analysis),
    }


def main(argv: list[str] | None = None) -> int:
    """Write figures 5-8 to ``--out`` (default docs/images/lessons/)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    for name, content in figures().items():
        (args.out / name).write_text(content, encoding="utf-8")
        print("wrote", args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
