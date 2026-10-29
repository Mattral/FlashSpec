"""Draw the 0.3.0 figures of docs/lessons.md (4 and 9) from FlashSpec's result JSONs.

    python scripts/article/make_030_figures.py [--out DIR]

Writes to docs/images/lessons/ by default. Reads only committed result files
in this repository:
- benchmarks/results/gpu/3c43e08/results/*.json  (0.3.0 release run, Tesla T4)
- benchmarks/results/gpu/3e38251/results/*.json  (Phase 6 run, Tesla T4, every arm alone)

Figure 4 (baselines) is redrawn from the 0.3.0 release run; figure 9
(acceptance vs throughput) uses the Phase 6 arms, which all ran in one
session. Visible labels are rounded for display; each mark's tooltip
(<title>) has the stored value. Colours and helpers are those of
make_phase6_figures.py (dataviz reference palette, light surface).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.arm_report import pooled_acceptance  # noqa: E402
from scripts.article.make_phase6_figures import (  # noqa: E402
    GRID,
    INK,
    INK2,
    OUT,
    REPO,
    S1,
    S2,
    S3,
    legend,
    svg,
    text,
)

RELEASE = REPO / "benchmarks/results/gpu/3c43e08/results"
PHASE6 = REPO / "benchmarks/results/gpu/3e38251/results"


def _docs(folder: Path) -> dict[tuple[str, str, str], dict]:
    """Load a bundle's documents by (method, dataset, mode)."""
    out = {}
    for path in sorted(folder.glob("*.json")):
        doc = json.loads(path.read_text())
        out[(doc["method"], doc["dataset"], doc["decoding"]["mode"])] = doc
    return out


# ── Figure 4: one run, three headlines (0.3.0 release run) ──────────────────────


def fig4() -> str:
    """Figure 4: one run against three baselines (0.3.0 release run)."""
    docs = _docs(RELEASE)
    width, height = 720, 360
    left, right, top, bottom = 56, 696, 96, 308
    cells = [
        ("humaneval", "greedy", "HumanEval, greedy"),
        ("humaneval", "sample", "HumanEval, T=0.7"),
        ("mt_bench", "greedy", "MT-Bench, greedy"),
        ("mt_bench", "sample", "MT-Bench, T=0.7"),
    ]
    series = [
        ("plain", S1, "vs eager HF generate"),
        ("hf_assisted", S2, "vs HF assisted generation"),
        ("plain_compiled", S3, "vs compiled HF generate (the fair one)"),
    ]
    ymax = 5.0

    def y(v: float) -> float:
        return bottom - (bottom - top) * v / ymax

    body = [
        text(left, 28, "One run, three headlines", 16, INK, weight=600),
        text(
            left,
            48,
            "Compiled FlashSpec speedup (median of per-prompt ratios) against three "
            "baselines, same T4 run (3c43e08, 0.3.0 defaults)",
        ),
    ]
    body += legend([(c, label) for _k, c, label in series], left, 63)
    for v in (0, 1, 2, 3, 4, 5):
        body.append(
            f"<line x1='{left}' x2='{right}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>"
        )
        body.append(text(left - 8, y(v) + 4, f"{v:g}×", 11, anchor="end"))
    body.append(
        f"<line x1='{left}' x2='{right}' y1='{y(1):.1f}' y2='{y(1):.1f}' stroke='{INK2}' "
        "stroke-dasharray='4 4'/>"
    )
    group = (right - left) / len(cells)
    bar = 30
    for i, (dataset, mode, label) in enumerate(cells):
        metrics = docs[("flashspec_compiled", dataset, mode)]["metrics"]
        plain = "hf_greedy" if mode == "greedy" else "hf_sample"
        x0 = left + i * group + (group - len(series) * (bar + 4)) / 2
        for j, (key, color, name) in enumerate(series):
            baseline = key.replace("plain", plain)
            value = float(metrics[f"speedup_vs_{baseline}_median"])
            x = x0 + j * (bar + 4)
            body.append(
                f"<rect x='{x:.1f}' y='{y(value):.1f}' width='{bar}' "
                f"height='{bottom - y(value):.1f}' rx='3' fill='{color}'><title>{escape(label)}, "
                f"{name}: {value!r}</title></rect>"
            )
            body.append(text(x + bar / 2, y(value) - 5, f"{value:.2f}", 10.5, INK, "middle", 600))
        body.append(text(left + i * group + group / 2, bottom + 20, label, 12, INK, "middle"))
    return svg(width, height, "Compiled FlashSpec speedup against three baselines (0.3.0)", body)


# ── Figure 9: acceptance is not the objective (Phase 6 arms, code edits) ────────


def fig9() -> str:
    """Figure 9: acceptance against throughput per arm on code edits."""
    docs = _docs(PHASE6)
    dataset = "humaneval_edit"
    arms = sorted(
        (m for (m, d, mode) in docs if d == dataset and mode == "greedy" and m.startswith("arm_")),
    )
    hf = float(
        docs[("hf_greedy_compiled", dataset, "greedy")]["metrics"]["workload_tokens_per_second"]
    )
    width, height = 720, 400
    left, right, top, bottom = 64, 690, 92, 340
    ymax = 180.0

    def x(v: float) -> float:
        return left + (right - left) * v

    def y(v: float) -> float:
        return bottom - (bottom - top) * v / ymax

    def colour(name: str) -> str:
        return S2 if "composite" in name else (S3 if "ngram" in name else S1)

    body = [
        text(left, 28, "Higher acceptance did not mean faster", 16, INK, weight=600),
        text(
            left,
            48,
            "Each arm alone on the code-edit set, greedy, one T4 session (3e38251)",
        ),
    ]
    body += legend(
        [(S1, "Qwen2.5-0.5B drafter"), (S2, "composite"), (S3, "n-gram lookup")], left, 63
    )
    for v in (0, 30, 60, 90, 120, 150, 180):
        body.append(
            f"<line x1='{left}' x2='{right}' y1='{y(v):.1f}' y2='{y(v):.1f}' stroke='{GRID}'/>"
        )
        body.append(text(left - 8, y(v) + 4, f"{v:g}", 11, anchor="end"))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(text(x(v), bottom + 18, f"{v:g}", 11, anchor="middle"))
    body.append(
        text(
            (left + right) / 2,
            bottom + 38,
            "acceptance (accepted / proposed draft tokens)",
            12,
            INK2,
            "middle",
        )
    )
    body.append(
        f"<text x='18' y='{(top + bottom) / 2:.1f}' font-size='12' fill='{INK2}' "
        f"text-anchor='middle' transform='rotate(-90 18 {(top + bottom) / 2:.1f})'>"
        "tokens per second</text>"
    )
    body.append(
        f"<line x1='{left}' x2='{right}' y1='{y(hf):.1f}' y2='{y(hf):.1f}' stroke='{INK2}' "
        "stroke-dasharray='4 4'/>"
    )
    body.append(text(left + 6, y(hf) - 6, "compiled HF generate", 11, INK2, "start"))
    # Label placement per arm (dx, dy, anchor), chosen after looking at the render.
    place = {
        "qwen-g2": (-4, 22, "end"),
        "qwen-g4": (0, -12, "middle"),
        "qwen-g6": (-10, -10, "end"),
        "ngram-g2": (10, 4, "start"),
        "ngram-g4": (-10, 4, "end"),
        "ngram-g6": (10, -6, "start"),
        "composite-g4": (10, 4, "start"),
    }
    for name in arms:
        doc = docs[(name, dataset, "greedy")]
        metrics = doc["metrics"]
        # From the per-step counts; acceptance_rate_mean in these files is a
        # running mean over earlier calls (fixed after 0.3.1).
        acc = pooled_acceptance(doc["arm_report"])
        assert acc is not None, name
        tps = float(metrics["workload_tokens_per_second"])
        cx, cy = x(acc), y(tps)
        body.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='6' fill='{colour(name)}' stroke='#fcfcfb' "
            f"stroke-width='2'><title>{escape(name.removeprefix('arm_'))}: acceptance "
            f"{acc!r}, {tps!r} tok/s</title></circle>"
        )
        short = name.removeprefix("arm_")
        dx, dy, anchor = place.get(short, (10, 4, "start"))
        body.append(text(cx + dx, cy + dy, short, 11, INK, anchor))
    return svg(width, height, "Acceptance vs throughput per arm on code edits (Phase 6)", body)


def figures() -> dict[str, str]:
    """Return figure file name -> SVG text for figures 4 and 9."""
    return {"fig4-baselines.svg": fig4(), "fig9-acceptance-vs-speed.svg": fig9()}


def main(argv: list[str] | None = None) -> int:
    """Write figures 4 and 9 to ``--out`` (default docs/images/lessons/)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    for name, content in figures().items():
        (args.out / name).write_text(content, encoding="utf-8")
        print("wrote", args.out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
