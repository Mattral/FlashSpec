"""Draw the README throughput chart from one bundle's result JSONs.

    python scripts/readme_chart.py --commit 3c43e08

Reads ``benchmarks/results/gpu/<commit>/results/*.json`` and plots, per
dataset and mode, ``metrics.tokens_per_second_median`` of compiled FlashSpec
(``flashspec_compiled``), compiled HuggingFace ``generate``
(``hf_*_compiled``) and eager HuggingFace ``generate`` (``hf_greedy`` /
``hf_sample``): the values in the README status table, from the same field.
Nothing is typed by hand and no value is printed on the chart; the exact
values are in the table above it in the README.

Writes ``docs/images/throughput_<commit>.png``. The README references it by
an absolute ``raw.githubusercontent.com`` URL, so PyPI can show it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["SERIES", "chart_rows", "main", "render"]

ROOT = Path(__file__).resolve().parents[1]

#: (label, method for greedy, method for T=0.7, colour): the reference
#: categorical palette's first three slots, in fixed order (validated light:
#: CVD and normal-vision separation pass; slot 3 needs the table as relief).
SERIES: tuple[tuple[str, str, str, str], ...] = (
    ("FlashSpec, compiled", "flashspec_compiled", "flashspec_compiled", "#2a78d6"),
    ("HF generate, compiled", "hf_greedy_compiled", "hf_sample_compiled", "#eb6834"),
    ("HF generate, eager", "hf_greedy", "hf_sample", "#1baf7a"),
)
_DATASETS = (("humaneval", "HumanEval"), ("mt_bench", "MT-Bench"))
_MODES = (("greedy", "greedy"), ("sample", "T=0.7"))
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_2 = "#52514e"
_GRID = "#e4e3df"


def chart_rows(results: Path) -> list[tuple[str, list[float]]]:
    """``[(row label, [tok/s per series])]`` in plot order, from the JSONs only."""
    docs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(results.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        mode = (doc.get("decoding") or {}).get("mode")
        docs[(str(doc.get("method")), str(doc.get("dataset")), str(mode))] = doc
    rows: list[tuple[str, list[float]]] = []
    for dataset, dataset_label in _DATASETS:
        for mode, mode_label in _MODES:
            values = []
            for _label, greedy_method, sample_method, _colour in SERIES:
                method = greedy_method if mode == "greedy" else sample_method
                doc = docs[(method, dataset, mode)]
                values.append(float(doc["metrics"]["tokens_per_second_median"]))
            rows.append((f"{dataset_label}, {mode_label}", values))
    return rows


def _bar(
    axis: Any,  # noqa: ANN401
    y: float,
    width: float,
    height: float,
    colour: str,
    radius_x: float,
    aspect: float,
) -> None:
    """Draw a horizontal bar, square at the baseline, rounded at the data end.

    ``radius_x`` is the corner radius in x units; ``aspect`` (x units per y
    unit, in pixels) keeps the corner circular on screen.
    """
    from matplotlib.patches import FancyBboxPatch

    # One patch, started left of the axis: the axes clip its left rounding,
    # so the baseline end is square and there is no seam between two patches.
    start = -3 * radius_x
    axis.add_patch(
        FancyBboxPatch(
            (start, y - height / 2),
            width - start,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius_x}",
            linewidth=0,
            facecolor=colour,
            mutation_aspect=1 / aspect,
        )
    )


def render(rows: list[tuple[str, list[float]]], out: Path) -> None:
    """Draw the grouped horizontal bar chart to ``out`` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_series = len(SERIES)
    bar = 0.22  # bar thickness in row units; the rest of the band is air
    gap = 0.02  # surface gap between adjacent bars
    figure, axis = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    figure.patch.set_facecolor(_SURFACE)
    axis.set_facecolor(_SURFACE)
    top = max(value for _label, values in rows for value in values)
    axis.set_xlim(0, top * 1.05)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([label for label, _values in reversed(rows)], color=_TEXT, fontsize=10)
    axis.tick_params(axis="x", colors=_TEXT_2, labelsize=9, length=0)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("tokens per second (median over 10 prompts)", color=_TEXT_2, fontsize=9)
    axis.grid(axis="x", color=_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(_GRID)
    axis.set_title(
        "Qwen2.5-1.5B-Instruct verified by Qwen2.5-0.5B-Instruct, Tesla T4, fp16, batch 1",
        color=_TEXT,
        fontsize=10.5,
        loc="left",
        pad=26,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colour, linewidth=0)
        for _name, _g, _s, colour in SERIES
    ]
    axis.legend(
        handles,
        [name for name, _g, _s, _c in SERIES],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=n_series,
        frameon=False,
        fontsize=9,
        labelcolor=_TEXT,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    figure.tight_layout()
    # Bars are drawn after layout, so the corner radius can be set in pixels.
    origin = axis.transData.transform((0, 0))
    px_x = axis.transData.transform((1, 0))[0] - origin[0]  # pixels per x unit
    px_y = axis.transData.transform((0, 1))[1] - origin[1]  # pixels per y unit
    radius_px = 6.0  # about 4px when the 1200px image is shown at GitHub's width
    for row, (_label, values) in enumerate(rows):
        centre = len(rows) - 1 - row
        offsets = [(index - (n_series - 1) / 2) * (bar + gap) for index in range(n_series)]
        for (_name, _g, _s, colour), value, offset in zip(SERIES, values, offsets, strict=True):
            _bar(axis, centre - offset, value, bar, colour, radius_px / px_x, px_y / px_x)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=_SURFACE, metadata={"Software": None})
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    """Draw the chart for ``--commit``'s bundle."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    results = ROOT / "benchmarks" / "results" / "gpu" / args.commit / "results"
    out = ROOT / "docs" / "images" / f"throughput_{args.commit}.png"
    render(chart_rows(results), out)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
