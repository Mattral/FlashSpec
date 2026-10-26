"""docs/lessons.md figures: the generated SVGs are exactly what the scripts produce from the JSONs.

Figures 4-9 are drawn by ``scripts/article/`` from committed result JSONs. If a
script, a JSON or an SVG changes without the others, this test fails. Figures
1-3 have no generator and are checked only for presence.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images" / "lessons"
LESSONS = ROOT / "docs" / "lessons.md"


@pytest.mark.parametrize("module", ["make_phase6_figures", "make_030_figures"])
def test_scripts_reproduce_the_committed_svgs(module: str) -> None:
    """Each generated figure is byte-for-byte the committed file."""
    import importlib

    scripts = importlib.import_module(f"scripts.article.{module}")
    figures = scripts.figures()
    assert figures
    for name, content in figures.items():
        assert (IMAGES / name).read_text(encoding="utf-8") == content, name


def test_lessons_page_references_existing_images_by_absolute_url() -> None:
    """Every image in lessons.md is an absolute GitHub URL to a committed SVG."""
    text = LESSONS.read_text(encoding="utf-8")
    urls = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(urls) == 9
    base = "https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/"
    for url in urls:
        assert url.startswith(base), url
        assert (IMAGES / url.removeprefix(base)).exists(), url
