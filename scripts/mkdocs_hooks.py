"""MkDocs hooks (``hooks:`` in ``mkdocs.yml``).

``llms.txt`` lives at the repository root, where agents look for it on
GitHub; this hook also publishes it at the root of the built docs site, so
``https://flashspec.readthedocs.io/en/latest/llms.txt`` serves the same file.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]


def on_post_build(config: Any, **_: Any) -> None:  # noqa: ANN401
    """Copy the repository's ``llms.txt`` into the built site."""
    shutil.copyfile(_ROOT / "llms.txt", Path(config["site_dir"]) / "llms.txt")
