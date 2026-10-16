"""Run the Triton GPU parity tests and write a log to download.

Colab and Drive do not update this git checkout. After the script
finishes, download ``colab_triton_tests.log`` and bring that file back.
Do not paste numbers from the log into the README until that file is
committed.

CI does not run these tests. The GPU workflow is ``workflow_dispatch``
only and this repository has no runner labelled for it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LOG_NAME = "colab_triton_tests.log"
PYTEST_ARGS = [
    sys.executable,
    "-m",
    "pytest",
    "tests/unit/test_verify_kernel.py",
    "-m",
    "gpu",
    "-q",
    "--tb=short",
    "-p",
    "no:cacheprovider",
    "-o",
    "addopts=",
]


def environment_lines() -> list[str]:
    """Describe torch and CUDA without failing when torch is missing.

    Returns
    -------
    list[str]
        Lines for the log header.
    """
    lines = [f"python: {sys.version.split()[0]}"]
    try:
        import torch
    except ImportError as exc:
        lines.append(f"torch: not importable ({exc})")
        lines.append("cuda: unavailable")
        return lines
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    return lines


def main() -> int:
    """Run the GPU tests and write the log in the working directory.

    Returns
    -------
    int
        Pytest's exit code, or 2 when pytest cannot start.
    """
    header = [
        "Triton GPU parity tests.",
        "CI does not run these tests. This repository has no GPU runner.",
        "Download this log. Colab does not write the git repository.",
        "",
        *environment_lines(),
        "",
        "command: " + " ".join(PYTEST_ARGS[1:]),
        "",
    ]
    try:
        completed = subprocess.run(
            PYTEST_ARGS,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        text = "\n".join([*header, f"pytest failed to start: {exc}", ""])
        path = Path(LOG_NAME)
        path.write_text(text, encoding="utf-8")
        print(text)
        print(f"DOWNLOAD: {path.resolve()}")
        return 2
    text = "\n".join(
        [
            *header,
            f"exit_code: {completed.returncode}",
            "",
            completed.stdout,
            completed.stderr,
        ]
    )
    path = Path(LOG_NAME)
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"DOWNLOAD: {path.resolve()}")
    print("Colab does not commit this file. Bring the log back to the repo.")
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
