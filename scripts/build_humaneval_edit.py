"""Build the HumanEval code-editing prompt set and the mixed-workload order.

Run once; the outputs are committed:

    python scripts/build_humaneval_edit.py <path to HumanEval.jsonl.gz>

Input: ``data/HumanEval.jsonl.gz`` from https://github.com/openai/human-eval
(MIT, Copyright (c) OpenAI), checked against the sha256 below.

Output 1, ``benchmarks/prompt_sets/humaneval_edit.json``: 10 copy-heavy
prompts. Each one shows a complete HumanEval function (the task prompt
followed by its canonical solution) and asks for an edit that keeps most
of the code: a Google-style docstring, descriptive variable names, or
type hints on local variables, rotating in that order. The answer
re-states most of the function, which is the case prompt lookup is for.

Selection, fixed before any measurement: tasks in file order whose
canonical solution has at least 5 non-empty lines, skipping the tasks in
``humaneval.json`` so the two sets do not overlap; the first 10.

Output 2, ``benchmarks/prompt_sets/mixed.json``: the order of the mixed
workload, the 30 prompts of ``mt_bench``, ``humaneval`` and
``humaneval_edit`` shuffled once with ``random.Random(0)``. The order is
stored, so it does not depend on the Python version that reads it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

_SOURCE_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
_SOURCE_COMMIT = "b9453e8f1865df0a0057a1383767caa400acfbcf"
_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "prompt_sets"
_N = 10
_MIN_SOLUTION_LINES = 5
_EDITS: tuple[tuple[str, str], ...] = (
    (
        "docstring",
        "Rewrite the docstring of this Python function in Google style, with Args and "
        "Returns sections. Keep the code itself unchanged and return the complete function.",
    ),
    (
        "rename",
        "Rename the variables in this Python function to longer, more descriptive names. "
        "Keep the behaviour unchanged and return the complete function.",
    ),
    (
        "type_hints",
        "Add type hints to every local variable in this Python function. Keep the "
        "behaviour unchanged and return the complete function.",
    ),
)


def build(source: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Return the edit-set payload and the mixed-order payload."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != _SOURCE_SHA256:
        raise SystemExit(f"{source} has sha256 {digest}; expected {_SOURCE_SHA256}.")
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    vendored = {
        item["id"]
        for item in json.loads((_DIR / "humaneval.json").read_text(encoding="utf-8"))["prompts"]
    }
    prompts: list[dict[str, str]] = []
    for row in rows:
        if row["task_id"] in vendored:
            continue
        solution = str(row["canonical_solution"])
        if sum(1 for line in solution.splitlines() if line.strip()) < _MIN_SOLUTION_LINES:
            continue
        kind, instruction = _EDITS[len(prompts) % len(_EDITS)]
        code = str(row["prompt"]).strip("\n") + "\n" + solution.rstrip() + "\n"
        prompts.append(
            {
                "id": f"{row['task_id']}:{kind}",
                "task_id": str(row["task_id"]),
                "edit": kind,
                "text": f"{instruction}\n\n```python\n{code}```",
            }
        )
        if len(prompts) == _N:
            break
    edit_payload: dict[str, object] = {
        "dataset": "humaneval_edit",
        "license": "MIT",
        "copyright": "Copyright (c) OpenAI",
        "source": "https://github.com/openai/human-eval/blob/"
        f"{_SOURCE_COMMIT}/data/HumanEval.jsonl.gz",
        "source_sha256": _SOURCE_SHA256,
        "selection": (
            f"tasks in file order whose canonical_solution has at least {_MIN_SOLUTION_LINES} "
            "non-empty lines, excluding the tasks in humaneval.json; the first "
            f"{_N}. Edits rotate docstring, rename, type_hints."
        ),
        "built_by": "scripts/build_humaneval_edit.py",
        "prompts": prompts,
    }
    order: list[dict[str, str]] = []
    for name in ("mt_bench", "humaneval"):
        payload = json.loads((_DIR / f"{name}.json").read_text(encoding="utf-8"))
        order += [{"dataset": name, "id": str(item["id"])} for item in payload["prompts"]]
    order += [{"dataset": "humaneval_edit", "id": item["id"]} for item in prompts]
    random.Random(0).shuffle(order)
    mixed_payload: dict[str, object] = {
        "dataset": "mixed",
        "description": (
            "mt_bench, humaneval and humaneval_edit interleaved in one fixed shuffled order "
            "(random.Random(0).shuffle, stored here). Licenses are those of the three sets."
        ),
        "built_by": "scripts/build_humaneval_edit.py",
        "order": order,
    }
    return edit_payload, mixed_payload


def main(argv: list[str]) -> int:
    """Write both JSON files next to the other prompt sets."""
    if len(argv) != 1:
        print(__doc__)
        return 2
    edit_payload, mixed_payload = build(Path(argv[0]))
    for name, payload in (("humaneval_edit", edit_payload), ("mixed", mixed_payload)):
        path = _DIR / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
