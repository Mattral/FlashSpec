"""Build the README's "API at a glance" block from the code, so names and defaults cannot drift.

    python scripts/readme_api.py           # rewrite the block in README.md
    python scripts/readme_api.py --check   # exit 1 if README.md's block is stale

Every name, signature, type and default in the block is read from the
installed ``flashspec`` package: ``inspect.signature`` for functions and
classes, ``model_fields`` for the pydantic configs, ``dataclasses.fields``
for result records, and the server's ``argparse`` parser for its flags. The
"What it does" column is the first line of each docstring.
``tests/unit/test_readme.py`` fails when the README block differs from what
this script renders.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import sys
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

__all__ = ["BEGIN", "END", "render", "replace_block"]

BEGIN = "<!-- BEGIN GENERATED: api (scripts/readme_api.py) -->"
END = "<!-- END GENERATED: api -->"
README = Path(__file__).resolve().parents[1] / "README.md"


def _entries() -> list[tuple[str, Any]]:
    """(import path, object), in the order a caller meets them."""
    import flashspec
    from flashspec import engine
    from flashspec.bandit import ThompsonSelector, UCB1Selector

    return [
        ("flashspec.FlashSpecConfig", flashspec.FlashSpecConfig),
        ("flashspec.SamplingConfig", flashspec.SamplingConfig),
        ("flashspec.BanditConfig", flashspec.BanditConfig),
        ("flashspec.SpeculativeEngine", flashspec.SpeculativeEngine),
        ("flashspec.SpeculativeEngine.generate", flashspec.SpeculativeEngine.generate),
        ("flashspec.GenerationResult", flashspec.GenerationResult),
        ("flashspec.engine.TargetModel", engine.TargetModel),
        ("flashspec.engine.HuggingFaceDrafter", engine.HuggingFaceDrafter),
        ("flashspec.engine.load_drafters", engine.load_drafters),
        ("flashspec.engine.NgramDrafter", engine.NgramDrafter),
        ("flashspec.engine.ArmSpec", engine.ArmSpec),
        ("flashspec.bandit.UCB1Selector", UCB1Selector),
        ("flashspec.bandit.ThompsonSelector", ThompsonSelector),
        ("flashspec.engine.DraftModel", engine.DraftModel),
        ("flashspec.register", flashspec.register),
    ]


def _type(annotation: Any) -> str:  # noqa: ANN401
    """Python-syntax type text for an annotation object or string."""
    if isinstance(annotation, str):
        return annotation
    if annotation is type(None):
        return "None"
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return " | ".join(_value(arg) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(_type(arg) for arg in args)
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{', '.join('...' if a is Ellipsis else _type(a) for a in args)}]"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _value(value: Any) -> str:  # noqa: ANN401
    """Default text: strings in double quotes, everything else as repr."""
    if isinstance(value, str):
        return f'"{value}"'
    return repr(value)


def _cell(text: str) -> str:
    """Escape the pipes of a table cell."""
    return text.replace("|", "\\|")


def _summary(obj: Any) -> str:  # noqa: ANN401
    doc = inspect.getdoc(obj) or ""
    line = doc.strip().splitlines()[0] if doc.strip() else ""
    for role in (":meth:", ":class:", ":func:"):
        line = line.replace(role, "")
    return line.replace("``", "`")


def _params(obj: Callable[..., Any]) -> list[str]:
    parameters = list(inspect.signature(obj).parameters.values())
    out: list[str] = []
    for parameter in parameters:
        if parameter.name == "self":
            continue
        text = parameter.name
        if parameter.annotation is not inspect.Parameter.empty:
            text += f": {_type(parameter.annotation)}"
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {_value(parameter.default)}"
        out.append(text)
    return out


def _fields(obj: Any) -> list[str]:  # noqa: ANN401
    """Fields of a pydantic model or a dataclass, with defaults."""
    out: list[str] = []
    if hasattr(obj, "model_fields"):
        for name, field in obj.model_fields.items():
            if field.default_factory is not None:
                default = f"{field.default_factory.__name__}()"  # type: ignore[union-attr]
            else:
                default = _value(field.default)
            out.append(f"{name}: {_type(field.annotation)} = {default}")
        return out
    for field in dataclasses.fields(obj):
        text = f"{field.name}: {_type(field.type)}"
        if field.default is not dataclasses.MISSING:
            text += f" = {_value(field.default)}"
        out.append(text)
    return out


def _protocol_methods(obj: Any) -> list[str]:  # noqa: ANN401
    methods = [
        name for name, member in vars(obj).items() if callable(member) and not name.startswith("_")
    ]
    return [f"{name}({', '.join(_params(getattr(obj, name)))})" for name in methods]


def _row(path: str, obj: Any) -> str:  # noqa: ANN401
    if hasattr(obj, "model_fields") or dataclasses.is_dataclass(obj):
        kind, args = "fields", _fields(obj)
    elif getattr(obj, "_is_protocol", False):
        kind, args = "methods", _protocol_methods(obj)
    else:
        kind, args = "arguments", _params(obj)
    listed = ", ".join(f"`{_cell(arg)}`" for arg in args) if args else "none"
    return f"| `{path}` | {_cell(_summary(obj))} | {kind}: {listed} |"


def _server_rows() -> list[str]:
    from flashspec.serve.__main__ import build_parser

    rows = []
    for action in build_parser()._actions:
        if not action.option_strings or action.dest == "help":
            continue
        flag = max(action.option_strings, key=len)
        default = "required" if action.required else f"`{_cell(_value(action.default))}`"
        choices = ", ".join(_value(c) for c in action.choices) if action.choices else ""
        repeat = " (repeatable)" if isinstance(action, argparse._AppendAction) else ""
        rows.append(f"| `{flag}`{repeat} | {default} | {_cell(choices)} |")
    return rows


def render() -> str:
    """Return the block's markdown, between (not including) the markers."""
    from flashspec import FlashSpecConfig, SamplingConfig
    from flashspec.engine.speculative import AUTO_STRATEGY
    from flashspec.utils.config import BanditConfig

    config, sampling, bandit = FlashSpecConfig(), SamplingConfig(), BanditConfig()
    lines = [
        "Generated from the code by `scripts/readme_api.py`; a test fails when it is stale.",
        "",
        "| API | What it does | Key arguments, with defaults |",
        "|---|---|---|",
        *(_row(path, obj) for path, obj in _entries()),
        "| `flashspec.TRITON_AVAILABLE` | `True` only on Linux with Triton installed | "
        "constant: `bool` |",
        "",
        "Defaults that change behaviour:",
        "",
        f"- `FlashSpecConfig.compile = {_value(config.compile)}`: the compiled path (static KV "
        "cache, CUDA graphs) turns on when the device is CUDA and Triton is installed, and stays "
        "off otherwise.",
        f"- `SamplingConfig.temperature = {_value(sampling.temperature)}` with "
        f"`variant = {_value(sampling.variant)}`: the default samples (Algorithm 1). "
        '`temperature=0.0` selects `variant="greedy"`.',
        f"- `SamplingConfig.gamma = {_value(sampling.gamma)}`: draft tokens per step.",
        f"- `FlashSpecConfig.drafter_strategy = {_value(config.drafter_strategy)}`: the arms "
        '`SpeculativeEngine(..., arms=None)` builds. `"auto"` runs '
        f"`{_value(AUTO_STRATEGY)}` (prompt lookup when the context has a match, else the model "
        "drafter; an `NgramDrafter` is added) when given one model drafter, and the "
        '`"bandit"` path otherwise; a call at batch > 1 or with `variant="typical"` uses the '
        'model drafter at `sampling.gamma`. `"bandit"` (the 0.2.0 behaviour) is one arm per '
        "drafter at `sampling.gamma`, chosen by the selector you pass. An explicit composite "
        "needs batch 1.",
        f"- `BanditConfig.n_arms = {_value(bandit.n_arms)}` when you pass a `BanditConfig` "
        "yourself: it must equal the number of arms, so pass `BanditConfig(n_arms=1)` for one "
        'arm. Under `"auto"` with no `bandit=`, or a composite with no `bandit=`, it is not a '
        "constraint.",
        f"- `FlashSpecConfig.device = {_value(config.device)}`, "
        f"`max_new_tokens = {_value(config.max_new_tokens)}`.",
        "",
        "Server flags (`python -m flashspec.serve`):",
        "",
        "| flag | default | choices |",
        "|---|---|---|",
        *_server_rows(),
    ]
    return "\n".join(lines) + "\n"


def replace_block(text: str, block: str) -> str:
    """Return ``text`` with the content between the markers replaced by ``block``."""
    start = text.index(BEGIN) + len(BEGIN)
    end = text.index(END)
    return text[:start] + "\n" + block + text[end:]


def main(argv: list[str] | None = None) -> int:
    """Rewrite the README block, or with ``--check`` report whether it is current."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--readme", type=Path, default=README)
    args = parser.parse_args(argv)
    text = args.readme.read_text(encoding="utf-8")
    updated = replace_block(text, render())
    if args.check:
        if updated != text:
            print(f"{args.readme}: the API block is stale; run python scripts/readme_api.py")
            return 1
        print(f"{args.readme}: the API block is current")
        return 0
    args.readme.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
