"""Fixed prompt sets for the benchmark harness.

This package does not import the harness. The harness imports this package.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

_DIR = Path(__file__).resolve().parent

REAL_DATASETS: frozenset[str] = frozenset({"humaneval", "mt_bench"})

_ALIASES: dict[str, str] = {
    "human_eval": "humaneval",
    "mtbench": "mt_bench",
}

ALPACA_REASON: str = (
    "Stanford Alpaca is CC BY-NC 4.0. Its instruction text is not in this "
    "tree, and no result file was written."
)


def canonical_dataset(name: str) -> str:
    """Return the canonical dataset slug.

    Parameters
    ----------
    name : str
        Config value. Case and hyphens are normalised.

    Returns
    -------
    str
        Lower-case slug. ``human_eval`` and ``mtbench`` map to the files
        in this package. Every other name is returned unchanged apart
        from that normalisation.
    """
    key = name.strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


def load_prompt_set(name: str) -> list[dict[str, str]]:
    """Load the fixed prompt records for one real dataset.

    Parameters
    ----------
    name : str
        Dataset name. ``alpaca`` is refused.

    Returns
    -------
    list[dict[str, str]]
        Records with ``id`` and ``text``, in file order.

    Raises
    ------
    RuntimeError
        If the name is Alpaca, unknown, or the JSON has no prompts.
    """
    slug = canonical_dataset(name)
    if slug == "alpaca":
        raise RuntimeError(ALPACA_REASON)
    if slug not in REAL_DATASETS:
        raise RuntimeError(f"No fixed prompt set named {name!r}. No result file was written.")
    path = _DIR / f"{slug}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise RuntimeError(f"{path.name} has no prompts. No result file was written.")
    records: list[dict[str, str]] = []
    for item in prompts:
        if not isinstance(item, dict) or "id" not in item or "text" not in item:
            raise RuntimeError(f"{path.name} prompt is missing id or text.")
        text = str(item["text"])
        if not text:
            raise RuntimeError(f"{path.name} contains an empty prompt.")
        records.append({"id": str(item["id"]), "text": text})
    return records


def tokenize_prompts(
    records: list[dict[str, str]],
    tokenizer: object,
    device: torch.device,
    max_length: int | None,
) -> list[torch.Tensor]:
    """Tokenize prompt texts with the target tokenizer.

    Parameters
    ----------
    records : list[dict[str, str]]
        Records from :func:`load_prompt_set`.
    tokenizer : object
        Object with ``encode`` or a callable that returns ``input_ids``.
    device : torch.device
        Destination device.
    max_length : int or None
        Keep the start of each encoding when set.

    Returns
    -------
    list[torch.Tensor]
        One tensor per record, shaped ``(1, n)`` and dtype long.

    Raises
    ------
    RuntimeError
        If an encoding is empty.
    """
    tensors: list[torch.Tensor] = []
    for record in records:
        ids = _encode(tokenizer, record["text"])
        if max_length is not None:
            ids = ids[:max_length]
        if not ids:
            raise RuntimeError(
                f"Tokenizer produced an empty encoding for prompt {record['id']}. "
                "No result file was written."
            )
        tensors.append(torch.tensor([ids], dtype=torch.long, device=device))
    return tensors


def _encode(tokenizer: object, text: str) -> list[int]:
    """Encode ``text`` with ``encode`` or a callable tokenizer.

    Parameters
    ----------
    tokenizer : object
        Target tokenizer.
    text : str
        Prompt text.

    Returns
    -------
    list[int]
        Token ids. A batch of one is unwrapped.
    """
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            raw = encode(text, add_special_tokens=True)
        except TypeError:
            raw = encode(text)
    else:
        raw = tokenizer(text)["input_ids"]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(item) for item in raw]
