"""Greedy divergences from compiled HuggingFace: fp16 exact tie, or not.

The width diagnostic (``benchmarks/width_diagnostic.py``, T4 run at
``125d1a1``, merged in #47) showed that fp32 greedy FlashSpec is
token-exact against compiled HuggingFace on every path, and that every
fp16 divergence is an exact top-2 tie on one side: two candidate tokens
with the same fp16 logit, which kernels of different shapes resolve
differently.

This module checks a measured divergence against that finding. It replays
the prompt through compiled HuggingFace with ``output_logits`` and through
the FlashSpec engine with its verify rows captured, and compares the two
rows that chose the diverging token. A divergence at token ``k`` is an
**allowed exact tie** when all of these hold (:func:`classify`):

1. the model dtype is float16 (in fp32 no divergence is allowed);
2. both replays reproduce the recorded outputs exactly;
3. both outputs agree on tokens ``[0, k)``;
4. the two tokens at ``k`` are the top-2 ids of the reference row and of
   the FlashSpec row (as a set);
5. one of the two rows has a top-1 / top-2 logit margin of exactly 0.0.

Anything else is a mismatch. Every value is copied from the replays;
nothing is estimated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

__all__ = [
    "ALLOWED_DTYPE",
    "RULE",
    "capture_verify_rows",
    "classify",
    "first_divergence",
    "hf_compiled_logits",
    "top2",
]

ALLOWED_DTYPE = "float16"

RULE = (
    "allowed exact tie: float16; both replays reproduce the recorded outputs; same tokens "
    "before the divergence; the two diverging tokens are the top-2 ids of both rows; one row's "
    "top-1/top-2 margin is exactly 0.0 (bundle 125d1a1 width diagnostic, #47)"
)


def first_divergence(got: list[int], want: list[int]) -> int | None:
    """First index where ``got`` and ``want`` differ, or ``None`` when equal."""
    limit = min(len(got), len(want))
    first = next((i for i in range(limit) if got[i] != want[i]), None)
    if first is None and len(got) != len(want):
        return limit
    return first


def top2(row: torch.Tensor) -> dict[str, Any]:
    """Top-2 ids, their logits, and the margin between them (row upcast to fp32)."""
    values, ids = row.float().topk(2)
    return {
        "ids": [int(item) for item in ids],
        "logits": [float(item) for item in values],
        "margin_top1_top2": float(values[0] - values[1]),
    }


def classify(
    *,
    dtype: str,
    reference: list[int],
    path: list[int],
    reference_replay: list[int],
    path_replay: list[int],
    reference_row: torch.Tensor | None,
    path_row: torch.Tensor | None,
) -> dict[str, Any]:
    """Apply the rule in the module docstring to one divergence.

    Parameters
    ----------
    dtype : str
        Model dtype name, e.g. ``"float16"``.
    reference, path : list of int
        Recorded outputs of compiled HuggingFace and of the FlashSpec method.
    reference_replay, path_replay : list of int
        The same prompt replayed by each.
    reference_row, path_row : torch.Tensor or None
        Logit rows that chose the token at the divergence (``None`` when a
        replay did not produce one).

    Returns
    -------
    dict
        The divergence position, each condition, both rows' top-2, and
        ``allowed_exact_tie``.
    """
    position = first_divergence(path, reference)
    entry: dict[str, Any] = {"position": position, "dtype": dtype}
    if position is None:
        entry["allowed_exact_tie"] = None
        entry["note"] = "no divergence"
        return entry
    replays_ok = reference_replay == reference and path_replay == path
    same_prefix = path[:position] == reference[:position]
    entry.update(
        {
            "reference_token": reference[position] if position < len(reference) else None,
            "path_token": path[position] if position < len(path) else None,
            "replays_reproduce_recorded": replays_ok,
            "same_prefix": same_prefix,
        }
    )
    tie = False
    if reference_row is not None and path_row is not None:
        ref_top, path_top = top2(reference_row), top2(path_row)
        chosen = {entry["reference_token"], entry["path_token"]}
        same_pair = chosen == set(ref_top["ids"]) == set(path_top["ids"])
        exact = ref_top["margin_top1_top2"] == 0.0 or path_top["margin_top1_top2"] == 0.0
        entry.update(
            {
                "reference_top2": ref_top,
                "path_top2": path_top,
                "max_abs_logit_difference": float(
                    (reference_row.float() - path_row.float()).abs().max()
                ),
                "diverging_tokens_are_both_rows_top2": same_pair,
                "one_margin_exactly_zero": exact,
            }
        )
        tie = same_pair and exact
    else:
        entry["note"] = "a replay produced no logit row at the divergence"
    entry["allowed_exact_tie"] = bool(dtype == ALLOWED_DTYPE and replays_ok and same_prefix and tie)
    return entry


def hf_compiled_logits(
    model: Any,  # noqa: ANN401
    prompt: torch.Tensor,
    max_new_tokens: int,
    static_cache: Any,  # noqa: ANN401
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy ``generate`` on the reused ``StaticCache`` with ``CompileConfig()``, plus logits.

    The keyword arguments are those of ``benchmarks.harness._hf_generate``
    for greedy decoding with no EOS stop, plus ``output_logits``.
    """
    from transformers.generation.configuration_utils import CompileConfig

    kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens,
        "eos_token_id": None,
        "use_cache": True,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "output_logits": True,
        "return_dict_in_generate": True,
    }
    if static_cache is not None:
        static_cache.reset()
        kwargs["past_key_values"] = static_cache
        kwargs["compile_config"] = CompileConfig()
    pad = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad is not None:
        kwargs["pad_token_id"] = pad
    with torch.no_grad():
        result = model.generate(prompt, attention_mask=torch.ones_like(prompt), **kwargs)
    tokens = [int(item) for item in result.sequences[0, prompt.shape[1] :].tolist()]
    return tokens, [step[0].float().cpu() for step in result.logits]


@contextmanager
def capture_verify_rows(target: Any, prompt_len: int) -> Iterator[dict[int, torch.Tensor]]:  # noqa: ANN401
    """Record the target's verify logit rows by generated-token position.

    Wraps ``target._verify_rows`` for the duration of the block. Row ``i``
    of a verify call predicts the token at ``committed - prompt_len + i``,
    where ``committed`` is the session's cached plus pending length before
    the call. A later step overwrites the rows it re-scores, so after the
    call each position holds the row that chose its token.
    """
    rows_by_position: dict[int, torch.Tensor] = {}
    verify_rows: Callable[[torch.Tensor], torch.Tensor] = target._verify_rows

    def _capture(draft: torch.Tensor) -> torch.Tensor:
        session = target._inc
        committed = int(session.cached_length) + int(session.pending_length)
        rows = verify_rows(draft)
        first = committed - prompt_len
        for offset in range(int(rows.shape[1])):
            rows_by_position[first + offset] = rows[0, offset].float().cpu()
        return rows

    target._verify_rows = _capture
    try:
        yield rows_by_position
    finally:
        del target._verify_rows  # back to the class method
