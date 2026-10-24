"""GPU diagnostic for the three Phase 4 greedy mismatches (see benchmarks/greedy_mismatch.py).

- float32: greedy FlashSpec must equal ``hf_greedy`` on each prompt.
- float16: no assertion; prints the target's top-2 logits and margin at
  the divergence position for both decoders, and the three token-0
  computations, so a near-tie can be told apart from a bug.

Run on a GPU (weights are downloaded)::

    pytest tests/integration/test_greedy_mismatch_gpu.py -m gpu -v -s -o addopts=
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
import torch

from benchmarks.greedy_mismatch import CASES, run_case

pytestmark = pytest.mark.gpu


def _case_id(case: dict[str, object]) -> str:
    pair = case["pair"]
    assert isinstance(pair, tuple)
    return f"{str(pair[0]).split('/')[1]}-{case['dataset']}-{case['index']}"


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_fp32_matches_hf_greedy(case: dict[str, object]) -> None:
    """float32: identical tokens to generate(do_sample=False)."""
    result = run_case(case, torch.float32, torch.device("cuda"))
    print(json.dumps(result, indent=2))
    assert result["matches_hf_greedy"], (
        f"fp32 diverged at {result['first_divergence']}: {result['at_first_divergence']}"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_fp16_margins_at_divergence(case: dict[str, object]) -> None:
    """float16: report, do not assert. The printed JSON is the evidence."""
    result = run_case(case, torch.float16, torch.device("cuda"))
    print(json.dumps(result, indent=2))
