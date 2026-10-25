"""Greedy gold test on TinyLlama-1.1B with a 160M drafter, on CUDA.

Same check as ``tests/unit/test_greedy_gold.py``: greedy FlashSpec must
emit exactly the tokens of ``model.generate(do_sample=False)``, here on
real checkpoints and real prompts (the first 5 vendored HumanEval
prompts, 64 new tokens each, gamma in {1, 2, 4, 6}), for three arm sets:
the model drafter, the prompt-lookup (n-gram) drafter alone, and both
under the bandit.

Run it on a GPU (Colab T4 is enough; weights are downloaded):

    pytest tests/integration/test_greedy_gold_gpu.py -m gpu -v --no-cov

float32 is the gold check. float16 is run as well but marked
non-strict xfail: a (1 + gamma)-token verify and a 1-token decode use
different kernels, and in half precision a near-tie argmax can flip.
That is a numerics property of fp16 decoding (HuggingFace assisted
generation has it too), not a bookkeeping error. An XPASS means the fp16
run matched exactly as well.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from benchmarks.prompt_sets import load_prompt_set, tokenize_prompts
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.ngram import NgramDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig

pytestmark = pytest.mark.gpu

_TARGET = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_DRAFTER = "Felladrin/Llama-160M-Chat-v1"
_N_PROMPTS = 5
_NEW_TOKENS = 64
_GAMMAS = (1, 2, 4, 6)
_DEVICE = "cuda"


@pytest.fixture(autouse=True)
def _quiet_single_arm_warning() -> Iterator[None]:
    logger = logging.getLogger("flashspec.engine.speculative")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(previous)


def _load(dtype: torch.dtype) -> tuple[Any, Any, list[torch.Tensor]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(_DEVICE)
    target = AutoModelForCausalLM.from_pretrained(_TARGET, torch_dtype=dtype).to(device).eval()
    drafter = AutoModelForCausalLM.from_pretrained(_DRAFTER, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(_TARGET)
    records = load_prompt_set("humaneval")[:_N_PROMPTS]
    if len(records) != _N_PROMPTS:
        raise AssertionError(
            f"humaneval has {len(records)} vendored prompts; this test needs {_N_PROMPTS}."
        )
    prompts = tokenize_prompts(records, tokenizer, device, None)
    return target, drafter, prompts


def _hf_greedy(model: Any, prompt: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
    with torch.no_grad():
        out = model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=_NEW_TOKENS,
            min_new_tokens=_NEW_TOKENS,
            eos_token_id=None,
            pad_token_id=model.config.pad_token_id or 0,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return out[:, prompt.shape[1] :]


def _check(dtype: torch.dtype) -> None:
    target_module, drafter_module, prompts = _load(dtype)
    expected = [_hf_greedy(target_module, prompt) for prompt in prompts]
    failures: list[str] = []
    for gamma, arm_set in [(g, a) for g in _GAMMAS for a in ("model", "ngram", "both")]:
        drafter = HuggingFaceDrafter(_DRAFTER, device=_DEVICE, model=drafter_module)
        drafters: list[Any] = {
            "model": [drafter],
            "ngram": [NgramDrafter()],
            "both": [NgramDrafter(), drafter],
        }[arm_set]
        target = TargetModel(_TARGET, torch.device(_DEVICE), dtype, model=target_module)
        config = FlashSpecConfig(
            device=_DEVICE,
            # The gold reference is eager generate(); keep FlashSpec eager too.
            compile=False,
            dtype=str(dtype).removeprefix("torch."),  # type: ignore[arg-type]
            max_new_tokens=_NEW_TOKENS,
            bandit=BanditConfig(n_arms=len(drafters), strategy="ucb1", window_size=0),
            sampling=SamplingConfig(gamma=gamma, temperature=0.0),
        )
        engine = SpeculativeEngine(
            config,
            drafters,
            target,
            UCB1Selector(n_arms=len(drafters), window_size=0),
        )
        for index, (prompt, reference) in enumerate(zip(prompts, expected, strict=True)):
            output = engine.generate(prompt).output_ids
            diverged = (output != reference).nonzero()
            if diverged.numel():
                position = int(diverged[0, 1])
                failures.append(
                    f"arms={arm_set} gamma={gamma} prompt={index}: "
                    f"first divergence at token {position} "
                    f"(flashspec {int(output[0, position])}, "
                    f"generate {int(reference[0, position])})"
                )
    assert not failures, "\n".join(failures)


def test_tinyllama_greedy_matches_generate_fp32() -> None:
    """float32: greedy FlashSpec == generate(do_sample=False), token for token."""
    _check(torch.float32)


@pytest.mark.xfail(
    strict=False,
    reason="fp16 near-tie argmax flips between batched verify and 1-token decode kernels",
)
def test_tinyllama_greedy_matches_generate_fp16() -> None:
    """float16: same check; a mismatch here is reported, not treated as a failure."""
    _check(torch.float16)
