"""The ``typical`` variant is labelled lossy in the API and warns at engine construction."""

from __future__ import annotations

import logging

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.hf_drafter import HuggingFaceDrafter
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.sampling import typical
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig


def _llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
    )
    return LlamaForCausalLM(config).eval()


def _engine(variant: str) -> SpeculativeEngine:
    config = FlashSpecConfig(
        drafter_strategy="bandit",
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        bandit=BanditConfig(n_arms=1, strategy="ucb1", window_size=0),
        sampling=SamplingConfig(gamma=2, variant=variant),  # type: ignore[arg-type]
    )
    return SpeculativeEngine(
        config,
        [HuggingFaceDrafter("d", device="cpu", model=_llama(1))],  # type: ignore[list-item]
        TargetModel("t", torch.device("cpu"), torch.float32, model=_llama(0)),
        UCB1Selector(n_arms=1, window_size=0),
    )


def test_docstrings_say_lossy() -> None:
    """The module, the function and the config field all say it."""
    assert typical.__doc__ is not None and "(lossy)" in typical.__doc__
    assert typical.typical_sample.__doc__ is not None
    assert "(lossy)" in typical.typical_sample.__doc__
    assert SamplingConfig.__doc__ is not None and "**lossy**" in SamplingConfig.__doc__


@pytest.mark.parametrize(("variant", "warns"), [("typical", True), ("rejection", False)])
def test_engine_warns_only_for_typical(
    caplog: pytest.LogCaptureFixture, variant: str, warns: bool
) -> None:
    """Selecting typical logs a lossy warning; rejection does not."""
    with caplog.at_level(logging.WARNING, logger="flashspec.engine.speculative"):
        _engine(variant)
    lossy = [r for r in caplog.records if "is lossy" in r.getMessage()]
    assert bool(lossy) is warns
