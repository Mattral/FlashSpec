"""Shared pytest fixtures for FlashSpec tests.

Provides tiny (4-layer, 128-hidden) toy models so no network calls or
real GPU weights are needed in unit or integration tests.

All GPU fixtures are decorated with ``@pytest.mark.gpu`` so they are
skipped in the fast CPU CI lane.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flashspec.utils.device import set_seed

# ── Constants ─────────────────────────────────────────────────────────────────
TOY_VOCAB_SIZE: int = 1_000
TOY_HIDDEN: int = 128
TOY_N_LAYERS: int = 4
TOY_BATCH_SIZE: int = 2
TOY_SEQ_LEN: int = 16
TOY_GAMMA: int = 4

LLAMA_VOCAB_SIZE: int = 32_000
MISTRAL_VOCAB_SIZE: int = 32_768


# ── Tiny causal LM ────────────────────────────────────────────────────────────

class _TinyCausalLM(nn.Module):
    """Minimal causal LM for testing: embedding → transformer → lm_head."""

    def __init__(self, vocab_size: int, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> "SimpleNamespace":  # type: ignore[name-defined]
        """Forward pass returning a namespace with ``logits``."""
        from types import SimpleNamespace
        x = self.embed(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return SimpleNamespace(logits=logits)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def toy_vocab_size() -> int:
    """Vocabulary size for toy models."""
    return TOY_VOCAB_SIZE


@pytest.fixture(scope="session")
def toy_model_cpu() -> _TinyCausalLM:
    """Tiny causal LM on CPU, fixed seed."""
    set_seed(42)
    model = _TinyCausalLM(TOY_VOCAB_SIZE, TOY_HIDDEN, TOY_N_LAYERS)
    model.eval()
    return model


@pytest.fixture()
def random_input_ids() -> torch.Tensor:
    """Random integer input IDs on CPU."""
    set_seed(0)
    return torch.randint(0, TOY_VOCAB_SIZE, (TOY_BATCH_SIZE, TOY_SEQ_LEN))


@pytest.fixture()
def random_draft_logprobs() -> torch.Tensor:
    """Random draft log-probabilities, shape (batch, gamma, vocab)."""
    set_seed(1)
    return torch.randn(TOY_BATCH_SIZE, TOY_GAMMA, TOY_VOCAB_SIZE).log_softmax(-1)


@pytest.fixture()
def random_target_logprobs() -> torch.Tensor:
    """Random target log-probabilities, shape (batch, gamma, vocab)."""
    set_seed(2)
    return torch.randn(TOY_BATCH_SIZE, TOY_GAMMA, TOY_VOCAB_SIZE).log_softmax(-1)


@pytest.fixture()
def random_draft_token_ids(toy_vocab_size: int) -> torch.Tensor:
    """Random draft token IDs, shape (batch, gamma)."""
    set_seed(3)
    return torch.randint(0, toy_vocab_size, (TOY_BATCH_SIZE, TOY_GAMMA))


@pytest.fixture()
def uniform_u() -> torch.Tensor:
    """Uniform random samples, shape (batch, gamma)."""
    set_seed(4)
    return torch.rand(TOY_BATCH_SIZE, TOY_GAMMA)


@pytest.fixture(scope="session")
def llama_vocab_size() -> int:
    """Llama-3 vocabulary size."""
    return LLAMA_VOCAB_SIZE


@pytest.fixture(scope="session")
def mistral_vocab_size() -> int:
    """Mistral vocabulary size."""
    return MISTRAL_VOCAB_SIZE
