"""Multi-drafter registration and HuggingFace drafter construction."""

from __future__ import annotations

import logging

import pytest
import torch

from flashspec.engine.drafter import (
    get_drafter,
    list_drafters,
    load_drafters,
    register,
    register_hf,
    unregister,
)
from flashspec.engine.hf_drafter import HuggingFaceDrafter, scaled_log_softmax


class _Tiny(torch.nn.Module):
    """Embedding plus a linear head. Enough to produce logits."""

    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.nn.Module:
        """Return an object with a ``logits`` attribute."""
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Drop demo names so tests do not leak into each other."""
    yield
    for name in ("draft-a", "draft-b", "only-one", "replaced"):
        unregister(name)


class TestRegistry:
    """Two real drafters can be registered without editing the engine."""

    def test_register_hf_loads_two_injected_models(self) -> None:
        """load_drafters returns one instance per name, in order."""
        register_hf("draft-a", "org/draft-a")
        register_hf("draft-b", "org/draft-b")
        models = {"draft-a": _Tiny(), "draft-b": _Tiny()}
        loaded = load_drafters(
            ["draft-a", "draft-b"],
            device="cpu",
            dtype=torch.float32,
            models=models,
        )
        assert len(loaded) == 2
        assert loaded[0].model_name == "org/draft-a"  # type: ignore[attr-defined]
        assert loaded[1].model_name == "org/draft-b"  # type: ignore[attr-defined]
        assert "draft-a" in list_drafters()
        assert get_drafter("draft-a").__name__.startswith("HuggingFaceDrafter")

    def test_duplicate_name_raises_unless_replaced(self) -> None:
        """The default registry contract still rejects a second registration."""

        @register("replaced")
        class _First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register("replaced")
            class _Second:
                pass

        @register("replaced", replace=True)
        class _Third:
            pass

        assert get_drafter("replaced") is _Third

    def test_unknown_name_is_registered_as_hub_id(self) -> None:
        """A hub id that is not registered becomes a drafter on load."""
        name = "local-injected-hub"
        unregister(name)
        try:
            loaded = load_drafters(
                [name, "draft-b"],
                device="cpu",
                models={name: _Tiny(), "draft-b": _Tiny()},
            )
        finally:
            unregister(name)
            unregister("draft-b")
        assert len(loaded) == 2

    def test_empty_names_raise(self) -> None:
        """An empty arm list is rejected."""
        with pytest.raises(ValueError, match="at least one"):
            load_drafters([], device="cpu")

    def test_single_drafter_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """One arm is valid and explicitly not an adaptive policy."""
        register_hf("only-one", "org/only-one", replace=True)
        with caplog.at_level(logging.WARNING):
            loaded = load_drafters(["only-one"], device="cpu", models={"only-one": _Tiny()})
        assert len(loaded) == 1
        assert any("cannot adapt" in record.message for record in caplog.records)


class TestHuggingFaceDrafter:
    """Draft tokens and log-probs from an injected module."""

    def test_generate_and_rescore_shapes(self) -> None:
        """generate_draft and compute_logprobs return the protocol shapes."""
        drafter = HuggingFaceDrafter("toy", device="cpu", model=_Tiny(vocab=16))
        context = torch.randint(0, 16, (2, 5))
        token_ids, logprobs = drafter.generate_draft(context, gamma=3)
        assert token_ids.shape == (2, 3)
        assert logprobs.shape[0] == 2
        assert logprobs.shape[1] == 3
        assert logprobs.shape[2] == 16
        rescored = drafter.compute_logprobs(context, token_ids)
        assert rescored.shape == logprobs.shape
        assert drafter.vocab_size == 16

    def test_temperature_is_applied_before_log_softmax(self) -> None:
        """Dividing logits by temperature matches scaled_log_softmax."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        manual = torch.log_softmax(logits / 2.0, dim=-1)
        assert torch.allclose(scaled_log_softmax(logits, 2.0), manual)
        with pytest.raises(ValueError):
            scaled_log_softmax(logits, 0.0)

    def test_bad_gamma_raises(self) -> None:
        """Gamma below 1 is rejected."""
        drafter = HuggingFaceDrafter("toy", device="cpu", model=_Tiny())
        with pytest.raises(ValueError, match="gamma"):
            drafter.generate_draft(torch.zeros(1, 2, dtype=torch.long), gamma=0)

    def test_pretrained_failure_is_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hub failures become RuntimeError and do not partially construct."""
        import transformers

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise OSError("missing weights")

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _boom)
        with pytest.raises(RuntimeError, match="Failed to load draft model"):
            HuggingFaceDrafter("org/does-not-exist", device="cpu")
