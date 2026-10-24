"""Queue, metrics, and the HTTP surface of the single-device server."""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch

from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.speculative import GenerationResult, SpeculativeEngine
from flashspec.serve.app import messages_to_prompt
from flashspec.serve.batching import QueueFullError, RequestBatcher
from flashspec.serve.metrics import ServeMetrics
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig


class _Peaked:
    """Minimal drafter and target used by the server tests."""

    def __init__(self, vocab: int = 16, peak: int = 2) -> None:
        self.vocab = vocab
        self.peak = peak

    def _logprobs(self, batch: int, steps: int) -> torch.Tensor:
        logits = torch.full((batch, steps, self.vocab), -4.0)
        logits[:, :, self.peak] = 4.0
        return torch.log_softmax(logits, dim=-1)

    def generate_draft(
        self, input_ids: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose the peak token."""
        batch = int(input_ids.shape[0])
        return torch.full((batch, gamma), self.peak, dtype=torch.long), self._logprobs(batch, gamma)

    def compute_logprobs(
        self, input_ids: torch.Tensor, draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Rescore under the same peak."""
        return self._logprobs(int(draft_token_ids.shape[0]), int(draft_token_ids.shape[1]))

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Target log-probs."""
        steps = gamma + 1 if include_bonus else gamma
        return self._logprobs(int(input_ids.shape[0]), steps)


def _engine() -> SpeculativeEngine:
    """One-arm engine. The bandit cannot adapt; the server still generates."""
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        max_new_tokens=4,
        bandit=BanditConfig(n_arms=1, window_size=0),
        sampling=SamplingConfig(gamma=2, temperature=1.0),
    )
    drafter = _Peaked()
    return SpeculativeEngine(
        config,
        [drafter],  # type: ignore[list-item]
        drafter,  # type: ignore[arg-type]
        UCB1Selector(n_arms=1, window_size=0),
    )


class _CountingEngine:
    """Delegates to a real engine and counts generate calls."""

    def __init__(self) -> None:
        self.inner = _engine()
        self.calls = 0
        self.batch_shapes: list[tuple[int, ...]] = []

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int | None = None
    ) -> GenerationResult:
        """Record the batch shape and generate."""
        self.calls += 1
        self.batch_shapes.append(tuple(input_ids.shape))
        return self.inner.generate(input_ids, max_new_tokens=max_new_tokens)


class TestMetrics:
    """Prometheus text exposition."""

    def test_render_contains_the_basic_series(self) -> None:
        """A finished request updates throughput, acceptance, and latency."""
        metrics = ServeMetrics()
        metrics.inc_active()
        metrics.observe_generation(
            tokens=4, acceptance_rate=0.5, latency_ms=12.0, tokens_per_second=100.0
        )
        metrics.dec_active()
        text = metrics.render()
        assert "flashspec_requests_active 0" in text
        assert "flashspec_tokens_generated_total 4" in text
        assert "flashspec_acceptance_rate 0.500000" in text
        assert 'flashspec_request_latency_ms_bucket{le="10"} 0' in text
        assert 'flashspec_request_latency_ms_bucket{le="25"} 1' in text
        assert text.endswith("\n")


class TestBatcher:
    """Fixed-size batching only stacks equal prompt lengths."""

    def test_equal_lengths_share_one_generate(self) -> None:
        """Two prompts of the same length become one batch."""
        engine = _CountingEngine()
        batcher = RequestBatcher(engine, batch_size=2, max_wait_ms=200, max_queue=8)  # type: ignore[arg-type]
        batcher.start()
        try:
            first = batcher.submit([1, 2, 3], max_new_tokens=2)
            second = batcher.submit([4, 5, 6], max_new_tokens=2)
            assert first.result(timeout=10).output_ids.shape[0] == 1
            assert second.result(timeout=10).n_tokens_generated == 2
        finally:
            batcher.stop()
        assert engine.calls == 1
        assert engine.batch_shapes == [(2, 3)]

    def test_different_lengths_do_not_share_a_batch(self) -> None:
        """A longer prompt is not padded onto a shorter one."""
        engine = _CountingEngine()
        batcher = RequestBatcher(engine, batch_size=4, max_wait_ms=50, max_queue=8)  # type: ignore[arg-type]
        batcher.start()
        try:
            short = batcher.submit([1, 2], max_new_tokens=2)
            long = batcher.submit([1, 2, 3, 4], max_new_tokens=2)
            short.result(timeout=10)
            long.result(timeout=10)
        finally:
            batcher.stop()
        assert engine.calls == 2

    def test_queue_full_raises_before_a_worker_starts(self) -> None:
        """A full queue fails the submitter instead of blocking forever."""
        batcher = RequestBatcher(_engine(), batch_size=1, max_queue=1)
        batcher.submit([1, 2], max_new_tokens=2)
        with pytest.raises(QueueFullError):
            batcher.submit([1, 2], max_new_tokens=2)

    def test_invalid_batch_size_raises(self) -> None:
        """batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size"):
            RequestBatcher(_engine(), batch_size=0)


class TestHttp:
    """OpenAI-compatible routes. Skipped when FastAPI is not installed."""

    def test_chat_completions_and_limits(self) -> None:
        """A chat request returns text, and the documented limits are visible."""
        fastapi = pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from flashspec.serve.app import create_app

        engine = _engine()
        app = create_app(
            engine,
            encode=lambda text: [min(ord(char), 15) for char in text] or [1],
            decode=lambda ids: "ok:" + ",".join(str(token) for token in ids),
            model_name="toy-target",
            batch_size=1,
            temperature=1.0,
            max_queue=4,
        )
        assert fastapi is not None
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            body = health.json()
            assert body["model"] == "toy-target"
            assert any("single device" in item for item in body["limitations"])
            assert "streaming" in " ".join(body["limitations"]).lower() or any(
                "stream" in item for item in body["limitations"]
            )

            rejected = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert rejected.status_code == 400

            temperature = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.2,
                },
            )
            assert temperature.status_code == 400

            chat = client.post(
                "/v1/chat/completions",
                json={
                    "model": "other-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 2,
                },
            )
            assert chat.status_code == 200
            payload = chat.json()
            assert payload["choices"][0]["message"]["content"].startswith("ok:")
            assert "flashspec_warning" in payload
            assert payload["usage"]["completion_tokens"] == 2

            completion = client.post(
                "/v1/completions",
                json={"prompt": "hello", "max_tokens": 2},
            )
            assert completion.status_code == 200
            assert completion.json()["choices"][0]["text"].startswith("ok:")

            metrics = client.get("/metrics")
            assert metrics.status_code == 200
            assert "flashspec_tokens_generated_total" in metrics.text
            assert "flashspec_acceptance_rate" in metrics.text

    def test_messages_require_strings(self) -> None:
        """Empty chats and non-string content are rejected by the helper."""
        assert messages_to_prompt([{"role": "user", "content": "hi"}]).endswith("assistant:")
        with pytest.raises(ValueError):
            messages_to_prompt([])


def test_queue_full_is_http_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """QueueFullError from the batcher becomes HTTP 429, not a 500."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from flashspec.serve.app import create_app
    from flashspec.serve.batching import RequestBatcher

    def _full(self: RequestBatcher, token_ids: list[int], max_new_tokens: int) -> None:
        raise QueueFullError("full")

    monkeypatch.setattr(RequestBatcher, "submit", _full)
    app = create_app(
        _engine(),
        encode=lambda text: [1, 2],
        decode=lambda ids: "x",
        batch_size=1,
        max_queue=1,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 2},
        )
    assert response.status_code == 429


def test_resolve_device_and_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI refuses a CUDA device when CUDA is absent."""
    from flashspec.serve.__main__ import build_parser, main, resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="Refusing to start"):
        resolve_device("cuda:0")
    args = build_parser().parse_args(["--target", "toy", "--drafter", "a", "--drafter", "b"])
    assert args.drafter == ["a", "b"]
    assert args.strategy == "ucb1"
    assert args.compile == "auto"
    with pytest.raises(RuntimeError, match="Refusing to start"):
        main(["--target", "toy", "--drafter", "a", "--device", "cuda"])


def _patch_server_loaders(monkeypatch: pytest.MonkeyPatch, n_drafters: int) -> object:
    """Replace model loading so the CLI can run without a checkpoint."""
    import flashspec.serve.__main__ as cli

    monkeypatch.setattr(cli, "load_drafters", lambda *args, **kwargs: [object()] * n_drafters)
    monkeypatch.setattr(cli, "TargetModel", lambda *args, **kwargs: object())
    configs: list[object] = []

    def _engine(config: object, *args: object, **kwargs: object) -> object:
        configs.append(config)
        return object()

    monkeypatch.setattr(cli, "SpeculativeEngine", _engine)
    monkeypatch.setattr(cli, "_seen_configs", configs, raising=False)

    class _Tok:
        pad_token = None
        eos_token = "<eos>"
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [3, 4]

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            return "ok"

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: _Tok())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return cli


def test_main_wires_thompson_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI builds the app and stops at the replaced uvicorn call."""
    cli = _patch_server_loaders(monkeypatch, n_drafters=2)
    seen: dict[str, object] = {}

    def _run(app: object, host: str, port: int) -> None:
        seen["host"] = host
        seen["port"] = port
        seen["app"] = app

    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = _run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    cli.main(
        [
            "--target",
            "toy",
            "--drafter",
            "a",
            "--drafter",
            "b",
            "--strategy",
            "thompson",
            "--host",
            "127.0.0.1",
            "--port",
            "9",
        ]
    )
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 9
    assert seen["app"] is not None
    # Completions stop at the tokenizer's EOS.
    (config,) = cli._seen_configs  # type: ignore[attr-defined]
    assert config.eos_token_ids == (2,)
    # The default --compile auto reaches the engine config unchanged.
    assert config.compile == "auto"


def test_main_one_drafter_warns_and_missing_uvicorn_raises(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One drafter is a warning. A missing uvicorn install raises ImportError."""
    cli = _patch_server_loaders(monkeypatch, n_drafters=1)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with caplog.at_level(logging.WARNING), pytest.raises(ImportError, match="flashspec\\[serve\\]"):
        cli.main(["--target", "toy", "--drafter", "only"])
    assert "will not adapt" in caplog.text


def test_prompt_helper_rejects_non_string_content() -> None:
    """Multipart content is outside the server's supported subset."""
    with pytest.raises(ValueError, match="string"):
        messages_to_prompt([{"role": "user", "content": ["not", "a", "string"]}])  # type: ignore[list-item]
