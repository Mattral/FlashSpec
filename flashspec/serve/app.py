"""Single-device HTTP server for FlashSpec.

The surface is a small OpenAI-compatible subset:

* ``POST /v1/chat/completions``
* ``POST /v1/completions``
* ``GET /health``
* ``GET /metrics``

Limitations, on purpose:

* One process and one device. There is no multi-GPU scheduling.
* No token streaming.
* No paged KV cache and no continuous batching.
* Requests share a batch only when their prompt lengths and
  ``max_tokens`` match. Anything else runs in its own batch.
* ``temperature`` and ``gamma`` are fixed when the process starts.
  A request that sends a different temperature is rejected.
* The JSON ``model`` field does not load a different checkpoint.
* Bandit state lives for the life of the process. It is not reset
  per request, and with one drafter it cannot adapt.

This module imports FastAPI inside :func:`create_app` so the rest of
FlashSpec imports on machines that did not install the ``serve`` extra.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from flashspec import __version__
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.sampling.rejection import active_residual_backend
from flashspec.serve.batching import QueueFullError, RequestBatcher
from flashspec.serve.metrics import ServeMetrics

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = [
    "LIMITATIONS",
    "create_app",
    "messages_to_prompt",
]

LIMITATIONS: tuple[str, ...] = (
    "single process and a single device (one GPU or CPU)",
    "no streaming responses",
    "no continuous batching and no paged KV cache",
    "fixed-size batches only group equal prompt lengths and equal max_tokens",
    "temperature and gamma are fixed at process start",
    "the request model field does not select a different checkpoint",
    "bandit statistics are process-global and are not reset per request",
    "a single drafter makes the bandit a no-op",
)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages into the single string the server encodes.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-style messages. ``content`` must be a string.

    Returns
    -------
    str
        A prompt ending with an ``assistant:`` cue.

    Raises
    ------
    ValueError
        If ``messages`` is empty or a content field is not a string.
    """
    if not messages:
        raise ValueError("messages must contain at least one item.")
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(
                "Only string message content is supported. Image and multipart content are not."
            )
        role = message.get("role", "user")
        if not isinstance(role, str):
            role = "user"
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n".join(parts)


class ChatMessage(BaseModel):
    """One chat message. Content must be a plain string."""

    role: str = "user"
    content: str


class ChatRequest(BaseModel):
    """Subset of the OpenAI chat completions request."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=32, ge=1, le=4096)
    temperature: float | None = None
    n: int = 1
    stream: bool = False


class CompletionRequest(BaseModel):
    """Subset of the OpenAI completions request."""

    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=4096)
    temperature: float | None = None
    n: int = 1
    stream: bool = False


def create_app(
    engine: SpeculativeEngine,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    *,
    model_name: str = "flashspec",
    batch_size: int = 1,
    max_wait_ms: float = 2.0,
    max_queue: int = 32,
    temperature: float = 1.0,
) -> FastAPI:
    """Build a FastAPI application bound to ``engine``.

    Parameters
    ----------
    engine : SpeculativeEngine
        Already-constructed engine. The server does not load weights.
    encode : callable
        Maps a prompt string to token ids.
    decode : callable
        Maps generated token ids to text. Padding ids (``-1``) are
        stripped before this is called.
    model_name : str
        Name reported by ``/health`` and by responses when the client
        does not send one.
    batch_size : int
        Fixed batch cap. See :class:`RequestBatcher`.
    max_wait_ms : float
        How long a batch waits for more equal-shaped requests.
    max_queue : int
        Queue capacity. Overflow returns HTTP 429.
    temperature : float
        Temperature the engine was built with. Requests that send a
        different value are rejected with HTTP 400.

    Returns
    -------
    fastapi.FastAPI
        The application. Import it only after installing
        ``flashspec[serve]``.

    Raises
    ------
    ImportError
        If FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the FlashSpec server. Install it with "
            "`pip install flashspec[serve]`."
        ) from exc

    metrics = ServeMetrics()
    batcher = RequestBatcher(
        engine,
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        max_queue=max_queue,
    )

    def _check_common(n: int, stream: bool, requested_temperature: float | None) -> None:
        if stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not implemented. Send stream=false.",
            )
        if n != 1:
            raise HTTPException(
                status_code=400,
                detail="Only n=1 is supported.",
            )
        if requested_temperature is not None and abs(requested_temperature - temperature) > 1e-6:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This server was started with temperature={temperature}. "
                    "Per-request temperature is not applied, so a mismatched "
                    "value is rejected instead of being silently ignored."
                ),
            )

    async def _generate(prompt: str, max_tokens: int) -> tuple[str, int, int, float, float]:
        try:
            token_ids = encode(prompt)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"encode failed: {exc}") from exc
        if not token_ids:
            raise HTTPException(status_code=400, detail="Encoder produced an empty token list.")
        metrics.inc_active()
        started = time.perf_counter()
        try:
            future = batcher.submit(token_ids, max_tokens)
        except QueueFullError as exc:
            metrics.dec_active()
            metrics.mark_rejected()
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            metrics.dec_active()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = await asyncio.wrap_future(future)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
        finally:
            metrics.dec_active()
        latency_ms = (time.perf_counter() - started) * 1000.0
        produced = [int(token) for token in result.output_ids.view(-1).tolist() if int(token) >= 0]
        text = decode(produced)
        metrics.observe_generation(
            tokens=len(produced),
            acceptance_rate=result.acceptance_rate,
            latency_ms=latency_ms,
            tokens_per_second=result.tokens_per_second,
        )
        return text, len(token_ids), len(produced), result.acceptance_rate, result.tokens_per_second

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start the batch worker for the life of the process."""
        batcher.start()
        try:
            yield
        finally:
            batcher.stop()

    app = FastAPI(
        title="FlashSpec",
        version=__version__,
        description=(
            "Single-device speculative decoding server. "
            "Not a multi-tenant production gateway. See GET /health."
        ),
        lifespan=_lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        """Report the one loaded model and the limits of this process."""
        return {
            "status": "ok",
            "model": model_name,
            "residual_backend": active_residual_backend(),
            "batch_size": batch_size,
            "temperature": temperature,
            "limitations": list(LIMITATIONS),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics_endpoint() -> str:
        """Prometheus text exposition of this process."""
        return metrics.render()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict[str, object]:
        """Generate an assistant reply for ``messages``."""
        _check_common(request.n, request.stream, request.temperature)
        try:
            prompt = messages_to_prompt([item.model_dump() for item in request.messages])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        text, prompt_tokens, completion_tokens, acceptance, _tps = await _generate(
            prompt, request.max_tokens
        )
        served_model = request.model or model_name
        payload: dict[str, object] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "flashspec_acceptance_rate": acceptance,
        }
        if request.model is not None and request.model != model_name:
            payload["flashspec_warning"] = (
                f"Requested model {request.model!r} was not loaded. "
                f"This process serves {model_name!r} only."
            )
        return payload

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest) -> dict[str, object]:
        """Generate a continuation of ``prompt``."""
        _check_common(request.n, request.stream, request.temperature)
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt must be a non-empty string.")
        text, prompt_tokens, completion_tokens, acceptance, _tps = await _generate(
            request.prompt, request.max_tokens
        )
        served_model = request.model or model_name
        payload = {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": served_model,
            "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "flashspec_acceptance_rate": acceptance,
        }
        if request.model is not None and request.model != model_name:
            payload["flashspec_warning"] = (
                f"Requested model {request.model!r} was not loaded. "
                f"This process serves {model_name!r} only."
            )
        return payload

    app.state.batcher = batcher
    app.state.metrics = metrics
    return app
