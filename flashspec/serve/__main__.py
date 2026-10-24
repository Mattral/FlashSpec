"""Command line entry point: ``python -m flashspec.serve``.

Loads one target and one or more drafters, then serves them on a single
device. Refuses to start when CUDA is requested but unavailable, rather
than pretending a GPU result is possible.
"""

from __future__ import annotations

import argparse
from typing import Literal

import torch

from flashspec.bandit.thompson import ThompsonSelector
from flashspec.bandit.ucb import UCB1Selector
from flashspec.engine.drafter import load_drafters
from flashspec.engine.speculative import SpeculativeEngine
from flashspec.engine.verifier import TargetModel
from flashspec.serve.app import create_app
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig
from flashspec.utils.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the server argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for ``python -m flashspec.serve``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Serve one FlashSpec engine over a small OpenAI-compatible HTTP API. "
            "Single device only. Streaming, continuous batching, and multi-GPU "
            "are not implemented."
        )
    )
    parser.add_argument(
        "--target",
        required=True,
        help="HuggingFace id or local path of the target.",
    )
    parser.add_argument(
        "--drafter",
        action="append",
        required=True,
        help="HuggingFace id or registry name. Repeat for a second drafter.",
    )
    parser.add_argument("--device", default="cpu", help="cuda:0, cpu, or auto. Default: cpu.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-queue", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--strategy", choices=("ucb1", "thompson"), default="ucb1")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument(
        "--compile",
        choices=("auto", "on", "off"),
        default="auto",
        help="Static KV cache + CUDA graphs. auto: on with CUDA and Triton, off otherwise.",
    )
    return parser


#: ``--compile`` value -> ``FlashSpecConfig.compile``.
_COMPILE_CHOICES: dict[str, bool | Literal["auto"]] = {"auto": "auto", "on": True, "off": False}


def resolve_device(requested: str) -> torch.device:
    """Resolve ``cpu``, ``auto``, or a CUDA device string.

    Parameters
    ----------
    requested : str
        User-supplied device. ``auto`` picks CUDA when it exists.

    Returns
    -------
    torch.device
        Device the engine will use.

    Raises
    ------
    RuntimeError
        If a CUDA device was requested and CUDA is not available.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {requested}, but CUDA is not available. "
            "Refusing to start. Pass --device cpu or --device auto."
        )
    return device


def main(argv: list[str] | None = None) -> None:
    """Load models and block in uvicorn.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector, excluding the program name. ``None`` reads
        ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    drafter_names: list[str] = list(args.drafter)
    if len(drafter_names) < 2:
        logger.warning(
            "Only one drafter was passed. The bandit will not adapt. "
            "Pass --drafter a second time with a different model if you "
            "want online draft selection to do anything."
        )
    drafters = load_drafters(
        drafter_names,
        device=device,
        dtype=dtype,
        temperature=args.temperature,
    )
    target = TargetModel(args.target, device, dtype)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # Stop each completion at the tokenizer's end-of-sequence token.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_ids = () if eos_id is None else (int(eos_id),)
    config = FlashSpecConfig(
        drafter_name=drafter_names[0],
        target_name=args.target,
        device=str(device),
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
        compile=_COMPILE_CHOICES[args.compile],
        bandit=BanditConfig(n_arms=len(drafters), strategy=args.strategy, window_size=100),
        sampling=SamplingConfig(gamma=args.gamma, temperature=args.temperature),
    )
    if args.strategy == "thompson":
        bandit = ThompsonSelector(n_arms=len(drafters), window_size=100)
    else:
        bandit = UCB1Selector(n_arms=len(drafters), window_size=100)
    engine = SpeculativeEngine(config, drafters, target, bandit)

    def encode(text: str) -> list[int]:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return [int(token) for token in encoded]

    def decode(token_ids: list[int]) -> str:
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))

    app = create_app(
        engine,
        encode,
        decode,
        model_name=args.target,
        batch_size=args.batch_size,
        max_wait_ms=args.max_wait_ms,
        max_queue=args.max_queue,
        temperature=args.temperature,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve FlashSpec. Install it with "
            "`pip install flashspec[serve]`."
        ) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
