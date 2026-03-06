"""Pull HuggingFace model checkpoints required for benchmarks.

Downloads the models listed in BENCHMARK_MODELS to the local HuggingFace
cache directory.  Requires a valid HF_TOKEN environment variable for
gated models.

Usage
-----
    export HF_TOKEN=hf_...
    python scripts/download_models.py
    python scripts/download_models.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# All models referenced in benchmarks/configs/*.yaml (AGENTS.md §17).
BENCHMARK_MODELS: list[str] = [
    # Target models.
    "meta-llama/Llama-3-8B-Instruct",
    "meta-llama/Llama-3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    # Draft models.
    "meta-llama/Llama-3-68M-Instruct",
    "meta-llama/Llama-3-1B-Instruct",
]


def _check_hf_token() -> str:
    """Read the HF_TOKEN environment variable or abort with a clear message.

    Returns
    -------
    str
        The HuggingFace access token.

    Raises
    ------
    SystemExit
        If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        sys.stderr.write(
            "ERROR: HF_TOKEN environment variable is not set.\n"
            "Set it with: export HF_TOKEN=hf_<your_token>\n"
            "Obtain a token at: https://huggingface.co/settings/tokens\n"
        )
        sys.exit(1)
    return token


def download_model(model_id: str, token: str, dry_run: bool) -> None:
    """Download a single model from HuggingFace Hub.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g. ``"meta-llama/Llama-3-8B-Instruct"``).
    token : str
        HuggingFace access token.
    dry_run : bool
        If ``True``, print the download command without executing it.

    Returns
    -------
    None
    """
    if dry_run:
        sys.stdout.write(f"  [DRY-RUN] Would download: {model_id}\n")
        return

    try:
        from huggingface_hub import snapshot_download  # type: ignore[import]
        sys.stdout.write(f"  Downloading {model_id} ...\n")
        local_path = snapshot_download(
            repo_id=model_id,
            token=token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        sys.stdout.write(f"  Saved to: {local_path}\n")
    except ImportError:
        sys.stderr.write(
            "ERROR: huggingface_hub not installed.\n"
            "Install with: pip install huggingface_hub\n"
        )
        sys.exit(1)
    except Exception as exc:
        sys.stderr.write(f"  ERROR downloading {model_id}: {exc}\n")


def main() -> None:
    """Parse CLI arguments and download all benchmark models."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model checkpoints for FlashSpec benchmarks",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which models would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model IDs to download (default: all benchmark models).",
    )
    args = parser.parse_args()

    token = _check_hf_token() if not args.dry_run else "dry-run-no-token-needed"

    models = args.models if args.models else BENCHMARK_MODELS

    sys.stdout.write(
        f"Downloading {len(models)} model(s) to HuggingFace cache "
        f"({'dry-run' if args.dry_run else 'live'}):\n"
    )
    for model_id in models:
        download_model(model_id, token, dry_run=args.dry_run)

    sys.stdout.write("Done.\n")


if __name__ == "__main__":
    main()
