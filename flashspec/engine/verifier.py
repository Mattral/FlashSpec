"""TargetModel wrapper for FlashSpec.

Wraps a HuggingFace causal language model as the verifier (target) model.
The wrapper handles batched parallel scoring of all gamma draft positions
in a single forward pass, and returns log-softmax probabilities.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

from flashspec.utils.logging import get_logger

__all__ = ["TargetModel"]

logger = get_logger(__name__)


class TargetModel:
    """Wrapper around a HuggingFace CausalLM used as the speculative verifier.

    The target model scores all ``gamma`` draft positions in a single forward
    pass by appending the draft tokens to the context and extracting the
    logits at each draft position.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
    device : torch.device
        Device to load the model onto.
    dtype : torch.dtype
        Compute dtype (``torch.float32``, ``torch.bfloat16``, or
        ``torch.float16``).

    Raises
    ------
    RuntimeError
        If the model cannot be loaded from ``model_name_or_path``.

    Notes
    -----
    Model weights are never committed to the repository (AGENTS.md §15).
    Weights are fetched at runtime via HuggingFace Hub or a local path.

    Examples
    --------
    >>> target = TargetModel("meta-llama/Llama-3-8B-Instruct", device, torch.bfloat16)
    >>> logprobs = target.score_draft(input_ids, draft_token_ids, gamma=4)
    >>> logprobs.shape
    torch.Size([1, 4, 32000])
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._device = device
        self._dtype = dtype
        logger.debug(
            "Loading target model",
            extra={"model": model_name_or_path, "device": str(device), "dtype": str(dtype)},
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                device_map=str(device),
            )
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load target model '{model_name_or_path}': {exc}"
            ) from exc
        logger.debug("Target model loaded", extra={"model": model_name_or_path})

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
    ) -> torch.Tensor:
        """Score all gamma draft positions in a single forward pass.

        Concatenates ``draft_token_ids`` to ``input_ids`` and runs one
        forward pass.  Returns log-softmax probabilities at each draft
        position (offset by 1 due to causal shift).

        Parameters
        ----------
        input_ids : torch.Tensor
            Context tokens.  Shape: ``(batch_size, seq_len)``, dtype int64.
        draft_token_ids : torch.Tensor
            Draft token IDs.  Shape: ``(batch_size, gamma)``, dtype int64.
        gamma : int
            Number of draft positions.  Must equal ``draft_token_ids.shape[1]``.

        Returns
        -------
        torch.Tensor
            Log-probabilities.  Shape: ``(batch_size, gamma, vocab_size)``,
            dtype float32.

        Raises
        ------
        ValueError
            If ``gamma`` does not match ``draft_token_ids.shape[1]``.
        RuntimeError
            If tensors are not on the model's device.

        Examples
        --------
        >>> logprobs = target.score_draft(input_ids, draft_token_ids, gamma=4)
        >>> logprobs.shape
        torch.Size([1, 4, 32000])
        """
        if gamma != draft_token_ids.shape[1]:
            raise ValueError(
                f"gamma={gamma} must match draft_token_ids.shape[1]="
                f"{draft_token_ids.shape[1]}."
            )
        input_ids = input_ids.to(self._device)
        draft_token_ids = draft_token_ids.to(self._device)

        # Concatenate context + draft tokens for a single forward pass.
        full_ids = torch.cat([input_ids, draft_token_ids], dim=1)

        with torch.no_grad():
            outputs = self._model(full_ids)

        logits = outputs.logits  # (B, context_len + gamma, vocab_size)

        # Causal offset: logits at position i predict token i+1.
        # Extract logits at positions (context_len - 1) .. (context_len + gamma - 2).
        context_len = input_ids.shape[1]
        draft_logits = logits[:, context_len - 1 : context_len + gamma - 1, :]  # (B, gamma, V)

        return torch.log_softmax(draft_logits.float(), dim=-1)

    @property
    def device(self) -> torch.device:
        """Device the model resides on.

        Returns
        -------
        torch.device
        """
        return self._device

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the target model.

        Returns
        -------
        int
        """
        return self._model.config.vocab_size  # type: ignore[attr-defined]
