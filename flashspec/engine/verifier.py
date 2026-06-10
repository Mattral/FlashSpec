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
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Score all gamma draft positions in a single forward pass.

        Concatenates ``draft_token_ids`` to ``input_ids`` and runs one
        forward pass.  Temperature is applied to the raw logits **before**
        ``log_softmax`` as required by §7 of AGENTS.md.

        Parameters
        ----------
        input_ids : torch.Tensor
            Context tokens.  Shape: ``(batch_size, seq_len)``, dtype int64.
        draft_token_ids : torch.Tensor
            Draft token IDs.  Shape: ``(batch_size, gamma)``, dtype int64.
        gamma : int
            Number of draft positions.  Must equal ``draft_token_ids.shape[1]``.
        temperature : float
            Sampling temperature.  Applied as ``logits / temperature`` **before**
            ``log_softmax``.  Must be > 0.  Default 1.0 (no scaling).

        Returns
        -------
        torch.Tensor
            Log-probabilities.  Shape: ``(batch_size, gamma, vocab_size)``,
            dtype float32.  Temperature scaling has already been applied.

        Raises
        ------
        ValueError
            If ``gamma`` does not match ``draft_token_ids.shape[1]``, or if
            ``temperature`` ≤ 0.
        RuntimeError
            If tensors are not on the model's device.

        Notes
        -----
        Temperature scaling must be applied before ``log_softmax``, not after
        (AGENTS.md §7).  This means the returned log-probs are
        ``log_softmax(logits / temperature)``, **not**
        ``log_softmax(logits) / temperature``.

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
        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0; got {temperature}."
            )
        input_ids = input_ids.to(self._device)
        draft_token_ids = draft_token_ids.to(self._device)

        # Concatenate context + draft tokens for a single forward pass.
        full_ids = torch.cat([input_ids, draft_token_ids], dim=1)

        with torch.no_grad():
            outputs = self._model(full_ids)

        logits = outputs.logits  # (B, context_len + gamma, vocab_size)

        # Causal offset: logits at position i predict token i+1.
        context_len = input_ids.shape[1]
        draft_logits = logits[
            :, context_len - 1 : context_len + gamma - 1, :
        ].float()  # (B, gamma, V)

        # §7: Apply temperature BEFORE log_softmax — never after.
        if temperature != 1.0:
            draft_logits = draft_logits / temperature

        return torch.log_softmax(draft_logits, dim=-1)

    @property
    def device(self) -> torch.device:
        """Device the model resides on.

        Returns
        -------
        torch.device
            The device passed to the constructor (e.g. ``cuda:0``).

        Notes
        -----
        All tensors passed to :meth:`score_draft` are moved to this device
        automatically; callers do not need to pre-move inputs.

        Examples
        --------
        >>> target.device
        device(type='cuda', index=0)
        """
        return self._device

    @property
    def vocab_size(self) -> int:
        """Vocabulary size of the target model.

        Returns
        -------
        int
            Number of tokens in the model's vocabulary, as reported by
            ``model.config.vocab_size``.

        Notes
        -----
        Used by the engine to validate that draft and target models share the
        same vocabulary before running speculative decoding.

        Examples
        --------
        >>> target.vocab_size
        32000
        """
        return self._model.config.vocab_size  # type: ignore[attr-defined]
