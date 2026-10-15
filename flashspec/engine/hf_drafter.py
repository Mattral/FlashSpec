"""HuggingFace causal language models as speculative drafters.

One class covers every registered draft model that is a standard
``AutoModelForCausalLM``. Pass an already-built ``model`` to skip the
download; tests and local toy runs use that path.
"""

from __future__ import annotations

from typing import Any

import torch

from flashspec.utils.logging import get_logger

__all__ = ["HuggingFaceDrafter", "scaled_log_softmax"]

logger = get_logger(__name__)


def scaled_log_softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature, then log-softmax.

    Temperature is applied to the raw logits **before** ``log_softmax``
    (AGENTS.md §7). Dividing log-probabilities by temperature is not
    the same operation and is not used here.

    Parameters
    ----------
    logits : torch.Tensor
        Raw next-token logits. The last dimension is the vocabulary.
    temperature : float
        Positive sampling temperature. ``1.0`` leaves logits unchanged.

    Returns
    -------
    torch.Tensor
        Temperature-scaled log-probabilities, same shape as ``logits``.

    Raises
    ------
    ValueError
        If ``temperature`` is not positive.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature}.")
    scaled = logits if temperature == 1.0 else logits / temperature
    return torch.log_softmax(scaled, dim=-1)


class HuggingFaceDrafter:
    """Draft model backed by a HuggingFace causal LM.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace hub id or a local directory. Recorded even when
        ``model`` is injected, so logs name the drafter.
    device : torch.device or str
        Device the model is moved to.
    dtype : torch.dtype
        Compute dtype used when loading from the hub. Injected modules
        are not recast; only their device is changed.
    temperature : float
        Sampling temperature applied before log-softmax. Default ``1.0``.
    model : nn.Module or None
        Pre-built module with a ``forward`` that returns ``.logits``.
        ``None`` loads ``model_name_or_path`` via ``transformers``.

    Raises
    ------
    RuntimeError
        If hub loading fails or ``transformers`` is not installed.
    ValueError
        If ``temperature`` is not positive.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        temperature: float = 1.0,
        model: Any | None = None,  # noqa: ANN401 — injected modules are duck-typed
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        self._name = model_name_or_path
        self._device = torch.device(device)
        self._dtype = dtype
        self._temperature = temperature
        if model is not None:
            self._model = model.to(self._device)
            self._model.eval()
            return
        self._model = self._load_pretrained(model_name_or_path)

    def _load_pretrained(self, model_name_or_path: str) -> Any:  # noqa: ANN401
        """Load a causal LM from the HuggingFace hub or a local path.

        Parameters
        ----------
        model_name_or_path : str
            Hub id or local directory.

        Returns
        -------
        Any
            A model in eval mode on ``self._device``.

        Raises
        ------
        RuntimeError
            If ``transformers`` is missing or ``from_pretrained`` fails.
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("transformers is required to load HuggingFace drafters.") from exc
        logger.info(
            "Loading draft model",
            extra={"model": model_name_or_path, "device": str(self._device)},
        )
        try:
            loaded = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=self._dtype,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load draft model '{model_name_or_path}': {exc}") from exc
        loaded = loaded.to(self._device)
        loaded.eval()
        return loaded

    @property
    def model_name(self) -> str:
        """Hub id or path this drafter was constructed with.

        Returns
        -------
        str
            The ``model_name_or_path`` argument.
        """
        return self._name

    @property
    def vocab_size(self) -> int:
        """Vocabulary size reported by the wrapped model.

        Returns
        -------
        int
            ``config.vocab_size`` when present, otherwise ``vocab_size``.

        Raises
        ------
        RuntimeError
            If neither attribute is an int.
        """
        config = getattr(self._model, "config", None)
        size = getattr(config, "vocab_size", None) if config is not None else None
        if not isinstance(size, int):
            size = getattr(self._model, "vocab_size", None)
        if not isinstance(size, int):
            raise RuntimeError(f"Draft model '{self._name}' does not expose an integer vocab_size.")
        return size

    def generate_draft(
        self,
        input_ids: torch.Tensor,
        gamma: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``gamma`` draft tokens and their full-vocab log-probs.

        Parameters
        ----------
        input_ids : torch.Tensor
            Context. Shape ``(batch, seq_len)``, dtype int64.
        gamma : int
            Number of draft tokens. Must be ``>= 1``.

        Returns
        -------
        draft_token_ids : torch.Tensor
            Shape ``(batch, gamma)``, dtype int64.
        draft_logprobs : torch.Tensor
            Temperature-scaled log-probs. Shape ``(batch, gamma, vocab)``.

        Raises
        ------
        ValueError
            If ``gamma < 1`` or ``input_ids`` is not 2-D.
        """
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1; got {gamma}.")
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D (batch, seq); got shape {tuple(input_ids.shape)}."
            )
        context = input_ids.to(self._device)
        token_steps: list[torch.Tensor] = []
        logprob_steps: list[torch.Tensor] = []
        for _ in range(gamma):
            step_logits = self._next_logits(context)
            step_logprobs = scaled_log_softmax(step_logits, self._temperature)
            next_id = torch.multinomial(step_logprobs.exp(), num_samples=1)
            token_steps.append(next_id)
            logprob_steps.append(step_logprobs.unsqueeze(1))
            context = torch.cat([context, next_id], dim=1)
        return torch.cat(token_steps, dim=1), torch.cat(logprob_steps, dim=1)

    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``draft_token_ids`` under this drafter in one forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Context. Shape ``(batch, seq_len)``.
        draft_token_ids : torch.Tensor
            Tokens to score. Shape ``(batch, gamma)``.

        Returns
        -------
        torch.Tensor
            Temperature-scaled log-probs. Shape ``(batch, gamma, vocab)``.
        """
        input_ids = input_ids.to(self._device)
        draft_token_ids = draft_token_ids.to(self._device)
        gamma = int(draft_token_ids.shape[1])
        full_ids = torch.cat([input_ids, draft_token_ids], dim=1)
        with torch.no_grad():
            logits = self._model(full_ids).logits
        context_len = input_ids.shape[1]
        draft_logits = logits[:, context_len - 1 : context_len + gamma - 1, :].float()
        return scaled_log_softmax(draft_logits, self._temperature)

    def _next_logits(self, context: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for ``context``.

        Parameters
        ----------
        context : torch.Tensor
            Token ids. Shape ``(batch, seq)``.

        Returns
        -------
        torch.Tensor
            Float32 logits. Shape ``(batch, vocab)``.
        """
        with torch.no_grad():
            logits = self._model(context).logits[:, -1, :]
        return logits.float()
