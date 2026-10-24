"""TargetModel wrapper for FlashSpec.

Wraps a HuggingFace causal language model as the verifier (target) model.
The wrapper handles batched parallel scoring of all gamma draft positions
in a single forward pass, and returns log-softmax probabilities.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM  # type: ignore[import-untyped]

from flashspec.engine.kv_cache import CacheUnavailableError, KvSession
from flashspec.utils.logging import get_logger

__all__ = ["TargetModel"]

logger = get_logger(__name__)


class TargetModel:
    """Wrapper around a HuggingFace CausalLM used as the speculative verifier.

    The target model scores all ``gamma`` draft positions in a single forward
    pass by appending the draft tokens to the context and extracting the
    logits at each draft position.

    Two interfaces are offered. :meth:`score_draft` takes the full context
    on every call. The incremental batch-1 interface, :meth:`begin`,
    :meth:`verify` (or :meth:`verify_greedy`) and :meth:`commit`, is what
    the engine uses: each step
    is exactly one forward of the pending token plus the draft, and
    committing the accepted prefix never runs the model.

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
        model: Any | None = None,  # noqa: ANN401 — injected modules are duck-typed
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._name = model_name_or_path
        if model is not None:
            self._model = model.to(device)
            self._model.eval()
            logger.debug(
                "Using injected target module",
                extra={"model": model_name_or_path, "device": str(device)},
            )
        else:
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
        self._kv = KvSession(self._model)
        self._verified = 0

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
        include_bonus: bool = False,
    ) -> torch.Tensor:
        """Score all gamma draft positions in a single forward pass.

        For a batch of one, the prefix is reused from ``past_key_values``
        and only the draft tokens are forwarded. Any other batch
        concatenates ``draft_token_ids`` to ``input_ids`` and runs one
        full forward. Temperature is applied to the raw logits **before**
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
        include_bonus : bool
            When true, also return the distribution of the token after the
            last draft token. The shape's middle dimension is then
            ``gamma + 1``. The engine samples that row only on a full accept
            when ``max_new_tokens`` still has room.

        Returns
        -------
        torch.Tensor
            Log-probabilities.  Shape: ``(batch_size, gamma, vocab_size)``,
            or ``(batch_size, gamma + 1, vocab_size)`` when ``include_bonus``
            is true.  Dtype float32.  Temperature scaling has already been
            applied.

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
                f"gamma={gamma} must match draft_token_ids.shape[1]={draft_token_ids.shape[1]}."
            )
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        input_ids = input_ids.to(self._device)
        draft_token_ids = draft_token_ids.to(self._device)
        if self._kv.supports(input_ids):
            try:
                scored = self._score_cached(
                    input_ids,
                    draft_token_ids,
                    temperature=temperature,
                    include_bonus=include_bonus,
                )
            except CacheUnavailableError:
                self._kv.disable()
            else:
                return scored

        # Concatenate context + draft tokens for a single forward pass.
        full_ids = torch.cat([input_ids, draft_token_ids], dim=1)

        with torch.no_grad():
            outputs = self._model(full_ids)

        logits = outputs.logits  # (B, context_len + gamma, vocab_size)

        # Causal offset: logits at position i predict token i+1.
        context_len = input_ids.shape[1]
        # Position context_len-1 predicts the first draft token. The slice of
        # length gamma stops one short of the logit at the last draft token,
        # which predicts the bonus token. include_bonus keeps that row.
        end = context_len + gamma if include_bonus else context_len + gamma - 1
        draft_logits = logits[:, context_len - 1 : end, :].float()

        # §7: Apply temperature BEFORE log_softmax — never after.
        if temperature != 1.0:
            draft_logits = draft_logits / temperature

        return torch.log_softmax(draft_logits, dim=-1)

    # ── Incremental batch-1 path (one forward per step) ─────────────────

    def begin(self, prompt: torch.Tensor) -> bool:
        """Prefill ``prompt`` for incremental verification.

        Parameters
        ----------
        prompt : torch.Tensor
            Shape ``(1, seq)``, int64.

        Returns
        -------
        bool
            True when :meth:`verify` and :meth:`commit` can be used. False
            when the model has no usable cache or the batch is not one;
            the caller then uses :meth:`score_draft`.
        """
        prompt = prompt.to(self._device)
        if not self._kv.supports(prompt):
            return False
        try:
            self._kv.start(prompt)
        except CacheUnavailableError:
            self._kv.disable()
            return False
        self._verified = 0
        return True

    def verify(self, draft_token_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Score a draft with one forward of ``pending + draft`` tokens.

        Parameters
        ----------
        draft_token_ids : torch.Tensor
            Shape ``(1, gamma)``, int64.
        temperature : float
            Applied to the raw logits **before** ``log_softmax`` (§7). Must
            be > 0.

        Returns
        -------
        torch.Tensor
            Log-probabilities. Shape ``(1, gamma + 1, vocab)``, float32.
            Row ``i < gamma`` scores draft token ``i``; row ``gamma`` is
            the bonus distribution after a full accept.

        Raises
        ------
        ValueError
            If ``temperature`` <= 0.
        CacheUnavailableError
            If the cache cannot be extended. The session is disabled; the
            caller restarts with :meth:`score_draft`.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        draft_token_ids = draft_token_ids.to(self._device)
        gamma = int(draft_token_ids.shape[1])
        # The first kept row is the last pending token, which predicts the
        # first draft token. The last row predicts the bonus token.
        rows = self._kv.advance(draft_token_ids, keep=gamma + 1)
        self._verified = gamma
        if temperature != 1.0:
            rows = rows / temperature
        return torch.log_softmax(rows, dim=-1)

    def verify_greedy(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Greedy verify: the same single forward as :meth:`verify`, argmax only.

        No softmax is computed. Rows are the target's argmax at each draft
        position and at the bonus position.

        Parameters
        ----------
        draft_token_ids : torch.Tensor
            Shape ``(1, gamma)``, int64.

        Returns
        -------
        torch.Tensor
            Target argmax ids. Shape ``(1, gamma + 1)``, int64.

        Raises
        ------
        CacheUnavailableError
            If the cache cannot be extended. The session is disabled.
        """
        draft_token_ids = draft_token_ids.to(self._device)
        gamma = int(draft_token_ids.shape[1])
        rows = self._kv.advance(draft_token_ids, keep=gamma + 1)
        self._verified = gamma
        return rows.argmax(dim=-1)

    def commit(self, n_accepted: int, next_token: torch.Tensor) -> None:
        """Keep ``n_accepted`` draft tokens and queue ``next_token``. Never forwards.

        Parameters
        ----------
        n_accepted : int
            Accepted draft tokens from the last :meth:`verify`, in
            ``[0, gamma]``.
        next_token : torch.Tensor
            The residual or bonus token. Shape ``(1, 1)``.

        Raises
        ------
        ValueError
            If ``n_accepted`` is outside ``[0, gamma]``.
        CacheUnavailableError
            If the cache cannot be cropped.
        """
        if not 0 <= n_accepted <= self._verified:
            raise ValueError(f"n_accepted must be in [0, {self._verified}]; got {n_accepted}.")
        self._kv.crop(self._verified - n_accepted)
        self._kv.push_pending(next_token.to(self._device).view(1, -1))
        self._verified = 0

    def check_cache(self, context: torch.Tensor) -> None:
        """Assert the incremental cache matches ``context`` (``FLASHSPEC_DEBUG_CACHE=1`` only)."""
        self._kv.check(context.to(self._device))

    def roll_cache_to(self, token_ids: torch.Tensor) -> None:
        """Crop or extend the cache until it matches ``token_ids``.

        The engine calls this with the accepted prefix after each step.
        A bonus token that was emitted is included. Rejected draft tokens
        are dropped. A batch other than one is ignored.

        Parameters
        ----------
        token_ids : torch.Tensor
            Accepted token ids. Shape ``(1, seq)`` or ``(batch, seq)``.
        """
        token_ids = token_ids.to(self._device)
        if not self._kv.supports(token_ids):
            return
        try:
            self._kv.sync(token_ids)
        except CacheUnavailableError:
            self._kv.disable()

    def _score_cached(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        temperature: float,
        include_bonus: bool,
    ) -> torch.Tensor:
        """Score a draft from the cached prefix.

        ``sync`` returns the logits that predict the first draft token.
        Forwarding the draft then returns one row per draft token, and
        the last of those rows predicts the bonus token.
        """
        first = self._kv.sync(input_ids)
        step = self._kv.append(draft_token_ids)
        if include_bonus:
            rows = torch.cat([first.unsqueeze(1), step], dim=1)
        else:
            rows = torch.cat([first.unsqueeze(1), step[:, :-1, :]], dim=1)
        if temperature != 1.0:
            rows = rows / temperature
        return torch.log_softmax(rows, dim=-1)

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
        config = getattr(self._model, "config", None)
        size = getattr(config, "vocab_size", None) if config is not None else None
        if not isinstance(size, int):
            size = getattr(self._model, "vocab_size", None)
        if not isinstance(size, int):
            raise RuntimeError(
                f"Target model '{self._name}' does not expose an integer vocab_size."
            )
        return size
