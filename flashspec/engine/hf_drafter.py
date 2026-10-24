"""HuggingFace causal language models as speculative drafters.

One class covers every registered draft model that is a standard
``AutoModelForCausalLM``. Pass an already-built ``model`` to skip the
download; tests and local toy runs use that path.
"""

from __future__ import annotations

from typing import Any

import torch

from flashspec.engine.kv_cache import CacheUnavailableError, KvSession
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
        else:
            self._model = self._load_pretrained(model_name_or_path)
        self._kv = KvSession(self._model)
        # Incremental mode: draft tokens forwarded by the last draft() (all
        # but the last), and the last one, which was not forwarded.
        self._forwarded_drafts = 0
        self._unforwarded: torch.Tensor | None = None

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
        if self._kv.supports(context):
            try:
                next_logits = self._kv.sync(context)
            except CacheUnavailableError:
                self._kv.disable()
            else:
                return self._sample_cached(next_logits, gamma)
        return self._sample_uncached(context, gamma)

    # ── Incremental batch-1 path (gamma forwards per step) ──────────────

    def begin(self, prompt: torch.Tensor) -> bool:
        """Prefill ``prompt`` for incremental drafting.

        Parameters
        ----------
        prompt : torch.Tensor
            The accepted sequence so far. Shape ``(1, seq)``, int64.

        Returns
        -------
        bool
            True when :meth:`draft`, :meth:`commit` and
            :meth:`push_pending` can be used. False when the model has no
            usable cache or the batch is not one; the caller then uses
            :meth:`generate_draft`.
        """
        prompt = prompt.to(self._device)
        if not self._kv.supports(prompt):
            return False
        try:
            self._kv.start(prompt)
        except CacheUnavailableError:
            self._kv.disable()
            return False
        self._forwarded_drafts = 0
        self._unforwarded = None
        return True

    def draft(self, gamma: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``gamma`` tokens with ``gamma`` forwards.

        The first forward carries the pending tokens. Each later forward
        carries the previous draft token. The last draft token is not
        forwarded: if it is accepted, :meth:`commit` queues it in front of
        the next token, and if it is rejected there is nothing to undo.

        Parameters
        ----------
        gamma : int
            Number of draft tokens. Must be ``>= 1``.

        Returns
        -------
        draft_token_ids : torch.Tensor
            Shape ``(1, gamma)``, dtype int64.
        draft_logprobs : torch.Tensor
            Temperature-scaled log-probs. Shape ``(1, gamma, vocab)``.

        Raises
        ------
        ValueError
            If ``gamma < 1``.
        CacheUnavailableError
            If the cache cannot be extended. The session is disabled.
        """
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1; got {gamma}.")
        token_steps: list[torch.Tensor] = []
        logprob_steps: list[torch.Tensor] = []
        next_ids: torch.Tensor | None = None
        for _ in range(gamma):
            next_logits = self._kv.advance(next_ids, keep=1)[:, -1, :]
            step_logprobs = scaled_log_softmax(next_logits, self._temperature)
            next_ids = torch.multinomial(step_logprobs.exp(), num_samples=1)
            token_steps.append(next_ids)
            logprob_steps.append(step_logprobs.unsqueeze(1))
        self._forwarded_drafts = gamma - 1
        self._unforwarded = next_ids
        return torch.cat(token_steps, dim=1), torch.cat(logprob_steps, dim=1)

    def commit(self, n_accepted: int, next_token: torch.Tensor) -> None:
        """Keep ``n_accepted`` draft tokens and queue ``next_token``. Never forwards.

        Parameters
        ----------
        n_accepted : int
            Accepted tokens from the last :meth:`draft`, in ``[0, gamma]``.
        next_token : torch.Tensor
            The residual or bonus token. Shape ``(1, 1)``.

        Raises
        ------
        ValueError
            If ``n_accepted`` is outside ``[0, gamma]``.
        CacheUnavailableError
            If the cache cannot be cropped.
        """
        gamma = self._forwarded_drafts + 1
        if self._unforwarded is None or not 0 <= n_accepted <= gamma:
            raise ValueError(f"n_accepted must be in [0, {gamma}] after draft(); got {n_accepted}.")
        next_token = next_token.to(self._device).view(1, -1)
        if n_accepted == gamma:
            # Every draft token was accepted. The last one was never
            # forwarded, so it goes in front of the bonus token.
            self._kv.push_pending(torch.cat([self._unforwarded, next_token], dim=1))
        else:
            self._kv.crop(self._forwarded_drafts - n_accepted)
            self._kv.push_pending(next_token)
        self._forwarded_drafts = 0
        self._unforwarded = None

    def push_pending(self, token_ids: torch.Tensor) -> None:
        """Queue tokens another drafter's step committed. Never forwards.

        The engine calls this on drafters the bandit did not pick, so
        their caches catch up in one forward the next time they draft.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(1, n)``.
        """
        self._kv.push_pending(token_ids.to(self._device).view(1, -1))

    def check_cache(self, context: torch.Tensor) -> None:
        """Assert the incremental cache matches ``context`` (``FLASHSPEC_DEBUG_CACHE=1`` only)."""
        self._kv.check(context.to(self._device))

    def roll_cache_to(self, token_ids: torch.Tensor) -> None:
        """Crop or extend the cache until it matches ``token_ids``.

        The engine calls this with the accepted prefix after each step,
        including a bonus token when one was appended. Rejected draft
        tokens are dropped. A batch other than one is ignored.

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

    def _sample_cached(
        self,
        next_logits: torch.Tensor,
        gamma: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``gamma`` tokens, forwarding one new token at a time."""
        token_steps: list[torch.Tensor] = []
        logprob_steps: list[torch.Tensor] = []
        for _ in range(gamma):
            step_logprobs = scaled_log_softmax(next_logits, self._temperature)
            next_id = torch.multinomial(step_logprobs.exp(), num_samples=1)
            token_steps.append(next_id)
            logprob_steps.append(step_logprobs.unsqueeze(1))
            next_logits = self._kv.append(next_id)[:, -1, :]
        return torch.cat(token_steps, dim=1), torch.cat(logprob_steps, dim=1)

    def _sample_uncached(
        self,
        context: torch.Tensor,
        gamma: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``gamma`` tokens, forwarding the growing context each step."""
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
        if self._kv.supports(input_ids):
            try:
                first = self._kv.sync(input_ids)
                step = self._kv.append(draft_token_ids)
            except CacheUnavailableError:
                self._kv.disable()
            else:
                # ``first`` predicts the first draft token. ``step`` starts
                # one position later, so its last row is not part of the
                # gamma draft scores.
                rows = torch.cat([first.unsqueeze(1), step[:, :-1, :]], dim=1)
                return scaled_log_softmax(rows, self._temperature)
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
