"""Thompson sampling bandit selector with Beta conjugate prior.

Uses a Beta(alpha, beta) prior on the acceptance rate for each arm.
The posterior is updated via conjugate Beta-Binomial updates.
"""

from __future__ import annotations

import numpy as np

from flashspec.bandit.base import ArmStats, DraftSelector
from flashspec.utils.logging import get_logger
from typing import Any

__all__ = ["ThompsonSelector"]

logger = get_logger(__name__)


class ThompsonSelector(DraftSelector):
    """Thompson sampling bandit selector for draft-model selection.

    Maintains a Beta(alpha_k, beta_k) posterior for each arm's acceptance
    rate.  At each round, one sample is drawn from each posterior and the
    arm with the highest sample is selected.

    Parameters
    ----------
    n_arms : int
        Number of draft-model arms.
    window_size : int
        Rolling window for acceptance statistics.  0 disables windowing.
    prior_alpha : float
        Initial alpha parameter of the Beta prior (successes + 1).
    prior_beta : float
        Initial beta parameter of the Beta prior (failures + 1).

    Raises
    ------
    ValueError
        If ``prior_alpha`` <= 0 or ``prior_beta`` <= 0.

    Notes
    -----
    The posterior parameters after ``n`` rounds with ``s`` successes are:

    .. math::

        \\alpha_k = \\alpha_0 + s_k, \\quad \\beta_k = \\beta_0 + (n_k - s_k)

    References
    ----------
    .. [1] Thompson (1933), "On the likelihood that one unknown probability
       exceeds another in view of the evidence of two samples."
    .. [2] Chapelle & Li (2011), "An Empirical Evaluation of Thompson
       Sampling", NeurIPS 2011.

    Examples
    --------
    >>> selector = ThompsonSelector(n_arms=3, prior_alpha=1.0, prior_beta=1.0)
    >>> arm = selector.select()
    >>> selector.update(arm, accepted=1)
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 500,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        if prior_alpha <= 0.0:
            raise ValueError(f"prior_alpha must be > 0; got {prior_alpha}.")
        if prior_beta <= 0.0:
            raise ValueError(f"prior_beta must be > 0; got {prior_beta}.")
        super().__init__(n_arms=n_arms, window_size=window_size)
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        # Posterior parameters (shape: n_arms).
        self._post_alpha: list[float] = [prior_alpha] * n_arms
        self._post_beta: list[float] = [prior_beta] * n_arms

    def select(self) -> int:
        """Sample from each arm's Beta posterior and return the argmax.

        Returns
        -------
        int
            Arm index in ``[0, n_arms)``.

        Notes
        -----
        One sample is drawn from Beta(α_k, β_k) for each arm k.  The arm
        with the largest sample is returned.  This exploration-exploitation
        trade-off naturally scales with posterior uncertainty: arms with fewer
        pulls have wider Beta distributions and are therefore more likely to
        be explored.

        Examples
        --------
        >>> arm = selector.select()
        >>> assert 0 <= arm < selector.n_arms
        """
        with self._lock:
            samples = [
                np.random.beta(self._post_alpha[k], self._post_beta[k])
                for k in range(self._n_arms)
            ]
            return int(np.argmax(samples))

    def update(self, arm: int, accepted: int) -> None:
        """Update the Beta posterior for ``arm`` and record the outcome.

        Parameters
        ----------
        arm : int
            Arm index that was pulled.
        accepted : int
            1 if the token was accepted, 0 otherwise.

        Raises
        ------
        ValueError
            If ``arm`` is not in ``[0, n_arms)``.

        Notes
        -----
        Conjugate Beta-Binomial update: α_k += accepted, β_k += (1 - accepted).
        The global round counter ``t`` is incremented and the arm's raw
        statistics are forwarded to :meth:`ArmStats.record`.

        Examples
        --------
        >>> selector.update(0, accepted=1)
        """
        if not (0 <= arm < self._n_arms):
            raise ValueError(f"arm must be in [0, {self._n_arms}); got {arm}.")
        with self._lock:
            self._arms[arm].record(accepted)
            self._post_alpha[arm] += float(accepted)
            self._post_beta[arm] += float(1 - accepted)
            self._t += 1
        logger.debug(
            "Thompson update",
            extra={
                "arm": arm,
                "accepted": accepted,
                "post_alpha": self._post_alpha[arm],
                "post_beta": self._post_beta[arm],
            },
        )

    def _state_dict(self) -> dict[str, Any]:
        return {
            "type": "thompson",
            "n_arms": self._n_arms,
            "window_size": self._window_size,
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "post_alpha": self._post_alpha,
            "post_beta": self._post_beta,
            "t": self._t,
            "arms": [a.to_dict() for a in self._arms],
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> "ThompsonSelector":
        obj = cls(
            n_arms=state["n_arms"],
            window_size=state["window_size"],
            prior_alpha=state["prior_alpha"],
            prior_beta=state["prior_beta"],
        )
        obj._t = state["t"]
        obj._post_alpha = state["post_alpha"]
        obj._post_beta = state["post_beta"]
        obj._arms = [ArmStats.from_dict(d) for d in state["arms"]]
        return obj
