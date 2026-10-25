"""Thompson sampling bandit selector.

Two posteriors, chosen by what the caller reports:

- **Reward** (what the engine passes: emitted tokens per step second). A
  Gaussian posterior on each arm's mean reward,
  ``N(mean_k, var_k / n_k)``, with ``var_k`` the arm's sample variance
  (the variance pooled over all arms while an arm has fewer than two
  observations). The draw is invariant to the reward's unit.
- **Accept count only** (``reward=None``). The original Beta(alpha, beta)
  posterior on the acceptance rate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flashspec.bandit.base import ArmStats, DraftSelector
from flashspec.utils.logging import get_logger

__all__ = ["ThompsonSelector"]

logger = get_logger(__name__)


class ThompsonSelector(DraftSelector):
    r"""Thompson sampling bandit selector for draft-model selection.

    With rewards (the engine's tokens per second), each round draws one
    sample per arm from a Gaussian posterior on its mean reward and picks
    the largest:

    .. math::

        \tilde{\mu}_k \sim \mathcal{N}\!\left(\hat{\mu}_k,\ \hat{\sigma}_k^2 / n_k\right)

    Arms not yet pulled are tried first, in index order. ``n_k``,
    :math:`\hat{\mu}_k` and :math:`\hat{\sigma}_k^2` follow the arm's
    window when windowing is on.

    Without rewards (``update(arm, accepted)`` only), it keeps a
    Beta(alpha_k, beta_k) posterior on the acceptance rate:
    ``alpha += accepted`` and ``beta += max(0, 1 - accepted)``.

    Parameters
    ----------
    n_arms : int
        Number of draft-model arms.
    window_size : int
        Rolling window for statistics.  0 disables windowing.
    prior_alpha : float
        Initial alpha parameter of the Beta prior (successes + 1).
    prior_beta : float
        Initial beta parameter of the Beta prior (failures + 1).
    seed : int or None
        Seed for this selector's own generator. ``None`` uses NumPy's
        global generator, as before.

    Raises
    ------
    ValueError
        If ``prior_alpha`` <= 0 or ``prior_beta`` <= 0.

    References
    ----------
    .. [1] Thompson (1933), "On the likelihood that one unknown probability
       exceeds another in view of the evidence of two samples."
    .. [2] Chapelle & Li (2011), "An Empirical Evaluation of Thompson
       Sampling", NeurIPS 2011.

    Examples
    --------
    >>> selector = ThompsonSelector(n_arms=3, seed=0)
    >>> arm = selector.select()
    >>> selector.update(arm, accepted=3, reward=41.5)
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 500,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if prior_alpha <= 0.0:
            raise ValueError(f"prior_alpha must be > 0; got {prior_alpha}.")
        if prior_beta <= 0.0:
            raise ValueError(f"prior_beta must be > 0; got {prior_beta}.")
        super().__init__(n_arms=n_arms, window_size=window_size)
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._seed = seed
        self._rng: Any = np.random.default_rng(seed) if seed is not None else np.random
        # Posterior parameters (shape: n_arms).
        self._post_alpha: list[float] = [prior_alpha] * n_arms
        self._post_beta: list[float] = [prior_beta] * n_arms
        # True once any update carried a reward: select on the reward posterior.
        self._reward_mode = False

    def select(self) -> int:
        """Draw one posterior sample per arm and return the argmax.

        Returns
        -------
        int
            Arm index in ``[0, n_arms)``.
        """
        with self._lock:
            if not self._reward_mode:
                samples = [
                    float(self._rng.beta(self._post_alpha[k], self._post_beta[k]))
                    for k in range(self._n_arms)
                ]
                return int(np.argmax(samples))
            for k, arm in enumerate(self._arms):
                if arm.n_pulls == 0:
                    return k
            pooled = self._pooled_variance()
            draws: list[float] = []
            for arm in self._arms:
                count = max(arm.reward_count, 1)
                variance = arm.reward_variance if arm.reward_count >= 2 else pooled
                spread = math.sqrt(max(variance, 0.0) / count)
                draws.append(arm.mean_reward + spread * float(self._rng.standard_normal()))
            return int(np.argmax(draws))

    def _pooled_variance(self) -> float:
        """Variance pooled over arms with two or more observations; 0 if none."""
        numerator = 0.0
        dof = 0
        for arm in self._arms:
            if arm.reward_count >= 2:
                numerator += arm.reward_variance * (arm.reward_count - 1)
                dof += arm.reward_count - 1
        return numerator / dof if dof else 0.0

    def update(self, arm: int, accepted: int, reward: float | None = None) -> None:
        """Record one round for ``arm``.

        Parameters
        ----------
        arm : int
            Arm index that was pulled.
        accepted : int
            Accept count. Updates the Beta posterior and is stored raw.
        reward : float or None
            Selection payoff. Once any reward is given, selection uses the
            reward posterior.

        Raises
        ------
        ValueError
            If ``arm`` is not in ``[0, n_arms)``.
        """
        if not (0 <= arm < self._n_arms):
            raise ValueError(f"arm must be in [0, {self._n_arms}); got {arm}.")
        with self._lock:
            self._arms[arm].record(accepted, reward=reward)
            self._post_alpha[arm] += float(accepted)
            # An accept count above 1 adds successes only; beta must stay > 0.
            self._post_beta[arm] += float(max(0, 1 - accepted))
            if reward is not None:
                self._reward_mode = True
            self._t += 1
        logger.debug(
            "Thompson update",
            extra={
                "arm": arm,
                "accepted": accepted,
                "reward": reward,
                "post_alpha": self._post_alpha[arm],
                "post_beta": self._post_beta[arm],
            },
        )

    def reset(self) -> None:
        """Reset statistics, posteriors and the round counter."""
        super().reset()
        with self._lock:
            self._post_alpha = [self._prior_alpha] * self._n_arms
            self._post_beta = [self._prior_beta] * self._n_arms
            self._reward_mode = False

    def _state_dict(self) -> dict[str, Any]:
        return {
            "type": "thompson",
            "n_arms": self._n_arms,
            "window_size": self._window_size,
            "prior_alpha": self._prior_alpha,
            "prior_beta": self._prior_beta,
            "seed": self._seed,
            "post_alpha": self._post_alpha,
            "post_beta": self._post_beta,
            "reward_mode": self._reward_mode,
            "t": self._t,
            "arms": [a.to_dict() for a in self._arms],
        }

    @classmethod
    def _from_state_dict(cls, state: dict[str, Any]) -> ThompsonSelector:
        obj = cls(
            n_arms=state["n_arms"],
            window_size=state["window_size"],
            prior_alpha=state["prior_alpha"],
            prior_beta=state["prior_beta"],
            seed=state.get("seed"),
        )
        obj._t = state["t"]
        obj._post_alpha = state["post_alpha"]
        obj._post_beta = state["post_beta"]
        obj._reward_mode = bool(state.get("reward_mode", False))
        obj._arms = [ArmStats.from_dict(d) for d in state["arms"]]
        return obj
