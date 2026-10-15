"""Controlled demonstration of online draft selection.

Two synthetic drafters disagree. The target distribution switches once,
halfway through the generation. The script times three policies on that
workload:

* fixed drafter A
* fixed drafter B
* UCB1 with both drafters and a short acceptance window

It writes the measured acceptance rates and throughputs and says whether
UCB beat the better fixed drafter. This is not a language-model
benchmark. A single drafter is included as a reminder that the bandit
then has no choice to make.

Theoretical UCB1 regret bounds are unchanged. They do not by themselves
mean a throughput win on a short conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402

from benchmarks.harness import detect_hardware, git_commit  # noqa: E402
from flashspec.bandit.ucb import UCB1Selector  # noqa: E402
from flashspec.engine.speculative import GenerationResult, SpeculativeEngine  # noqa: E402
from flashspec.utils.config import BanditConfig, FlashSpecConfig, SamplingConfig  # noqa: E402
from flashspec.utils.device import set_seed  # noqa: E402


class _PeakedDrafter:
    """Drafter that always proposes one vocabulary id."""

    def __init__(self, vocab: int, peak: int, peak_logit: float = 8.0) -> None:
        self.vocab = vocab
        self.peak = peak
        self.peak_logit = peak_logit

    def _logprobs(self, batch: int, steps: int) -> torch.Tensor:
        logits = torch.full((batch, steps, self.vocab), -8.0)
        logits[:, :, self.peak] = self.peak_logit
        return torch.log_softmax(logits, dim=-1)

    def generate_draft(
        self, input_ids: torch.Tensor, gamma: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propose ``gamma`` copies of the peak token.

        Parameters
        ----------
        input_ids : torch.Tensor
            Context. Only the batch size is used.
        gamma : int
            Speculation length.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Token ids ``(batch, gamma)`` and log-probs ``(batch, gamma, vocab)``.
        """
        batch = int(input_ids.shape[0])
        token_ids = torch.full((batch, gamma), self.peak, dtype=torch.long)
        return token_ids, self._logprobs(batch, gamma)

    def compute_logprobs(
        self, input_ids: torch.Tensor, draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """Score tokens under the same peak the drafter proposes.

        Parameters
        ----------
        input_ids : torch.Tensor
            Unused except for the batch size.
        draft_token_ids : torch.Tensor
            Shape ``(batch, gamma)``.

        Returns
        -------
        torch.Tensor
            Log-probs of shape ``(batch, gamma, vocab)``.
        """
        batch, gamma = draft_token_ids.shape
        return self._logprobs(int(batch), int(gamma))


class _SwitchingTarget:
    """Target whose peak token changes after the context grows."""

    def __init__(self, vocab: int, peak_a: int, peak_b: int, switch_after: int) -> None:
        self.vocab = vocab
        self.peak_a = peak_a
        self.peak_b = peak_b
        self.switch_after = switch_after

    def score_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
        gamma: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return a peaked distribution that flips at ``switch_after``.

        Parameters
        ----------
        input_ids : torch.Tensor
            Context. The length selects the peak.
        draft_token_ids : torch.Tensor
            Draft tokens. Only the batch and gamma dimensions are used.
        gamma : int
            Speculation length.
        temperature : float
            Applied before log-softmax. Must be positive.

        Returns
        -------
        torch.Tensor
            Log-probs of shape ``(batch, gamma, vocab)``.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        peak = self.peak_a if int(input_ids.shape[1]) < self.switch_after else self.peak_b
        batch = int(input_ids.shape[0])
        logits = torch.full((batch, gamma, self.vocab), -8.0)
        logits[:, :, peak] = 8.0
        if temperature != 1.0:
            logits = logits / temperature
        return torch.log_softmax(logits, dim=-1)


def _run_policy(
    drafters: list[Any],
    target: _SwitchingTarget,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    gamma: int,
    window_size: int,
) -> GenerationResult:
    """Generate once with the given drafter list.

    Parameters
    ----------
    drafters : list
        One entry per bandit arm.
    target : _SwitchingTarget
        Verifier.
    prompt : torch.Tensor
        Shape ``(1, seq)``.
    max_new_tokens : int
        Generation cap.
    gamma : int
        Speculation length.
    window_size : int
        UCB acceptance window. ``0`` keeps the full history.

    Returns
    -------
    GenerationResult
        Engine result for this policy.
    """
    config = FlashSpecConfig(
        device="cpu",
        dtype="float32",
        drafter_name="mixed-demo",
        target_name="switching-target",
        max_new_tokens=max_new_tokens,
        bandit=BanditConfig(n_arms=len(drafters), strategy="ucb1", window_size=window_size),
        sampling=SamplingConfig(gamma=gamma, temperature=1.0),
    )
    engine = SpeculativeEngine(
        config,
        drafters,
        target,  # type: ignore[arg-type]
        UCB1Selector(n_arms=len(drafters), window_size=window_size),
    )
    return engine.generate(prompt, max_new_tokens=max_new_tokens)


def run_demo(
    *,
    max_new_tokens: int = 24,
    gamma: int = 4,
    vocab: int = 32,
    seed: int = 0,
    window_size: int = 4,
) -> dict[str, Any]:
    """Run the mixed-workload comparison and return a measured report.

    Parameters
    ----------
    max_new_tokens : int
        Tokens requested from each policy.
    gamma : int
        Speculation length.
    vocab : int
        Synthetic vocabulary size.
    seed : int
        Torch seed.
    window_size : int
        UCB window. Short windows track the switch more quickly and also
        forget the first phase, which is the point of the demo.

    Returns
    -------
    dict
        Measured acceptance and throughput for fixed A, fixed B, UCB,
        and a one-arm control. ``improved_acceptance_vs_best_fixed`` is
        true only when UCB's acceptance is strictly higher.
    """
    torch.set_num_threads(1)
    set_seed(seed)
    prompt_length = 4
    switch_after = prompt_length + max_new_tokens // 2
    target = _SwitchingTarget(vocab, peak_a=3, peak_b=9, switch_after=switch_after)
    drafter_a = _PeakedDrafter(vocab, peak=3)
    drafter_b = _PeakedDrafter(vocab, peak=9)
    prompt = torch.randint(0, vocab, (1, prompt_length))

    def _once(drafters: list[Any]) -> GenerationResult:
        set_seed(seed)
        return _run_policy(
            drafters,
            target,
            prompt,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            window_size=window_size,
        )

    policies = {
        "fixed_a": _once([drafter_a]),
        "fixed_b": _once([drafter_b]),
        "ucb": _once([drafter_a, drafter_b]),
        "single_drafter_control": _once([drafter_a]),
    }
    acceptance = {name: result.acceptance_rate for name, result in policies.items()}
    throughput = {name: result.tokens_per_second for name, result in policies.items()}
    best_fixed_alpha = max(acceptance["fixed_a"], acceptance["fixed_b"])
    best_fixed_tps = max(throughput["fixed_a"], throughput["fixed_b"])
    alpha_improved = acceptance["ucb"] > best_fixed_alpha + 1e-6
    throughput_improved = throughput["ucb"] > best_fixed_tps
    if alpha_improved or throughput_improved:
        conclusion = (
            "On this controlled switch, UCB improved at least one of "
            "acceptance or throughput relative to the better fixed drafter."
        )
    else:
        conclusion = (
            "On this controlled switch, UCB did not beat the better fixed "
            "drafter. A regret bound does not guarantee a win on a short "
            "horizon; read the numbers rather than the theorem."
        )
    return {
        "status": "measured",
        "measurement_kind": "controlled_mixed_workload_simulation",
        "git_commit": git_commit(),
        "hardware": detect_hardware(),
        "model": "synthetic_peaked_drafters",
        "method": "ucb_vs_fixed",
        "gamma": gamma,
        "dataset": "midstream_distribution_switch",
        "n_samples": 1,
        "metrics": {
            "acceptance_rate": acceptance,
            "tokens_per_second": throughput,
        },
        "unmeasured": [
            "speedup_vs_vanilla_ar",
            "llama_acceptance",
            "gpu_throughput",
        ],
        "improved_acceptance_vs_best_fixed": alpha_improved,
        "improved_throughput_vs_best_fixed": throughput_improved,
        "best_fixed_acceptance": best_fixed_alpha,
        "best_fixed_tokens_per_second": best_fixed_tps,
        "single_drafter_matches_fixed_a": (
            abs(acceptance["single_drafter_control"] - acceptance["fixed_a"]) < 1e-6
        ),
        "settings": {
            "max_new_tokens": max_new_tokens,
            "vocab": vocab,
            "seed": seed,
            "window_size": window_size,
            "switch_after_context_length": switch_after,
            "peak_a": 3,
            "peak_b": 9,
        },
        "notes": (
            "Synthetic peaked distributions, not a trained model. "
            "The target peak flips once. fixed_a and fixed_b each match "
            "only one phase. single_drafter_control is the same run shape "
            "as fixed_a and exists to show that one arm cannot adapt. " + conclusion
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the demo and write JSON.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector excluding the program name.
    """
    parser = argparse.ArgumentParser(description="Mixed-workload bandit demonstration")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", default="benchmarks/results/bandit_mixed_workload.json")
    args = parser.parse_args(argv)
    report = run_demo(max_new_tokens=args.max_new_tokens)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["notes"])
    print(
        f"acceptance fixed_a={report['metrics']['acceptance_rate']['fixed_a']:.3f} "
        f"fixed_b={report['metrics']['acceptance_rate']['fixed_b']:.3f} "
        f"ucb={report['metrics']['acceptance_rate']['ucb']:.3f}"
    )
    print(
        f"acceptance improved vs best fixed: {report['improved_acceptance_vs_best_fixed']}; "
        f"throughput improved vs best fixed: {report['improved_throughput_vs_best_fixed']}"
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
