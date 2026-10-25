# Online Bandit Draft Selection

## Motivation

Different draft models have different acceptance rates across input distributions.
A short repetitive prompt may favour a small n-gram drafter; a long
code-generation task may favour a 1B-parameter LLM drafter.

FlashSpec treats draft-model selection as a K-armed bandit problem and can
adapt the choice online.

> **Experimental.** The selector was measured choosing among seven arms
> once, in the Phase 6 run on a Tesla T4 (bundle `3e38251`, values in
> `benchmarks/results/gpu/3e38251/`). On the mixed workload neither UCB1
> nor Thompson beat the best fixed arm, so the bandit is not a candidate
> for the default; the composite strategies below are. The regret results
> further down are synthetic simulations.

---

## Selectors

### Arms

An arm is a drafter and a gamma (`flashspec.engine.ArmSpec`). Arms may
share a drafter: Qwen2.5-0.5B at gamma 2, 4 and 6 is one model with one KV
cache and three arms. A *composite* arm (`ArmSpec(lookup, gamma,
fallback=model)`) uses the prompt-lookup proposal on a step where one
exists and the model drafter otherwise. The choice needs no extra device
read: the next step's lookup is computed on the device before the step's
one read, which then carries its length. `fallback_gamma` sets how many
tokens the model drafter drafts on a fallback step (`gamma` when unset):
`ArmSpec(lookup, 6, fallback=model, fallback_gamma=2)` looks up 6 tokens and
drafts 2. The target verifies every step at `max(gamma, fallback_gamma) + 1`
on the compiled path.

### Drafter strategies

`FlashSpecConfig.drafter_strategy` (server: `--drafter-strategy`) builds the
arms when the engine is not given `arms=`:

- `"auto"` (default): `"composite-g6-f2"` when the engine is given one
  model drafter, the `"bandit"` path otherwise; a call at batch > 1 or with
  `variant="typical"` uses the model drafter at `sampling.gamma`. One INFO
  line names the choice and the reason.
- `"bandit"` (the 0.2.0 behaviour): one arm per drafter at
  `sampling.gamma`, chosen by the selector.
- `"composite-g4"`: one composite arm, lookup 4 / fallback 4 (the Phase 6
  composite arm).
- `"composite-g6-f2"`: one composite arm, lookup 6 / fallback 2.

A composite strategy needs one model drafter (an `NgramDrafter` is added
when none is given), uses a single bandit arm (`bandit.n_arms` defaults to
1), and runs at batch 1; set explicitly, any other call raises. The
pre-registered comparison in `benchmarks/configs/qwen_t4_composite.yaml`
picked `"composite-g6-f2"` as the target of `"auto"` (bundle `fdd3be9`; see
[Benchmarks](benchmarks.md#default-drafter-strategy-composite-g4-against-composite-g6-f2)).

A contextual bandit (context: the next lookup's match length) is recorded as
future work in the README; it is not implemented.

The reward of a step is its emitted tokens (accepted drafts plus the
residual or bonus) divided by its wall time, including any catch-up
forward an idle model drafter makes on that step.
`GenerationResult.arm_steps` records every step (arm, drafter used,
proposed, accepted, emitted, seconds, reward, catch-up tokens), and
`benchmarks/arm_report.py` turns those records into a per-run report.

The bandit's value for an arm is the mean of its per-step rewards. That
is not the same as the arm's throughput (total tokens / total time) when
step times vary within an arm, as they do for a composite arm (cheap
lookup steps, dearer model steps); the report prints both.

### UCB1 (`bandit/ucb.py`)

Score for arm k at round t:

```
score_k = μ̂_k / s + c · √(2 log t / n_k)
```

- `μ̂_k`: empirical mean reward (optionally windowed). A call that omits
  `reward` uses the accept count, which is what the regret tests record.
  The raw accept count stays on the arm either way.
- `s`: the largest reward seen so far (`normalise_rewards=True`, the
  default). UCB1's bonus is sized for rewards in `[0, 1]`. Next to a reward
  in tokens per second (tens to hundreds) an unscaled bonus is negligible,
  and the selector stops exploring after one pull per arm. Dividing by the
  running maximum makes the choice independent of the unit; for rewards in
  `[0, 1]` that reach 1 it changes nothing.
- `c`: exploration constant (default 1.0).
- `n_k`: number of times arm k has been pulled.
- `t`: total rounds elapsed.

### Thompson Sampling (`bandit/thompson.py`)

With a reward (what the engine passes), each round draws one sample per arm
from a Gaussian posterior on its mean reward and selects the argmax:

```
sample_k ~ Normal(μ̂_k, σ̂_k² / n_k)
```

`σ̂_k²` is the arm's sample variance, or the variance pooled over arms while
the arm has fewer than two observations. Unpulled arms are tried first.
The draw does not depend on the reward's unit. `seed=` gives the selector
its own generator.

Without a reward (`update(arm, accepted)` only), it keeps the Beta(α_k, β_k)
posterior on the acceptance rate:

```
α_k += accepted      β_k += max(0, 1 - accepted)
```

### Oracle (`bandit/oracle.py`)

Always selects the arm with the highest *true* acceptance rate.
Used only in regret upper-bound experiments. The serving path does not
construct an oracle.

---

## Bandit regret (moved from the README) — measured, synthetic (T=10,000, K=3 arms)

In this simulation UCB1 and Thompson sampling stay inside the
O(√(KT log T)) bound over 10,000 rounds.
UCB1 cumulative regret at T=10,000: **100.2** (theory bound: ~526).
Thompson cumulative regret at T=10,000: **18.9**.

![Bandit regret: cumulative regret of UCB1, Thompson and Oracle over 10,000 rounds; UCB1 100.2, Thompson 18.9, Oracle 0.0 at T=10,000](https://raw.githubusercontent.com/Mattral/FlashSpec/main/paper/figures/bandit_regret.jpg)

The figure is `paper/figures/bandit_regret.jpg`. The saved output is
`notebooks/02_bandit_analysis.ipynb` cell 3 (0-based index, execution
count 4): UCB1 100.2, Thompson 18.9, Oracle 0.0. This is a synthetic
K=3 arm experiment, not a throughput gain on a language model.

**In short (experimental).** UCB1 score for arm k at round t:
`μ̂ₖ + c · √(2 log t / nₖ)`. Unpulled arms are selected first. The reward
is emitted tokens per step second. In a synthetic K=3 simulation UCB1
regret at T=10,000 is 100.2 (theory bound ≈ 526) and Thompson's is 18.9.
On a language model it was measured once, over seven arms (Phase 6, bundle
`3e38251`), and did not beat the best fixed arm on the mixed workload.

---

## UCB1 regret bound — proof sketch

**Theorem** (Auer et al., 2002): For K arms and T rounds, UCB1 satisfies:

```
E[R_T] ≤ Σ_{k ≠ k*} [ 8 log(T) / Δ_k + (1 + π²/3) Δ_k ]
```

where `Δ_k = μ* - μ_k` is the suboptimality gap of arm k and `k*` is the
optimal arm.

**Consequence**: `E[R_T] = O(√(K T log T))` (substituting the worst-case gap).

**Sketch of proof:**

1. **Exploration phase** — any arm with fewer than `8 log(t) / Δ_k²` pulls must
   have been selected, since its UCB exceeds the optimal arm's empirical mean.
2. **Exploitation phase** — once arm k* has been pulled enough times, its UCB
   score dominates all suboptimal arms' UCB scores with high probability (via
   Chernoff–Hoeffding).
3. **Combining** — summing over all suboptimal arms and rounds gives the bound.

**Empirical verification**: `tests/unit/test_bandit.py::TestUCB1Selector::test_ucb1_regret_within_theoretical_bound`
runs T=10,000 rounds with K=3 arms and asserts `empirical_regret / theoretical_upper_bound < 1.5`.

---

## Empirical regret convergence

`paper/figures/bandit_regret.jpg` shows cumulative regret for UCB1,
Thompson, and Oracle over T=10,000 rounds with K=3 arms and true rates
[0.3, 0.7, 0.5]. The source is `notebooks/02_bandit_analysis.ipynb` cell 3
(0-based index, execution count 4), which is in this checkout. The sketch
below is illustrative. The recorded cumulative regret at T=10,000 is
UCB1 100.2 and Thompson 18.9, against a theory bound of about 526. This
is a synthetic arm experiment, not language-model throughput.

```
Cumulative Regret vs Round (K=3, Δ_best=0.4)

Regret
  400 |
      |  Thompson ········
  300 |             UCB1 ──────
      |
  200 |
      |
  100 |      Oracle ──────
      |
    0 +──────────────────────────
      0    2000   5000   10000   t
```

Both UCB1 and Thompson track O(√(KT log T)); Oracle achieves near-zero regret.
The unit test `test_ucb1_regret_within_theoretical_bound` is the check that
still runs in this tree.

---

## What the bound does not say

The regret bound is about cumulative acceptance regret against the best
fixed arm. It does not say that UCB1 beats the better drafter on a short
generation, and it does not say anything when only one drafter is loaded.
`SpeculativeEngine` logs a warning in that case. `load_drafters` does too.

`scripts/demo_bandit_mixed.py` runs a controlled simulation: two peaked
drafters and a target whose peak flips halfway through the generation.
It reports whether UCB1 improved acceptance or throughput relative to the
better fixed drafter on that run. Read the JSON it writes. A loss there
means the bound did not produce a short-horizon win. That is a measured
outcome, not a failure of the theorem.

Registering real drafters:

```python
import torch
from flashspec.engine.drafter import load_drafters

drafters = load_drafters(
    ["org/draft-a", "org/draft-b"],
    device="cpu",
    dtype=torch.float32,
    temperature=1.0,
)
```

Unknown names are registered as HuggingFace causal LMs. Pass `models=`
to inject already-built modules and skip the hub.

## Windowed statistics

When `window_size > 0` (default 500), `mean_accept_rate` uses only the
last `window_size` rounds.

**Justification**: The acceptance rate of a draft model depends on the
current prompt distribution.  When the topic shifts (e.g., a new conversation
turn with different vocabulary), the historical acceptance rates become stale.
A windowed mean allows both UCB1 and Thompson to track non-stationary
distributions without requiring an explicit reset.

The window is implemented as a `collections.deque(maxlen=window_size)` per arm.
Thread-safety is provided by the per-selector `threading.Lock`.

---

## JSON serialisation

All selectors support `to_json()` / `from_json()` for checkpointing and resuming
across generation sessions.

```python
selector = UCB1Selector(n_arms=2, window_size=500)
# ... run for 1000 rounds ...
state = selector.to_json()          # serialize to string
restored = UCB1Selector.from_json(state)  # restore exactly
assert restored.t == selector.t    # round counter preserved
```

---

## Adding a new selector

1. Subclass `DraftSelector` from `flashspec.bandit.base`.
2. Implement `select()`, `update()`, `_state_dict()`, `_from_state_dict()`.
3. Export from `flashspec/bandit/__init__.py`.
4. Add unit tests in `tests/unit/test_bandit.py` following existing patterns.
5. Register in `BanditConfig.strategy` literal type in `flashspec/utils/config.py`.

---

## References

1. Auer et al. (2002), "Finite-time Analysis of the Multiarmed Bandit Problem",
   *Machine Learning* 47(2-3):235-256.
2. Thompson (1933), "On the likelihood that one unknown probability exceeds another
   in view of the evidence of two samples", *Biometrika* 25(3-4):285-294.
3. Chapelle & Li (2011), "An Empirical Evaluation of Thompson Sampling", NeurIPS 2011.
