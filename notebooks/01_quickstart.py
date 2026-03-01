# %% [markdown]
# # FlashSpec Quickstart
#
# This notebook demonstrates how to build a `SpeculativeEngine` with a tiny
# toy model (no GPU or real weights required).

# %%
import sys
sys.path.insert(0, "..")  # run from notebooks/

import torch
from flashspec import FlashSpecConfig, BanditConfig, SamplingConfig
from flashspec.bandit import UCB1Selector
from flashspec.sampling.rejection import rejection_sample
from flashspec.utils.device import set_seed

set_seed(42)

# %% [markdown]
# ## 1. Build configuration

# %%
config = FlashSpecConfig(
    device="cpu",
    dtype="float32",
    drafter_name="toy",
    target_name="toy",
    max_new_tokens=64,
    bandit=BanditConfig(n_arms=1, strategy="ucb1"),
    sampling=SamplingConfig(gamma=4, temperature=1.0),
)
print(config.model_dump_json(indent=2))

# %% [markdown]
# ## 2. Simulate one speculative step

# %%
BATCH, GAMMA, VOCAB = 2, 4, 1000

set_seed(7)
draft_lp = torch.randn(BATCH, GAMMA, VOCAB).log_softmax(-1)
target_lp = torch.randn(BATCH, GAMMA, VOCAB).log_softmax(-1)
draft_ids = torch.randint(0, VOCAB, (BATCH, GAMMA))
ctx = torch.randint(0, VOCAB, (BATCH, 16))

accepted_ids, first_rejection, alpha = rejection_sample(
    input_ids=ctx,
    draft_logprobs=draft_lp,
    target_logprobs=target_lp,
    draft_token_ids=draft_ids,
    gamma=GAMMA,
)

print(f"Accepted token IDs:\n{accepted_ids}")
print(f"First rejection per sequence: {first_rejection.tolist()}")
print(f"Mean acceptance rate α = {alpha:.3f}")

# %% [markdown]
# ## 3. Bandit selection demo

# %%
selector = UCB1Selector(n_arms=2, window_size=100)

import numpy as np
rng = np.random.default_rng(0)
true_rates = [0.4, 0.75]

print("Round | Arm | α    | Arm-0 pulls | Arm-1 pulls")
print("------|-----|------|-------------|------------")
for t in range(30):
    arm = selector.select()
    accepted = int(rng.random() < true_rates[arm])
    selector.update(arm, accepted)
    if (t + 1) % 5 == 0:
        print(
            f"{t+1:5d} | {arm}   | {accepted} "
            f"| {selector._arms[0].n_pulls:11d} | {selector._arms[1].n_pulls}"
        )

print(f"\nFinal mean accept rates: arm0={selector._arms[0].mean_accept_rate:.3f}, "
      f"arm1={selector._arms[1].mean_accept_rate:.3f}")
