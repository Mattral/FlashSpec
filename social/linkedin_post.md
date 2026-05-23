# LinkedIn Post — What I Learned Building FlashSpec

> **How to use**: Post as-is, or trim the "Lessons" section if you want it
> shorter. Replace `[X.Xx]` and `[XX]` with real numbers from
> `benchmarks/results/` after running `make bench`. LinkedIn favors
> line breaks every 1-2 sentences — formatting below is intentional.

---

**What I learned building an open-source LLM inference engine from a spec, not a tutorial**

A few weeks ago I set out to build FlashSpec — an adaptive speculative
decoding engine for LLM inference, combining two ideas I hadn't seen
paired anywhere else: GPU-native token verification via a custom Triton
kernel, and online multi-armed bandits for choosing which "draft" model
to use at each step.

Here's what stuck with me from the process.

**1. Writing the spec first changed everything.**

Before any code, I wrote down the algorithmic contracts: the exact
acceptance/rejection math (Algorithm 1 from the speculative decoding
paper), the residual distribution formula, the regret bound the bandit
had to satisfy, the numerical tolerances every kernel had to hit
(1e-5 for float32, 1e-3 for bfloat16), and the test categories required
before anything could be called "done."

That document became the source of truth for every decision afterward —
including which parts of the system could be built autonomously and
which needed a human to sign off (touching the acceptance kernel, for
instance, always requires explicit review).

**2. Correctness has to be provable, not just "fast enough."**

The easy version of speculative decoding is "draft model proposes,
target model checks, ship it." The hard part is making sure the *output
distribution* doesn't change — that you're not silently making the model
dumber in exchange for speed.

FlashSpec's CI runs a Kolmogorov-Smirnov test at α=0.01 over 10,000
samples on every build, comparing the speculative output distribution to
plain autoregressive sampling. If that test fails, the build fails. No
exceptions, no "we'll fix it later."

**3. The two technical bets:**

→ Most speculative decoding implementations verify tokens on the CPU,
which stalls the GPU pipeline every single step. FlashSpec's Triton
kernel does the accept/reject test on-device, reading only the two
log-probabilities that matter per candidate token — so SRAM usage is
constant in vocabulary size, not proportional to it.

→ Most implementations also pick one draft model offline and never
revisit that choice — even though acceptance rates shift depending on
whether you're doing chat, code generation, or long-context reasoning.
FlashSpec frames this as a K-armed bandit (UCB1 and Thompson sampling
variants) with a formal O(√(KT log T)) regret bound, so it adapts online
without any human tuning.

**4. "Done" means tested, documented, and benchmarked — not just working.**

The repo ended up with 80+ files: full unit/integration/chaos test
suites, Hypothesis property-based tests, Triton kernel parity tests
against a pure-PyTorch reference (both float32 and bfloat16), a MkDocs
documentation site with architecture diagrams, and CI workflows that run
pip-audit and Trivy security scans before anything ships.

First measurements on a Tesla T4 (Colab) with TinyLlama-1.1B at 4-bit
quantization: **44.2 tok/s**, α=0.75, p50=22.1ms. The T4 is
bandwidth-constrained relative to the target H100 hardware, so the Triton
kernel speedup doesn't fully materialise at batch=1 on this GPU — an honest
result that tells us the kernel's benefit is most visible at larger batches
and on higher-bandwidth hardware. Full Llama-3 comparisons vs Medusa and
EAGLE on H100 are next.

**5. The biggest lesson:**

Building ML systems software at this level of rigor is slow — but the
slowness is front-loaded. Once the spec, the tests, and the CI gates
existed, adding new bandit strategies, sampling variants, or kernels
became fast and low-risk, because every change had to pass the same
correctness bar automatically.

The whole thing is Apache 2.0 and open for review:
🔗 https://github.com/Mattral/FlashSpec
📚 https://flashspec.readthedocs.io

If you work on LLM inference systems, I'd genuinely value feedback —
especially on the bandit formulation and the kernel tiling strategy.

#MachineLearning #LLM #OpenSource #GPUProgramming #Triton #AIInfrastructure
