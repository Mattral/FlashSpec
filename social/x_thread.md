# X (Twitter) Thread — FlashSpec Launch

> **How to use**: Post tweet 1, then reply to your own tweet with tweet 2, and
> so on (a "thread"). Each block below is one tweet (under 280 chars).
> Replace `[X.Xx]` and `[XX]` placeholders with real numbers from
> `benchmarks/results/` AFTER running `make bench` — do not post with
> placeholders still in the text.

---

**1/**
Spent the last few weeks building FlashSpec — an open-source speculative
decoding engine for LLM inference.

Two ideas I couldn't find anywhere else, combined:
🔹 GPU-native token verification (no CPU round-trip)
🔹 online bandits to pick the draft model

🧵

---

**2/**
The problem: speculative decoding lets a small "draft" model propose tokens
that a big "target" model verifies in one pass.

But most implementations verify on the CPU — which means every step stalls
the GPU pipeline waiting for a sync.

---

**3/**
FlashSpec's fix: a Triton kernel that does the accept/reject test entirely
on-device.

The trick — you only need 2 scalars per candidate token (log p and log q),
not the full vocab distribution. So SRAM usage is O(1) in vocab size, not
O(V).

---

**4/**
Second problem: which draft model do you use?

Most systems pick one offline and never revisit it. But acceptance rates
shift with the prompt — code vs. chat vs. long context all favor different
drafters.

---

**5/**
FlashSpec treats drafter selection as a K-armed bandit (UCB1 / Thompson
sampling) and adapts online — no human tuning, with a provable
O(√(KT log T)) regret bound.

If the best drafter changes mid-conversation, it notices and recovers.

---

**6/**
The part I'm most proud of isn't the speed — it's that the output
distribution is *provably identical* to plain autoregressive decoding from
the target model.

CI runs a KS test at α=0.01 over 10,000 samples on every build. If that test
goes red, the build fails. No exceptions.

---

**7/**
Full repo: 80+ files, 95%+ test coverage target, Hypothesis property-based
tests, Triton kernel parity tests (float32 + bfloat16), adversarial bandit
chaos tests, MkDocs site, Docker + k8s deploy configs.

Built spec-first — the whole architecture was written down before a line of
code existed.

---

**8/**
Early numbers vs vanilla autoregressive decoding on Llama-3-8B
(H100, γ=4): [X.Xx]× throughput, [XX]% token acceptance.

Full benchmark suite (vs Medusa, EAGLE) running now — results landing in
`benchmarks/results/` this week.

---

**9/**
Apache 2.0, fully open source.

Repo: https://github.com/Mattral/FlashSpec
Docs: https://flashspec.readthedocs.io

If you work on LLM inference and have thoughts on the bandit formulation or
the kernel design, I'd love a review. ⭐ if useful.

---

## Alternative shorter version (4 tweets, if you want something tighter)

**1/**
Open-sourced FlashSpec: a speculative decoding engine that (1) verifies
tokens on-GPU with a Triton kernel (O(1) SRAM in vocab size) and
(2) picks the draft model online via multi-armed bandits.

🧵

**2/**
Why it matters: most speculative decoding setups verify on CPU (pipeline
stall) and pick one drafter offline (suboptimal as prompts shift).
FlashSpec fixes both — adaptively, with a regret bound on the bandit side.

**3/**
And it's provably correct: output distribution matches plain
autoregressive sampling exactly. CI enforces this with a KS test
(α=0.01, N=10,000) on every commit.

**4/**
Apache 2.0. 80+ files, full test suite, docs, Docker/k8s configs.
Benchmarks vs Medusa/EAGLE landing this week.
https://github.com/Mattral/FlashSpec — feedback welcome 🙏
