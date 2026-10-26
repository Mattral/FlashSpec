# Engineering notes: making FlashSpec faster

*FlashSpec is a small library for lossless speculative decoding of HuggingFace causal language models at batch 1. You give it your model and a smaller drafter with the same tokenizer (or no second model at all, using prompt lookup), and it returns your model's own output, faster.*

- *Version 0.1.5 was slower than plain decoding: 0.61× on a Tesla T4.*
- *Version 0.2.0 was faster than HuggingFace `generate` compiled the same way: 1.08× to 1.49× (Qwen2.5-1.5B verified by Qwen2.5-0.5B, one T4, batch 1).*
- *Version 0.3.0 is 1.26× to 1.67× on the same setup, with a default drafting strategy that a pre-registered measurement chose. The "adaptive" bandit that was the project's original headline lost to a two-line rule along the way.*

*Almost none of the gains came from clever math. They came from finding out where the time went, what we were really comparing against, and which of our checks were checking what we thought.*

---

## The idea in one paragraph

A small **drafter** guesses the next γ tokens. The large **target** model checks all γ guesses in one forward pass and keeps the longest correct prefix, plus one token of its own. If the drafter is usually right, you get several tokens for roughly the price of one target forward. The rejection rule (Leviathan et al., Algorithm 1) keeps the output distribution exactly the target's, so the speedup costs nothing in quality. Speed is then simple arithmetic: **tokens per step ÷ cost per step**. Every lesson below is about one side of that fraction, or about how we measured it.

---

## 1. Count the forwards before you write a kernel

FlashSpec 0.1.5 shipped a Triton kernel for the accept/reject test and a hand-written assembly kernel for the residual distribution. Both were real engineering, and both optimised something that takes microseconds next to model forwards that take tens of milliseconds.

The real problem was structural. Every step ran the target **twice**: once to verify the draft, and once more just to push the accepted token into its KV cache. A forward-count test on a toy model showed it in one line: `[8, 4, 1, 4, 1, 4, 1, ...]`. Every `1` is a whole target forward spent on one token.

![Figure 1](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig1-step-structure.svg)

*Figure 1. A closed-form model, not a measurement. With two target forwards per step you need an acceptance rate around 0.7 just to break even; with one, break-even falls to about 0.4.*

The fix was bookkeeping: the engine already knows how many tokens were accepted, so it crops each cache by that count and carries the next token as "pending" into the next forward. **In practice:** write a test that asserts the number and length of model forwards per step. It runs on CPU in a second, it is deterministic, and it catches the class of bug that dominates everything else.

---

## 2. At batch 1, a model costs its layers, not its parameters

Speculative decoding pays only if drafting is cheap compared with verifying. The intuition says a 0.5B drafter costs a third of a 1.5B target. We measured it.

![Figure 3](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig3-forward-cost.svg)

*Figure 3. Measured on the T4. In eager PyTorch the 0.5B drafter's one-token forward costs 90% of the target's whole verify forward.*

At batch 1 a GPU is not compute-bound. It is bound by launching hundreds of small kernels and by reading weights. The drafter has a third of the parameters but 24 layers against the target's 28, and its cost ratio (0.90) tracks the layer ratio (0.86), not the parameter ratio (0.33).

![Figure 2](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig2-step-time.svg)

*Figure 2. One speculative step, measured. Drafting four tokens is 77% of the eager step. CUDA graphs (`torch.compile(mode="reduce-overhead")` on a preallocated static KV cache) take the step from 167 ms to 59 ms, but drafting is still two-thirds of it.*

When the drafter's cost will not go down, change the kind of drafter. **Prompt lookup** takes the last three (then two, then one) tokens, finds their most recent earlier occurrence in the context, and proposes what followed. It runs on the GPU as one vectorised compare, with fixed shapes and no host read, so a lookup step costs only the target's verify:

![Figure 8](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig8-arm-step-cost.svg)

*Figure 8. Every arm alone on the T4. A lookup step costs about 21–22 ms at any γ, which is the verify forward and little else. A Qwen-drafted step costs 41–79 ms, because each drafted token is another drafter forward.*

**In practice:** put the measured drafter-to-target cost ratio *c* in your benchmark output next to acceptance. Do not estimate it from parameter counts.

---

## 3. Acceptance rate is not the objective

Most write-ups on speculative decoding report acceptance rate, and it is tempting to optimise it. It is only the numerator's half of the fraction.

![Figure 9](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig9-acceptance-vs-speed.svg)

*Figure 9. Every arm alone on the code-edit set, one T4 session. The Qwen drafter at γ=2 had 0.94 of its proposals accepted and produced 70 tokens/s. Prompt lookup at γ=6 had only 0.45 accepted and produced 161 tokens/s, because its proposals cost almost nothing.*

The same thing decided FlashSpec's default. The composite strategy uses lookup when the context has a match and the model drafter otherwise. Two versions went into a head-to-head run:

- **lookup 4 / model 4** (look up 4 tokens; on no match, the model drafts 4);
- **lookup 6 / model 2** (look up 6; on no match, the model drafts 2).

Lookup 6 / model 2 had a *lower* acceptance rate on every prompt set and was faster on every set, by 8% on the mixed workload. Longer lookups cost nothing extra when they miss, and shorter model drafts waste fewer expensive drafter forwards. It also changed how the release reads. The 0.3.0 README shows a lower acceptance column than 0.2.0 next to a higher throughput column, which looks like a regression to anyone reading acceptance as the score.

**In practice:** optimise and report tokens per second, with acceptance and step cost as the two explanations of it. When a mechanism changes what a "proposal" is, say so next to the metric.

---

## 4. The baseline is the headline

Here is one run reported three ways:

![Figure 4](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig4-baselines.svg)

*Figure 4. 0.3.0 release run, same compiled FlashSpec numbers throughout. Against eager HuggingFace it is 2.6×–3.4×; against HuggingFace assisted generation with the same drafter it is 3.7×–4.5×; against HuggingFace `generate` with the same static cache and CUDA graphs it is 1.26×–1.67×.*

Comparing a compiled engine with an eager baseline mostly measures the compiler. The fair comparison compiles both sides, and that is the only number in FlashSpec's headline.

Baselines also drift without anyone touching them. Qwen2.5-Instruct ships generation defaults (`top_k=20`, `top_p=0.8`, `repetition_penalty=1.1`), and `generate` applies the repetition penalty **even with `do_sample=False`**. Our "plain greedy" baseline was not plain greedy until the harness reset every checkpoint to a neutral `GenerationConfig()` and recorded what it replaced. **In practice:** a baseline is a configuration, not a function name. Pin it, record it in the result, and compile it the way you compile yourself.

---

## 5. "Lossless" is a claim about numerics, so compare like with like

Algorithm 1 is exact in real arithmetic, and greedy speculative decoding is exact when argmax is exact. GPUs in fp16 are neither. Verifying several tokens in one batched forward and decoding one token at a time run different kernels, which round differently. When the top two logits are tied, the argmax can flip.

In Phase 6 every greedy mismatch against compiled HuggingFace came from the engines that verify 7 tokens at a time, and none from those that verify 3 or 5. Was it a padding bug or rounding? Three cheap tests settled it:

1. **Do different code paths with the same shape agree?** Four methods fed completely different tokens into their width-7 calls and still produced bit-identical outputs. Two of them never padded after their first step, and they mismatched anyway.
2. **Is it exact in fp32?** On the T4, every path matched both HuggingFace references token for token, and changing the padded slots' contents never changed a real row by a single bit.
3. **What are the margins where fp16 diverges?**

![Figure 7](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig7-fp16-ties.svg)

*Figure 7. At each fp16 divergence one side's top two logits were exactly equal (margin 0) and the other side's differed by one fp16 step. Unpadded width 7 diverged at the same tokens. It is the kernel shape, not the padding.*

Then we turned the diagnosis into a check. The next comparison run replayed every greedy divergence. It counted one as an allowed exact tie only if:

- both replays reproduce the recorded outputs;
- the prefixes match;
- the two diverging tokens are the top two of both rows;
- one row's margin is exactly zero.

Anything else counted as a mismatch.

| Comparison (greedy, vs compiled HuggingFace) | Same tokens | Where they differ |
|---|---|---|
| 0.2.0, model drafter at width 5 (20 prompts) | 20/20 | — |
| Phase 6, width-3 and width-5 engines (60 per engine) | all | — |
| Phase 6, width-7 engines (60 per engine) | 54/60 | three exact fp16 ties; fp32 exact |
| 0.3.0 release, default strategy (width 7, 20 prompts) | 17/20 (HumanEval 9/10, MT-Bench 8/10) | the same three ties: every output identical, token for token, to the tie-checked run |

Every result stores per-prompt token ids and a sha256, so the last row is a lookup, not a claim. **In practice:** test exactness in fp32, where it must hold; on the GPU, report match counts against a reference with the *same* numerics; and when fp16 disagrees, measure the margin before you hunt for a bug. A chi-squared test on emitted tokens covers sampling. It is committed together with a test showing that it fails on a deliberately wrong residual.

---

## 6. Make the measurement pipeline refuse bad evidence, and audit what it derives

Every GPU number comes from a Colab notebook run by a person, which is exactly where numbers get retyped, rounded, run from the wrong branch, or measured with a feature silently switched off. So the pipeline refuses all of that:

- the notebook pins an exact commit;
- the bundle carries per-file hashes and a dirty-tree flag;
- the notebook will not download unless every result says the compiled path and the default strategy actually ran;
- an ingest script **generates** the README tables, so nobody types a measured number.

Long runs save each finished block to Drive and resume. The release is built from the measured commit, so measured branches take `main` by merge, never by rebase.

The less obvious failure is in the numbers the pipeline *derives*. Compiled methods cannot be hooked to count forwards, because a hook forces a recompile. So the harness derived drafter forwards as "one prefill plus γ per step". That was true for every drafter we had, and became silently false for the composite: lookup steps run no drafter at all, and its fallback drafts 2, not γ. The forward-cost column would have been wrong in the release tables. The fix derives the count from per-step records. A test checks that, on identical steps, the derived count equals what a hook measures on the uncompiled engine.

**In practice:** every derived metric encodes an assumption about the mechanism. When the mechanism changes, re-derive and cross-check against a measured count.

---

## 7. A path nobody has run end to end is broken until shown otherwise

Several things had unit tests and had never once been run end to end. Each one was broken:

- **The bandit with one option.** Every run up to 0.2.0 had exactly one drafter, so the "adaptive" selector chose nothing. When it finally had real choices, neither selector worked with the reward the engine sends, tokens per second:
  - UCB1's exploration bonus is sized for rewards between 0 and 1, so it stopped exploring after one pull per arm;
  - Thompson sampling ignored the reward entirely.
- **The documented server command.** `pip install "flashspec[serve]"` and the README's `python -m flashspec.serve` crashed at model load. The server loads with `device_map`, which needs `accelerate`, and the extra did not include it. We found it by running the exact command from the README.
- **The lossy `typical` variant.** Its sampler had unit tests, but no test had ever counted the tokens it produced. When the first draft token was rejected, it emitted nothing, so a batch-1 call stopped after one step with zero tokens. It also kept typical tokens *after* a rejected one, emitting a sequence with holes.

Now every Python block in the README runs in CI on tiny models, and the server command and request body are checked against the server's own parser.

**In practice:** for every path you document (each variant, each extra, each command), have one test that runs it end to end and checks the obvious output, such as "did it produce the requested number of tokens?".

---

## 8. Write the bar down before the run, including what happens if you miss

Phase 6 was meant to make the "adaptive" claim true. The exit criteria went into the benchmark config **before** the run:

- on a mixed workload, beat the best single fixed strategy;
- on each prompt set, come within 5% of that set's best;
- be faster than compiled HuggingFace everywhere;
- keep every correctness check green.

![Figure 5](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig5-bandit-vs-arms.svg)

*Figure 5. The bandit reached 0.95 of the best arm on the mixed workload and 0.76 on code edits.*

It failed the first two criteria. The changelog says so, and the bandit stays experimental. Writing the bar first stopped us from picking a flattering summary afterwards (HumanEval at 0.9505 "passes", but the criterion is about every set) and made the failure cheap to accept.

The next pre-registered test, choosing the default drafting strategy, passed. But it exposed gaps in a rule we thought was complete:

- *What if the faster variant fails the correctness condition: does the other one win by default?* (We decided no: neither becomes the default.)
- *Does "every prompt set" include the mixed workload, which had no HuggingFace run?*
- *Exactly which greedy mismatches count as allowed?* This became the tie rule in lesson 5.

**In practice:** pre-register the metric, the threshold and the decision, and also the branch where the winner fails a gate. Ambiguity resolved after the data is in is a choice made with the data in view.

---

## 9. Test your favourite explanation with data you already logged

The obvious explanation for the bandit's loss was overhead. It verifies 7 tokens on every step, so its cheap arms paid for a wide forward. That was plausible, and mostly wrong.

![Figure 6](https://raw.githubusercontent.com/Mattral/FlashSpec/main/docs/images/lessons/fig6-why-bandit-lost.svg)

*Figure 6. The first bars re-price the bandit's own choices at each arm's stand-alone step cost; the last two are oracles that pick the best arm per prompt set or per prompt with hindsight. All of it is arithmetic on logged per-step records, not new runs.*

Re-pricing moved the bandit by −1% to +5%. The real causes were three:

- **Exploration.** Seven arms over about 1,400 steps: 75% of its steps went to arms other than the best one.
- **The reward was the wrong average.** The selector maximised the mean of per-step tokens-per-second. For the composite arm, whose steps mix cheap lookups and expensive drafts, that mean overstated its real throughput by 1.24×–1.28×. The average of ratios is not the ratio of totals.
- **No context.** The best arm changed between consecutive prompts 16 times out of 29.

The oracles also capped any fix: perfect per-prompt choice would gain only about 8% over the best single arm. **In practice:** log per-step records (choice, tokens, seconds) and price a fix before you build it.

---

## 10. A learner has to beat the best rule you can write with the same information

The best single strategy on the mixed workload was not a learner. It was the composite rule: *use the lookup proposal if there is one, otherwise ask the model drafter.* That rule is a contextual bandit with one context bit (does the context contain a match?), and the bit is free to observe.

So 0.3.0 ships the rule, not the learner. A pre-registered comparison picked lookup 6 / model 2, as in lesson 3. `drafter_strategy="auto"` runs it whenever you give the engine one model drafter, and falls back to the model drafter for calls it cannot run, such as batch sizes above 1. A learned contextual bandit is written down as future work, with criteria fixed in advance. It was not built, because the oracle ceiling said it had a few percent at most to win.

**In practice:** put the best hand-written rule in the comparison from day one, and let it win when it wins.

---

## 11. Changing a default changes what your tests test

Making the composite the default looked like a one-line change. Switching the config default made 169 tests fail or error, including the core distribution test. That test relies on the default to build a model-drafter engine at γ=4 and batch 10,000.

The failures were not the dangerous part. We redesigned the default to behave like `compile="auto"`:

- choose the composite when it can run;
- otherwise fall back to the model drafter, with one log line saying which strategy ran and why.

Most failures went away, but many of the now-passing tests had silently started exercising the composite instead of the path they were written for. One even passed only because its batch size happened to trigger the fallback. The fix was to pin the strategy explicitly in every test, benchmark method and diagnostic that means the model-drafter path, so each one keeps meaning what it meant. The default is `"auto"` rather than the composite's own name, so an explicit setting is always distinguishable from the default and always respected.

**In practice:** correctness tests should not depend on defaults. Pin the configuration a test means, and when you change a default, also review the tests that still pass.

---

## Takeaways

1. Assert forward counts per step before optimising kernels.
2. Measure the drafter-to-target cost ratio. At batch 1 it follows layers and launch overhead; if it won't go down, change the kind of drafter.
3. Optimise tokens per second, not acceptance; a cheaper, less accurate drafter can win.
4. Compile the baseline too, and record its configuration in the result.
5. Check exactness in fp32 and against the same numerics; measure the margin before calling a mismatch a bug, then turn the diagnosis into a check.
6. Make the pipeline reject untrusted runs, and re-derive every derived metric when the mechanism changes.
7. Run every documented path end to end at least once, and check its obvious output.
8. Pre-register the bar, including what happens when the winner fails a gate.
9. Price your favourite explanation with logged per-step data before building its fix.
10. Benchmark any learner against the best hand-written rule with the same information.
11. Pin the configuration each correctness test means; a changed default can make tests pass while testing something else.

*Sources: result JSONs in the FlashSpec repository under `benchmarks/results/gpu/`: `8d036d7` (0.2.0 release), `3e38251` (Phase 6), `fdd3be9` (default-strategy comparison) and `3c43e08` (0.3.0 release). Also the width diagnostics in `benchmarks/results/diagnostics/`, `scripts/phase6_analysis.py`, `scripts/composite_report.py` and the CHANGELOG. Figure 1 is a closed-form model. Figures 2 and 3 are from the 0.2.0 JSONs and have no generator script. Figures 5–8 are drawn by `scripts/article/make_phase6_figures.py`, and figures 4 and 9 by `scripts/article/make_030_figures.py`; both write to `docs/images/lessons/`, and a test checks that they reproduce the committed SVGs byte for byte. Numbers in the text and figure labels are rounded for reading; the JSONs and each mark's tooltip keep the stored values.*
