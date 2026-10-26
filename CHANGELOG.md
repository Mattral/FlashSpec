# Changelog

All notable changes to FlashSpec are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).
Versioning follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## [Unreleased]

### Added
- Container image on GitHub Container Registry, `ghcr.io/mattral/flashspec:<version>` and
  `:latest`, built by `.github/workflows/container.yml`. It builds when a release is published
  or by hand with a version input, uses only `GITHUB_TOKEN`, and its actions are SHA-pinned. It
  smoke-tests every image on a CPU runner (import, then the server with a tiny model) before
  pushing, and the OCI source label links the package to this repository.

### Changed
- `deploy/Dockerfile` installs the released `flashspec[serve]==<version>` from PyPI instead of
  the source tree. It uses `nvidia/cuda:13.0.0-base-ubuntu24.04` (pinned by digest) with torch
  `2.11.0+cu130` and transformers `5.20.2`, runs as a non-root user, and serves by default.
  `deploy/docker-compose.yml` and the k8s manifest use the published image.

## [0.3.0] — 2026-10-26 — Composite drafter by default, prompt lookup, separate gammas

Development status stays Alpha. The measured scope is still one model pair
on one GPU at batch 1; read the README tables, which are generated from the
result JSONs, for the numbers.

### Highlights
- **New default drafter strategy, chosen by a pre-registered rule.**
  `FlashSpecConfig.drafter_strategy="auto"` runs `composite-g6-f2` when you
  give the engine one model drafter: prompt lookup (up to 6 tokens) when the
  context has a match, else the model drafter drafts 2. The rule was fixed
  in `benchmarks/configs/qwen_t4_composite.yaml` before the run, and
  composite-g6-f2 passed it (bundle `fdd3be9`).
  `drafter_strategy="bandit"` restores 0.2.0.
- **Prompt-lookup drafter** (`NgramDrafter`). It needs no second model and
  uses exact Algorithm 1 with a one-hot draft distribution, so it is
  lossless. Composite arms use it with separate lookup and fallback gammas.
- **Fixes:** `variant="typical"` (lossy) produced zero tokens at batch 1;
  the `serve` extra now installs `accelerate`, which the server needs.
- **README restructured** for a 30-second decision, a 5-minute trial and AI
  agents: a tested quickstart, recipes, an API table generated from the
  code, `llms.txt`, and a throughput chart drawn from the result JSONs by
  `scripts/readme_chart.py` (its values are the table's). Stale statements in
  `docs/postponed.md`, `docs/architecture.md`, `PUBLISHING.md`,
  `CITATION.cff` and `.zenodo.json` were brought up to date.

### Measured (bundle `3c43e08`, run from `main`)
- `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`,
  one Tesla T4, fp16, batch 1, 10 MT-Bench and 10 HumanEval prompts through
  the chat template, greedy and T = 0.7, 128 new tokens, with the defaults
  `compile="auto"` (resolved on, no eager fallback) and
  `drafter_strategy="auto"` (resolved to `composite-g6-f2` in every
  FlashSpec result). The config is the 0.2.0 release config plus
  `drafter_strategy: auto`.
- Compiled FlashSpec was faster than HuggingFace `generate` compiled the
  same way in all four dataset and mode cells: by at least 1.5x on both
  HumanEval cells and by less on MT-Bench. It was also faster than eager
  `generate` and HuggingFace assisted generation in all four.
- The eager path (`compile=False`) was faster than plain `generate` in all
  four cells (0.2.0: slower in 3 of 4) and slower than compiled `generate`
  in all four.
- Greedy compiled FlashSpec matched `hf_greedy_compiled` on 9/10 HumanEval
  and 8/10 MT-Bench prompts. On all 20 prompts both sides' outputs are
  token-for-token those of the tie-checked comparison run (bundle
  `fdd3be9`), so the three divergences (HumanEval/53 at token 94, MT-Bench
  81 at 74 and 101 at 44) are the fp16 exact ties found there and in #47.
- The README GPU tables are generated from this bundle only
  (`scripts/ingest_gpu_bundle.py --readme-commit 3c43e08`). The 0.2.0
  bundle `8d036d7` stays under `benchmarks/results/gpu/`.

### Upgrade notes
- **Default behaviour changed** (see "Changed (behaviour)" below). One model
  drafter at batch 1 now runs the composite strategy, and the engine adds
  the lookup drafter itself. For the 0.2.0 behaviour pass
  `FlashSpecConfig(drafter_strategy="bandit")`, or
  `--drafter-strategy bandit` to the server.
- The `acceptance` column now counts lookup proposals too (accepted /
  proposed over every step). It is lower than 0.2.0's model-drafter value
  while throughput is higher, so do not compare the two columns directly.
- `pip install "flashspec[serve]"` now installs `accelerate`.

### Known issues
- **The first call compiles.** With `compile="auto"` on CUDA the first
  `generate` compiles and captures CUDA graphs. In the release run the
  first compiled call took 223225.92456299992 ms (the stored `cold_start`
  value), over three minutes. Later calls reuse the graphs. Use
  `compile=False` to skip it; the eager path is slower than compiled
  `generate`.
- **fp16 greedy ties.** In fp16, kernels of different shapes can resolve an
  exact top-2 tie differently, so greedy FlashSpec can differ from
  HuggingFace at such a tie (the three cases above). fp32 is token-exact on
  every path (T4 width diagnostic, #47).
- Eager greedy FlashSpec differed from eager `hf_greedy` on MT-Bench prompt
  82 at token 97. That is the same prompt and position recorded in Phase 4
  and in the Phase 6 legacy bandit, and that prompt is fp32-exact. This run
  did not tie-check it.
- **Batch > 1** uses the generic loop. Under `"auto"` such calls use the
  model drafter at `sampling.gamma`. Batch > 1 has not been measured.
- **The bandit is experimental.** Measured once over seven arms (Phase 6,
  bundle `3e38251`), it did not beat the best fixed arm.
- One model pair and one GPU (Tesla T4) were measured; no other GPU, pair
  or batch size.
- PyTorch's "CUDAGraph supports dynamic shapes by recording a new graph for
  each distinct input size" warning still appeared once in the release
  run's log. Its source compiled function is not identified yet.

### Changed (behaviour)
- **The default drafter strategy is now `drafter_strategy="auto"`.** Like
  `compile="auto"`, it picks per engine and logs one INFO line with the
  choice and the reason (`engine.drafter_strategy`,
  `engine.drafter_strategy_reason`). Given one model drafter and no `arms=`,
  the engine runs `"composite-g6-f2"`: prompt lookup up to 6 tokens when the
  context has a match, else the model drafter drafts 2. It adds an
  `NgramDrafter` when none is passed. A call it cannot run that way (batch
  > 1, or `variant="typical"`) uses the model drafter at `sampling.gamma`,
  with one INFO line; several drafters get the 0.2.0 one-arm-per-drafter
  path. Under `"auto"`, `config.bandit.n_arms` is only checked when you pass
  a `BanditConfig`. An explicit `drafter_strategy` is always respected, and
  an explicit composite at batch > 1 raises a `ValueError` naming the fix.
  **To get the 0.2.0 behaviour back, pass
  `FlashSpecConfig(drafter_strategy="bandit")`.** The server's
  `--drafter-strategy` defaults to `auto` as well.
- Why: the pre-registered comparison (`benchmarks/configs/qwen_t4_composite.yaml`,
  bundle `fdd3be9`, Tesla T4) passed for composite-g6-f2: composite-g6-f2 had the higher mixed-workload greedy throughput, was faster than compiled HuggingFace on HumanEval, MT-Bench and HumanEval-edit in greedy and at T=0.7, and matched compiled HuggingFace greedy output except at three fp16 exact ties (HumanEval/53, MT-Bench 81 and 101, the ties the #47 diagnostic found). Values in `benchmarks/results/gpu/fdd3be9/`
  (`scripts/composite_report.py`). The README tables are still the 0.2.0
  model-drafter run (`8d036d7`).
- Tests, benchmark harness methods and diagnostics that mean the
  model-drafter path pin `drafter_strategy="bandit"`, so they test and
  measure what they did before. That includes one added keyword in
  `tests/integration/test_output_distribution.py`; its Algorithm 1 check and
  the residual are unchanged.

### Fixed
- `sampling.variant="typical"` (lossy) produced too few tokens. A step
  that rejected the first draft token emitted nothing, so a batch-1 call
  stopped after one step with zero tokens, and typical draft tokens after a
  rejected one were kept, so a step could emit a non-contiguous sequence.
  `typical_sample` now keeps only the prefix before the first rejection and
  fills the rejected position with one token drawn from the target
  distribution there (seeded by `sampling.seed`), in the layout
  `rejection_sample` uses. The acceptance criterion is unchanged, and the
  variant stays lossy.
- The `serve` extra installs `accelerate` (>= 1.1.0).
  `python -m flashspec.serve` loads the target with `device_map`, which
  transformers only accepts with `accelerate`, so a server installed with
  `pip install "flashspec[serve]"` failed at model load.

### Added
- Separate lookup and fallback gammas for a composite arm:
  `ArmSpec(lookup, gamma, fallback=model, fallback_gamma=g)`. The model
  drafter drafts `fallback_gamma` tokens on fallback steps (`gamma` when
  unset); the verify width is `max gamma + 1`.
- `FlashSpecConfig.drafter_strategy`: `"auto"` (the default; see
  Changed), `"bandit"` (the 0.2.0 behaviour), `"composite-g4"` (lookup 4 /
  fallback 4, the Phase 6 composite arm) or `"composite-g6-f2"` (lookup 6 /
  fallback 2). Without `arms=`, the
  engine builds the strategy's arms (`flashspec.engine.arms.strategy_arms`).
  A composite strategy is one arm and defaults `bandit.n_arms` to 1. The
  server takes `--drafter-strategy`; an explicit composite takes one
  `--drafter` and requires `--batch-size 1`, and the engine adds the
  prompt-lookup drafter.
- Default-strategy comparison, pre-registered:
  `benchmarks/configs/qwen_t4_composite.yaml` (notebook cell 3f) runs both
  composite variants against compiled HuggingFace on the Phase 6 prompt
  sets; `scripts/composite_report.py` applies the rule fixed in the config.
  The runner's `greedy_tie_check: true` replays each greedy divergence from
  compiled HuggingFace and classifies it with `benchmarks/tie_check.py`
  (fp16 exact tie, as found in #47, or a mismatch). Result: see Changed
  (behaviour).
- Prompt-lookup (n-gram) drafter, `flashspec.engine.NgramDrafter` (PLAN
  Phase 6a). It proposes the continuation of the most recent earlier
  occurrence of the last 3, 2 or 1 context tokens, searched on the device
  with fixed shapes (`unfold` + compare, no loop over the context, no
  device read). It may propose 0 to gamma tokens; the real length stays a
  device tensor, and positions past it are never accepted. A step with no
  proposal emits one target token from its single target forward.
- Sampling with a lookup drafter uses Algorithm 1 with a one-hot `q`
  (`flashspec.sampling.rejection_step_onehot`): accept with probability
  `min(1, p(x))`; on a rejection, sample `max(0, p - onehot(x))`
  renormalised, through the same residual code as the model drafter. No
  smoothed `q`, and no `log(0)`.
- Lookup drafters run on the batch-1 fast path only; the batch > 1 loop
  raises a `ValueError` naming that.
- Per-arm gamma (PLAN Phase 6b): `SpeculativeEngine(..., arms=[ArmSpec(drafter,
  gamma), ...])`. Arms may share a drafter (one model, one KV cache, several
  gammas). Without `arms` the engine keeps one arm per drafter at
  `sampling.gamma`, as before. The acceptance rate now divides by what each
  step proposed (`AcceptanceTracker.record(n_accepted, n_proposed)`).
- `FlashSpecConfig.compile_fixed_shapes` (default `True`): on the compiled
  path, the target's verify calls are padded at the end to one width,
  `gamma_max + 1`, and a drafter's calls to `(1, 2, 4, 8, 16, 32)`. Padded
  positions are moved out of the static cache right after the forward and
  their logit rows are dropped, so they are never kept or accepted. Variable
  gammas, short lookup proposals and catch-ups then replay a few CUDA graphs
  instead of recording one per length. `False` keeps the 0.2.0 shapes.
- An idle model drafter catches up in one forward of everything it missed
  (Phase 6c); under compile that call is padded to the next drafter width,
  or runs eagerly when longer than 32 tokens. Its time is part of the step
  that uses the drafter, so it counts against that arm's reward.
- Composite arm (PLAN Phase 6d): `ArmSpec(lookup, gamma, fallback=model)`
  uses the lookup proposal when one exists and the model drafter
  otherwise. The next step's lookup is computed on the device before the
  step's one read, which carries its length, so the choice adds no read per
  step (one per `generate` call, before the first step).
- `GenerationResult.arm_steps`: one record per step (arm, drafter used,
  proposed, accepted, emitted, seconds, reward, catch-up tokens), and
  `benchmarks/arm_report.py` for per-run reports: pulls over time,
  per-arm acceptance and tokens per step, cumulative reward.
- Prompt sets (PLAN Phase 6e): `humaneval_edit`, 10 copy-heavy code-editing
  prompts built from HumanEval (MIT) by `scripts/build_humaneval_edit.py`,
  and `mixed`, the 30 MT-Bench, HumanEval and HumanEval-edit prompts in one
  stored shuffled order.
- Phase 6 measurement (PLAN Phase 6f): `benchmarks/phase6.py` (configs with
  `runner: phase6`) runs every fixed arm alone, the UCB1 and Thompson
  bandits over all arms, compiled and eager HuggingFace and HF assisted, on
  each prompt set and on the mixed workload (one engine per bandit for the
  whole block). Every FlashSpec document carries the arm report, CUDA-graph
  captures per wrapper group, per-prompt token ids with sha256, and greedy
  matches against eager and compiled HuggingFace.
  `benchmarks/configs/qwen_t4_phase6.yaml` is the T4 config (notebook cell
  3d); `scripts/phase6_report.py` evaluates the exit criteria from the JSONs.
- **Phase 6 result (bundle `3e38251`, one Tesla T4): the bandit failed its main
  exit criterion.** On the mixed workload neither UCB1 nor Thompson beat the
  best fixed arm, and UCB1 was not within 5% of each prompt set's best fixed
  arm; it was faster than compiled HuggingFace on every set. The values are in
  `benchmarks/results/gpu/3e38251/` (`scripts/phase6_report.py`,
  `scripts/phase6_analysis.py`). The bandit stays experimental.
- Greedy mismatches against compiled HuggingFace in that bundle were all in
  the verify-width-7 group. `benchmarks/width_diagnostic.py` (notebook cell
  3e) on the T4 found every path token-exact in fp32. In fp16, each divergence
  is at an exact top-2 tie on one side, and the unpadded width-7 path diverges
  identically, so the mismatches come from fp16 rounding that depends on the
  kernel shape, not from padding. Pad content never reaches a real row (bit
  identical). Diagnostics are in `benchmarks/results/diagnostics/`.
- Resumable Phase 6 bundles: `gpu_bundle.py --checkpoint-dir` (notebook cell
  3d uses a Google Drive folder per commit) saves each finished block (one
  prompt set and mode) with a marker naming the commit, the config's sha256
  and the session, plus that session's log so far. A rerun restores intact
  blocks, runs only the missing ones, and assembles one zip whose manifest
  lists the blocks and the session that produced each, with every session's
  log under `logs/`. Ingest checks that every block and document traces to
  one listed session. The runner prints progress per block with an ETA.
- `HuggingFaceDrafter.set_temperature()` and `pending_length`.
- `scripts/ingest_gpu_bundle.py` checks Phase 6 documents and keeps them
  out of the existing README tables.
- Benchmarks: `flashspec_compiled_legacy` (compiled, `compile_fixed_shapes=False`)
  and a `cuda_graphs` section on compiled methods: CUDA-graph captures
  counted at `torch.cuda.CUDAGraph.capture_begin`, plus the call shapes
  each static session sent down the captured path.

### Changed
- `UCB1Selector` divides mean rewards by the largest reward seen so far
  (`normalise_rewards=True`). With the engine's tokens-per-second reward the
  unscaled exploration bonus was negligible, so UCB1 stopped exploring after
  one pull per arm. For rewards in `[0, 1]` that reach 1 nothing changes.
  Saved states without the field restore unscaled.
- `ThompsonSelector` samples a Gaussian posterior on the reward when one
  is passed (the engine always passes one); before, it ignored the reward
  and sampled a Beta posterior on the accept count. Without a reward it
  keeps the Beta posterior, now with `beta += max(0, 1 - accepted)` so an
  accept count above 1 can no longer drive beta to zero or below. New
  `seed=` argument.

## [0.2.0] — 2026-10-25 — Compiled fast path, lossless checks, release hygiene

Development status stays Alpha. The measured scope is one model pair on
one GPU; read the README tables, which are generated from the result
JSONs, for the numbers.

### Highlights
- One target forward and one host read per speculative step (batch 1),
  greedy decoding (temperature 0), EOS handling, seeded generation, and a
  bandit reward of emitted tokens per step second.
- `FlashSpecConfig.compile="auto"` (the default): on CUDA with Triton the
  batch-1 path runs on static KV caches and replays the fixed-shape
  drafter decode and target verify calls as CUDA graphs; elsewhere it
  stays eager. One log line says which and why.
- Every benchmark result stores per-prompt output token ids and sha256
  hashes; greedy compiled FlashSpec is checked token for token against
  compiled HuggingFace `generate`.

### Measured (bundle `8d036d7`, run from `main`)
- `Qwen/Qwen2.5-1.5B-Instruct` verified by `Qwen/Qwen2.5-0.5B-Instruct`,
  one Tesla T4, fp16, batch 1, gamma 4, 10 MT-Bench and 10 HumanEval
  prompts through the chat template, greedy and T = 0.7, 128 new tokens,
  `compile="auto"` (resolved on, no eager fallback).
- FlashSpec with the compiled path was faster than HuggingFace `generate`
  compiled the same way in all four dataset and mode cells: by at least
  1.3x on both HumanEval cells and by less on MT-Bench. It was also faster
  than eager `generate` and HuggingFace assisted generation in all four.
- Greedy compiled FlashSpec produced the same tokens as
  `hf_greedy_compiled` on 10/10 HumanEval and 10/10 MT-Bench prompts.
- The eager path (`compile=False`) was slower than plain `generate` in 3
  of the 4 cells.
- The README GPU tables are generated from this bundle only
  (`scripts/ingest_gpu_bundle.py --readme-commit 8d036d7`); the earlier
  feature-branch bundles `36134c0` and `52993e4` stay under
  `benchmarks/results/gpu/` as history.

### Upgrade notes
- **The first call compiles.** With `compile="auto"` on CUDA + Triton,
  the first `generate` compiles the drafter and target forwards and
  captures CUDA graphs, which can take minutes; later calls in the
  process reuse them. Pass `FlashSpecConfig(compile=False)` (or
  `--compile off` to the server) to stay eager. The first-call time was
  not recorded in bundle `8d036d7`; the harness records it from the next
  run on (`cold_start`).
- `sampling.variant="typical"` is lossy and now says so (docs and a
  warning at engine construction).
- `publish.yml` must be run on a `v<version>` tag (see `PUBLISHING.md`).

### Known issues
- Eager fp16 greedy FlashSpec diverged from eager `hf_greedy` on MT-Bench
  prompt index 1 at token 44 (`flashspec_ucb`, bundle `8d036d7`). Likely
  an fp16 near-tie between the batched verify and a one-token decode;
  not confirmed. Compiled FlashSpec and compiled HuggingFace both differ
  from eager `hf_greedy` on HumanEval prompt index 2 at token 94, and
  match each other there.
- The bandit has only been measured with one drafter (single arm), so it
  has not been shown to choose between drafters.
- Batch sizes above 1 use the generic loop: the largest equal-length
  group runs per loop and the KV cache is used only at group size 1.
- One model pair and one GPU (Tesla T4) were measured; no other GPU,
  pair or batch size.
- PyTorch may warn "CUDAGraph supports dynamic shapes by recording a new
  graph for each distinct input size ... observed 9 distinct sizes"
  during a compiled run. FlashSpec's static-cache sessions are the only
  compiled callers whose input shapes vary (drafter decode of 1 or 2
  tokens, target verify of gamma or gamma + 1 tokens, and the
  `logits_to_keep` int), while HuggingFace's compiled decode has fixed
  shapes. A CPU stand-in did not reproduce more than two shapes per
  graph, so the count is not explained yet; bundle runs now copy the
  warning into `run.log`. Follow-up: pad the drafter and verify calls to
  one fixed shape each.

### Added (harness, for the next run)
- `cold_start` in every result: the wall time of each method's first
  `generate` in a block (compile and CUDA-graph capture included for
  compiled methods in the first block of a process), whether it was a
  warmup, and whether it was the first block in the process. The README
  status block prints it when present.
- `run.log` in a bundle now also carries PyTorch warnings (CUDA-graph and
  recompile messages), FlashSpec log lines (including the compile
  decision) and Python warnings, timestamped.
- `scripts/ingest_gpu_bundle.py --readme-commit <commit>` builds the
  README GPU blocks from one bundle only. The status block adds generated
  sentences for the compiled greedy match and the cold start.

### Detailed changes since 0.1.5

### Changed (release hygiene)
- `publish.yml` must run on a `v<version>` tag: a `verify` job checks
  that the ref is a tag, that it equals `"v"` + the `pyproject.toml`
  version, that the tagged commit is on `main`, runs
  `pytest tests/unit tests/integration -m "not gpu"`, and builds the
  sdist and wheel; the `publish` job keeps the `pypi` environment
  reviewer gate and Trusted Publishing. `PUBLISHING.md` has the steps.
- Every action in every workflow is pinned to a commit SHA (the SHA each
  tag in use pointed to, so behaviour is unchanged).
- Removed `gpu_tests.yml` and `benchmark.yml`: they targeted self-hosted
  A10G/H100 runners that don't exist. GPU checks run by hand on a Colab
  T4 (`notebooks/04_gpu_measurement.ipynb`).
- `ci.yml` runs on every pull request, including stacked PRs whose base
  is another branch, and its lint job now runs `make typecheck`.
- `mypy --strict flashspec` passes. Fixed: a non-Windows `ctypes.windll`
  access, the serve CLI's selector type, the HF drafter's `.to()` and
  logits types, and the Triton-unavailable stub's type (`NoReturn`).
  Configured in `pyproject.toml`: Triton and onnx are typed as `Any`
  whether or not they are installed, and Triton's `@jit`/`@autotune` are
  allowed as untyped decorators in the two kernel modules. The stale
  `type: ignore` comments are gone.
- Removed the references to `AGENTS.md` and its section numbers (the file
  was never in the repository); the rules they pointed at are stated
  where they apply.
- `sampling.variant="typical"` is labelled lossy in its module,
  function and `SamplingConfig` docs, and the engine logs a warning when
  it is selected.

### Added
- Every benchmark result JSON stores `outputs`: per prompt, in prompt
  order, the generated token ids, their count and a sha256 (of the ids as
  decimal text joined by `,`). Two methods made the same tokens for a
  prompt exactly when the hashes match. A greedy `flashspec_compiled`
  result run next to `hf_greedy_compiled` also records
  `greedy_token_match_compiled` against it: both use compiled numerics,
  so this is the lossless check for the compiled path (the existing
  `greedy_token_match` stays against eager `hf_greedy`). The ingest script
  rejects a bundle whose results lack `outputs`, whose hashes don't match
  their ids, or whose greedy `flashspec_compiled` lacks the compiled
  match. The README status table gains a "greedy match vs compiled HF"
  column; it is `n/a` for bundle `52993e4`, which predates the field.
- Phase 5 (no GPU numbers yet): `FlashSpecConfig.compile` runs the
  batch-1 path on a preallocated static KV cache
  (`flashspec/engine/static_cache.py`) and captures the fixed-shape
  drafter decode and target verify calls with
  `torch.compile(mode="reduce-overhead")` CUDA graphs, using the same
  `CompileConfig()` HuggingFace `generate` uses for a static cache.
  Cropping moves the cache write position; `cache_position` is computed
  on the device; the prefill stays eager; a compile failure falls back to
  eager and is recorded. CUDA only; on other devices the flag logs a
  warning and changes nothing. The static path is checked eagerly on CPU
  (same logits as the dynamic cache, greedy equal to `generate`, same
  sampled tokens as the cache-free engine).
- Benchmark methods `flashspec_compiled`, `hf_greedy_compiled` and
  `hf_sample_compiled` (HuggingFace `generate` on a reused `StaticCache`,
  compiled with `CompileConfig()`), so a graph-captured FlashSpec is
  compared with a graph-captured HuggingFace. Every result records
  `forward_cost` (c = drafter 1-token forward ms / target verify forward
  ms, timed eager and compiled); FlashSpec results also record
  `step_cost` (c from the step profile) and `compile` (requested,
  effective, fallback reason). `benchmarks/configs/qwen_t4_compile.yaml`
  and notebook cell 3b run the Qwen pair with every method, plus the
  greedy-mismatch diagnostic, into one bundle.
- Phase 4 GPU measurement setup (no GPU numbers yet).
  - `benchmarks/configs/qwen_t4.yaml` (Qwen2.5-1.5B-Instruct verified,
    Qwen2.5-0.5B-Instruct drafting) and `benchmarks/configs/tinyllama_t4.yaml`
    (TinyLlama-1.1B-Chat, Llama-160M-Chat): float16, batch 1, gamma 4,
    128 new tokens, 2 warmup prompts, 10 MT-Bench + 10 HumanEval prompts
    through the target's chat template (`chat_template: true`), greedy
    and temperature 0.7 (`temperatures: [0.0, 0.7]`).
  - Per method: tok/s median, p25, p75 and IQR over prompts; forward
    counts; peak GPU memory; host reads and NaN/inf logit counts from an
    untimed probe generation; `greedy_token_match` against `hf_greedy`
    (FlashSpec and `hf_assisted`). Method order rotates per prompt.
  - The drafter and target tokenizers must be identical (vocab and added
    tokens), checked at load. Checkpoints load once per config.
  - `benchmarks/gpu_bundle.py` runs configs into one zip (result JSONs,
    `manifest.json` with hashes, `profile.txt`, `run.log`).
    `scripts/ingest_gpu_bundle.py` validates a bundle (layout, hashes,
    commit, clean tree, GPU, real prompts, allowed hardware fields,
    required sections) and rebuilds the README `gpu-results` block from
    the JSONs only. Notebook cell 3a runs both configs and downloads the
    zip.
- The MT-Bench and HumanEval prompt sets have 10 prompts each (were 4),
  extended with the same selection rules; the first four keep their
  positions.
- Per-step profiling: `MetricsConfig(profile_steps=True)` records
  draft, verify, sample and bookkeeping time for every step
  (`flashspec/metrics/step_profile.py`, CUDA events on GPU), returned as
  `GenerationResult.step_profile`. Off by default.
- `flashspec/metrics/instrument.py`: `ForwardRecorder` (forward calls and
  their input lengths) and `count_host_syncs` (CUDA sync debug mode, or a
  CPU proxy that counts tensor reads).
- Benchmark baselines `hf_greedy`, `hf_sample` and `hf_assisted`
  (HuggingFace assisted generation with the drafter), a `model: toy_llama`
  random tiny Llama pair, and `benchmarks/configs/tiny_llama_cpu.yaml`.
  FlashSpec result files now carry `step_profile`, `forward_counts` and
  `host_syncs`.
- Greedy decoding: `SamplingConfig(temperature=0)` selects
  `variant="greedy"`. The drafter proposes argmax tokens (no softmax, no
  log-probs), the target's single verify forward returns argmax ids, and
  draft tokens are accepted up to the first mismatch; the next token is
  `target_argmax[n_accepted]`, which covers both the mismatch and the
  bonus case (`flashspec/sampling/greedy.py`). The output equals
  `model.generate(do_sample=False)` token for token
  (`tests/unit/test_greedy_gold.py`; the TinyLlama GPU version is
  `tests/integration/test_greedy_gold_gpu.py`, runnable from
  `notebooks/04_gpu_measurement.ipynb`). Sampling at temperature > 0 is
  unchanged.
  The GPU gold test was run on Colab on `perf/phase-2-greedy`
  (2026-10-24; GPU: Tesla T4, torch 2.11.0+cu130, CUDA 13.0):
  `test_tinyllama_greedy_matches_generate_fp32` PASSED,
  `test_tinyllama_greedy_matches_generate_fp16` XPASS (fp16 also matched
  token for token); 1 passed, 1 xpassed in 126.04s. That run covered
  4 HumanEval prompts, not the 5 the test asks for: only 4 were vendored
  and the test sliced `[:5]` without checking. The test now asserts the
  prompt count.
- The harness runs greedy configs (`temperature: 0`,
  `benchmarks/configs/tiny_llama_cpu_greedy.yaml`): `vanilla_ar` decodes
  greedily, `hf_sample` is skipped, every result records `decoding`, and
  the FlashSpec result records `greedy_token_match` against the greedy
  baseline for each measured prompt.

- Stop tokens: `FlashSpecConfig.eos_token_ids` and
  `SpeculativeEngine.generate(..., eos_token_ids=...)`. A row stops right
  after its first stop token (kept); the rest of the row is `-1`. The
  serve CLI sets it from the tokenizer, so completions end at EOS.

### Changed (public claims)
- Truth pass on the README, `pyproject.toml` description, `docs/index.md`,
  `docs/bandit.md`, `docs/kernels.md`, `paper/joss/paper.md`,
  `paper/flashspec.tex`, `CITATION.cff` and `.zenodo.json`. They now say
  what was measured and nothing more: one model pair
  (Qwen2.5-1.5B-Instruct verified by Qwen2.5-0.5B-Instruct), one Tesla
  T4, fp16, batch 1, with the compiled path on; the eager path was slower
  than plain HuggingFace `generate` in 7 of 8 cells of the earlier run.
  The bandit is "experimental, single-arm measured only". The Triton and
  assembly kernels moved under "Experimental kernels (not on the critical
  path)": the engine does not call them. The README decode-loop diagram
  no longer shows the Triton kernel, the install notes no longer call it
  GPU-accelerated, and the Docker lines build the image locally (none is
  published). The draft preprint no longer claims a distribution proof
  for the lossy `typical` variant, an expected chaos-test failure (fixed
  by #29), or novelty for bandit draft selection. Measured tables were
  moved, not edited; the software title in `CITATION.cff` and
  `.zenodo.json` changed, and the deposited preprint's title and DOI did
  not.

### Fixed
- Greedy first token now comes from the same computation as
  HuggingFace `generate`: the target prefills the whole prompt with
  `logits_to_keep=1` and scores the first draft token from that row. It
  used to hold the last prompt token back and score it as row 0 of the
  first `[last prompt token + draft]` forward. The two are equal in exact
  arithmetic but not in fp16. On TinyLlama MT-Bench prompt 131 the
  fp32 top-2 margin at token 0 is 0.0013 logits, and the two paths
  differed by up to 0.0195 logits in fp16 on CPU, enough to change the
  argmax (the Phase 4 T4 run diverged at token 0 on that prompt). After
  the change, the token-0 logits equal HuggingFace's exactly in fp32
  (max difference 0.0 on CPU). `benchmarks/greedy_mismatch.py` and
  `tests/integration/test_greedy_mismatch_gpu.py` rerun the three Phase 4
  mismatches on a GPU: fp32 must match, fp16 reports the top-2 margins.
- The HuggingFace baselines were not plain greedy or plain temperature
  sampling on checkpoints that ship generation defaults. Qwen2.5-Instruct
  ships `top_k=20`, `top_p=0.8` and `repetition_penalty=1.1`, and
  `generate` applied them (the repetition penalty even with
  `do_sample=False`). Loaded checkpoints now get a neutral
  `GenerationConfig()` (the shipped settings are recorded in
  `generation_config_overrides`), and sampled baselines pass `top_k=0`
  and `top_p=1.0`, because `GenerationConfig()` resolves `top_k` to 50
  when sampling. Earlier CPU toy `hf_sample` and `vanilla_ar` rows
  sampled with that top-k of 50. The TinyLlama checkpoint ships no
  sampling defaults, so its greedy rows are unaffected.
- `SamplingConfig.seed` is applied. Each `generate` call seeds a private
  `torch.Generator` used for every draw (drafts, acceptance uniforms,
  residual, bonus), so one seed reproduces one output and the global RNG
  is not touched. `seed=None` keeps the global RNG, as before.
- The bandit reward is emitted tokens (accepted drafts plus the residual
  or bonus token) per step-second. It used to count accepted drafts only,
  which ignored the token every step emits and penalised larger gamma.
  The accept count passed to the bandit is unchanged.
- `test_adversarial_swap_recovery_within_200_rounds` checked the wrong
  quantity ([issue 27](https://github.com/Mattral/FlashSpec/issues/27)).
  UCB1 does switch: at seed 55 it picks the new best arm in 197 of the 200
  rounds after the swap. The old best arm is then pulled only 3 times, so
  its per-arm window still holds pre-swap rounds (0.81), and comparing
  windowed means failed. The test now checks the selections (new best arm
  in at least 45 of the last 50 rounds) and is no longer marked xfail.

Postponed work is listed in [`docs/postponed.md`](docs/postponed.md).
No H100 run was made for 0.2.0.

### Changed
- `FlashSpecConfig.compile` now defaults to `"auto"`: the static KV cache
  and CUDA-graph path turns on when the engine device is CUDA and Triton
  is installed, and stays off otherwise (CPU, or CUDA without Triton).
  The engine logs one line saying which and why
  (`flashspec.utils.device.resolve_compile`; `engine.compile_decision`
  holds it). `True` and `False` still force it on or off. The serve CLI
  takes `--compile {auto,on,off}` (default `auto`). The benchmark
  harness keeps `flashspec_ucb` eager explicitly; `flashspec_compiled`
  takes `flashspec_compile: true | auto` from the config (default
  `true`, as in the Phase 5 bundles) and records the requested value
  and the decision. The greedy gold test and the greedy-mismatch
  diagnostic pin `compile=False`, since their reference is eager
  `generate()`.
- Batch-1 fast path (`variant` `"rejection"` or `"greedy"`): one
  device-to-host read per step instead of several, no
  `torch.cuda.synchronize()` per step, `exp()` only on the one vocab row
  the residual or bonus needs, a preallocated sequence buffer instead of
  `torch.cat` growth. `rejection_step` keeps Algorithm 1's acceptance
  test and residual formula; with a fixed seed the emitted tokens are
  unchanged. Batches > 1 and `variant="typical"` keep the context-based
  loop.
- Batch-1 speculative steps run one target forward instead of two, and
  `gamma` drafter forwards instead of `gamma + 1`. The target used to
  re-forward the residual or bonus token alone after every verify to
  catch its cache up. The caches now crop by the rejected count and hold
  the next token as pending, so it rides along in the next verify (or
  first draft) forward. New incremental methods: `TargetModel.begin`,
  `verify`, `commit`; `HuggingFaceDrafter.begin`, `draft`, `commit`,
  `push_pending`. The context-based methods are unchanged and still used
  for batches larger than one and for wrappers that lack the new methods.
  With a fixed seed the emitted tokens are identical to before.
- The prompt prefill keeps one logit row (`logits_to_keep=1`) instead of
  materialising `(seq, vocab)` logits.
- `FLASHSPEC_DEBUG_CACHE=1` asserts after every step that each cache
  matches the accepted sequence.
- `paper/flashspec.tex` is a draft preprint for a later Zenodo version
  (October 2026, software 0.1.5, ORCID 0009-0005-0853-8301, no
  affiliation). The citable record remains
  10.5281/zenodo.20766119 (June 2026) until the maintainer deposits the
  draft. The draft is not an arXiv submission, and the JOSS-format text
  is not under review.
- The NF4 figures 44.2 tok/s, acceptance 0.75, and p50 22.1 ms are
  UNVERIFIED. An older notebook cell assigns them, and no timing log is
  in the repository, so they are not used as a result. The JSON file and
  the notebook output stay. The paper table and the README table are
  removed.
- PyPI publishes through `.github/workflows/publish.yml` with Trusted
  Publishing. The `pypi` environment requires a reviewer. Token upload
  instructions are not part of this repository.
- `torch.onnx.export` stays on the legacy tracer (`dynamo=False`).
  Torch 2.9 makes the dynamo exporter the default, and that path imports
  `onnxscript`, which the `onnx` extra does not install. The parity tests
  export a tensor-returning copy of the toy model, because the toy forward
  returns a namespace and the tracer rejects that. The variable-sequence
  check uses an embedding and a linear layer. The transformer trace bakes
  the example sequence length into attention.
- The CPU coverage gate is 83% line coverage, the measured
  unit-and-integration coverage rounded down. The lint job runs
  `ruff check`, `ruff format --check`, and import-linter.
  The layer order puts `serve` and `engine` above the sibling modules,
  and `utils` at the base. `mypy --strict` is `make typecheck` and is
  not part of that job.
  `test_adversarial_swap_recovery_within_200_rounds` is an expected
  failure at seed 55
  ([issue 27](https://github.com/Mattral/FlashSpec/issues/27)).
  The mark is strict. A LaTeX job builds `paper/flashspec.tex` and
  uploads the PDF. The PyPI project URL for the paper points at
  10.5281/zenodo.20766119.
- The PyPI project `flashspec` at 0.1.2 was yanked by the maintainer.
  Do not install `flashspec==0.1.2`. There is no `v0.1.2` tag.
  `flashspec-asm-kernel` 0.1.2 is a different package and remains the
  optional Linux wheel.

### Added
- 0.1.5 is on PyPI as `flashspec-0.1.5-py3-none-any.whl` and
  `flashspec-0.1.5.tar.gz`. The git tag `v0.1.5` points at the changelog
  cut, and the GitHub Release is
  https://github.com/Mattral/FlashSpec/releases/tag/v0.1.5.
  Release titles for v0.1.0, v0.1.1, v0.1.3, v0.1.4, and v0.1.5 use the
  form `FlashSpec vX.Y.Z — short title`. The tag commits did not move.

### Added (0.2.0 measurement setup, no numbers yet)
- `benchmarks/configs/qwen_t4_compile_auto.yaml`: `qwen_t4_compile.yaml`
  plus `flashspec_compile: auto`, so `flashspec_compiled` runs the way a
  default user gets it. Notebook cell 1 takes `COMMIT` (a commit on
  `main`, checked out exactly and verified to be on `main`), and cell 3c
  runs this config into one bundle. It refuses to download unless every
  `flashspec_compiled` result records `requested: auto`, `effective: true`
  and no eager fallback. The 0.2.0 README tables come from that bundle.

## [0.1.5] — 2026-10-16 — Honest CPU path, optional ASM, small server

Development status stays Alpha. This version does not claim a production
deployment or an H100 speedup. It collects the 0.1.5 tree, including the
notes that were waiting under Unreleased. PyPI's previous upload is 0.1.4.

### Changed
- The README quickstart loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with
  `Felladrin/Llama-160M-Chat-v1` and injects those modules into
  `load_drafters` and `TargetModel`. Calling `TargetModel` with only a
  hub id passes `device_map` and needs `accelerate`, which is not a
  dependency. Both checkpoints report vocabulary 32000. A Llama-3 model
  reports vocabulary 128256 and must not be paired with that drafter.
  `notebooks/01_quickstart.py` stays the toy demo.
- `docs/kernels.md` titles the H100 section "Roofline analysis, not a
  profile". The figures stay arithmetic from the NVIDIA H100 whitepaper.
  No H100 profile was added.
- `SpeculativeEngine.generate` is documented as serial over context
  length. Each loop forwards only the largest equal-length group. The KV
  cache rolls only when that group has one row. A larger group
  full-forwards. Shorter output rows are filled with `-1`. That fill is
  not an attention mask, and the engine does not pad prompts into one
  masked batch.
- `docs/architecture.md` states the expected tokens per target forward as
  `(1 - α^(γ+1)) / (1 - α)`, about 3.05 at α=0.75 and γ=4. The γ·α+1
  line is removed. That figure is not a measured speedup. The overview
  no longer says the residual is KS-verified on every CI run. The check
  is the chi-squared test, and the lint job is off, so the CI jobs that
  need it do not run.
- The paper's expected-token formula is
  `(1 - α^(γ+1)) / (1 - α)`, about 3.05 at α=0.75 and γ=4.
  The sentence that called that figure a measured throughput improvement
  is removed, because the NF4 JSON has no autoregressive baseline.
- The H100 comparison table is removed from `paper/flashspec.tex`.
  Those tok/s figures are unmeasured design targets. Medusa and EAGLE
  are not implemented. The figures stay listed in
  `benchmarks/results/PROVENANCE.md`.
- README result tables use three significant figures. Full precision and
  the run notes stay in `benchmarks/results/` and
  `benchmarks/results/README.md`.
- The README and the papers no longer say that most implementations move
  accept/reject to the CPU. They describe the Triton kernel and the
  pure-PyTorch reference, and the measured Tesla T4 batch-1 result in
  which the Triton kernel was slower. The papers also stop saying that CI
  runs lint, type checking, a coverage gate, or a Kolmogorov-Smirnov test.
  The lint job is off, so the jobs that depend on it do not run. The suite's
  distribution check is a chi-squared test.
- The bandit `update` accepts an optional `reward`. The engine passes
  accepted draft tokens divided by that step's wall time. The raw accept
  count is still recorded. Thompson's Beta posterior stays on that count,
  because tokens per second are not a Bernoulli trial. Omitting `reward`
  keeps the previous UCB mean. The bonus token is not part of the count.
- The `asm` extra requires `flashspec-asm-kernel` >= 0.1.2 and < 2 on
  Linux x86-64 (`sys_platform == "linux"`). The kernel runs only when
  each probability row and the residual output row are CPU float32,
  contiguous, and do not require grad. A missing library, including the
  0.1.0 `OSError` or `ImportError`, keeps the pure-PyTorch residual.
  Release 0.1.2 ships `libflashspec_asm.so` in a manylinux wheel.

### Fixed
- The drafter, the verifier, and the autoregressive baseline reuse
  `past_key_values` for a batch of one. After each step both caches are
  cropped back to the accepted prefix, including the bonus token when
  one was emitted. A model that does not return a cache, and any batch
  larger than one, still forwards the full sequence.
- A fully accepted speculative step now emits the bonus token. The
  verifier returns ``gamma + 1`` positions when ``include_bonus`` is
  set, and the engine appends one sample from the last row only when
  every draft token was accepted and ``max_new_tokens`` still has room.
  A rejection does not emit that token.

### Removed
- The README visits badge (`hits.sh`) and the pepy.tech total-download badge. The PyPI version, Python versions, monthly downloads, docs, DOI, citation, license, and Colab badges stay.
- The disabled CI coverage job. Pytest still fails under 95% coverage
  via `addopts`. The lint job stays off until ruff, ruff format, and
  mypy --strict would pass.
- The README table of unmeasured Llama-3-8B H100 design-target tok/s
  for vanilla autoregressive decoding, Medusa, EAGLE, and FlashSpec.
  No H100 comparison exists, and Medusa and EAGLE are not implemented.
  Those figures were design targets. They were not removed from a
  result file. The same table was later removed from `paper/flashspec.tex`.
  The figures stay listed as unmeasured design targets in
  `benchmarks/results/PROVENANCE.md`.

### Added
- CPU float32 TinyLlama-1.1B-Chat with Felladrin/Llama-160M-Chat-v1,
  HumanEval/23, 16 new tokens, no warmup, one thread, git `f59b5b4`.
  FlashSpec 1.7507264092133707 tok/s, vanilla autoregressive decoding
  4.2977579080394 tok/s, speedup 0.40735807988124606, acceptance
  0.11363636363636363. The files are
  `benchmarks/results/tinyllama_cpu_humaneval_flashspec_ucb_f59b5b4.json`
  and `benchmarks/results/tinyllama_cpu_humaneval_vanilla_ar_f59b5b4.json`.
- `notebooks/colab_triton_tests.py` runs the Triton GPU parity tests and
  writes a log to download. Those tests are not run in CI.
- Fixed HumanEval and MT-Bench prompt sets under `benchmarks/prompt_sets/`.
  The harness tokenizes them with the target tokenizer and records the
  prompt ids beside acceptance. Alpaca text is not included
  (CC BY-NC 4.0). No acceptance figure for these sets is published yet.
- Chi-squared tests that the first emitted token, and the second when
  gamma is 4, follow the target distribution for fixed `p != q`. An
  engine test covers a toy target whose next token depends on context.
  The older gamma=1, draft-equals-target test only checks acceptance.
- `notebooks/04_gpu_measurement.ipynb` builds flashspec-asm from source
  when the installed package does not select the ASM backend.
- CPU tests for typical acceptance, the Linux AVX2 flag reader, ONNX
  export guards, and the serve command wiring.

### Added
- Optional extra `asm`: `flashspec-asm-kernel` on Linux x86-64 only.
  The CPU residual uses that kernel when the ABI is 1, AVX2 is present,
  and the inputs are contiguous float32. Every other case, including a
  kernel exception, stays on pure PyTorch. AVX2 `vmaxps` maps NaN to 0;
  PyTorch preserves NaN. Finite inputs are required to match within 1e-5.
  Upstream Linux timings live in flashspec-asm `bench/results/phase5_benchmark.txt`
  and are not copied into this repo.
- Optional extra `serve`: one worker, a queue, batches only when prompt
  length and `max_tokens` match, and a hand-rolled Prometheus text endpoint.
  `POST /v1/chat/completions` and `POST /v1/completions`. No streaming,
  no continuous batching, no paged KV, no multi-GPU. Temperature and gamma
  are fixed at process start. The JSON `model` field does not switch
  checkpoints.
- `load_drafters` / `register_hf` for two or more HuggingFace drafters.
  A single arm logs that the bandit cannot adapt.
- `benchmarks/harness.py` writes a measured JSON or nothing. CUDA without
  a GPU, and a checkpoint that fails to load, write no file.
  Checkpoint loads use the config `dtype` (`float32`, `float16`, or
  `bfloat16`). A drafter and target with different vocabulary sizes
  raise before any result file is written.
  `benchmarks/configs/tiny_cpu.yaml` is an in-process random network.
  `benchmarks/configs/tinyllama_gpu.yaml` and
  `notebooks/04_gpu_measurement.ipynb` are the T4-or-newer measurement.
  The notebook downloads a zip; Colab Drive is not the repo.
- `benchmarks/bench_residual.py` and `scripts/demo_bandit_mixed.py`.

### Changed
- `SpeculativeEngine.generate` keeps one context per row. Rows that accept
  different lengths no longer have to be rectangular for `Tensor.view`.
  Tokens after the first rejection are discarded. A fully accepted step
  keeps the draft tokens. A residual whose mass is ~0 (draft matches
  target) samples from the target distribution instead of calling
  multinomial on an all-zero vector.
- `benchmarks/results/baseline.json` is a schema template with empty
  metrics. The previous all-zero H100 document was not a run.
- Gamma and draft-size CSVs carry a `measurement_kind` column.
- Version synced to 0.1.5 in `pyproject.toml`, `flashspec/__init__.py`,
  and `CITATION.cff`.

### Fixed
- PyPI `flashspec-asm-kernel` 0.1.0 is a source distribution with no
  `libflashspec_asm.so`. The package imports, then the first call raises
  `ImportError`. That error keeps the pure-PyTorch residual, so
  `import flashspec` succeeds on Linux after `pip install flashspec[asm]`.
- `benchmarks/bench_residual.py` writes `timestamp` and lists the
  end-to-end metric names in `unmeasured`. `residual_cpu.json` records
  the `cb02ca1` commit time `2026-10-15T06:26:21Z` for the timestamp
  the original script omitted. Its millisecond values are unchanged.

### Measured here
CPU toy run, git `da8182b`, one thread. The host machine is not named
in this changelog. The Linux ASM library did not load, so ASM and the
C baseline are unmeasured on that run. Full figures are in the JSON
files under `benchmarks/results/`.

- Toy random transformer, `n_samples=3`, γ=4, one drafter:
  FlashSpec 807.876431892803 tok/s, acceptance 0.3115079365079365,
  speedup vs vanilla AR 0.3478541784757594. Vanilla AR
  2322.725711556072 tok/s. Speculative decoding was slower on this toy.
- Residual microbench, 30 iterations, pure PyTorch: V=1024 residual
  0.018213333290380735 ms, verification 0.4179066666741467 ms; V=4096
  residual 0.01968333341816712 ms, verification 0.522750000012214 ms.
- Synthetic two-drafter switch, `n_samples=1`: UCB acceptance
  0.5555555555555556 and 5794.162381923592 tok/s, against fixed-drafter
  peaks of 0.2 acceptance and 3215.4340838834096 tok/s. That win is
  this controlled switch. It is not a language-model result, and those
  tok/s are not comparable to the toy end-to-end row.
- GitHub-hosted Linux runner, git `4e48c55`, torch 2.14.1+cpu, AVX2,
  30 iterations. `flashspec-asm` loaded (`backend=asm`). Residual means
  in ms: vocab 1024 PyTorch 0.011835766664110755, C 0.010876566682327393,
  ASM 0.01513639999378332; vocab 4096 PyTorch 0.011573599992213227,
  C 0.014012900006340109, ASM 0.014001300003959235; vocab 32000 PyTorch
  0.030213800005185476, C 0.04299199999877601, ASM 0.024774433317512983.
  Verification with ASM selected was slower than pure PyTorch at all
  three sizes. File: `benchmarks/results/residual_linux_gha.json`.
- Second GitHub-hosted Linux run, git `1b0cda3`, timestamp
  `2026-10-15T07:18:40.745328+00:00`, same Python and torch, backend
  `asm`. Residual means in ms: vocab 1024 PyTorch 0.0068616666718905135,
  C 0.006093900015002873, ASM 0.008056400001047829; vocab 4096 PyTorch
  0.006851300016326907, C 0.007845133351717474, ASM 0.008722400002625363;
  vocab 32000 PyTorch 0.012439799998749853, C 0.02345496665914955, ASM
  0.015084966666260394. ASM residual means are higher than PyTorch at
  all three sizes. The vocab-32000 ordering from the first file did not
  repeat. File: `benchmarks/results/residual_linux_gha_1b0cda3.json`.
- Colab Tesla T4, git `cb02ca1`, Python 3.13.15, torch 2.11.0+cu130,
  CUDA 13.0. TinyLlama-1.1B-Chat float16 with drafter
  `Felladrin/Llama-160M-Chat-v1`, `n_samples=8`, synthetic token ids,
  γ=4. FlashSpec 17.576979879136424 tok/s, acceptance
  0.20834964273981926, speedup vs vanilla AR 0.6061833262227975.
  Vanilla AR 29.006488905704696 tok/s. Speculative decoding was slower
  on this pair. Process peak memory 2.3778347969055176 GB. The ASM
  cell on that runtime failed and wrote no file.

---

## [0.1.4] — 2026-06-15 — First real GPU measurements

### Added
- First real GPU measurements (Tesla T4, TinyLlama-1.1B-Chat NF4, FlashSpec UCB1,
  γ=4): **44.2 tok/s**, α=0.75, p50=22.1ms.
  Result committed to `benchmarks/results/flashspec_ucb_tiny_llama.json`.
- Bandit regret empirically measured (T=10,000, K=3 arms, true_rates=[0.3,0.7,0.5]):
  UCB1=100.2, Thompson=18.9, Oracle=0.0. Theory bound O(√(KT log T))≈526.
  Figure committed to `paper/figures/bandit_regret.jpg`.
- Gamma sweep results committed to `benchmarks/results/gamma_sweep.csv`
  (acceptance rates genuine; throughput values reflect sampling kernel only,
  not end-to-end inference — see CSV for details).
- Draft model size sweep committed to `benchmarks/results/draft_size_sweep.csv`.
- Notebooks updated with real Colab T4 execution outputs:
  `notebooks/01_quickstart.ipynb`, `02_bandit_analysis.ipynb`,
  `03_kernel_profiling.ipynb`.
- Triton kernel profiling on Tesla T4 committed (notebook 03). Reference
  kernel: 0.07–0.72ms. Triton kernel: faster at batch=8 (1.6×), slower at
  batch=1 (0.2×) on T4 due to HBM bandwidth constraints — expected.
  No H100 profile has been committed.
- README updated: all real T4 measurements separated from H100 design targets;
  ⊛ symbol marks every unverified number.



### Fixed
- **Critical**: `pip install flashspec` failed completely on Windows and
  macOS with `ERROR: No matching distribution found for triton>=3.0.0` —
  Triton publishes official PyPI wheels for Linux only, but `triton>=3.0.0`
  was a hard core dependency, making the package uninstallable on those
  platforms. (Reported against v0.1.0–v0.1.2.)
  - `pyproject.toml`: moved `triton` out of `dependencies` into a new
    `gpu` extra with a `platform_system == 'Linux'` environment marker:
    `pip install flashspec[gpu]`.
  - `flashspec/kernels/__init__.py`: now imports `verify_kernel` and
    `gather_kernel` inside a `try/except ImportError`. If Triton is
    unavailable, `verify_tokens` and `gather_accepted` are bound to a
    wrapper that raises an actionable `ImportError` only when *called*
    (not at import time), explaining `pip install flashspec[gpu]`
    (Linux-only) and pointing to the cross-platform
    `flashspec.kernels._reference` module (identical numerics, verified in
    `tests/unit/test_verify_kernel.py`). Adds module-level
    `flashspec.kernels.TRITON_AVAILABLE: bool`.
  - `tests/unit/test_verify_kernel.py`: added
    `TestKernelsPackageWithoutTriton` (3 tests, CPU-only, no GPU/Triton
    required) verifying the package imports cleanly and raises the
    actionable error when Triton is absent.
  - `.github/workflows/{ci,gpu_tests,benchmark}.yml`: updated `pip install`
    steps to request the `gpu` extra wherever Triton-backed kernel tests
    or benchmarks actually run (all on `ubuntu-22.04`/self-hosted Linux
    runners, where the extra resolves normally).
  - `README.md`: rewrote the Installation section — `pip install flashspec`
    is now correctly described as cross-platform (CPU reference kernels),
    with `pip install flashspec[gpu]` for Linux+CUDA accelerated kernels.
  - Bumped version `0.1.0` → `0.1.3` in `pyproject.toml` and
    `flashspec/__init__.py` (0.1.0–0.1.2 are broken on non-Linux and
    should be marked yanked on PyPI after this release).

---
- Initial project scaffold following AGENTS.md specification.
- `flashspec.utils.config`: Pydantic v2 configuration models (`FlashSpecConfig`,
  `BanditConfig`, `SamplingConfig`, `MetricsConfig`). All immutable (`frozen=True`).
- `flashspec.utils.logging`: Structured JSON logger via `get_logger`.
- `flashspec.utils.device`: Device detection, seeding (`set_seed`), and MFU helpers.
- `flashspec.kernels._reference`: Pure-PyTorch reference implementations of
  `verify_tokens_reference` and `gather_accepted_reference` (tests use only).
- `flashspec.kernels.verify_kernel`: Triton token-verification kernel with
  `@triton.autotune` over 4 tile configs; handles float32 and bfloat16.
- `flashspec.kernels.gather_kernel`: Triton gather/scatter helpers.
- `flashspec.sampling.rejection`: Algorithm 1 (Leviathan et al. 2023) exact
  speculative sampling with correct residual distribution. Temperature applied
  by caller before log-softmax per §7.
- `flashspec.sampling.typical`: Typical-acceptance sampling variant.
- `flashspec.bandit.base`: `DraftSelector` abstract base class with JSON
  serialisation contract and windowed statistics; thread-safe via `threading.Lock`.
- `flashspec.bandit.ucb`: UCB1 selector satisfying O(√(KT log T)) regret bound.
- `flashspec.bandit.thompson`: Thompson sampling selector with Beta conjugate prior.
- `flashspec.bandit.oracle`: Oracle selector for regret upper-bound experiments.
- `flashspec.engine.drafter`: `DraftModel` protocol and entry-point registry.
- `flashspec.engine.verifier`: `TargetModel` wrapper with temperature applied
  before `log_softmax` (§7 compliance).
- `flashspec.engine.speculative`: `SpeculativeEngine` main orchestrator.
- `flashspec.metrics.acceptance`: Token acceptance rate tracker.
- `flashspec.metrics.throughput`: Tokens/s and MFU tracker.
- `flashspec.metrics.latency`: p50/p95/p99 latency tracker.
- `flashspec.export.onnx`: ONNX export for draft models (opset ≥ 17).
- `tests/unit/test_verify_kernel.py`: 13 CPU reference tests + 6 GPU Triton
  parity tests (`@pytest.mark.gpu`) + 1 Hypothesis property test.
- `tests/unit/test_sampling.py`: All §5.2 mandatory invariants + 3 Hypothesis
  property tests (shape, first_rejection range, padding invariant).
- `tests/unit/test_bandit.py`: Convergence, regret bound, serialisation,
  adversarial swap, thread-safety + 3 Hypothesis property tests.
- `tests/unit/test_acceptance.py`: Full AcceptanceTracker coverage.
- `tests/unit/test_config.py`: Pydantic validation + 3 Hypothesis property tests.
- `tests/integration/test_e2e_greedy.py`: 4 tests (greedy equiv + 3 edge cases).
- `tests/integration/test_e2e_sampling.py`: §2.1 hard gate at N=10,000 samples
  + 4 edge cases. Replaces the previous N=1,000 fast gate.
- `tests/integration/test_onnx_parity.py`: §2.4 parity for batch 1..8, batch
  9..32, variable seq_len, invalid opset edge case.
- `tests/chaos/test_bandit_adversarial.py`: 0%, 100%, many-arms, swap scenarios.
- `benchmarks/run_all.py`: §6-compliant runner — `set_num_threads(1)`, 50
  warm-up, mean±std over 200 steps, `synchronize()`, `max_memory_allocated()`.
- `benchmarks/compare_baselines.py`: FlashSpec vs Medusa vs EAGLE vs AR.
- `benchmarks/sweep_draft_sizes.py`: Acceptance rate vs parameter count ablation.
- `benchmarks/sweep_gamma.py`: γ sweep (speculation length vs throughput).
- `benchmarks/benchmark_kernels.py`: §13.5 kernel roofline benchmark (HBM
  bandwidth model, memory-efficiency ratio, per-shape JSON output).
- `benchmarks/results/baseline.json`: Stub with exact §14 JSON schema.
- `scripts/check_regression.py`: Per-model regression thresholds from §6
  (Llama-3-8B < 1.8×, Llama-3-70B < 2.2×, Mistral-7B < 1.7×, kernel < 1 ms).
- `scripts/profile_kernel.py`: Nsight/torch.profiler latency table.
- `scripts/export_draft.py`: CLI ONNX export with parity verification.
- `scripts/download_models.py`: HuggingFace Hub checkpoint downloader.
- `scripts/update_benchmark_results.py`: §14 schema validator + git commit.
- `docs/architecture.md`: Mermaid sequence + component diagrams + correctness
  guarantee section explaining Algorithm 1 distribution preservation.
- `docs/kernels.md`: Tiling strategy, SRAM analysis, roofline for both kernels.
- `docs/bandit.md`: UCB1 regret proof sketch, convergence figure, windowed
  statistics justification.
- `docs/benchmarks.md`: Reproduction guide for every paper number.
- `mkdocs.yml` at repo root: MkDocs site configuration (Material theme,
  mkdocstrings, Mermaid support).
- `.readthedocs.yaml`: Read the Docs build config (Python 3.11, ubuntu-22.04,
  MkDocs Material auto-install). Required by Read the Docs for all new projects.
- `docs/requirements.txt`: Pinned doc dependencies for reproducible RTD builds.
- `paper/flashspec.tex`: 9-section LaTeX paper (Introduction, Background,
  Method, Experiments, Ablation, Analysis, Related Work, Conclusion,
  Reproducibility Statement).
- `paper/flashspec.bib`: All 10 references (Leviathan, Auer, Thompson, Cai,
  Li, Chen, Chapelle, Dao, Triton, Mattral self-citation).
- `paper/figures/.gitkeep`: Vector-figure output directory for the paper
  (PDF/SVG only, generated by notebooks 02/03).
- `notebooks/01_quickstart.ipynb`: Colab-ready quickstart, < 10 min runtime.
- `notebooks/02_bandit_analysis.ipynb`: Regret curve plots → `paper/figures/`.
- `notebooks/03_kernel_profiling.ipynb`: Reference vs Triton latency table.
- `.github/workflows/ci.yml`: 6-job pipeline in exact §12.1 order — lint
  (+ pip-audit CRITICAL CVE gate), test-cpu, test-chaos, coverage (≥ 95%),
  onnx-parity, docs-build. All jobs on ubuntu-22.04 + TRITON_CPU_BACKEND=1.
- `.github/workflows/gpu_tests.yml`: test-gpu (A10G, N=10,000 KS gate) +
  bench-quick (fails if throughput < 1.5× AR).
- `.github/workflows/benchmark.yml`: bench-full (H100) → Trivy image scan →
  `update_benchmark_results.py` → git commit + push → Slack alert on failure.
- `CONTRIBUTING.md`: Commit format, branching, PR rules, autonomous vs
  human-approval boundaries (§9, §13.1, §13.2).
- `deploy/Dockerfile`: CUDA 12.4 image.
- `deploy/docker-compose.yml` and `deploy/k8s/flashspec-deployment.yaml`.
- `social/x_thread.md`: X (Twitter) launch thread (long + short variants).
- `social/linkedin_post.md`: LinkedIn launch post with engineering narrative.
- `paper/joss/paper.md` + `paper/joss/paper.bib`: JOSS submission (673 words,
  within the 250–1000 word requirement; all citation keys verified present).
- `paper/joss/README.md`: JOSS submission checklist and step-by-step guide.
- `CITATION.cff`: GitHub-native citation metadata (renders "Cite this
  repository" button); includes `preferred-citation` block.
- `.zenodo.json`: Zenodo archival metadata, activates on first GitHub Release.
- `PUBLISHING.md`: Master guide sequencing GitHub Release → Zenodo → JOSS →
  arXiv → social posts, with clear blockers identified per deliverable.

### Fixed
- `engine/speculative.py`: Renamed local variable `tps` → `tokens_per_second`
  (§3.4 naming convention violation).
- `engine/drafter.py`: Replaced bare `except Exception: return` with logged
  exception (§13.6 banned silent-except pattern).
- `sampling/rejection.py`: Removed unused `temperature` parameter — temperature
  is the caller's responsibility per §7.
- `engine/verifier.py`: Applied temperature scaling to raw logits **before**
  `log_softmax` (§7 compliance). Added `temperature` parameter to `score_draft`.
- `engine/speculative.py`: Passes `temperature` to `score_draft` instead of
  (incorrectly) forwarding it to `rejection_sample`.
- `engine/verifier.py`: Fixed two lines exceeding 100 characters (§3.1);
  `# type: ignore` annotations now carry their explanatory comment on the
  same line as required by §13.2.
- `metrics/throughput.py`: Replaced `tps` with `tokens_per_second` in
  docstring example (§3.4).
- `benchmarks/results/baseline.json`: Removed non-spec `_note` field; schema
  now matches §14 exactly.
- `benchmarks/run_all.py`: Removed non-spec `schema_version` from result dict;
  added `torch.cuda.synchronize()` around all timed regions (§6).
- `scripts/update_benchmark_results.py`: Removed `schema_version` from required
  keys to match §14 schema.
- `tests/integration/test_e2e_sampling.py`: Upgraded CI hard gate from N=1,000
  to N=10,000 (§2.1 requirement). Added 4 edge cases.
- `tests/unit/test_acceptance.py`, `tests/unit/test_bandit.py`,
  `tests/unit/test_config.py`: Added failure messages to all bare `assert`
  statements (§5.3); `test_config.py` rewritten with 3 Hypothesis property tests.
- `docs/mkdocs.yml`: Moved from `docs/` to repo root (MkDocs requirement).
  `docs/mkdocs.yml` stub retained per §1 layout tree.
- `flashspec/kernels/_reference.py`: Fixed a 102-character line (§3.1).
- `AGENTS.md`: Copied into the repository root (§1 — was missing).
- `paper/flashspec.bib`: **Critical** — repaired a malformed `triton2019`
  entry left over from a prior edit. The `@inproceedings{triton2019,` header
  and `title` field had been accidentally deleted, leaving orphaned fields
  (`author`, `booktitle`, `year`) with no entry header — this would have
  caused `bibtex` to fail and `make paper` to error out. Verified post-fix:
  10 entries, balanced braces, all 9 `\cite{}` keys in `flashspec.tex`
  resolve correctly (`mattral2025flashspec` is defined-but-unused by design —
  it is the self-citation for external use, not cited within the paper).

---

## [0.1.0] — 2025-11-01

- Initial release (scaffold only; no real model weights required).
