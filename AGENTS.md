# FlashSpec — Agent & Copilot Instruction Manual

> **Canonical reference for every AI code editor, copilot, and automated agent
> working on this repository.**
> Human reviewers treat this file as the project constitution.
> If a generated change contradicts any rule here, reject it without discussion.

---

## 0. Quick orientation

| Item | Value |
|---|---|
| Project name | `flashspec` |
| One-line purpose | Adaptive speculative-decoding inference engine with Triton-optimised verification and online bandit draft selection |
| Target publication venues | MLSys 2026 · NeurIPS 2025 Efficient NLP Workshop · ICLR 2026 |
| Primary language | Python 3.11 + CUDA C++ / Triton |
| Primary frameworks | PyTorch 2.x · Triton 3.x · HuggingFace Transformers |
| Hardware targets | NVIDIA H100 SXM5 (primary) · A100 80 GB · A10G (CI) |
| License | Apache 2.0 |
| Style authority | This file > any linter default > personal preference |

Read **all** sections before writing a single line of code or documentation.

---

## 1. Repository layout (authoritative)

```
flashspec/
├── AGENTS.md                        ← you are reading this
├── README.md                        ← project overview and quick-start
├── CHANGELOG.md                     ← semver log, updated on every PR
├── CONTRIBUTING.md                  ← external contributor guide
├── LICENSE
├── pyproject.toml                   ← single source of truth for deps + tooling
├── Makefile                         ← canonical task runner (see §11)
│
├── flashspec/                       ← installable Python package
│   ├── __init__.py                  ← public API surface, nothing else
│   ├── py.typed                     ← PEP 561 marker
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── speculative.py           ← SpeculativeEngine orchestrator
│   │   ├── drafter.py               ← DraftModel protocol + registry
│   │   └── verifier.py              ← TargetModel wrapper
│   │
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── verify_kernel.py         ← Triton token-verification kernel (core)
│   │   ├── gather_kernel.py         ← Triton gather / scatter helpers
│   │   └── _reference.py            ← Pure-PyTorch reference impl (tests use this)
│   │
│   ├── bandit/
│   │   ├── __init__.py
│   │   ├── base.py                  ← DraftSelector abstract class
│   │   ├── ucb.py                   ← UCB1 selector
│   │   ├── thompson.py              ← Thompson sampling selector
│   │   └── oracle.py                ← Oracle selector (upper-bound baseline)
│   │
│   ├── sampling/
│   │   ├── __init__.py
│   │   ├── rejection.py             ← Standard speculative-sampling (Algorithm 1)
│   │   └── typical.py               ← Typical-acceptance variant
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── acceptance.py            ← Token-acceptance-rate tracker
│   │   ├── throughput.py            ← Tokens/s, wall-clock, MFU
│   │   └── latency.py               ← p50/p95/p99 latency tracker
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   └── onnx.py                  ← ONNX export for draft models
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                ← Pydantic v2 config models
│       ├── logging.py               ← structured JSON logger
│       └── device.py                ← device detection helpers
│
├── benchmarks/
│   ├── README.md                    ← how to reproduce every number in the paper
│   ├── run_all.py                   ← single entry-point for all benchmarks
│   ├── compare_baselines.py         ← flashspec vs Medusa vs EAGLE vs vanilla AR
│   ├── sweep_draft_sizes.py         ← acceptance rate vs draft model size ablation
│   ├── sweep_gamma.py               ← speculation length γ sweep
│   ├── configs/
│   │   ├── llama3_8b.yaml
│   │   ├── llama3_70b.yaml
│   │   └── mistral_7b.yaml
│   └── results/                     ← committed CSV/JSON benchmark artifacts
│       └── .gitkeep
│
├── tests/
│   ├── unit/
│   │   ├── test_verify_kernel.py    ← numerical parity: Triton vs reference
│   │   ├── test_sampling.py         ← distribution correctness (KS test)
│   │   ├── test_bandit.py           ← regret bounds, convergence
│   │   ├── test_acceptance.py       ← acceptance-rate tracker correctness
│   │   └── test_config.py           ← Pydantic model validation
│   ├── integration/
│   │   ├── test_e2e_greedy.py       ← greedy speculative == greedy autoregressive
│   │   ├── test_e2e_sampling.py     ← output distribution equivalence
│   │   └── test_onnx_parity.py      ← ONNX draft == PyTorch draft (atol 1e-5)
│   ├── chaos/
│   │   └── test_bandit_adversarial.py  ← bandit under adversarial acceptance rates
│   └── conftest.py                  ← shared fixtures, tiny toy models
│
├── notebooks/
│   ├── 01_quickstart.ipynb          ← colab-ready, ≤10 min runtime
│   ├── 02_bandit_analysis.ipynb     ← UCB vs Thompson regret plots
│   └── 03_kernel_profiling.ipynb    ← Nsight / torch.profiler walkthrough
│
├── docs/
│   ├── mkdocs.yml
│   ├── architecture.md
│   ├── kernels.md
│   ├── bandit.md
│   ├── benchmarks.md
│   └── api/                         ← auto-generated from docstrings
│
├── paper/
│   ├── flashspec.tex                ← LaTeX source (MLSys / NeurIPS template)
│   ├── flashspec.bib
│   ├── figures/                     ← vector figures (PDF/SVG only, no PNGs)
│   └── Makefile
│
├── scripts/
│   ├── profile_kernel.py            ← Nsight Systems / torch.profiler helper
│   ├── export_draft.py              ← CLI: export a draft model to ONNX
│   └── download_models.py           ← pull HF checkpoints for benchmarks
│
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s/
│       └── flashspec-deployment.yaml
│
└── .github/
    └── workflows/
        ├── ci.yml                   ← lint + unit + integration (CPU)
        ├── gpu_tests.yml            ← GPU integration (self-hosted runner)
        └── benchmark.yml            ← nightly benchmark regression
```

**Rules enforced by CI:**
- No Python file outside `flashspec/`, `benchmarks/`, `tests/`, `scripts/`, or `notebooks/`.
- No import of `flashspec.kernels` from inside `flashspec.engine` without going through `flashspec.kernels.__init__`.
- No circular imports. Verified by `import-linter`.

---

## 2. Core algorithmic contracts

These contracts are **immutable**. No agent may change them without a corresponding
change to the paper's algorithm pseudocode and an update to the relevant test.

### 2.1 Speculative decoding correctness (Algorithm 1)

Given target model distribution $p$ and draft model distribution $q$:

```
for i in 1..γ:
    x_i ~ q(· | context)
accept_i = min(1, p(x_i | context) / q(x_i | context))
```

The output distribution of `SpeculativeEngine.generate()` **must be identical**
to autoregressive sampling from the target model $p$.
This is verified by `tests/integration/test_e2e_sampling.py` using a KS test
at significance level α = 0.01 over 10,000 samples.
**CI fails if this test is red. No exceptions.**

### 2.2 Triton verification kernel contract

`flashspec.kernels.verify_kernel.verify_tokens(draft_logprobs, target_logprobs, u)`
must satisfy:

```
max(|output - reference_output|) < 1e-5   (float32)
max(|output - reference_output|) < 1e-3   (bfloat16)
```

where `reference_output` is computed by `flashspec.kernels._reference.verify_tokens_reference`.
Verified by `tests/unit/test_verify_kernel.py` on random inputs of shape
`(batch=8, gamma=8, vocab=32000)`.

### 2.3 Bandit regret bound

The UCB1 selector must satisfy expected cumulative regret:

```
E[R_T] ≤ O(√(K T log T))
```

where K = number of draft arms and T = number of selection rounds.
Verified empirically by `tests/unit/test_bandit.py` against a stochastic
oracle over T=10,000 rounds. The test asserts
`empirical_regret / theoretical_upper_bound < 1.5`.

### 2.4 ONNX parity contract

Draft model ONNX export must satisfy:

```
max(|onnx_logits - pytorch_logits|) < 1e-5   (all batch sizes 1..32)
```

Verified by `tests/integration/test_onnx_parity.py`.

---

## 3. Language and style standards

### 3.1 Python

| Rule | Detail |
|---|---|
| Version | 3.11 only. No `match` abuse; use `match` where it genuinely improves readability. |
| Type annotations | **All** function signatures fully annotated. `mypy --strict` must pass with zero errors. |
| Docstrings | NumPy docstring style for all public functions and classes. Every parameter, return value, and raised exception documented. |
| Line length | 100 characters. |
| Formatter | `ruff format` (black-compatible). Never manually format. |
| Linter | `ruff check` with rules: E, W, F, I, B, C4, UP, ANN, N, D (pydocstyle). |
| Import order | `ruff` isort profile. stdlib → third-party → local. Absolute imports only. |
| f-strings | Always over `.format()` or `%`. |
| Exceptions | Always raise with a message. Never `raise ValueError` alone. |
| Assertions | `assert` for invariants inside non-public functions only. Never for user-facing validation (use Pydantic or explicit raises). |
| `__all__` | Defined in every `__init__.py`. Public API is explicit, not implicit. |
| Dataclasses | Use `@dataclass(slots=True, frozen=True)` for value objects. Pydantic v2 `BaseModel` for config / serialised objects. |
| Logging | Use `flashspec.utils.logging.get_logger(__name__)` only. Never `print()` in library code. |

### 3.2 Triton kernels

| Rule | Detail |
|---|---|
| Kernel naming | `snake_case` with suffix `_kernel` for `@triton.jit` functions. |
| Autotuning | Every kernel with a tiling parameter must have a `@triton.autotune` decorator with at least 4 configs covering small/medium/large shapes. |
| Meta-parameters | All compile-time constants passed via `tl.constexpr`. No magic numbers. |
| Numerics | All arithmetic in float32 minimum. BF16 accumulation requires explicit upcast before reduction. |
| Bounds checking | Always use `mask = offsets < n_elements` with `other=0.0` on masked loads. No silent OOB. |
| Reference impl | Every kernel has a pure-PyTorch reference in `flashspec/kernels/_reference.py` with identical signature. Tests compare them. |
| Comments | Explain the tiling strategy and memory access pattern at the top of each kernel. Link to the relevant section in `docs/kernels.md`. |

### 3.3 C++ extensions (if any)

- Standard: C++17.
- No raw owning pointers. Use `std::unique_ptr` / `std::shared_ptr`.
- All CUDA kernel launches checked with `TORCH_CHECK(cudaGetLastError() == cudaSuccess, ...)`.
- CMake minimum version 3.21.

### 3.4 Naming conventions

| Concept | Convention | Example |
|---|---|---|
| Acceptance rate | `alpha` (float in [0, 1]) | `alpha: float` |
| Speculation length | `gamma` (int ≥ 1) | `gamma: int = 4` |
| Draft model | `drafter` | `drafter: DraftModel` |
| Target model | `target` or `verifier` | `target: TargetModel` |
| Token ids | `input_ids` | `input_ids: torch.Tensor` |
| Logits (raw) | `logits` | shape `(batch, seq, vocab)` |
| Log-probs | `logprobs` | shape `(batch, seq, vocab)` |
| Throughput | `tokens_per_second` | never `tps` or `tok/s` in code |
| Batch size | `batch_size` | never `B` in signatures |
| Sequence length | `seq_len` | never `T` in signatures |

---

## 4. Mathematical and algorithmic requirements

### 4.1 Speculative sampling

Implement **exactly** Algorithm 1 from Leviathan et al. (2023) "Fast Inference
from Transformers via Speculative Decoding" (arXiv:2211.17192).
Do not approximate, simplify, or alter the acceptance criterion.

The adjusted distribution for the residual token (when a draft token is rejected) is:

```python
residual = torch.clamp(p - q, min=0.0)
residual = residual / residual.sum(dim=-1, keepdim=True)
```

This **must** be implemented exactly as above — no temperature rescaling, no
`softmax` applied to the residual.

### 4.2 Verification kernel design

The Triton verification kernel must implement parallel acceptance testing across
all γ draft tokens in a **single kernel launch** (no Python loop over draft positions).
The kernel operates on pre-computed draft and target log-probabilities to avoid
a full forward pass.

Kernel input shape: `(batch_size, gamma, vocab_size)` for both `draft_logprobs`
and `target_logprobs`, plus uniform samples `u` of shape `(batch_size, gamma)`.

Kernel output: `accepted` boolean mask of shape `(batch_size, gamma)` and
`first_rejection` integer tensor of shape `(batch_size,)` indicating the position
of the first rejection per sequence (or `gamma` if all accepted).

Tiling strategy: tile over the vocab dimension to fit SRAM. The acceptance
criterion requires only `draft_logprobs[:, :, token_id]` and
`target_logprobs[:, :, token_id]` for the actual draft token — not the full
vocab sweep. Design the kernel accordingly.

### 4.3 Online bandit

The adaptive draft selector must implement at minimum:
- **UCB1** (Auer et al., 2002): `score_k = mu_k + sqrt(2 log(t) / n_k)`
- **Thompson Sampling** with Beta(α, β) conjugate prior on acceptance rate.
- An **Oracle** selector that always picks the arm with the highest true acceptance
  rate (requires a ground-truth oracle, used only for regret upper-bound experiments).

The bandit state must be:
- Serialisable to / from JSON (for checkpoint / resume).
- Thread-safe if called from multiple generation workers simultaneously.
- Resettable per-context-window (the acceptance rate of a draft model depends on
  the distribution of the current prompt; the bandit should support windowed
  statistics with configurable window size W).

### 4.4 Draft model registry

Implement a registry pattern so new draft models can be registered without
modifying engine code:

```python
@flashspec.engine.drafter.register("llama3-1b")
class Llama3_1B_Drafter(DraftModel):
    ...
```

The registry must be discoverable via Python entry points so external packages
can extend it.

---

## 5. Testing standards

Every PR must pass all of the following. CI blocks merges on any failure.

### 5.1 Test categories and required coverage

| Category | Location | Coverage requirement | Notes |
|---|---|---|---|
| Unit | `tests/unit/` | ≥ 95% line coverage on `flashspec/` | `pytest-cov` |
| Integration | `tests/integration/` | All happy paths + 3 edge cases per module | Run on GPU in CI |
| Chaos | `tests/chaos/` | Adversarial acceptance rates, 0% and 100% cases | |
| Property-based | Inline with unit tests | All kernel inputs tested with `hypothesis` | Shapes, dtypes, edge values |

### 5.2 Mandatory test invariants

Every test file must include these invariants for its module:

**Kernel tests (`test_verify_kernel.py`)**
- Numerical parity with reference at float32 and bfloat16.
- All-accept case (u < alpha for all positions): `first_rejection == gamma`.
- All-reject case: `first_rejection == 0`.
- Batch size 1 and batch size 64.
- Vocab size 32,000 (Llama) and 32,768 (Mistral).
- Property: output is identical regardless of values of `draft_logprobs` and
  `target_logprobs` for tokens **not** selected by the draft.

**Sampling tests (`test_sampling.py`)**
- Distribution equivalence (KS test, α=0.01, N=10,000).
- Token-conservation: total tokens generated equals total tokens consumed.
- Determinism: identical seed produces identical output.
- No-speedup correctness: gamma=1 must match standard autoregressive sampling exactly.

**Bandit tests (`test_bandit.py`)**
- Convergence: UCB1 selects the best arm > 90% of the time after T=1,000 rounds.
- Regret bound: empirical regret within 1.5× theoretical bound.
- Serialisation round-trip: `bandit.to_json()` → `Bandit.from_json()` preserves state exactly.
- Adversarial: bandit recovers optimal arm within 200 rounds after a sudden swap
  of best/worst arm (non-stationary test).

### 5.3 Test writing rules

- No `time.sleep()` in tests. Use mocks for timing-dependent behaviour.
- No network calls in unit or integration tests. All model weights mocked or
  loaded from `tests/conftest.py` fixtures (tiny 4-layer, 128-hidden toy models).
- Every test function has a one-sentence docstring explaining what property it verifies.
- Tests are named `test_<what>_<when>_<expected>`. Example:
  `test_verify_kernel_all_accept_returns_gamma`.
- Fixtures that create GPU tensors must be decorated with `@pytest.mark.gpu`.
  CPU-only tests run in the fast CI lane; GPU tests run in the nightly lane.
- No `assert True` or empty asserts. Every assert has a failure message.

---

## 6. Performance standards

These numbers are targets for the paper's results table and **must be met
on H100 SXM5 before submission**. They are also enforced by the nightly
benchmark regression in `.github/workflows/benchmark.yml`.

| Metric | Baseline (autoregressive) | FlashSpec target | Regression threshold |
|---|---|---|---|
| Throughput (Llama-3-8B, greedy, batch=1) | 1× (reference) | ≥ 2.0× | < 1.8× fails |
| Throughput (Llama-3-70B, greedy, batch=1) | 1× | ≥ 2.5× | < 2.2× fails |
| Throughput (Mistral-7B, greedy, batch=1) | 1× | ≥ 1.9× | < 1.7× fails |
| Verification kernel overhead | — | ≤ 0.5 ms per step (H100) | > 1 ms fails |
| Bandit selection overhead | — | ≤ 50 µs per call | > 200 µs fails |
| GPU memory overhead vs AR | — | ≤ 15% | > 25% fails |

Benchmark results are stored in `benchmarks/results/` as JSON, committed to
`main` on each nightly run, and plotted automatically by `notebooks/`.

The benchmark runner must:
- Pin CPU threads (`torch.set_num_threads(1)`) for single-batch experiments.
- Run 50 warm-up steps before measurement.
- Report mean ± std over 200 steps.
- Use `torch.cuda.synchronize()` around all timed regions.
- Log GPU memory with `torch.cuda.max_memory_allocated()`.

---

## 7. Numerical standards

- All acceptance-probability comparisons operate in **log-space** to avoid
  underflow. Never compute `p(x) / q(x)` directly in float32 for long sequences.
- Kernel accumulation uses float32 minimum, regardless of input dtype.
- Any division must guard against zero denominator:
  ```python
  safe_div = lambda a, b: a / b.clamp(min=1e-9)
  ```
- Temperature scaling must be applied **before** log-softmax, not after.
- All random seeds in tests set via `torch.manual_seed` + `numpy.random.seed` +
  `random.seed` in a single `set_seed(seed: int)` utility.

---

## 8. Documentation standards

### 8.1 README.md

Must contain, in order:
1. One-line description + badges (CI, coverage, PyPI, arXiv).
2. A 3-command quick-start that reproduces a real throughput number on a single GPU.
3. A results table comparing FlashSpec against Medusa, EAGLE, and vanilla AR
   on at least two models and two datasets.
4. Architecture diagram (Mermaid or SVG — no PNGs).
5. Installation instructions (pip + from source + Docker).
6. Citation block (BibTeX).
7. Links to paper, docs, and benchmarks.

### 8.2 Code docstrings

Every public function and class must have a NumPy-style docstring with:
- One-sentence summary line.
- Parameters section (name, type, description for each).
- Returns section.
- Raises section (if applicable).
- Notes section explaining algorithmic choices or references.
- At minimum one usage example in the docstring (for public API functions).

Template:
```python
def verify_tokens(
    draft_logprobs: torch.Tensor,
    target_logprobs: torch.Tensor,
    u: torch.Tensor,
    gamma: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel token acceptance test across all draft positions.

    Implements the acceptance criterion from Algorithm 1 of Leviathan et al.
    (2023) in a single Triton kernel launch. Operates on pre-computed log-
    probabilities to avoid redundant forward passes.

    Parameters
    ----------
    draft_logprobs : torch.Tensor
        Log-probabilities under the draft model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    target_logprobs : torch.Tensor
        Log-probabilities under the target model.
        Shape: ``(batch_size, gamma, vocab_size)``, dtype float32 or bfloat16.
    u : torch.Tensor
        Uniform samples for acceptance testing.
        Shape: ``(batch_size, gamma)``, dtype float32. Values in [0, 1).
    gamma : int
        Speculation length. Must equal ``draft_logprobs.shape[1]``.

    Returns
    -------
    accepted : torch.Tensor
        Boolean acceptance mask. Shape: ``(batch_size, gamma)``.
    first_rejection : torch.Tensor
        Index of the first rejection per sequence, or ``gamma`` if all accepted.
        Shape: ``(batch_size,)``, dtype int32.

    Raises
    ------
    ValueError
        If ``draft_logprobs`` and ``target_logprobs`` shapes do not match.
    RuntimeError
        If tensors are not on the same CUDA device.

    Notes
    -----
    The acceptance criterion is ``u_i < exp(log p(x_i) - log q(x_i))``,
    clipped to [0, 1]. This is numerically equivalent to ``u_i < p(x_i)/q(x_i)``
    but avoids underflow for low-probability tokens.

    References
    ----------
    .. [1] Leviathan et al. (2023), "Fast Inference from Transformers via
       Speculative Decoding", arXiv:2211.17192, Algorithm 1.

    Examples
    --------
    >>> accepted, first_rejection = verify_tokens(
    ...     draft_logprobs, target_logprobs, u, gamma=4
    ... )
    >>> assert accepted.shape == (batch_size, 4)
    """
```

### 8.3 Architecture documentation

`docs/architecture.md` must contain:
- A sequence diagram showing the full speculative decoding loop (drafter → verifier → bandit → output).
- A component diagram showing module dependencies.
- A section explaining the correctness guarantee and why the output distribution
  matches the target.

`docs/kernels.md` must contain:
- For each Triton kernel: the tiling strategy, SRAM usage calculation, and
  roofline analysis showing where the kernel is memory-bound vs compute-bound.

`docs/bandit.md` must contain:
- The regret bound proof sketch for UCB1.
- A figure showing empirical regret convergence.
- The justification for windowed statistics.

### 8.4 Paper (`paper/flashspec.tex`)

The paper must follow the MLSys 2026 / NeurIPS format. Required sections:
1. Abstract (≤ 150 words; must contain the throughput numbers).
2. Introduction with clear problem statement and contributions.
3. Background: speculative decoding, online bandits.
4. Method: FlashSpec system design, kernel design, bandit formulation.
5. Experiments: baselines (Medusa, EAGLE, vanilla AR), datasets (MT-Bench,
   HumanEval, Alpaca), models (Llama-3-8B, Llama-3-70B, Mistral-7B).
6. Ablation: gamma sweep, draft model size sweep, bandit vs fixed draft.
7. Analysis: acceptance rate distribution, bandit convergence, kernel profiling.
8. Related work.
9. Conclusion.
10. Reproducibility statement (required by all venues).

---

## 9. Git and version control standards

### 9.1 Commit messages

Follow Conventional Commits 1.0.0 strictly:

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

Types: `feat` · `fix` · `perf` · `refactor` · `test` · `docs` · `ci` · `chore` · `bench`

Scope (required): `kernel` · `bandit` · `engine` · `sampling` · `export` · `bench` · `paper` · `ci`

Subject rules:
- Imperative mood: "add" not "adds" or "added".
- No period at the end.
- ≤ 72 characters.
- Reference the invariant broken or fixed when fixing a correctness bug.

Examples:
```
feat(kernel): add Triton verification kernel with dynamic gamma support
perf(kernel): tile over vocab_size dimension for H100 SRAM efficiency
fix(sampling): correct residual distribution for zero-probability tokens
test(bandit): add adversarial acceptance-rate swap convergence test
bench(engine): add Llama-3-70B throughput vs Medusa comparison
docs(kernels): add SRAM usage analysis for verify_kernel
```

### 9.2 Branching

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Stable, paper-reproducible | Never force-push. All CI must pass. |
| `dev` | Integration target | All PRs merge here first. |
| `feat/<name>` | Feature development | Branch from `dev`, PR back to `dev`. |
| `fix/<name>` | Bug fixes | Branch from `main` for hotfixes, `dev` otherwise. |
| `bench/<name>` | Benchmark experiments | Branch from `dev`. Results committed as CSV. |
| `paper/<section>` | Paper writing | Branch from `main`. Docs only. |

### 9.3 PR requirements

Every PR must include:
- A description of **what** changed and **why**.
- A reference to the invariant or performance contract affected (§2 or §6).
- A test demonstrating the change (new test or pointer to existing test now passing).
- Updated CHANGELOG.md entry under `[Unreleased]`.
- No decrease in test coverage (CI enforces ≥ 95%).

### 9.4 Tags and releases

- Tag format: `v{MAJOR}.{MINOR}.{PATCH}` (semver).
- Every tag must have a corresponding GitHub Release with the CHANGELOG section.
- The arXiv paper cites the GitHub tag matching the submitted version.

---

## 10. Dependency management

All dependencies are declared in `pyproject.toml`. No `requirements.txt`
(except auto-generated for Docker).

```toml
[project]
requires-python = ">=3.11"

[project.dependencies]
torch = ">=2.2.0"
triton = ">=3.0.0"
transformers = ">=4.40.0"
pydantic = ">=2.0.0"
numpy = ">=1.26.0"

[project.optional-dependencies]
onnx = ["onnx>=1.16", "onnxruntime-gpu>=1.18"]
dev = ["pytest", "pytest-cov", "hypothesis", "ruff", "mypy", "import-linter"]
bench = ["pandas", "matplotlib", "seaborn", "lm-eval>=0.4"]
docs = ["mkdocs-material", "mkdocstrings[python]"]
```

Rules:
- Never pin to an exact version in `[project.dependencies]`. Use `>=` with the
  minimum tested version.
- The `[project.optional-dependencies]` section is the single source of truth.
  No ad-hoc pip installs in CI scripts.
- Before adding any new dependency, justify it in the PR description. Reject
  any dependency that duplicates functionality already in PyTorch or the stdlib.

---

## 11. Makefile targets (canonical task runner)

Every common task must be runnable via `make`. CI uses only these targets.

```makefile
make lint          # ruff check + ruff format --check + mypy --strict
make format        # ruff format (auto-fix)
make test          # pytest tests/unit tests/integration -x --cov=flashspec
make test-gpu      # pytest tests/ -m gpu -x (requires CUDA)
make test-chaos    # pytest tests/chaos -x
make bench         # python benchmarks/run_all.py --config benchmarks/configs/
make bench-quick   # benchmarks on toy model, no real weights (CI smoke test)
make docs          # mkdocs build
make docs-serve    # mkdocs serve (local)
make paper         # cd paper && make (latexmk)
make docker-build  # docker build -t flashspec:latest .
make docker-test   # docker run flashspec:latest make test
make clean         # remove __pycache__, .coverage, dist/, build/
```

---

## 12. CI/CD pipeline specification

### 12.1 `ci.yml` — runs on every push and PR to `dev` and `main`

```
Jobs (in order):
  lint        → make lint                           # fail-fast
  test-cpu    → make test                           # unit + integration, CPU
  test-chaos  → make test-chaos                     # adversarial bandit tests
  coverage    → coverage report, fail if < 95%
  onnx-parity → pytest tests/integration/test_onnx_parity.py
  docs-build  → make docs
```

All jobs run on `ubuntu-22.04`. Triton CPU mode enabled via
`TRITON_CPU_BACKEND=1` env var in test jobs.

### 12.2 `gpu_tests.yml` — runs nightly and on PR to `main`

```
Jobs:
  test-gpu    → make test-gpu     # self-hosted A10G runner
  bench-quick → make bench-quick  # smoke throughput, fails if < 1.5× AR
```

### 12.3 `benchmark.yml` — runs nightly on `main`

```
Jobs:
  bench-full  → make bench        # full benchmark suite, H100 runner
              → python scripts/update_benchmark_results.py
              → git commit benchmarks/results/ -m "bench: nightly results $(date)"
              → git push
```

Nightly benchmark failures do **not** block merges but do post a Slack/email alert.

---

## 13. Agent-specific rules

These rules are written explicitly for AI code editors and copilots.

### 13.1 What agents are permitted to do autonomously

- Write new Python functions and classes that follow all rules in this document.
- Write new Triton kernels following the kernel standards in §3.2.
- Write new tests following the test standards in §5.
- Refactor existing code that does not change public API.
- Update docstrings.
- Update `CHANGELOG.md`.
- Fix linter errors flagged by `make lint`.

### 13.2 What agents must never do without explicit human approval

- Change any function or class in `flashspec/kernels/` that has a corresponding
  test in `tests/unit/test_verify_kernel.py` (correctness-critical).
- Modify the acceptance criterion in `flashspec/sampling/rejection.py`.
- Change `pyproject.toml` dependencies.
- Force-push or rebase `main`.
- Delete any file in `benchmarks/results/`.
- Modify `paper/` directory contents.
- Change the public API surface defined in `flashspec/__init__.py`.
- Add any `# type: ignore` annotation without a comment explaining why.
- Disable or skip any test without a linked issue.

### 13.3 Code generation quality gates

Before proposing any code change, the agent must internally verify:

1. **Type correctness**: Would `mypy --strict` pass?
2. **Test coverage**: Is there a test that would fail if this code were wrong?
3. **Invariant preservation**: Does this change preserve the mathematical contracts in §2?
4. **Naming**: Does every identifier follow the conventions in §3.4?
5. **Docstring**: Is there a complete NumPy docstring following the template in §8.2?
6. **No magic numbers**: Are all constants named and defined at module level?
7. **No silent failures**: Are all error cases handled with explicit exceptions?

If any gate fails, the agent must fix the issue before proposing the change.

### 13.4 Handling ambiguous requirements

If a task is ambiguous, the agent must:
1. State the ambiguity explicitly in a comment at the top of the proposed diff.
2. Choose the interpretation that is most conservative (least likely to break invariants).
3. Propose an alternative interpretation in the comment.

Never silently pick an interpretation. Never silently omit a feature because
it is hard.

### 13.5 Triton kernel generation checklist

When generating or modifying a Triton kernel, verify every item:

- [ ] `@triton.jit` decorator present.
- [ ] `@triton.autotune` decorator present with ≥ 4 configs.
- [ ] All tiling block sizes are `tl.constexpr` parameters.
- [ ] All loads use `mask=` parameter.
- [ ] No Python-level loop over the tiled dimension.
- [ ] A corresponding reference implementation exists in `_reference.py`.
- [ ] A numerical parity test exists in `tests/unit/test_verify_kernel.py`.
- [ ] A comment at the top explains the tiling strategy and links to `docs/kernels.md`.
- [ ] Kernel handles both `float32` and `bfloat16` inputs.
- [ ] Kernel is benchmarked in `benchmarks/` with roofline analysis.

### 13.6 Banned patterns

These patterns are rejected in code review regardless of context:

```python
# BANNED: silent except
try:
    ...
except Exception:
    pass

# BANNED: bare raise
raise ValueError

# BANNED: print in library code
print("debug")

# BANNED: mutable default argument
def foo(x, cache={}):
    ...

# BANNED: torch.no_grad() in tests that check gradients
# BANNED: hardcoded device string
tensor = torch.tensor([1.0]).cuda()        # use device= parameter

# BANNED: global mutable state outside of registry singletons
_global_state = {}                         # use class-level state

# BANNED: numpy in Triton kernels
# BANNED: Python-level loop inside @triton.jit function body
# BANNED: magic tiling constants without autotune
BLOCK_SIZE = 128  # inside kernel without autotune config
```

---

## 14. Benchmarking and reproducibility standards

Every number that appears in the paper must be:
1. Reproducible by running `make bench`.
2. Stored in `benchmarks/results/` as a JSON file with the following schema:

```json
{
  "timestamp": "2025-11-01T14:32:00Z",
  "git_commit": "abc1234",
  "hardware": {
    "gpu": "NVIDIA H100 SXM5 80GB",
    "driver": "550.54.15",
    "cuda": "12.4"
  },
  "model": "meta-llama/Llama-3-8B-Instruct",
  "method": "flashspec_ucb",
  "gamma": 4,
  "dataset": "mt_bench",
  "n_samples": 200,
  "metrics": {
    "tokens_per_second_mean": 142.3,
    "tokens_per_second_std": 4.1,
    "speedup_vs_ar_mean": 2.31,
    "speedup_vs_ar_std": 0.08,
    "acceptance_rate_mean": 0.73,
    "acceptance_rate_std": 0.05,
    "p50_latency_ms": 38.1,
    "p95_latency_ms": 44.2,
    "p99_latency_ms": 47.8,
    "gpu_memory_gb": 18.4
  }
}
```

3. Have a corresponding row in the paper's results table.
4. Be plotted by a notebook in `notebooks/`.

Benchmark configs in `benchmarks/configs/*.yaml` are version-controlled and
must not be modified after results for a given config are committed.

---

## 15. Security and safety

- No model weights committed to the repository. Weights are fetched at runtime
  via HuggingFace Hub or a local path specified in config.
- No API keys, tokens, or credentials in any file. Use environment variables.
  CI uses GitHub Secrets.
- `pyproject.toml` must pin `transformers`, `torch`, and `triton` to known-good
  versions. Run `pip-audit` in CI and fail on any CRITICAL CVE.
- Docker image scanned with Trivy in CI; CRITICAL CVEs block the build.

---

## 16. Glossary

| Term | Definition as used in this codebase |
|---|---|
| AR | Autoregressive (baseline generation method) |
| gamma (γ) | Number of speculative draft tokens per step |
| alpha (α) | Token acceptance rate: fraction of draft tokens accepted by target |
| drafter | Small/fast draft model that proposes candidate tokens |
| target / verifier | Large target model that accepts or rejects draft tokens |
| bandit | Online algorithm that selects which draft model to use |
| arm | One draft model in the bandit's action set |
| regret | Cumulative loss vs oracle arm selection |
| MFU | Model FLOPs Utilisation (fraction of peak FLOP/s achieved) |
| speedup | Ratio of FlashSpec tokens/s to autoregressive tokens/s (wall-clock) |
| roofline | Compute vs memory bandwidth bound analysis for a kernel |

---

## 17. Reference implementations and baselines

When implementing a new feature, always check these repositories first:

| System | Repo | What to learn from it |
|---|---|---|
| Speculative Decoding (original) | [jaymody/speculative-sampling](https://github.com/jaymody/speculative-sampling) | Reference Algorithm 1 implementation |
| Medusa | [FasterDecoding/medusa](https://github.com/FasterDecoding/medusa) | Multi-head speculation baseline |
| EAGLE | [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) | Feature-level draft model baseline |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Production speculative decoding reference |
| DeepSpeed-FastGen | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | System-level speculative decoding |

FlashSpec must be benchmarked against Medusa and EAGLE on identical hardware
and dataset before any performance claim is made in the paper.

---

## 18. Paper authorship and citation policy

- The paper is authored by Min Htet Myet (Mattral).
- All experiments are conducted by the author unless explicitly noted.
- Every external result cited in the paper must cite the original published source,
  not a blog post or GitHub README.
- The paper must not claim results that are not in `benchmarks/results/`.
- The reproducibility statement must specify exact hardware, software versions,
  random seeds, and the make target to reproduce each table.

BibTeX entry for the repository (update when arXiv number is assigned):

```bibtex
@misc{mattral2025flashspec,
  title   = {{FlashSpec}: Adaptive Speculative Decoding with Online Bandit
             Draft Selection and Triton-Optimised Verification},
  author  = {Myet, Min Htet},
  year    = {2025},
  note    = {arXiv preprint. \url{https://github.com/Mattral/FlashSpec}},
}
```

---

*End of AGENTS.md — version 1.0.0*
*This file is the project constitution. All contributors (human and AI) are bound by it.*
