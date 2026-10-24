# Contributing to FlashSpec

Thank you for contributing.  This document describes how to open a pull
request. The lint job runs `ruff check`, `ruff format --check`,
import-linter, and `mypy --strict` (`make typecheck`). CI runs on every
pull request, including stacked ones whose base is another branch.

---

## Quick checklist

Before opening a PR, run:

```bash
make lint          # ruff check, ruff format --check, and import-linter
make typecheck     # mypy --strict flashspec. Part of the CI lint job.
make test          # unit + integration on CPU. Fails under 83% line coverage.
make test-chaos    # adversarial bandit tests. One test is an expected failure.
make bench-quick   # toy benchmark smoke-test
```

The coverage floor is the measured unit-and-integration coverage, rounded
down.

GPU checks have no CI runner. They run by hand on a Colab T4 through
`notebooks/04_gpu_measurement.ipynb`, which writes a results bundle that
`scripts/ingest_gpu_bundle.py` validates.

---

## Commit message format

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org) **strictly**.

```
<type>(<scope>): <subject>
```

### Types

`feat` · `fix` · `perf` · `refactor` · `test` · `docs` · `ci` · `chore` · `bench`

### Scopes (required)

`kernel` · `bandit` · `engine` · `sampling` · `export` · `bench` · `paper` · `ci`

### Subject rules

- Imperative mood: "add" not "adds" or "added".
- No period at the end.
- ≤ 72 characters total (type + scope + subject).
- Reference the invariant broken or fixed when fixing a correctness bug.

### Examples

```
feat(kernel): add Triton verification kernel with dynamic gamma support
perf(kernel): tile over vocab_size dimension for H100 SRAM efficiency
fix(sampling): correct residual distribution for zero-probability tokens
test(bandit): add adversarial acceptance-rate swap convergence test
bench(engine): add Llama-3-70B throughput vs Medusa comparison
docs(kernels): add SRAM usage analysis for verify_kernel
```

---

## Branching strategy

| Branch | Purpose | Rules |
|--------|---------|-------|
| `main` | Stable history | Do not force-push. Open a branch and a pull request. |
| `feat/<name>` | Feature development | Branch from `main`, pull request back to `main`. |
| `fix/<name>` | Bug fixes | Branch from `main`. |
| `bench/<name>` | Benchmark experiments | Branch from `main`. Commit a result file only from a run. |
| `docs/<name>` | Documentation and the draft paper | Branch from `main`. |

---

## PR requirements

Every PR must include:

1. **What changed and why** — a clear description in the PR body.
2. **What the change preserves** — name the sampling, batching, or measurement contract it touches.
3. **Test evidence** — a pointer to the new or existing test that now passes.
4. **`CHANGELOG.md` entry** under `[Unreleased]`.
5. **Coverage** — pytest fails under 83% line coverage on the unit and
   integration suites. That floor is the measured coverage, rounded down.

---

## What you may change autonomously

- New Python functions and classes that follow the style rules in this file.
- New Triton kernels that keep the numerical contract in `docs/kernels.md`.
- New tests under `tests/unit/`, `tests/integration/`, or `tests/chaos/`.
- Refactoring that does **not** change the public API.
- Docstring updates.
- `CHANGELOG.md` entries.
- Linter-flagged fixes.

## What requires explicit human approval

- Any change to `flashspec/kernels/` that has a corresponding test in
  `tests/unit/test_verify_kernel.py` (correctness-critical).
- Modifying the acceptance criterion in `flashspec/sampling/rejection.py`.
- Changing `pyproject.toml` dependencies.
- Changing the benchmark result schema in `benchmarks/`.
- Adding a new public function to the `flashspec` top-level namespace.
- Any change to the paper after results have been committed.

---

## Code style

All style rules are enforced by `make lint` (ruff + import-linter) and
`make typecheck` (mypy).

Key rules:

- NumPy-style docstrings on every public function and class.
- `logger.debug(...)` instead of `print(...)` in library code.
- Never use `tps` as a variable name; use `tokens_per_second`.
- Never use `.cuda()` directly; pass `device` as a parameter.
- All random seeds set via `flashspec.utils.device.set_seed(seed)`.
- Every `assert` in tests must have a failure message.

---

## Testing

- All new code must have tests in `tests/unit/`, `tests/integration/`, or `tests/chaos/`.
- Test names follow `test_<what>_<when>_<expected>`.
- GPU tests must be decorated with `@pytest.mark.gpu`.
- No `time.sleep()` in tests. No network calls. No real model weights.

---

## Adding a new draft model

1. Create a class implementing the `DraftModel` protocol
   (`flashspec/engine/drafter.py`).
2. Register it with `@flashspec.engine.drafter.register("your-name")`.
3. Or register it via a Python entry point in your own package:
   ```toml
   [project.entry-points."flashspec.drafters"]
   your-name = "your_package:YourDrafterClass"
   ```

---

## Adding a new bandit selector

1. Subclass `DraftSelector` (`flashspec/bandit/base.py`).
2. Implement `select()`, `update()`, `_state_dict()`, `_from_state_dict()`.
3. Add it to `flashspec/bandit/__init__.py` and the `BanditConfig.strategy` literal.
4. Add unit tests in `tests/unit/test_bandit.py`.

---

## Benchmark contributions

- Add a YAML config in `benchmarks/configs/`.
- Do **not** modify existing configs after their first committed result.
- Run `make bench` and commit the JSON output in `benchmarks/results/`.
- Add a row to the paper's results table and update `benchmarks/README.md`.

---

## Versioning and releases

Tags follow semver: `v{MAJOR}.{MINOR}.{PATCH}`.
Every tag must have a corresponding GitHub Release with the relevant
CHANGELOG section. Releases exist for v0.1.0, v0.1.1, v0.1.3, v0.1.4,
and v0.1.5. There is no v0.1.2 tag: the `flashspec` package at 0.1.2 was
yanked, and `flashspec==0.1.2` is not an install target.
`paper/flashspec.tex` is a draft preprint. It is not on arXiv.

---

## Getting help

Open a GitHub issue on `Mattral/FlashSpec`.
