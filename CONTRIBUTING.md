# Contributing to FlashSpec

Thank you for contributing.  This document describes everything you need
to know to open a pull request that will pass CI on the first try.

---

## Quick checklist

Before opening a PR, run:

```bash
make lint          # ruff + mypy + import-linter — must pass with zero errors
make test          # unit + integration, CPU — must pass at ≥ 95% coverage
make test-chaos    # adversarial bandit tests — must pass
make bench-quick   # toy benchmark smoke-test — must pass
```

---

## Commit message format (§9.1)

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

## Branching strategy (§9.2)

| Branch | Purpose | Rules |
|--------|---------|-------|
| `main` | Stable, paper-reproducible | Never force-push. All CI must pass. |
| `dev` | Integration target | All PRs merge here first. |
| `feat/<name>` | Feature development | Branch from `dev`, PR back to `dev`. |
| `fix/<name>` | Bug fixes | Branch from `main` for hotfixes, `dev` otherwise. |
| `bench/<name>` | Benchmark experiments | Branch from `dev`. Results committed as CSV. |
| `paper/<section>` | Paper writing | Branch from `main`. Docs only. |

---

## PR requirements (§9.3)

Every PR must include:

1. **What changed and why** — a clear description in the PR body.
2. **Invariant or performance contract reference** — cite the relevant §2 or §6 item.
3. **Test evidence** — a pointer to the new or existing test that now passes.
4. **`CHANGELOG.md` entry** under `[Unreleased]`.
5. **No coverage regression** — CI enforces ≥ 95% line coverage.

---

## What you may change autonomously (§13.1)

- New Python functions and classes following all rules in `AGENTS.md`.
- New Triton kernels following the kernel standards in `AGENTS.md §3.2`.
- New tests following the test standards in `AGENTS.md §5`.
- Refactoring that does **not** change the public API.
- Docstring updates.
- `CHANGELOG.md` entries.
- Linter-flagged fixes.

## What requires explicit human approval (§13.2)

- Any change to `flashspec/kernels/` that has a corresponding test in
  `tests/unit/test_verify_kernel.py` (correctness-critical).
- Modifying the acceptance criterion in `flashspec/sampling/rejection.py`.
- Changing `pyproject.toml` dependencies.
- Changing the benchmark result schema in `benchmarks/`.
- Adding a new public function to the `flashspec` top-level namespace.
- Any change to the paper after results have been committed.

---

## Code style

All style rules are enforced by `make lint` (ruff + mypy + import-linter).

Key rules (full list in `AGENTS.md §3`):

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

## Versioning and releases (§9.4)

Tags follow semver: `v{MAJOR}.{MINOR}.{PATCH}`.
Every tag must have a corresponding GitHub Release with the relevant
CHANGELOG section.  The arXiv paper cites the GitHub tag matching the
submitted version.

---

## Getting help

Open a GitHub Discussion or email `mattral@example.com`.
