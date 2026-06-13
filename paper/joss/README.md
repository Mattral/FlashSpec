# JOSS Submission Guide for FlashSpec

This directory contains `paper.md` and `paper.bib` formatted for the
[Journal of Open Source Software](https://joss.theoj.org/) (JOSS).

JOSS review focuses on the **software**, not on novel research claims —
reviewers check that the software does what it says, has tests, has docs,
and that the paper accurately describes it. This is a good fit for
FlashSpec given its test suite and documentation are already complete.

---

## Pre-submission checklist

Before submitting, confirm each of these (JOSS reviewers will check all of
them):

- [ ] **Repository is public** on GitHub (`Mattral/FlashSpec`)
- [ ] **License file present** — `LICENSE` (Apache 2.0) ✅ already done
- [ ] **Installation instructions** work as written — test
      `pip install -e ".[dev]"` on a clean machine
- [ ] **Example usage** runs — `notebooks/01_quickstart.ipynb` executes
      end-to-end without errors
- [ ] **Automated tests exist and pass** — `make test` ✅ already done
- [ ] **Community guidelines** — `CONTRIBUTING.md` ✅ already done
- [ ] **paper.md compiles** — see "Local preview" below
- [ ] **Version tag exists** — create a GitHub Release (e.g. `v0.1.0`)
      before or during review; reviewers cite a specific version
- [ ] **paper.md word count is 250–1000 words** (excluding YAML header,
      references) — current draft is ~750 words ✅
- [ ] **ORCID filled in** — `paper.md` has a placeholder
      `0000-0000-0000-0000`; replace with your real ORCID
      (register free at https://orcid.org if you don't have one)
- [ ] **Submission date set** — replace `date: TODO` in `paper.md`

---

## Local preview (compile paper.md to PDF)

JOSS uses a Docker-based compiler. From the repo root:

```bash
docker run --rm \
  --volume "$(pwd)/paper/joss:/data" \
  --user "$(id -u):$(id -g)" \
  --env JOURNAL=joss \
  openjournals/inara
```

This produces `paper/joss/paper.pdf`. Review it for formatting issues
before submitting.

---

## Submission steps

1. Go to https://joss.theoj.org/papers/new
2. Sign in with GitHub
3. Enter the repository URL: `https://github.com/Mattral/FlashSpec`
4. Enter the path to the paper: `paper/joss/paper.md`
5. JOSS will run automated checks (license, DOI archive link, etc.) and
   open a pre-review issue on the JOSS GitHub repo
6. A JOSS editor will assign reviewers (typically 1–2 reviewers,
   2–8 weeks)
7. Reviewers open issues against `Mattral/FlashSpec` for any problems —
   respond and fix as normal GitHub issues
8. On acceptance, JOSS mints a DOI and publishes the paper

---

## Relationship to the arXiv preprint

JOSS and arXiv serve different purposes and are **not mutually
exclusive**:

| | JOSS (`paper/joss/`) | arXiv (`paper/flashspec.tex`) |
|---|---|---|
| Focus | The *software* — does it work, is it documented, is it tested | The *research* — the bandit formulation, regret bounds, benchmark results |
| Length | 250–1000 words | Full paper (8–15 pages) |
| Review | Open peer review on GitHub, 2–8 weeks | None (preprint) — or full peer review if later submitted to a venue |
| Output | Citable DOI | arXiv ID, citable but not peer-reviewed |
| Numbers required | None — describes architecture and test methodology | Full benchmark tables (Section 4 of `flashspec.tex`) |

**Recommended order**: submit to arXiv first (no benchmark blocker beyond
what's needed for Section 4 — but per AGENTS.md §18, real numbers from
`benchmarks/results/` are required before that submission). Submit to
JOSS in parallel or shortly after — JOSS's `paper.md` makes no specific
performance claims, so it does **not** need to wait for `make bench`.

**FlashSpec is ready for JOSS submission right now**, modulo the three
placeholders listed in the checklist above (ORCID, date, GitHub Release
tag).
