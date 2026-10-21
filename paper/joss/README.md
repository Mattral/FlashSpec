# JOSS Submission Guide for FlashSpec

This directory contains `paper.md` and `paper.bib` formatted for the
[Journal of Open Source Software](https://joss.theoj.org/) (JOSS).

JOSS review focuses on the **software**, not on novel research claims —
reviewers check that the software does what it says, has tests, has docs,
and that the paper accurately describes it. This directory is a
pre-submission draft. It has not been submitted, and it is not under
review.

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
- [ ] **Automated tests exist and pass** — tests exist. `make test` is
      configured to fail under 95% coverage, and that gate has not been
      met. Do not check this box from a skipped CI job.
- [ ] **Community guidelines** — `CONTRIBUTING.md` is present. It describes
      the current branch and CI state.
- [ ] **paper.md compiles** — see "Local preview" below. This checkout has
      not produced `paper.pdf` in this pass.
- [ ] **Version tag exists** — GitHub Releases exist for v0.1.0, v0.1.1,
      v0.1.3, v0.1.4, and v0.1.5. There is no v0.1.2 tag. The `flashspec`
      package at 0.1.2 was yanked.
- [ ] **paper.md word count is 250–1000 words** (excluding YAML header and
      references) — recount before submission. This draft does not claim
      a checked word count.
- [x] **ORCID filled in** — `0009-0005-0853-8301`. No affiliation is listed.
- [x] **Draft date set** — `date: 2026-10-16` is the draft date, not a
      submission date.

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

## Relationship to the Zenodo draft

`paper/flashspec.tex` is the draft research preprint for a later Zenodo
version. It is not an arXiv submission. `AGENTS.md` is not in this
checkout. The citable archive is still
[10.5281/zenodo.20766119](https://doi.org/10.5281/zenodo.20766119)
(June 2026) until the maintainer deposits a new version.

| | JOSS draft (`paper/joss/`) | Zenodo draft (`paper/flashspec.tex`) |
|---|---|---|
| Focus | The software: what it does, how it is tested | The method, the bound, and the committed measurements |
| Status | Not submitted | Not deposited |
| Numbers | Must match the committed pairs | Tables cite the JSON and notebook cells |

Do not describe either draft as under review. Do not install
`flashspec==0.1.2`.
