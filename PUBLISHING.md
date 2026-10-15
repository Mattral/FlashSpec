# FlashSpec Publication Guide

This document indexes everything prepared for announcing and publishing
FlashSpec, and gives you a clear sequence of actions.

---

## What's ready right now (no GPU needed)

| Deliverable | Location | Status |
|---|---|---|
| X (Twitter) thread | `social/x_thread.md` | Ready — has 2 placeholder numbers |
| LinkedIn post | `social/linkedin_post.md` | Ready — has 2 placeholder numbers |
| JOSS paper | `paper/joss/paper.md` + `paper.bib` | Ready — 3 small placeholders (ORCID, date, version tag) |
| Zenodo metadata | `.zenodo.json`, `CITATION.cff` | Ready — activates on first GitHub Release |

## What's blocked on `make bench` (needs GPU + weights)

| Deliverable | Location | Blocker |
|---|---|---|
| arXiv preprint | `paper/flashspec.tex` | Table 1, Figures 1–2, abstract numbers — all currently placeholders per §18 |
| X thread numbers | `social/x_thread.md` tweet 8 | `[X.Xx]`, `[XX]%` |
| LinkedIn numbers | `social/linkedin_post.md` point 4 | `[X.Xx]`, `[XX]%` |

---

## Recommended sequence

### Step 0 — Run the benchmarks

**Status: partially done.**

The 0.1.4 changelog records a Tesla T4 / TinyLlama-1.1B-Chat NF4 run
(44.2 tok/s, α=0.75, p50=22.1ms) in
`benchmarks/results/flashspec_ucb_tiny_llama.json`. That file is one
prompt. Its speedup, p95, p99, and GPU-memory fields are `0.0` because
they were not measured. Do not copy those zeros into a new file. The
gamma and draft-size CSVs are labeled synthetic (`measurement_kind`
column). They are not H100 end-to-end results.

Still required before those numbers can be published as paper Table 1:

- An H100 (or whatever GPU you actually have) run whose JSON is committed
- A vanilla autoregressive baseline on that same machine
- Medusa and EAGLE are not implemented, so they cannot be rows until they exist

```bash
export HF_TOKEN=hf_your_token
python scripts/download_models.py
python benchmarks/run_all.py --config benchmarks/configs/llama3_8b.yaml
```

Commit a result file only after reading its `hardware`, `measurement_kind`,
and `notes`. Name the commit after the machine that produced the JSON.
An H100 commit message is accurate only when that string is in the file.

### Step 1 — Create a GitHub Release (5 minutes)

This single action unblocks Zenodo:

1. GitHub repo → "Releases" → "Draft a new release"
2. Tag: `v0.1.0`
3. Title: `FlashSpec v0.1.0 — Initial Release`
4. Description: paste the relevant section from `CHANGELOG.md`
5. Publish

### Step 2 — Archive on Zenodo (10 minutes, gets you a DOI immediately)

1. Go to https://zenodo.org, sign in with GitHub
2. Settings → GitHub → toggle on `Mattral/FlashSpec`
3. If you already created the release in Step 1, Zenodo auto-archives it
   and mints a DOI. If not, create the release now and it triggers
   automatically.
4. Copy the DOI badge Zenodo gives you
5. Add it to `README.md` (near the top, with the other badges) and to
   `CITATION.cff` under `preferred-citation` → add a `doi:` field

**Why Zenodo first**: it requires zero new writing, gives you a citable
DOI within minutes, and the JOSS submission process explicitly checks for
an archive link — having one ready makes JOSS review smoother.

### Step 3 — Submit to JOSS (today, if Step 0 isn't done yet)

JOSS's `paper.md` doesn't make performance claims, so it does **not**
need to wait for benchmarks. Fill in the 3 placeholders in
`paper/joss/paper.md` (ORCID, date, and reference the v0.1.0 tag from
Step 1), then follow `paper/joss/README.md` to submit. Expect 2–8 weeks
for review.

### Step 4 — Submit to arXiv (after Step 0)

Once `benchmarks/results/` has real numbers:

1. Fill in the abstract's `[X.Xx]` placeholder in `paper/flashspec.tex`
2. Fill in Table 1 with real numbers from `benchmarks/results/*.json`
3. Generate figures:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/02_bandit_analysis.ipynb
   jupyter nbconvert --to notebook --execute notebooks/03_kernel_profiling.ipynb
   ```
4. Compile: `cd paper && make`
5. Submit the resulting PDF + `paper/` source to arxiv.org
   - Categories: `cs.LG` (primary) + `cs.DC` (cross-list)
   - Takes 1–2 business days for moderation

### Step 5 — Post on X and LinkedIn (after Step 0 and Step 4)

Fill in the real numbers in `social/x_thread.md` and
`social/linkedin_post.md`. Best sequence:

1. Post once the arXiv ID exists — include the link in both posts
2. X first (faster-moving audience), LinkedIn within the same day
3. Reply to your own X thread with the LinkedIn link for cross-traffic

---

## Why Preprint.org wasn't chosen as primary

Preprint.org (ResearchGate's preprint server) has lower visibility in the
ML/CS community than arXiv and is not indexed by Google Scholar as
reliably. **arXiv is the standard for ML systems papers** and is what
reviewers, conference PCs, and other researchers will look for. Use
arXiv as primary; there's no need for a second preprint server once
arXiv + Zenodo (for the software) + JOSS (for peer-reviewed software
citation) are in place — between the three, you have priority,
citability, and peer review covered.

---

## Quick links once everything is live

Update this list as IDs become available:

- GitHub: https://github.com/Mattral/FlashSpec
- Docs: https://flashspec.readthedocs.io
- Zenodo DOI: `TODO — paste after Step 2`
- JOSS DOI: `TODO — paste after Step 3 acceptance`
- arXiv ID: `TODO — paste after Step 4`
