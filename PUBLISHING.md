# Publishing

Status as of October 2026. This file records what is already public and
what stays with the maintainer. It does not start a deposit.

## Already public

| Item | Where |
|---|---|
| Package 0.1.5 | PyPI: wheel `flashspec-0.1.5-py3-none-any.whl` and sdist `flashspec-0.1.5.tar.gz` |
| GitHub Releases | v0.1.0, v0.1.1, v0.1.3, v0.1.4, v0.1.5. Titles use `FlashSpec vX.Y.Z — short title`. |
| Citable preprint | Zenodo [10.5281/zenodo.20766119](https://doi.org/10.5281/zenodo.20766119), deposited 2026-06-19. Creator Min Htet Myet, ORCID [0009-0005-0853-8301](https://orcid.org/0009-0005-0853-8301). |
| Code | https://github.com/Mattral/FlashSpec |

The PyPI project `flashspec` at 0.1.2 was yanked by the maintainer. Do not
install `flashspec==0.1.2`. There is no `v0.1.2` tag. The optional package
`flashspec-asm-kernel` 0.1.2 is a different artifact and remains the Linux
wheel named by the `asm` extra.

## Drafts that are not deposits

- `paper/flashspec.tex` is the draft for a later Zenodo version. It is
  dated October 2026, describes software 0.1.5, and names no affiliation.
  It has not been deposited. It is not an arXiv submission.
- `paper/joss/paper.md` is a JOSS-format draft. It is not under review.
- Social posts under `social/` are drafts. Do not treat them as published
  announcements.

Depositing a new Zenodo version stays with the maintainer. A GitHub Release
can cause Zenodo to archive a tag when the GitHub integration is enabled.
Do not edit a published release in order to mint a paper DOI.

## Not this release

- v0.2.0 stays reserved for a measured datacenter-GPU Llama run. Do not
  tag it before that measurement exists.
- No H100 generation result is committed. Medusa and EAGLE are not
  implemented.
- The 95% coverage gate, `mypy --strict`, and the lint job are not green
  gates. Do not describe them as passed in a release note.
