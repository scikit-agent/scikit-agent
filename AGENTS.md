# Notes for coding assistants

This file is for AI coding assistants working in this repository. It carries
operations and gotchas only. Every rule about how to write code, prose or
documentation lives in one of the documents below, and where this file and one
of them disagree, the document wins.

| Document                          | What it covers                                                    |
| --------------------------------- | ----------------------------------------------------------------- |
| `.github/CONTRIBUTING.md`         | development environment, pre-commit, tests, coverage, docs builds |
| `docs/community/index.md`         | contributor guidelines, the project's AI policy, code of conduct  |
| `docs/community/documentation.md` | the documentation standard: docstrings, page conventions, voice   |

## The project's AI policy binds you

Read it in `docs/community/index.md` before contributing generated work. In
short: a human contributor is responsible for anything submitted and must be
able to explain it, use of AI is disclosed in the pull request, and agents do
not open pull requests on their own. Propose changes and leave committing,
pushing and submitting to the human you are working with.

## Commands

```bash
uv sync --extra test --extra docs          # set up
uv run --no-sync pytest                    # the default suite
uv run --no-sync pytest -n auto            # same, across cores
uv run --no-sync pytest -n auto --runoracle  # before changing a solver, loss or benchmark model
pre-commit run                             # formatting and lint on changed files
uv run --no-sync python -m sphinx -b html -W --keep-going docs docs/_build
```

Building the docs needs graphviz installed. The first docs build executes every
gallery example and takes minutes; later builds take seconds.

## Gotchas

- `docs/auto_examples/` is generated on every build and is not in the
  repository. Edit the example in `examples/` instead.
- A cross-reference to a Python object that does not resolve renders as plain
  code and does not fail the `-W` build. Check that a new one is a link in the
  built page.
- Prettier owns markdown wrapping at 80 columns and ruff owns Python at 88. Do
  not hand-wrap against either; run `pre-commit run` and let the hooks lay it
  out.
- Stage explicit paths rather than `git add -A`. The working tree usually holds
  untracked files that are not meant for the repository.
- Add an entry under `[Unreleased]` in `CHANGELOG.md` for a user-visible change,
  and keep it short.
