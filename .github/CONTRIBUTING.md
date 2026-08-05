See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Setting up a development environment

Install [uv](https://docs.astral.sh/uv/), then sync the project into a local
`.venv`. `uv sync` is exact, so list every extra you need in one command (`test`
covers the pytest stack used below; `docs` is for local Sphinx builds):

```bash
uv sync --extra test --extra docs
```

# Pre-commit

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

```bash
uv run --no-sync pytest
```

# Coverage

```bash
uv run --no-sync pytest --cov=skagent
```

# Building docs

Building the docs requires graphviz to be installed (`apt-get install graphviz`
or `brew install graphviz`). To build them the same way CI does:

```bash
uv run --no-sync python -m sphinx -b html -W --keep-going docs docs/_build
```

To build and serve them with live reload while you edit:

```bash
uv run --no-sync sphinx-autobuild docs docs/_build/html
```

# Building an SDist and wheel

```bash
uv build
```
