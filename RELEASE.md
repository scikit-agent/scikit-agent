# Releasing scikit-agent

This describes how maintainers cut a release. Releases are published to PyPI by
GitHub Actions; nothing is built or uploaded from a maintainer's laptop.

## How it works

The version is not stored in a file. `hatch-vcs` derives it from the most recent
git tag, so **the tag is the release**. There is no version-bump commit to make.

Publishing is triggered by a **published GitHub Release**, not by the tag alone.
`.github/workflows/cd.yml` runs two jobs:

- `dist` builds the sdist and wheel and inspects them. It runs on every push,
  pull request, and manual dispatch, so packaging breakage surfaces before a
  release, not during one.
- `publish` uploads to PyPI. It is gated on
  `github.event_name == 'release' && github.event.action == 'published'` and
  runs in the `pypi` GitHub environment, authenticating to PyPI via trusted
  publishing (OIDC). There is no API token to rotate.

Trusted publishing binds to a specific repository and workflow, so releases must
be made from `scikit-agent/scikit-agent`. A tag pushed to a fork will not
publish.

## Steps

1. **Check that `main` is green.** In particular the `dist` job, whose log lists
   the exact contents of the sdist and wheel. Skim it: the sdist should contain
   the source tree and the wheel should contain only `skagent/` and its
   `.dist-info`.

2. **Update `CHANGELOG.md`.** Rename the `## [Unreleased]` heading to
   `## [X.Y.Z] - YYYY-MM-DD`, add a fresh empty `## [Unreleased]` above it, and
   update the link references at the bottom of the file:

   ```markdown
   [Unreleased]:
     https://github.com/scikit-agent/scikit-agent/compare/vX.Y.Z...main
   [X.Y.Z]: https://github.com/scikit-agent/scikit-agent/releases/tag/vX.Y.Z
   [previous]:
     https://github.com/scikit-agent/scikit-agent/compare/vA.B.C...vX.Y.Z
   ```

   Open this as a pull request and merge it. The changelog must be on `main`
   before the tag, so the tag captures it.

3. **Tag the merge commit and push the tag** to the canonical repository:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push skagent vX.Y.Z   # whichever remote is scikit-agent/scikit-agent
   ```

   Tags are `vX.Y.Z`. The leading `v` is stripped when deriving the version, so
   tag `v1.2.0` produces version `1.2.0`.

4. **Publish a GitHub Release** on that tag, with the changelog section as the
   body. Publishing it starts the upload; saving a draft does not.

5. **Verify.** Watch the `publish` job, then confirm the new version appears at
   <https://pypi.org/project/scikit-agent/> and installs cleanly into an empty
   environment:

   ```bash
   uv run --with scikit-agent --no-project -- python -c "import skagent; print(skagent.__version__)"
   ```

## Version numbers

The project follows [Semantic Versioning](https://semver.org/). While the
Development Status classifier in `pyproject.toml` says Pre-Alpha and the major
version is 0, the public API may change in minor releases; say so plainly in the
changelog when it does.

Pre-releases use PEP 440 suffixes on the tag — `v1.2.0rc1`. Mark the GitHub
Release as a pre-release so PyPI records it as such and
`pip install scikit-agent` continues to resolve to the last stable version.

## Things that will bite you

**Do not build locally for upload.** Hatchling's default sdist selection
includes everything not excluded by `.gitignore`, which means untracked scratch
files, virtualenvs, and downloaded artifacts sitting in the working tree get
packed into the sdist. The repository has historically accumulated exactly this
kind of debris. CI builds from a clean checkout, which is the point. `uv build`
is fine for inspecting a local build; just never upload its output.

**A version can be uploaded to PyPI only once**, and deleting it does not free
the number. If a release is broken, fix forward with a new patch version and
[yank](https://pypi.org/help/#yanked) the bad one — yanking hides it from
resolution while leaving it installable by exact pin, so it does not break
anyone who already depends on it.

**The `hatchling` upper bound in `pyproject.toml` is deliberate.** hatchling
1.32.0 raised the default core metadata version to 2.5, which `twine` rejects as
invalid, failing the `dist` job. The pin holds core metadata at 2.4. Relax it
once twine and PyPI accept 2.5.

**License metadata is expressed once.** `pyproject.toml` declares
`license = "MIT"` (a PEP 639 expression). PyPI rejects a distribution that also
carries a `License :: ...` classifier, so do not add one back.

**To rehearse without spending a version number**, add a `repository-url` of
`https://test.pypi.org/legacy/` to the `pypa/gh-action-pypi-publish` step and
release a pre-release tag; it lands on TestPyPI. Remove the line afterwards.
TestPyPI needs its own trusted publisher registration.

## One-time setup

Already done for this project; recorded here so it can be re-verified or
repeated:

- **PyPI**: a trusted publisher for project `scikit-agent`, owner
  `scikit-agent`, repository `scikit-agent`, workflow `cd.yml`, environment
  `pypi`. Before the first upload this is registered as a _pending_ publisher,
  since the project does not yet exist.
- **GitHub**: an environment named `pypi` on the repository. The `publish` job
  references it and fails without it. Restrict it to protected tags if you want
  a human gate on uploads.
