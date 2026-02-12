# Build and release

This document describes how to build Neurolight locally and how the CI/CD pipeline produces releases.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## Versioning

Version is **derived from Git at build time** via [versioningit](https://github.com/jwodder/versioningit):

- On a **tagged** commit (e.g. `v1.0.1`): version is `1.0.1`.
- On an **untagged** commit: version is something like `1.0.1.dev3+gabc1234` (distance from last tag + commit).

No manual version bump in the repo is required; tagging is the source of truth.

## Local build

### Run from source

```bash
uv sync
uv run python src/main.py
```

Or, after installing the package (e.g. `uv pip install -e .`):

```bash
uv run neurolight
```

### Wheel and sdist

From the repo root (with Git and tags available):

```bash
uv sync
uv build
```

Outputs go to `dist/` (e.g. `neurolight_prototype-1.0.1-py3-none-any.whl` and a source tarball).

### Standalone executable (PyInstaller)

Install the project and the `build` extra (PyInstaller), then run the spec from the repo root:

```bash
uv sync --extra build
uv run pyinstaller neurolight.spec --noconfirm --clean
```

- **Windows**: `dist/neurolight.exe`
- **macOS / Linux**: `dist/neurolight` (no extension)

Using the same Python version (3.10) and the lockfile (`uv.lock`) produces reproducible builds. The spec file (`neurolight.spec`) uses no hardcoded user paths.

## CI/CD

There are three triggers that drive the pipeline:

| Trigger | What happens | Who it's for |
|---|---|---|
| **PR merge** | Build artifacts (wheel, sdist) | Developers / testers |
| **Push version tag** (`v*`) | Full release (packages + executables) | End users |
| **Manual dispatch** | On-demand build | Developers |

### CI (`.github/workflows/ci.yml`)

Runs on **every push and every pull request**:

1. **Lint** — Ruff check and format.
2. **Test** — `pytest tests/`.
3. **Build** — `uv build` (wheel/sdist).

When a PR is merged into `main` (push event), two additional things happen:

4. **Artifact upload** — the wheel and sdist from the build job are uploaded as workflow artifacts for testing.
5. **Release prep** — after lint, test, and build succeed, the workflow computes the next semver tag and pushes it. The bump type defaults to **patch** but can be overridden by including `[minor]` or `[major]` in the merge commit message. If no tags exist yet, it starts at `v1.0.0`.

### CD (`.github/workflows/cd.yml`)

Runs when a **version tag** matching `v*` is pushed (typically by the release-prep step above, but can also be pushed manually):

1. **CI check** — polls the commit status API for up to 10 minutes to confirm that CI passed on the tagged commit before proceeding.
2. **build-python** — checks out the tag, runs `uv build`, and uploads `dist/*` (wheel and sdist).
3. **build-exe** — matrix build across Windows, macOS, and Linux. On each runner: installs the `build` extra, runs PyInstaller with the spec file, and uploads the platform executable (e.g. `neurolight-v1.0.1-windows-amd64.exe`).
4. **release** — downloads all artifacts, validates them, and creates a GitHub Release for the tag with:
   - Python wheel and sdist
   - Standalone executables for all three platforms
   - Auto-generated release notes

### Manual / on-demand builds

Any workflow with a `workflow_dispatch` trigger can be run from the GitHub Actions tab using the **Run workflow** button. This is useful for producing a one-off build without merging a PR or pushing a tag.

## Summary flow

```
PR merged into main
  └─► CI (lint → test → build → upload artifacts)
        └─► release-prep pushes tag vX.Y.Z
              └─► CD (ci-check → build-python + build-exe → GitHub Release)

Manual tag push (git tag v… && git push origin v…)
  └─► CD (same as above)

Manual dispatch (Actions tab → Run workflow)
  └─► On-demand build
```
