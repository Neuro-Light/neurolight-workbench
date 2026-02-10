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

### CI (`.github/workflows/ci.yml`)

Runs on **every push and every pull request**:

1. **Lint**: Ruff check and format.
2. **Test**: `pytest tests/`.
3. **Build**: `uv build` (wheel/sdist).
4. **Release prep** (only on **push to `main`**): After lint, test, and build succeed, the workflow creates the next patch version tag (e.g. `v1.0.0` → `v1.0.1`) on the current commit and pushes it. If no tags exist yet, it creates `v1.0.0`.

Artifacts from the build job on `main` are uploaded (e.g. `dist/`) for that run.

### CD (`.github/workflows/cd.yml`)

Runs on **push of a tag** matching `v*` (e.g. after CI pushes `v1.0.1`):

1. **build-python**: Checkout the tag, run `uv build`, upload `dist/*` (wheels and sdist) as an artifact.
2. **build-exe**: Matrix over Windows, macOS, and Linux. On each runner: checkout tag, `uv sync --extra build`, run `pyinstaller neurolight.spec`, and upload the executable as an artifact (e.g. `neurolight-1.0.1-windows-amd64.exe`).
3. **release**: Download all artifacts and create a GitHub Release for the tag with:
   - Python wheel(s) and sdist
   - Standalone executables for Windows, macOS, and Linux

Release notes are generated automatically.

## Summary flow

1. PR is merged into `main` → CI runs (lint, test, build).
2. CI creates and pushes tag `vX.Y.Z` on the merged commit.
3. CD is triggered by the new tag → builds Python packages and platform executables → publishes the GitHub Release with all assets.
