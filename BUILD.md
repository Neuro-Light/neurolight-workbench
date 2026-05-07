# Build and Release

This document describes how to build Neurolight Workbench locally and how the CI/CD pipeline produces releases.

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- macOS builds also require Xcode command-line tools

## Versioning

Version is **derived from Git at build time** via [versioningit](https://github.com/jwodder/versioningit):

- On a **tagged** commit (e.g. `v1.0.1`): version is `1.0.1`.
- On an **untagged** commit: version is something like `1.0.1.dev3+gabc1234` (distance from last tag + commit hash).

No manual version bump in the repo is required; tagging is the source of truth.

## Local Build

### Run from source

```bash
uv sync
uv run python src/main.py
```

Or, after installing the package (`uv pip install -e .`):

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
- **macOS** (onedir): `dist/Neurolight/` + `dist/Neurolight.app` bundle

Using the same Python version (3.10) and the lockfile (`uv.lock`) produces reproducible builds. The spec file (`neurolight.spec`) uses no hardcoded user paths.

### macOS app bundle

The repo ships helper shell scripts for macOS packaging:

```bash
# Build the .app bundle
./build-macos-app.sh
open dist/Neurolight.app
```

### macOS DMG installer

Once the app bundle exists, wrap it in a drag-to-install DMG:

```bash
./build-macos-dmg.sh
open dist/Neurolight.dmg
```

The resulting `Neurolight.dmg` is **unsigned** when built locally. For a fully signed and notarized DMG, use the CI pipeline (see [macOS release assets](#macos-release-assets-githubworkflowsciyml) below).

### macOS unsigned pkg installer (alternative)

```bash
./build-macos-pkg.sh
open dist/Neurolight-unsigned.pkg
```

This uses `pkgbuild` and `productbuild` to produce a component + product archive. It is suitable for local testing only — broad distribution requires Developer ID signing and notarization.

### Enable multiprocessing in frozen macOS builds

Alignment multiprocessing is disabled by default in frozen apps to avoid Qt import issues in spawned workers. Opt-in with:

```bash
NEUROLIGHT_ENABLE_MP=1 open dist/Neurolight.app
```

## CI/CD

There are three triggers that drive the pipeline:

| Trigger | What happens | Who it's for |
|---|---|---|
| **Every push / PR** | Lint, test, build artifacts | All contributors |
| **PR merged into `main`** | Same as above + auto-tag next semver | Developers |
| **Version tag push** (`v*`) | Full release (packages + executables + GitHub Release) | End users |
| **Manual dispatch** | On-demand build | Developers |

### CI (`.github/workflows/ci.yml`)

Runs on **every push and every pull request**:

1. **Lint** – On PRs: runs `ruff check --fix` and `ruff format`, commits any fixes back to the branch. On all events: runs `ruff check` and `ruff format --check` to enforce standards.
2. **Test** – `pytest tests/` with coverage; uploads `coverage.xml` to Codecov.
3. **Build** – `uv build` (wheel + sdist).

When a PR is merged into `main` (push to `main`), two additional things happen:

4. **Artifact upload** – wheel and sdist from the build job are uploaded as workflow artifacts.
5. **Release prep** – after lint, test, and build succeed, computes the next semver tag and pushes it. The bump type defaults to **patch**; include `[minor]` or `[major]` in the merge commit message to bump those components instead. If no tags exist yet, starts at `v1.0.0`.

   > Requires a `RELEASE_TOKEN` GitHub Actions secret (a PAT) so the pushed tag triggers downstream workflows. Tags pushed with `GITHUB_TOKEN` do not trigger other workflows.

### macOS Release Assets (`.github/workflows/ci.yml`)

Runs when a **version tag** matching `v*` is pushed (triggered by the release-prep step):

1. Builds the macOS `.app` bundle and DMG via `build-macos-app.sh` and `build-macos-dmg.sh`
2. Imports the Developer ID Application certificate from the `DEVELOPER_ID_CERT` / `CERT_PASSWORD` secrets
3. Code-signs `dist/Neurolight.app` with the `--options runtime` flag
4. Re-packages the signed app into a new DMG
5. Notarizes the DMG via `xcrun notarytool` using the `APPLE_ID`, `TEAM_ID`, and `APP_SPECIFIC_PASSWORD` secrets
6. Staples the notarization ticket
7. Uploads `Neurolight.dmg` to the GitHub Release

### CD (`.github/workflows/cd.yml`)

Runs when a **version tag** matching `v*` is pushed:

1. **CI check** – polls the GitHub check-runs API for up to 10 minutes to confirm that `lint`, `test`, `build`, and `release-prep` all passed on the tagged commit before proceeding.
2. **build-python** – checks out the tag, runs `uv build`, and uploads `dist/*` (wheel + sdist).
3. **build-exe** – matrix build across **Windows** and **macOS** (Linux exe temporarily disabled). On each runner: installs the `build` extra, runs PyInstaller with the spec file, and uploads the platform executable:
   - `neurolight-vX.Y.Z-windows-amd64.exe`
   - `neurolight-vX.Y.Z-macos-universal`
4. **release** – downloads all artifacts, validates them, and creates a GitHub Release for the tag with:
   - Python wheel and sdist
   - Standalone executables for Windows and macOS
   - Auto-generated release notes

### Manual / on-demand builds

Any workflow with a `workflow_dispatch` trigger can be run from the GitHub Actions tab using the **Run workflow** button. Useful for producing a one-off build without merging a PR or pushing a tag.

## Required GitHub Secrets

| Secret | Used by | Purpose |
|---|---|---|
| `RELEASE_TOKEN` | `ci.yml` release-prep | PAT that allows the tag push to trigger `cd.yml` |
| `CODECOV_TOKEN` | `ci.yml` test | Optional for public repos; required for private repos |
| `DEVELOPER_ID_CERT` | `ci.yml` macOS release | Base64-encoded Developer ID Application `.p12` certificate |
| `CERT_PASSWORD` | `ci.yml` macOS release | Password for the `.p12` certificate |
| `APPLE_ID` | `ci.yml` macOS release | Apple ID email for notarytool |
| `TEAM_ID` | `ci.yml` macOS release | Apple Developer Team ID for notarytool |
| `APP_SPECIFIC_PASSWORD` | `ci.yml` macOS release | App-specific password for notarytool |

## Summary Flow

```
Every push / PR
  └─► CI (lint → test → build)

PR merged into main
  └─► CI (lint → test → build → upload artifacts)
        └─► release-prep pushes tag vX.Y.Z
              ├─► macOS release assets (sign + notarize DMG → attach to release)
              └─► CD
                    └─► ci-check → build-python + build-exe (Windows + macOS)
                          └─► GitHub Release (wheel + sdist + exes + signed DMG)

Manual tag push (git tag v… && git push origin v…)
  └─► Same as the CD branch above

Manual dispatch (Actions tab → Run workflow)
  └─► On-demand build
```
