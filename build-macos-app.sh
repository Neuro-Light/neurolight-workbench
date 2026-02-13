#!/usr/bin/env bash
# Build Neurolight as a macOS .app bundle.
# Run from the project root. Output: dist/Neurolight.app

set -euo pipefail

cd "$(dirname "$0")"

echo "Installing dependencies (including PyInstaller)..."
uv sync --extra build

echo "Building Neurolight.app..."
uv run pyinstaller neurolight.spec --noconfirm --clean

if [[ -d dist/Neurolight.app ]]; then
  echo ""
  echo "Done! Neurolight.app is in dist/"
  echo "  Open it: open dist/Neurolight.app"
  echo "  Or drag dist/Neurolight.app to Applications"
else
  echo "Build may have produced different output. Check dist/"
  ls -la dist/ 2>/dev/null || true
  exit 1
fi
