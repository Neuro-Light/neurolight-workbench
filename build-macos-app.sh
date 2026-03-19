#!/usr/bin/env bash
# Build Neurolight as a macOS .app bundle.
# Run from the project root. Output: dist/Neurolight.app

set -euo pipefail

cd "$(dirname "$0")"

LOGO_PNG="logo.png"
ICONSET_DIR="build/Neurolight.iconset"
ICON_ICNS="build/Neurolight.icns"

if [[ ! -f "$LOGO_PNG" ]]; then
  echo "Missing $LOGO_PNG"
  echo "Please add logo.png at the repo root."
  exit 1
fi

echo "Generating app icon from $LOGO_PNG..."
rm -rf "$ICONSET_DIR" "$ICON_ICNS"
mkdir -p "$ICONSET_DIR"

# Build iconset sizes expected by iconutil
sips -z 16 16 "$LOGO_PNG" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -z 32 32 "$LOGO_PNG" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$LOGO_PNG" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -z 64 64 "$LOGO_PNG" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$LOGO_PNG" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -z 256 256 "$LOGO_PNG" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$LOGO_PNG" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -z 512 512 "$LOGO_PNG" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$LOGO_PNG" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$LOGO_PNG" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null

iconutil -c icns "$ICONSET_DIR" -o "$ICON_ICNS"

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
