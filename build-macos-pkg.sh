#!/usr/bin/env bash
# Build an unsigned macOS installer .pkg from dist/Neurolight.app
#
# Produces:
#   dist/Neurolight-unsigned.pkg
#
# Notes:
# - This is *not* notarized. On other Macs, Gatekeeper may warn or block.
# - For distribution, you typically sign + notarize (Developer ID).

set -euo pipefail

cd "$(dirname "$0")"

APP="dist/Neurolight.app"
if [[ ! -d "$APP" ]]; then
  echo "Missing $APP"
  echo "Run: ./build-macos-app.sh"
  exit 1
fi

OUT_DIR="dist"
PKGROOT="build/pkgroot"
COMPONENT_PKG="build/Neurolight.component.pkg"
FINAL_PKG="$OUT_DIR/Neurolight-unsigned.pkg"

rm -rf "$PKGROOT" "$COMPONENT_PKG" "$FINAL_PKG"
mkdir -p "$PKGROOT/Applications" "$OUT_DIR" "build"

echo "Staging app into pkgroot..."
ditto "$APP" "$PKGROOT/Applications/Neurolight.app"

echo "Building component package..."
pkgbuild \
  --root "$PKGROOT" \
  --identifier "com.neurolight.workbench" \
  --version "1.0.0" \
  --install-location "/" \
  "$COMPONENT_PKG"

echo "Building product archive..."
productbuild \
  --package "$COMPONENT_PKG" \
  "$FINAL_PKG"

echo ""
echo "Done! Created $FINAL_PKG"
echo "Try opening it with: open \"$FINAL_PKG\""
