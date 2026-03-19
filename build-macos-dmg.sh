#!/usr/bin/env bash
# Build a distributable macOS DMG for drag-and-drop install.
#
# Produces:
#   dist/Neurolight.dmg

set -euo pipefail

cd "$(dirname "$0")"

APP="dist/Neurolight.app"
DMG_STAGE="build/dmg-stage"
DMG_TMP="build/Neurolight.tmp.dmg"
DMG_FINAL="dist/Neurolight.dmg"
VOL_NAME="Neurolight"

if [[ ! -d "$APP" ]]; then
  echo "Missing $APP"
  echo "Run: ./build-macos-app.sh"
  exit 1
fi

rm -rf "$DMG_STAGE" "$DMG_TMP" "$DMG_FINAL"
mkdir -p "$DMG_STAGE" "dist" "build"

echo "Staging app for DMG..."
cp -R "$APP" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"

echo "Creating DMG..."
hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$DMG_STAGE" \
  -ov \
  -format UDZO \
  "$DMG_TMP" >/dev/null

mv "$DMG_TMP" "$DMG_FINAL"

echo ""
echo "Done! Created $DMG_FINAL"
echo "Open with: open \"$DMG_FINAL\""
