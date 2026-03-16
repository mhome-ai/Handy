#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_OUT_DIR="$REPO_DIR/app"
BUNDLE_APP="$REPO_DIR/src-tauri/target/release/bundle/macos/Handy.app"

cd "$REPO_DIR"

echo "== building Handy.app =="
npm run tauri build -- --bundles app

if [ ! -d "$BUNDLE_APP" ]; then
  echo "Build finished but Handy.app was not found at: $BUNDLE_APP" >&2
  exit 1
fi

mkdir -p "$APP_OUT_DIR"
rm -rf "$APP_OUT_DIR/Handy.app"
cp -R "$BUNDLE_APP" "$APP_OUT_DIR/Handy.app"

echo "== packaged app copied to =="
echo "$APP_OUT_DIR/Handy.app"
