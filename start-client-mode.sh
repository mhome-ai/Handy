#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://192.168.86.21:9999}"
APP_SUPPORT_DIR="$HOME/Library/Application Support/com.pais.handy"
SETTINGS_PATH="$APP_SUPPORT_DIR/settings_store.json"
REPO_DIR="/Users/alimao/.openclaw/workspace/handy"

mkdir -p "$APP_SUPPORT_DIR"

python3 - <<PY
import json
from pathlib import Path

settings_path = Path(${SETTINGS_PATH@Q})
base_url = ${BASE_URL@Q}

if settings_path.exists():
    try:
        data = json.loads(settings_path.read_text())
    except Exception:
        data = {}
else:
    data = {}

data["client_mode_enabled"] = True
data["client_mode_base_url"] = base_url

settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(f"updated {settings_path} -> client_mode_enabled=true, client_mode_base_url={base_url}")
PY

cd "$REPO_DIR"

if [ -d "/Applications/Handy.app" ]; then
  echo "launching installed Handy.app"
  open -a "/Applications/Handy.app"
else
  echo "Handy.app not found in /Applications, starting from repo"
  npm run tauri dev
fi
