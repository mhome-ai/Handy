#!/usr/bin/env sh
set -eu

BASE_URL="${1:-http://192.168.86.21:9999}"
APP_SUPPORT_DIR="$HOME/Library/Application Support/ai.mhome.footy"
SETTINGS_PATH="$APP_SUPPORT_DIR/settings_store.json"
REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

mkdir -p "$APP_SUPPORT_DIR"

SETTINGS_PATH="$SETTINGS_PATH" BASE_URL="$BASE_URL" python3 - <<'PY'
import json
import os
from pathlib import Path

settings_path = Path(os.environ['SETTINGS_PATH'])
base_url = os.environ['BASE_URL']

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

if [ -d "/Applications/Footy.app" ]; then
  echo "launching installed Footy.app"
  open -a "/Applications/Footy.app"
else
  echo "Footy.app not found in /Applications, starting from repo"
  npm run tauri dev
fi
