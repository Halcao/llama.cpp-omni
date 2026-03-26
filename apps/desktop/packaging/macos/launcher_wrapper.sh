#!/bin/bash
# macOS .app launcher — uses bundled standalone Python

DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES="$DIR/../Resources"
APP_SUPPORT="$HOME/Library/Application Support/Comni"
LOG_FILE="$APP_SUPPORT/comni_service.log"

mkdir -p "$APP_SUPPORT"

PYTHON="$RESOURCES/python/bin/python3"

if [ ! -x "$PYTHON" ]; then
    osascript -e 'display alert "Bundled Python Missing" message "The app bundle appears corrupted.\nPlease re-download Comni." as critical'
    exit 1
fi

cd "$RESOURCES"
exec "$PYTHON" apps/desktop/menubar_app.py 2>>"$LOG_FILE"
