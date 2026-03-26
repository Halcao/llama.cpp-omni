#!/bin/bash
# llama.cpp-omni Desktop App - Quick Start
#
# Usage:
#   bash apps/start.sh --model-dir /path/to/MiniCPM-o-4_5-gguf
#   bash apps/start.sh --model-dir /path/to/gguf --http
#   bash apps/start.sh --model-dir /path/to/gguf --port 9090

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=================================================="
echo "  llama.cpp-omni Desktop App"
echo "=================================================="

# Check Python
PYTHON="${PYTHON_CMD:-python3}"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: Python3 not found. Set PYTHON_CMD to your python path."
    exit 1
fi

# Install deps if needed
if ! "$PYTHON" -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
fi

# Check llama-server binary
SERVER_BIN=""
for candidate in \
    "$REPO_ROOT/build/bin/llama-server" \
    "$REPO_ROOT/build/bin/Release/llama-server"; do
    if [ -f "$candidate" ]; then
        SERVER_BIN="$candidate"
        break
    fi
done

if [ -z "$SERVER_BIN" ]; then
    echo ""
    echo "  llama-server not found. Building..."
    echo ""
    cd "$REPO_ROOT"
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --target llama-server -j
    SERVER_BIN="$REPO_ROOT/build/bin/llama-server"
fi

echo "  Server binary: $SERVER_BIN"
echo "=================================================="
echo ""

# Launch
cd "$SCRIPT_DIR"
exec "$PYTHON" desktop/launcher.py "$@"
