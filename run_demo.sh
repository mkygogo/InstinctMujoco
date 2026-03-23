#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_ACTIVATE="$WORKSPACE_DIR/InstinctMJ/.venv/bin/activate"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "Virtual environment not found: $VENV_ACTIVATE" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
source "$VENV_ACTIVATE"

python run_parkour_mujoco.py \
    --command-x 0.5 \
    --terrain stairs \
    --use-depth \
    "$@"
