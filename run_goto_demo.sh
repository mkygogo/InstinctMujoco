#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Look for virtual environment in common locations
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    VENV_ACTIVATE=""  # already activated
elif [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/../InstinctMJ/.venv/bin/activate" ]]; then
    VENV_ACTIVATE="$SCRIPT_DIR/../InstinctMJ/.venv/bin/activate"
else
    echo "No virtual environment found. Please activate one or create .venv/ in the project root." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
[[ -n "${VENV_ACTIVATE:-}" ]] && source "$VENV_ACTIVATE"

python run_goto_demo.py --terrain pyramid --use-depth "$@"
