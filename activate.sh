#!/bin/bash

# Ensure the script is sourced, not executed directly.
# Running as ./activate.sh creates a subshell, which prevents environment
# changes from persisting in the current terminal.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ Error: This script must be sourced to affect your current terminal."
    echo "👉 Please run:  source activate.sh"
    exit 1
fi

# --- Step 1: Activate the shared conda environment ---
# Prevent conda from prepending its own label to PS1 — we set our own below.
export CONDA_CHANGEPS1=false

# 'conda activate' is a shell function that requires conda's hooks to be loaded.
# These are normally sourced by .bashrc in interactive terminals, but may not
# be present when sourcing this script directly, so we load them explicitly.
CONDA_BASE=$(conda info --base 2>/dev/null)

if [[ -z "$CONDA_BASE" ]]; then
    echo "❌ Error: Could not determine the conda base path."
    echo "   Make sure 'conda' is available in your PATH."
    return 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate davian-py3110
echo "✅ Conda environment 'davian-py3110' activated (Python 3.11 interpreter)."

# --- Step 2: Activate the project-specific virtual environment ---
# davian-py3110 is a shared env for all Python 3.11 projects.
# The .venv holds packages specific to THIS project, managed by uv.

# Prevent venv from prepending its own label to PS1.
export VIRTUAL_ENV_DISABLE_PROMPT=1

VENV_PATH="$(dirname "${BASH_SOURCE[0]}")/.venv"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "⚠️  Warning: No .venv found at '$VENV_PATH'."
    echo "   Run 'uv pip sync' to create and populate it."
    return 1
fi

source "$VENV_PATH/bin/activate"
echo "✅ Project venv activated (project-specific packages)."

# --- Step 3: Set a concise, unified prompt ---
# Save the original PS1 so deactivate.sh can fully restore it.
export _PREV_PS1="$PS1"

# Dynamically read the active Python version (e.g. "3.11") — avoids hardcoding
# in case the conda env interpreter is ever updated.
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

# Format: (cns:pyX.Y) ~/current/dir $
# - (cns:pyX.Y) : project tag + Python version from the active conda env
# - \w         : current working directory (abbreviated with ~)
# - \$         : $ for normal user, # for root
PS1="\[\e[0;32m\](cns:py${PYTHON_VERSION})\[\e[0m\] \[\e[0;34m\]\w\[\e[0m\] \$ "
export PS1