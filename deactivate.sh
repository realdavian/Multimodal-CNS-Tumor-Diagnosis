#!/bin/bash

# Ensure the script is sourced, not executed directly.
# Running as ./deactivate.sh creates a subshell, so deactivation
# would not persist in the current terminal.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "❌ Error: This script must be sourced to affect your current terminal."
    echo "👉 Please run:  source deactivate.sh"
    exit 1
fi

# Deactivation order is the reverse of activation:
# first the project venv, then the shared conda environment.

# --- Step 1: Deactivate the project-specific virtual environment ---
# The 'deactivate' function is injected into the shell by 'source .venv/bin/activate'.
# We check if it exists before calling it to avoid errors if the venv is not active.
if declare -f deactivate > /dev/null 2>&1; then
    deactivate
    echo "✅ Project venv deactivated."
else
    echo "ℹ️  Project venv was not active, skipping."
fi

# --- Step 2: Deactivate the shared conda environment ---
# We check CONDA_DEFAULT_ENV to see if a conda environment is currently active.
if [[ -n "$CONDA_DEFAULT_ENV" && "$CONDA_DEFAULT_ENV" != "base" ]]; then
    conda deactivate
    echo "✅ Conda environment '$CONDA_DEFAULT_ENV' deactivated."
else
    echo "ℹ️  No active conda environment to deactivate, skipping."
fi

# --- Step 3: Restore the original prompt ---
# Undo the CONDA_CHANGEPS1 and VIRTUAL_ENV_DISABLE_PROMPT overrides.
unset CONDA_CHANGEPS1
unset VIRTUAL_ENV_DISABLE_PROMPT

# Restore PS1 to what it was before activation.
if [[ -n "$_PREV_PS1" ]]; then
    PS1="$_PREV_PS1"
    export PS1
    unset _PREV_PS1
fi
