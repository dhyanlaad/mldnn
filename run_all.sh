#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found at $python_bin" >&2
    echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

make -C benchmark

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mldnn-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/mldnn-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

experiments=(
    experiments.exp1_deterministic
    experiments.exp2_stochastic_ou
    experiments.exp3_gbm
    experiments.exp4_cir
    experiments.exp5_trig
)

for experiment in "${experiments[@]}"; do
    echo "==> Running $experiment"
    "$python_bin" -m "$experiment"
done
