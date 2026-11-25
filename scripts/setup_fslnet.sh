#!/usr/bin/env bash
# FSL-Net (Barrabes et al., ICML 2025; code and weights CC BY-NC 4.0) in its own environment, at the commit used:
# github.com/AI-sandbox/FSL-Net @ adbb87608bba7e1da9d7d7207047efc5550d23c2. CPU-only PyTorch.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${FSLNET_VENV:-$PWD/.venv-fslnet}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install numpy==2.5.5 "git+https://github.com/AI-sandbox/FSL-Net.git@adbb87608bba7e1da9d7d7207047efc5550d23c2"
echo "export FSLNET_PYTHON=$VENV/bin/python"
