"""Paths. Every location can be overridden by an environment variable.

GRWR_DATA    downloaded datasets and third-party files (default: ./data)
GRWR_CACHE   caches written on first use (default: ./cache)
GRWR_OUT     per-unit p-values and scored cells (default: ./outputs)
FSLNET_PYTHON  Python interpreter of the FSL-Net environment (default: ./.venv-fslnet/bin/python)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir(var, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(var, os.path.join(ROOT, default))))


DATA = _dir("GRWR_DATA", "data")
CACHE = _dir("GRWR_CACHE", "cache")
OUT = _dir("GRWR_OUT", "outputs")

CATS = os.path.join(DATA, "cats")
LT_WALKS = os.path.join(DATA, "causal_chambers", "lt_walks_v1")
WT_WALKS = os.path.join(DATA, "causal_chambers", "wt_walks_v1")
WT_META = os.path.join(DATA, "causal_chambers", "wt_meta")
LTOWN = os.path.join(DATA, "ltown")
ARM_CORE = os.path.join(DATA, "third_party", "arm_core.py")
FSLNET_PYTHON = os.path.abspath(os.path.expanduser(
    os.environ.get("FSLNET_PYTHON", os.path.join(ROOT, ".venv-fslnet", "bin", "python"))))
