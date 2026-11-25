#!/usr/bin/env bash
# Regenerate Figure 1, Figure 2, the per-setting table and every number of the text from results/ (a few seconds).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
$PY figure1.py
$PY figure2.py
$PY table_settings.py
$PY text_numbers.py
