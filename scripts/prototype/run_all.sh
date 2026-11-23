#!/usr/bin/env bash
# Features of every stored disc, 1948-2023, and where their saddles sit.
# Usage: bash scripts/prototype/run_all.sh [Y0 Y1]      (default 1948 2023; about 10 minutes on 10 cores)
set -euo pipefail
Y0=${1:-1948}
Y1=${2:-2023}
PY=${PY:-python}
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(dirname "$(dirname "$HERE")")/output/prototype"
mkdir -p "$OUT"
"$PY" "$HERE/extract_features.py"       --years "$Y0" "$Y1" | tee "$OUT/extract_features_${Y0}_${Y1}.out"
"$PY" "$HERE/saddles_and_hysteresis.py" --years "$Y0" "$Y1" | tee "$OUT/saddles_and_hysteresis_${Y0}_${Y1}.out"
