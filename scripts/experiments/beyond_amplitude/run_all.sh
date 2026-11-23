#!/usr/bin/env bash
# Detector comparison and track-end statistics, full chain (about 40 minutes on 2 cores).
# Needs data/processed_data/prototype/feat_{sup,sub}_1948_2023.npz (scripts/prototype/run_all.sh) and
# data/processed_data/experiments/beyond_amplitude/{hgt.sfc,land}.nc (see data/README.md).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
O=../../../output/experiments/beyond_amplitude
mkdir -p $O
export OMP_NUM_THREADS=1
$PY descriptors.py --jobs 2
$PY -c "from common import table; table('sup'); table('sub')"
for s in detectors.py detectors_monthly.py lifecycle.py; do
  a=""; [ $s = detectors.py ] && a="--curve"
  $PY $s --type sup $a > $O/${s%.py}_sup.out & $PY $s --type sub $a > $O/${s%.py}_sub.out; wait
done
