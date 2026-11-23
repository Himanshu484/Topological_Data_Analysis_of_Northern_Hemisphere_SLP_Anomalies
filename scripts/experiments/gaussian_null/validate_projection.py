"""
Check that anomaly + sparse projection reproduces the stored discs and the
stored prototype features.

    python scripts/experiments/gaussian_null/validate_projection.py
"""
import os
import time

import numpy as np

from common import ROOT, START_YEAR, build_anomalies, projection_operator, discs
from basins import Disc

t0 = time.time()
anom, lat, lon = build_anomalies()
print(f'anomaly {anom.shape} {anom.dtype} in {time.time() - t0:.0f}s')
t0 = time.time()
P, valid, shape = projection_operator()
print(f'operator {P.shape}, nnz {P.nnz}, {time.time() - t0:.0f}s')
rng = np.random.default_rng(1)
Zs = {t: np.load(os.path.join(ROOT, f'data/processed_data/prototype/feat_{t}_1948_2023.npz')) for t in ('sup', 'sub')}
mx = 0.0
nd = npk = 0
for k in range(30):
    y, d = int(rng.integers(1948, 2024)), int(rng.integers(1, 366))
    sub, sup = discs(anom[(y - START_YEAR) * 365 + d - 1].ravel(), P, valid, shape)
    for t, g in (('sub', sub), ('sup', sup)):
        s = np.load(os.path.join(ROOT, f'data/processed_data/SLP_data_years/{y}/slp_{t}_{y}_day_{d}.npy'))
        m = s != s[0, 0]
        assert (m == (g != g[0, 0])).all()
        mx = max(mx, np.abs(s[m] - g[m]).max())
        F = Disc(g).features()
        Z = Zs[t]
        sel = (Z['year'] == y) & (Z['day'] == d)
        a = sorted(zip(np.round(Z['birth'][sel], 1), np.round(Z['death'][sel], 1)))
        b = sorted(zip(np.round(F['saddle'], 1), np.round(F['peak'], 1)))
        nd += a != b
        npk += len(a)
print(f'max |disc - stored disc| = {mx:.3e} Pa; days whose (saddle, peak) set differs from the '
      f'prototype npz at 0.1 Pa: {nd} of 60 ({npk} features)')
