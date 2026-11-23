"""
Validate basins.py against scripts/prototype/pers_features.py (pairs, peak,
saddle, parent) and against brute-force scipy.ndimage labelling (basin and
half-depth areas), on real discs.

    python scripts/experiments/gaussian_null/validate_basins.py
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
from basins import Disc, PAREA, RE_KM  # noqa: E402
from pers_features import features as py_features  # noqa: E402

S8 = np.ones((3, 3), bool)


def brute_area(g, sea, peak_idx, level):
    lab, _ = ndimage.label((g > level) & ~sea, structure=S8)
    L = lab.ravel()[peak_idx]
    return PAREA[lab == L].sum()


def main():
    rng = np.random.default_rng(0)
    g0 = np.load(os.path.join(ROOT, 'data/processed_data/SLP_data_years/2000/slp_sub_2000_day_1.npy'))
    valid = g0 != g0[0, 0]
    print(f'disc area {PAREA[valid].sum() / 1e6:.2f} M km^2 vs hemisphere {2 * np.pi * RE_KM ** 2 / 1e6:.2f}')
    npair = nbad = nb = nbb = 0
    tc = tp = 0.0
    for k in range(40):
        y, d, t = int(rng.integers(1948, 2024)), int(rng.integers(1, 366)), ['sup', 'sub'][k % 2]
        g = np.load(os.path.join(ROOT, f'data/processed_data/SLP_data_years/{y}/slp_{t}_{y}_day_{d}.npy'))
        sea = g == g[0, 0]
        t0 = time.time(); Fp = py_features(g, sea); tp += time.time() - t0
        t0 = time.time(); D = Disc(g, sea); F = D.features()
        half = D.level_area(F['peak_idx'], F['peak'] - 0.5 * (F['peak'] - F['saddle'])); tc += time.time() - t0
        A = sorted((int(a[2]), int(a[3]), int(a[4])) for a in Fp)
        B = sorted(zip(F['peak_idx'].tolist(), F['saddle_idx'].tolist(), F['parent_idx'].tolist()))
        npair += len(A)
        nbad += A != B
        # brute-force areas for the deep features
        for i in np.where(F['peak'] - F['saddle'] >= 500)[0][:6]:
            gg = g.astype(np.float64)
            a1 = brute_area(gg, sea, F['peak_idx'][i], F['saddle'][i])
            a2 = brute_area(gg, sea, F['peak_idx'][i], F['peak'][i] - 0.5 * (F['peak'][i] - F['saddle'][i]))
            nb += 1
            nbb += (abs(a1 - F['basin_km2'][i]) > 1e-6 * a1) + (abs(a2 - half[i]) > 1e-6 * a2)
    print(f'{npair} pairs on 40 discs; discs whose (peak, saddle, parent) set differs from pers_features: {nbad}')
    print(f'{nb} deep features checked by brute force; area mismatches: {nbb}')
    print(f'time per disc: python {tp / 40 * 1e3:.0f} ms, C incl. half-depth {tc / 40 * 1e3:.1f} ms')


if __name__ == '__main__':
    main()
