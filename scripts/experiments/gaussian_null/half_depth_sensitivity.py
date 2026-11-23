"""
Sensitivity of the footprint to the level fraction (Table 8 of the paper).

The paper's footprint of a feature is the connected component of {g > level}
that contains its peak, with level = peak - f * depth and f = 1/2.  This script
repeats the area at f = 1/3, 1/2 and 2/3 for every feature of depth >= 10 hPa,
for the real field (main period) and for a few regenerated ft / iaaft
surrogates (same seeds as run_surrogates.py, so the fields are identical to
the ones behind metrics.json).

    python half_depth_sensitivity.py --reals 0 1 2
    GN_SOURCE=era5 python half_depth_sensitivity.py --reals 0 1 2

Writes output/experiments/gaussian_null{,_era5}/half_depth_sensitivity.json.
Check: the f = 1/2 medians must reproduce half_med_{C,A}_10 in metrics.json.
"""
import argparse
import json
import os
import time

import numpy as np

from basins import Disc
from common import MAIN_PERIOD, OUT, START_YEAR, build_anomalies, discs, projection_operator
from surrogates import Period

FRACS = {'1/3': 1 / 3, '1/2': 0.5, '2/3': 2 / 3}
TAU_PA = 1000.0


def areas(field):
    """{sign: {frac: array of km2}} for all features of depth >= 10 hPa in a (T, 37, 144) field."""
    P, valid, shape = projection_operator()
    out = {c: {k: [] for k in FRACS} for c in 'CA'}
    for i in range(field.shape[0]):
        sub, sup = discs(field[i].ravel(), P, valid, shape)
        for c, g in (('C', sup), ('A', sub)):
            D = Disc(g)
            F = D.features()
            depth = F['peak'] - F['saddle']
            k = depth >= TAU_PA
            for name, f in FRACS.items():
                out[c][name].append(D.level_area(F['peak_idx'][k], F['peak'][k] - f * depth[k]))
    return {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in out.items()}


def summarise(A):
    return {c: {k: {'median_1e6km2': float(np.median(v) / 1e6), 'n': int(len(v)),
                    'req_km': float(np.sqrt(np.median(v) / np.pi))} for k, v in d.items()} for c, d in A.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reals', nargs='*', type=int, default=[0, 1, 2])
    a = ap.parse_args()
    t0 = time.time()
    y0, y1 = MAIN_PERIOD
    anom, _, _ = build_anomalies()
    seg = anom[(y0 - START_YEAR) * 365:(y1 - START_YEAR + 1) * 365].copy()
    del anom
    res = {'real': summarise(areas(seg))}
    print(f'real done {time.time() - t0:.0f}s', res['real'], flush=True)
    P = Period(seg)
    del seg
    for r in a.reals:
        for kind in ('ft', 'iaaft'):
            rng = np.random.default_rng(1000 * y0 + r)
            S = P.ft(rng) if kind == 'ft' else P.iaaft(rng)[0]
            res[f'{kind}_r{r}'] = summarise(areas(S))
            del S
            print(f'{kind} r{r} done {time.time() - t0:.0f}s', res[f'{kind}_r{r}'], flush=True)
            json.dump(res, open(os.path.join(OUT, 'half_depth_sensitivity.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
