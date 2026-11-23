"""
Tracker-free persistence: the probability that a feature of depth >= 10 hPa on
day t has a feature of the same sign and depth >= 5 hPa within r km of its
position on day t + k.  No linking, no gate: it complements the track
statistics, whose long-track shares depend on the linking radius
(track_sensitivity.py).

    python lagged_coherence.py [n_realisations]          # R1 1979-2023
    GN_SOURCE=era5 python lagged_coherence.py

Writes output/experiments/gaussian_null{,_era5}/lagged_coherence.json:
    {dataset: {sign: {radius: [p(k) for k in LAGS]}}}
Check: lag 1, 500 km reproduces coh500_{C,A} in metrics.json.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from analyse import SIGN, W, load
from common import MAIN_PERIOD, OUT, REAL

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'prototype'))
from tracking import hav, pix2ll  # noqa: E402

LAGS = [1, 2, 3, 4, 5, 7, 10, 15, 30, 365]      # 365 d: same season, another year = chance level
RADII = [500, 1000]


def coherence(F, tau_hi=10, tau_lo=5):
    hi, lo = F['depth'] >= tau_hi, F['depth'] >= tau_lo
    lat, lon = pix2ll(*np.divmod(F['peak_idx'], W))
    s, T = F['slot'] - F['slot0'], F['T']
    sh, sl = s[hi], s[lo]
    lah, loh, lal, lol = lat[hi], lon[hi], lat[lo], lon[lo]
    days = np.arange(T)
    a0, a1 = np.searchsorted(sh, days), np.searchsorted(sh, days, 'right')
    b0, b1 = np.searchsorted(sl, days), np.searchsorted(sl, days, 'right')
    out = {r: [] for r in RADII}
    for k in LAGS:
        hit, n = {r: 0 for r in RADII}, 0
        for t in range(T - k):
            if a1[t] == a0[t]:
                continue
            n += a1[t] - a0[t]
            if b1[t + k] == b0[t + k]:
                continue
            D = hav(lah[a0[t]:a1[t]], loh[a0[t]:a1[t]], lal[b0[t + k]:b1[t + k]], lol[b0[t + k]:b1[t + k]]).min(1)
            for r in RADII:
                hit[r] += int((D <= r).sum())
        for r in RADII:
            out[r].append(hit[r] / n)
    return out


def one(name):
    res = {}
    for tag in ('sup', 'sub'):
        F = load(REAL, tag, MAIN_PERIOD) if name == 'real' else load(name, tag)
        res[SIGN[tag]] = {str(r): v for r, v in coherence(F).items()}
    print('done', name, flush=True)
    return name, res


def main():
    nreal = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    y0, y1 = MAIN_PERIOD
    names = ['real'] + [f'{kind}_{y0}_{y1}_r{r}' for kind in ('ft', 'iaaft') for r in range(nreal)]
    with Pool(int(os.environ.get('GN_WORKERS', 3))) as pool:
        res = dict(pool.imap_unordered(one, names))
    res['_lags'] = LAGS
    json.dump(res, open(os.path.join(OUT, 'lagged_coherence.json'), 'w'), indent=1)
    print('wrote lagged_coherence.json')


if __name__ == '__main__':
    main()
