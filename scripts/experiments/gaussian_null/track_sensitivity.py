"""
Sensitivity of the track statistics, and of their real / surrogate ratios, to
the linking radius R and to the two thresholds.

Tracks are gated nearest-neighbour with two-level thresholds (analyse.track_stats):
features are followed down to `lo` and a track is kept if it reaches `hi`.

    python track_sensitivity.py                 # R1 1979-2023, surrogates r0-r4
    GN_SOURCE=era5 python track_sensitivity.py

Writes output/experiments/gaussian_null{,_era5}/track_sensitivity.json:
    {config: {dataset: {sign: {n_per_yr, meanL, pct1, pct_ge5, pct_ge10, median_step_km}}}}
"""
import json
import os
import sys
from multiprocessing import Pool

from analyse import SIGN, load, track_stats
from common import MAIN_PERIOD, OUT, REAL

CONFIGS = [(5, 10, 500), (5, 10, 750), (5, 10, 1000), (5, 10, 1250), (5, 10, 1500),
           (10, 10, 1000), (3, 10, 1000), (7.5, 15, 1000), (10, 20, 1000)]
KEEP = ('n_per_yr', 'meanL', 'pct1', 'pct_ge5', 'pct_ge10', 'median_step_km')


def one(name):
    out = {}
    for tag in ('sup', 'sub'):
        F = load(REAL, tag, MAIN_PERIOD) if name == 'real' else load(name, tag)
        for lo, hi, R in CONFIGS:
            ts = track_stats(F, lo=lo, hi=hi, R=R)
            out.setdefault(f'{lo:g}/{hi:g}/{R:g}', {})[SIGN[tag]] = {k: float(ts[k]) for k in KEEP}
    print('done', name, flush=True)
    return name, out


def main():
    nreal = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    y0, y1 = MAIN_PERIOD
    names = ['real'] + [f'{kind}_{y0}_{y1}_r{r}' for kind in ('ft', 'iaaft') for r in range(nreal)]
    res = {}
    with Pool(int(os.environ.get('GN_WORKERS', 3))) as pool:
        for name, out in pool.imap_unordered(one, names):
            for cfg, v in out.items():
                res.setdefault(cfg, {})[name] = v
    json.dump(res, open(os.path.join(OUT, 'track_sensitivity.json'), 'w'), indent=1)
    print('wrote track_sensitivity.json')


if __name__ == '__main__':
    main()
