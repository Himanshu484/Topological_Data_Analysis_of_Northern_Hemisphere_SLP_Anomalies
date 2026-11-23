"""
Terrain check for the daily analysis: are the established differences from the
null fields changed when every feature over terrain higher than 1500 m is
removed from the reanalysis and from the surrogates alike?

Sea-level pressure over high terrain is an extrapolation below ground.  The
six-hourly analysis (scripts/experiments/era5_6h) found an artefact there in
R1, and a mask was introduced after the fact; this script applies the same
mask to the daily analysis, so that the mask is a robustness check of the
whole paper and not a device of one section.

    python terrain_check.py                 # R1 1979-2023, all realisations     (~10 min on 7 cores)
    GN_SOURCE=era5 python terrain_check.py  # ERA5 1979-2022

Writes output/experiments/gaussian_null{,_era5}/terrain_check.json:
    {dataset: {'full' | 'low': {'C' | 'A': statistics}}}
with, per variant and sign: features per day above 5, 10, 20, 30 hPa; TP2 of
the stored features (depth >= 2 hPa, 98.6 % of the total); median half-depth
footprint above 10 hPa and at matched count; nesting; the area-depth exponent;
lagged coherence within 500 km; and the trajectory statistics of the lifted
tracker for every setting of scripts/paper/trajectory_analysis.py (all settings
for the reanalysis and realisations 0-4, the reference setting for the rest).

The nesting level is computed in the full field and kept: masking removes
features, it does not redefine the hierarchy.  Orography: R1, 2.5 degrees,
nearest grid point (data/processed_data/experiments/beyond_amplitude/hgt.sfc.nc).
scripts/paper/make_terrain_outputs.py applies the decision rule and writes the table.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from analyse import SIGN, W, depth_area_slope, load, nesting_level
from common import DATA, MAIN_PERIOD, OUT, REAL, ROOT
from lagged_coherence import LAGS, coherence

sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'paper'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from tracking import pix2ll  # noqa: E402
from trajectory_analysis import CONFIGS  # noqa: E402
import lifted_tracker as lt  # noqa: E402

LIMIT = 1500.0
TAUS = (5, 10, 20, 30)
TRACK_KEYS = ('mean_length', 'pct_1d', 'pct_ge5d', 'pct_ge10d', 'median_step_km')
_H = None


def over_high_terrain(lat, lon):
    global _H
    if _H is None:
        import netCDF4 as nc
        _H = np.asarray(nc.Dataset(os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'beyond_amplitude',
                                                'hgt.sfc.nc'))['hgt'][0].filled(np.nan), float)
    i = np.clip(np.round((90 - lat) / 2.5).astype(int), 0, 72)
    j = np.round((lon % 360) / 2.5).astype(int) % 144
    return _H[i, j] > LIMIT


def subset(F, k):
    n = len(k)
    return {key: (v[k] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n else v) for key, v in F.items()}


def stats(F, labels, ref_n10=None):
    d, T = F['depth'], F['T']
    S = {f'N{t}': float((d >= t).sum() / T) for t in TAUS}
    S['TP2s'] = float((d ** 2).sum() / T)
    k = d >= 10
    S['half_med_10'] = float(np.median(F['half_km2'][k]) / 1e6)
    S['nest_frac_10'] = float(np.mean(F['parent_idx'][k] >= 0))
    S['nest_ge2_10'] = float(np.mean(F['lev'][k] >= 2))
    S['slope_half'] = depth_area_slope(d[~F['pole']], F['half_km2'][~F['pole']])[0]
    n10 = S['N10'] if ref_n10 is None else ref_n10               # footprint at the count of the reanalysis
    tau = float(np.sort(d)[::-1][int(round(n10 * T)) - 1])
    S['mtau'], S['mhalf_med'] = tau, float(np.median(F['half_km2'][d >= tau]) / 1e6)
    S['coh500'] = coherence(F)[500]
    o = np.lexsort((F['peak_idx'], F['slot']))
    lat, lon = pix2ll(*np.divmod(F['peak_idx'][o], W))
    day = (F['slot'][o] - F['slot0']).astype(int)
    birth, death = F['saddle'][o].astype(float) / 100, F['peak'][o].astype(float) / 100
    S['tracks'] = {}
    for lab in labels:
        tracks, _, links = lt.track(day, birth, death, lat, lon, int(T), **CONFIGS[lab])
        s = lt.summary(tracks, links)
        S['tracks'][lab] = {key: s[key] for key in TRACK_KEYS}
        S['tracks'][lab]['tracks_per_year'] = s['n_tracks'] / (T / 365)
    return S


def one(job):
    name, ref = job
    r = None if name == 'real' else int(name.split('_r')[-1])
    labels = list(CONFIGS) if (r is None or r < 5) else ['reference']
    out = {'full': {}, 'low': {}}
    for tag in ('sup', 'sub'):
        c = SIGN[tag]
        F = load(REAL, tag, MAIN_PERIOD) if name == 'real' else load(name, tag)
        F['lev'] = nesting_level(F)
        lat, lon = pix2ll(*np.divmod(F['peak_idx'], W))
        high = over_high_terrain(lat, lon)
        out['full'][c] = stats(F, labels, None if ref is None else ref['full'][c])
        out['low'][c] = stats(subset(F, ~high), labels, None if ref is None else ref['low'][c])
        out['full'][c]['share_high_10'] = float(np.mean(high[F['depth'] >= 10]))
    print('done', name, flush=True)
    return name, out


def main():
    y0, y1 = MAIN_PERIOD
    names = sorted({os.path.basename(p)[5:-8] for p in glob.glob(os.path.join(DATA, f'feat_*_{y0}_{y1}_r*_sup.npz'))})
    names = [n for n in names if n.startswith(('ft_', 'iaaft_')) and os.path.exists(os.path.join(DATA, f'diag_{n}.json'))]
    res = dict([one(('real', None))])
    ref = {v: {c: res['real'][v][c]['N10'] for c in 'CA'} for v in ('full', 'low')}
    with Pool(int(os.environ.get('GN_WORKERS', 6))) as pool:
        res.update(dict(pool.imap_unordered(one, [(n, ref) for n in names])))
    res['_meta'] = dict(lags=LAGS, limit_m=LIMIT, configs=list(CONFIGS), period=[y0, y1])
    json.dump(res, open(os.path.join(OUT, 'terrain_check.json'), 'w'), indent=1, default=float)
    print('wrote terrain_check.json', len(names), 'surrogates')


if __name__ == '__main__':
    main()
