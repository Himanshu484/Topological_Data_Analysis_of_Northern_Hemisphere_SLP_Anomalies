"""
Metrics of the 6-hourly ERA5 experiment for every feature file present:
ERA5 at 0.25 and 1.5 degrees with its surrogates at 1.5 degrees, and NCEP-NCAR R1
at 2.5 degrees ('r1') with its surrogates.

    python analyse6h.py           # writes output/experiments/era5_6h/metrics_6h.json

Per data set and sign (C = 1-cyclones / lows, A = 1-anticyclones / highs):
  counts    N_tau per field for tau = 2 ... 40 hPa (annual, DJF, JJA), all extrema, TP1, TP2, largest depth
  geometry  median half-depth footprint of features >= 10 hPa (also at the count-matched threshold), nesting
  coherence lagged coherence c_k within 250 and 500 km for lags of 6 h ... 7 d, and at one year (chance)
  terrain   features per field over terrain above and below 1500 m
  low       everything over terrain above 1500 m removed: counts, coherence, tracks (terrain control)
  matched   (surrogates) coherence and tracks again with both thresholds moved to match the reanalysis counts
  tracks    lifted Wasserstein tracker (src/lifted_tracker.py), thresholds 5/10 hPa, lambda 0.01 hPa/km,
            for linking radii of 150 ... 1000 km per 6-hour step

A track of n fields is counted as lasting n/4 days, so that a feature seen in
every field of one day lasts one day, as in the daily analysis.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

import grid as G
import lifted_tracker as lt

TAUS = [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40]
LAGS = [1, 2, 4, 8, 12, 20, 28, G.SPY]
RADII = [250, 500]
TRACK_R = [150, 250, 375, 500, 750, 1000]
SEASONS = {'ANN': list(range(12)), 'DJF': [11, 0, 1], 'JJA': [5, 6, 7]}


def load(name, res, sign):
    Z = np.load(os.path.join(G.DATA, f'feat_{name}_{res}_{sign}.npz'))
    c = G.GRIDS[res]['c']
    W = 2 * c + 1
    o = np.lexsort((Z['peak_idx'], Z['slot']))
    F = {k: Z[k][o] for k in ('slot', 'peak', 'saddle', 'peak_idx', 'parent_idx', 'half_km2')}
    F['depth'] = (F['peak'].astype(np.float64) - F['saddle']) / 100
    F['lat'], F['lon'] = G.pix2ll(F['peak_idx'] // W, F['peak_idx'] % W, c)
    F['agg'] = Z['agg'].astype(np.float64)
    F['W'] = W
    return F


def nesting_level(F):
    W2 = F['W'] ** 2
    key = F['slot'].astype(np.int64) * W2 + F['peak_idx']
    o = np.argsort(key)
    ks = key[o]
    pk = np.where(F['parent_idx'] >= 0, F['slot'].astype(np.int64) * W2 + F['parent_idx'], -1)
    pos = np.clip(np.searchsorted(ks, pk), 0, len(ks) - 1)
    prow = np.where((pk >= 0) & (ks[pos] == pk), o[pos], -1)
    assert ((pk < 0) | (prow >= 0)).all(), 'a parent is missing from the stored features'
    lev = np.zeros(len(key), int)
    cur = prow.copy()
    while (cur >= 0).any():
        m = cur >= 0
        lev[m] += 1
        cur[m] = prow[cur[m]]
    return lev


def over_high_terrain(F, limit=1500.0):
    """True for features whose extremum lies over terrain above `limit` metres (R1 2.5 degree orography, nearest point)."""
    import netCDF4 as nc
    h = np.asarray(nc.Dataset(os.path.join(G.ROOT, 'data', 'processed_data', 'experiments', 'beyond_amplitude',
                                           'hgt.sfc.nc'))['hgt'][0].filled(np.nan), float)
    i = np.clip(np.round((90 - F['lat']) / 2.5).astype(int), 0, 72)
    j = np.round((F['lon'] % 360) / 2.5).astype(int) % 144
    return h[i, j] > limit


def terrain(name, res, sign, limit=1500.0):
    """Features per field over terrain above and below `limit` metres."""
    F = load(name, res, sign)
    high = over_high_terrain(F, limit)
    out = {}
    for tau in (5, 10, 20):
        k = F['depth'] >= tau
        out[f'N{tau}_high'] = float((k & high).sum() / G.T)
        out[f'N{tau}_low'] = float((k & ~high).sum() / G.T)
    k = (F['depth'] >= 10) & high                             # diurnal dependence over high terrain: per field, by synoptic hour
    out['N10_high_by_hour'] = [float((k & (F['slot'] % G.SPD == h)).sum() / (G.T / G.SPD)) for h in range(G.SPD)]
    return out


def coherence(F, tau_hi=10, tau_lo=5, lags=None):
    hi, lo = F['depth'] >= tau_hi, F['depth'] >= tau_lo
    sh, sl = F['slot'][hi], F['slot'][lo]
    lah, loh, lal, lol = F['lat'][hi], F['lon'][hi], F['lat'][lo], F['lon'][lo]
    t = np.arange(G.T)
    a0, a1 = np.searchsorted(sh, t), np.searchsorted(sh, t, 'right')
    b0, b1 = np.searchsorted(sl, t), np.searchsorted(sl, t, 'right')
    out = {r: [] for r in RADII}
    for k in (lags or LAGS):
        hit, n = {r: 0 for r in RADII}, 0
        for s in range(G.T - k):
            if a1[s] == a0[s]:
                continue
            n += a1[s] - a0[s]
            if b1[s + k] == b0[s + k]:
                continue
            D = lt.great_circle(lah[a0[s]:a1[s]], loh[a0[s]:a1[s]], lal[b0[s + k]:b1[s + k]], lol[b0[s + k]:b1[s + k]]).min(1)
            for r in RADII:
                hit[r] += int((D <= r).sum())
        for r in RADII:
            out[r].append(hit[r] / n)
    return {str(r): v for r, v in out.items()}


def tracks(F, R, tau_lo=5.0, tau_hi=10.0, lam=0.01):
    T_, keep, links = lt.track(F['slot'], F['saddle'] / 100.0, F['peak'] / 100.0, F['lat'], F['lon'], G.T,
                               tau_lo=tau_lo, tau_hi=tau_hi, R=float(R), lam=lam)
    n = np.array([len(m) for m in T_])
    step = links['step'][~np.isnan(links['step'])]
    return dict(per_year=len(n) / G.NY, pct_single_step=float(100 * np.mean(n == 1)), pct_lt1d=float(100 * np.mean(n < 4)),
                mean_days=float(n.mean() / 4), pct_ge5d=float(100 * np.mean(n >= 20)), pct_ge10d=float(100 * np.mean(n >= 40)),
                max_days=float(n.max() / 4), median_step_km=float(np.median(step)), mean_step_km=float(step.mean()),
                pct_steps_over_0p8R=float(100 * np.mean(step > 0.8 * R)),
                survival_days=[float(np.mean(n >= 4 * d)) for d in range(1, 31)])


def metrics(name, res, sign, ref_n10=None):
    F = load(name, res, sign)
    d = F['depth']
    month = G.MONTH_OF_DOY[(F['slot'] % G.SPY) // G.SPD]
    fmonth = G.MONTH_OF_DOY[(np.arange(G.T) % G.SPY) // G.SPD]
    M = {}
    for s, ms in SEASONS.items():
        nf = np.isin(fmonth, ms).sum()
        ks = np.isin(month, ms)
        for tau in TAUS:
            M[f'N{tau:g}_{s}'] = float(((d >= tau) & ks).sum() / nf)
    agg = F['agg']
    M['Nall'] = float(agg[:, 0].mean())
    M['TP1'] = float(agg[:, 1].mean() / 100)
    M['TP2'] = float(agg[:, 2].mean() / 1e4)
    M['maxdepth'] = float(agg[:, 3].mean() / 100)
    k = d >= 10
    M['half_med_10'] = float(np.median(F['half_km2'][k]) / 1e6)
    M['half_req_km_10'] = float(np.sqrt(np.median(F['half_km2'][k]) / np.pi))
    lev = nesting_level(F)
    M['nest_frac_10'] = float(np.mean(lev[k] >= 1))
    M['nest_ge2_10'] = float(np.mean(lev[k] >= 2))
    M['depth_over_amp_median_10'] = float(np.median(d[k] * 100 / np.abs(F['peak'][k])))
    if ref_n10 is not None:                                  # count-matched threshold for the surrogates
        kk = int(round(ref_n10 * G.T))
        tau = float(np.sort(d)[::-1][kk - 1])
        M['mtau'] = tau
        M['mhalf_med'] = float(np.median(F['half_km2'][d >= tau]) / 1e6)
    else:
        M['mhalf_med'] = M['half_med_10']
    M['coherence'] = coherence(F)
    M['tracks'] = {str(R): tracks(F, R) for R in TRACK_R}
    return M


def matched(name, res, sign, ref):
    """Count-matched control: the surrogate's two thresholds are moved so that it has as many features per
    field as the reanalysis has above 10 and above 5 hPa; coherence and trajectories are then recomputed."""
    F = load(name, res, sign)
    ds = np.sort(F['depth'])[::-1]
    th = {t: float(ds[int(round(ref[sign][t] * G.T)) - 1]) for t in ('10', '5')}
    return dict(tau_hi=th['10'], tau_lo=th['5'], coherence=coherence(F, th['10'], th['5']),
                tracks={str(R): tracks(F, R, tau_lo=th['5'], tau_hi=th['10']) for R in TRACK_R})


def low_terrain(name, res, sign):
    """Terrain control: every feature over terrain above 1500 m is removed, then counts, coherence and trajectories
    are recomputed.  Sea-level pressure there is an extrapolation below ground, and in six-hourly R1 it has a
    diurnal artefact (highs over high terrain are twice as frequent at 00 UTC as at 06 UTC and half of them are
    gone six hours later)."""
    F = load(name, res, sign)
    k = ~over_high_terrain(F)
    F = {key: (v[k] if isinstance(v, np.ndarray) and v.shape[:1] == k.shape else v) for key, v in F.items() if key != 'agg'}
    out = {f'N{tau}': float((F['depth'] >= tau).sum() / G.T) for tau in (5, 10, 20)}
    out['half_med_10'] = float(np.median(F['half_km2'][F['depth'] >= 10]) / 1e6)
    out['coherence'] = coherence(F)
    out['tracks'] = {str(R): tracks(F, R) for R in TRACK_R}
    return out


def one_low(job):
    name, res = job
    out = {s: low_terrain(name, res, s) for s in 'CA'}
    print('done low terrain', name, res, flush=True)
    return f'{name}_{res}', out


def one_tracks(job):
    """Recompute only the trajectory statistics (used when a statistic is added to tracks())."""
    name, res, th = job
    out = {}
    for s in 'CA':
        F = load(name, res, s)
        out[s] = {str(R): tracks(F, R) for R in TRACK_R}
        if th is not None:
            out[s + '_m'] = {str(R): tracks(F, R, tau_lo=th[s][1], tau_hi=th[s][0]) for R in TRACK_R}
    print('done tracks', name, res, flush=True)
    return f'{name}_{res}', out


def one_matched(job):
    name, res, ref = job
    out = {s: matched(name, res, s, ref) for s in 'CA'}
    print('done matched', name, res, flush=True)
    return f'{name}_{res}', out


def one(job):
    name, res, ref = job
    out = {s: metrics(name, res, s, None if ref is None else ref[s]) for s in 'CA'}
    print('done', name, res, flush=True)
    return f'{name}_{res}', out


def main():
    res = {}
    files = sorted(glob.glob(os.path.join(G.DATA, 'feat_*_C.npz')))
    sets = [os.path.basename(f)[5:-6].rsplit('_', 1) for f in files]          # (name, res)
    reals = [(n, r) for n, r in sets if n == 'real']
    out_path = os.path.join(G.OUT, 'metrics_6h.json')
    old = json.load(open(out_path)) if os.path.exists(out_path) else {}
    with Pool(int(os.environ.get('ERA6H_WORKERS', 4))) as pool:
        for key, out in pool.imap_unordered(one, [(n, r, None) for n, r in reals if f'{n}_{r}' not in old]):
            old[key] = out
        ref = {r: {s: old[f'real_{r}'][s]['N10_ANN'] for s in 'CA'} for _, r in reals}       # each null against its own reanalysis
        todo = [(n, r, ref[r]) for n, r in sets if n != 'real' and f'{n}_{r}' not in old
                and os.path.exists(os.path.join(G.DATA, f'diag_{n}_{r}.json'))]
        for key, out in pool.imap_unordered(one, todo):
            old[key] = out
        refm = {r: {s: {'10': old[f'real_{r}'][s]['N10_ANN'], '5': old[f'real_{r}'][s]['N5_ANN']} for s in 'CA'} for _, r in reals}
        todo = [(n, r, refm[r]) for n, r in sets if n != 'real' and f'{n}_{r}' in old and 'matched' not in old[f'{n}_{r}']['C']]
        for key, out in pool.imap_unordered(one_matched, todo):
            for s in 'CA':
                old[key][s]['matched'] = out[s]
    todo = [(n, r) for n, r in sets if f'{n}_{r}' in old and 'low' not in old[f'{n}_{r}']['C']]
    if todo:
        with Pool(int(os.environ.get('ERA6H_WORKERS', 4))) as pool:
            for key, out in pool.imap_unordered(one_low, todo):
                for s in 'CA':
                    old[key][s]['low'] = out[s]
    stale = [(n, r, {s: (old[f'{n}_{r}'][s]['matched']['tau_hi'], old[f'{n}_{r}'][s]['matched']['tau_lo']) for s in 'CA'} if n != 'real' else None)
             for n, r in sets if f'{n}_{r}' in old and 'mean_step_km' not in old[f'{n}_{r}']['C']['tracks'][str(TRACK_R[0])]]
    if stale:
        with Pool(int(os.environ.get('ERA6H_WORKERS', 4))) as pool:
            for key, out in pool.imap_unordered(one_tracks, stale):
                for s in 'CA':
                    old[key][s]['tracks'] = out[s]
                    if s + '_m' in out:
                        old[key][s]['matched']['tracks'] = out[s + '_m']
    for n, r in sets:
        for s in 'CA':
            if f'{n}_{r}' in old and 'N10_high_by_hour' not in old[f'{n}_{r}'][s].get('terrain', {}):
                old[f'{n}_{r}'][s]['terrain'] = terrain(n, r, s)
    res = old
    res['_meta'] = dict(lags_steps=LAGS, radii_km=RADII, track_R_km=TRACK_R, taus=TAUS)
    json.dump(res, open(out_path, 'w'), indent=1)
    print('wrote', out_path, sorted(k for k in res if not k.startswith('_')))


if __name__ == '__main__':
    main()
