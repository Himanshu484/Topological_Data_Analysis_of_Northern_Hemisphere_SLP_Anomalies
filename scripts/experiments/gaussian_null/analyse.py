"""
Real vs Gaussian (ft) vs amplitude-adjusted (iaaft) surrogates: all metrics,
the verdict table, and figures.

    python analyse.py            # writes output/experiments/gaussian_null/{metrics.json, tables.md, fig_*.png}
"""
import glob
import json
import os
import sys

import numpy as np

from common import DATA, OUT, START_YEAR, ROOT, REAL, MAIN_PERIOD, PERIODS as PERIOD_LIST

sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
from tracking import pix2ll, hav, run as nn_run, tracks as nn_tracks  # noqa: E402

W = 172
TAUS = [2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40]
SEASONS = {'ANN': list(range(12)), 'DJF': [11, 0, 1], 'MAM': [2, 3, 4], 'JJA': [5, 6, 7], 'SON': [8, 9, 10]}
MONTH_OF_DOY = np.repeat(np.arange(12), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
SIGN = {'sup': 'C', 'sub': 'A'}
PERIODS = {str(p[0])[2:]: p for p in PERIOD_LIST}
MAIN = f'real_{MAIN_PERIOD[0]}_{MAIN_PERIOD[1]}'


def load(name, tag, period=None):
    Z = np.load(os.path.join(DATA, f'feat_{name}_{tag}.npz'))
    slot0, T = int(Z['slot0']), int(Z['nslots'])
    F = {k: Z[k] for k in ('slot', 'peak', 'saddle', 'peak_idx', 'parent_idx', 'basin_km2', 'half_km2', 'pole')}
    agg = Z['agg'].astype(np.float64)
    if period is not None:
        s0, s1 = (period[0] - START_YEAR) * 365, (period[1] - START_YEAR + 1) * 365
        k = (F['slot'] >= s0) & (F['slot'] < s1)
        F = {kk: v[k] for kk, v in F.items()}
        agg = agg[s0 - slot0:s1 - slot0]
        slot0, T = s0, s1 - s0
    F['depth'] = (F['peak'].astype(np.float64) - F['saddle']) / 100
    F['slot0'], F['T'], F['agg'] = slot0, T, agg
    F['month'] = MONTH_OF_DOY[F['slot'] % 365]
    return F


def nesting_level(F):
    """0 = parent is the rim, 1 = parent's parent is the rim, ..."""
    key = F['slot'].astype(np.int64) * (W * W) + F['peak_idx']
    o = np.argsort(key)
    ks = key[o]
    pk = np.where(F['parent_idx'] >= 0, F['slot'].astype(np.int64) * (W * W) + F['parent_idx'], -1)
    pos = np.searchsorted(ks, pk)
    pos = np.clip(pos, 0, len(ks) - 1)
    prow = np.where((pk >= 0) & (ks[pos] == pk), o[pos], -1)
    assert ((pk < 0) | (prow >= 0)).all(), 'a parent is missing from the stored features'
    lev = np.zeros(len(key), int)
    cur = prow.copy()
    while (cur >= 0).any():
        m = cur >= 0
        lev[m] += 1
        cur[m] = prow[cur[m]]
    return lev


def next_day_coherence(F, tau_hi=10, tau_lo=5, R=(500, 1000)):
    hi = F['depth'] >= tau_hi
    lo = F['depth'] >= tau_lo
    lat, lon = pix2ll(*np.divmod(F['peak_idx'], W))
    s = F['slot'] - F['slot0']
    T = F['T']
    sh, sl = s[hi], s[lo]
    lah, loh, lal, lol = lat[hi], lon[hi], lat[lo], lon[lo]
    a0, a1 = np.searchsorted(sh, np.arange(T)), np.searchsorted(sh, np.arange(T), 'right')
    b0, b1 = np.searchsorted(sl, np.arange(T)), np.searchsorted(sl, np.arange(T), 'right')
    hit = {r: 0 for r in R}
    n = 0
    for t in range(T - 1):
        if a1[t] == a0[t]:
            continue
        n += a1[t] - a0[t]
        if b1[t + 1] == b0[t + 1]:
            continue
        D = hav(lah[a0[t]:a1[t]], loh[a0[t]:a1[t]], lal[b0[t + 1]:b1[t + 1]], lol[b0[t + 1]:b1[t + 1]])
        dm = D.min(1)
        for r in R:
            hit[r] += int((dm <= r).sum())
    return {r: hit[r] / n for r in R}


def track_stats(F, lo=5.0, hi=10.0, R=1000.0):
    k = F['depth'] >= lo
    G = {'slot': F['slot'][k] - F['slot0'], 'depth': F['depth'][k]}
    o = np.lexsort((F['peak_idx'][k], G['slot']))
    G = {kk: v[o] for kk, v in G.items()}
    G['plat'], G['plon'] = pix2ll(*np.divmod(F['peak_idx'][k][o], W))
    G['birth'] = np.zeros(len(o)); G['death'] = G['depth']
    G['nslots'] = F['T']
    G['starts'] = np.searchsorted(G['slot'], np.arange(G['nslots']))
    G['ends'] = np.searchsorted(G['slot'], np.arange(G['nslots']), 'right')
    res = nn_run(G, 'nn', R=R)
    T = [m for m in nn_tracks(res) if (G['depth'][m] >= hi).any()]
    L = np.array([len(m) for m in T])
    step = res['step'][~np.isnan(res['step'])]
    return dict(n_per_yr=len(L) / (F['T'] / 365), meanL=L.mean(), pct1=100 * np.mean(L == 1),
                pct_ge5=100 * np.mean(L >= 5), pct_ge10=100 * np.mean(L >= 10), maxL=int(L.max()),
                surv=[float(np.mean(L >= j)) for j in range(1, 31)], median_step_km=float(np.median(step)))


def depth_area_slope(depth, area, lo=5, hi=40):
    k = (depth >= lo) & (depth <= hi) & (area > 0)
    edges = np.logspace(np.log10(lo), np.log10(hi), 9)
    xb, yb = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = k & (depth >= a) & (depth < b)
        if m.sum() >= 30:
            xb.append(np.log10(np.median(depth[m]))); yb.append(np.log10(np.median(area[m])))
    slope = np.polyfit(xb, yb, 1)[0] if len(xb) >= 3 else np.nan
    return float(slope), [10 ** x for x in xb], [10 ** y / 1e6 for y in yb]


def matched(F, tag, n_per_day):
    """Track and coherence stats at the threshold giving the real N(>=10 hPa) per day."""
    c = SIGN[tag]
    k = int(round(n_per_day * F['T']))
    tau = float(np.sort(F['depth'])[::-1][k - 1])
    M = {f'mtau_{c}': tau}
    coh = next_day_coherence(F, tau_hi=tau, tau_lo=tau / 2)
    M[f'mcoh500_{c}'], M[f'mcoh1000_{c}'] = coh[500], coh[1000]
    ts = track_stats(F, lo=tau / 2, hi=tau)
    for kk in ('n_per_yr', 'meanL', 'pct1', 'pct_ge5', 'pct_ge10', 'median_step_km'):
        M[f'mtrk_{kk}_{c}'] = ts[kk]
    M[f'mtrk_surv_{c}'] = ts['surv']
    # half-depth area of the matched top-N features
    kk = F['depth'] >= tau
    M[f'mhalf_med_{c}'] = float(np.median(F['half_km2'][kk]) / 1e6)
    return M


def metrics(F, tag, with_tracks=True):
    c = SIGN[tag]
    M, T, d = {}, F['T'], F['depth']
    for sname, ms in SEASONS.items():
        ks = np.isin(F['month'], ms)
        ndays = T * len(ms) / 12 if sname != 'ANN' else T
        ndays = np.isin(MONTH_OF_DOY[np.arange(F['slot0'], F['slot0'] + T) % 365], ms).sum()
        for tau in TAUS:
            M[f'N{tau:g}_{c}_{sname}'] = float(((d >= tau) & ks).sum() / ndays)
    agg = F['agg']
    M[f'Nall_{c}'] = float(agg[:, 0].mean())
    M[f'TP1_{c}'] = float(agg[:, 1].mean() / 100)
    M[f'TP2_{c}'] = float(agg[:, 2].mean() / 1e4)
    M[f'maxdepth_{c}'] = float(agg[:, 3].mean() / 100)
    for tau in (5, 10):
        k = d >= tau
        np_ = ~F['pole']
        M[f'basin_med_{c}_{tau}'] = float(np.median(F['basin_km2'][k]) / 1e6)
        M[f'basin_med_nopole_{c}_{tau}'] = float(np.median(F['basin_km2'][k & np_]) / 1e6)
        M[f'basin_mean_{c}_{tau}'] = float(np.mean(F['basin_km2'][k]) / 1e6)
        M[f'pole_frac_{c}_{tau}'] = float(np.mean(F['pole'][k]))
        M[f'half_med_{c}_{tau}'] = float(np.median(F['half_km2'][k]) / 1e6)
        M[f'half_req_km_{c}_{tau}'] = float(np.sqrt(np.median(F['half_km2'][k]) / np.pi))
        M[f'nest_frac_{c}_{tau}'] = float(np.mean(F['parent_idx'][k] >= 0))
    lev = nesting_level(F)
    k = d >= 10
    M[f'nest_mean_{c}_10'] = float(lev[k].mean())
    M[f'nest_ge2_{c}_10'] = float(np.mean(lev[k] >= 2))
    M[f'nest_hist_{c}_10'] = [float(np.mean(lev[k] == j)) for j in range(6)]
    # basins of nested features only (their saddle is not the ~0 hPa rim contour)
    kn = k & (F['parent_idx'] >= 0)
    M[f'basin_med_nested_{c}_10'] = float(np.median(F['basin_km2'][kn]) / 1e6)
    M[f'basin_med_rim_{c}_10'] = float(np.median(F['basin_km2'][k & (F['parent_idx'] < 0)]) / 1e6)
    np_ = ~F['pole']
    for nm, ar in (('half', F['half_km2']), ('basin', F['basin_km2'])):
        s, xb, yb = depth_area_slope(d[np_], ar[np_])
        M[f'slope_{nm}_{c}'] = s
        M[f'curve_{nm}_{c}'] = [xb, yb]
    # size by depth bin (half-depth), for the scaling table
    for lo_, hi_ in ((5, 10), (10, 20), (20, 40)):
        m = (d >= lo_) & (d < hi_) & np_
        M[f'half_med_{c}_{lo_}_{hi_}'] = float(np.median(F['half_km2'][m]) / 1e6)
    coh = next_day_coherence(F)
    M[f'coh500_{c}'], M[f'coh1000_{c}'] = coh[500], coh[1000]
    if with_tracks:
        ts = track_stats(F)
        for kk, v in ts.items():
            M[f'trk_{kk}_{c}'] = v
    return M


def annual(F, col, scale):
    v = F['agg'][:, col] / scale
    return v.reshape(-1, 365).mean(1)


def one_dataset(nm, ref_n10):
    """Metrics of one surrogate realisation, cached on the feature files' mtimes."""
    cache = os.path.join(OUT, 'cache', nm + '.json')
    stamp = [os.path.getmtime(os.path.join(DATA, f'feat_{nm}_{t}.npz')) for t in ('sup', 'sub')]
    if os.path.exists(cache):
        C = json.load(open(cache))
        if C['stamp'] == stamp and C['ref'] == ref_n10:
            return nm, C['metrics']
    M = {}
    for tag in ('sup', 'sub'):
        F = load(nm, tag)
        M.update(metrics(F, tag))
        M.update(matched(F, tag, ref_n10[SIGN[tag]]))
    json.dump({'stamp': stamp, 'ref': ref_n10, 'metrics': M}, open(cache, 'w'), default=float)
    return nm, M


def main():
    from multiprocessing import Pool
    os.makedirs(os.path.join(OUT, 'cache'), exist_ok=True)
    names = sorted({os.path.basename(p)[5:-8] for p in glob.glob(os.path.join(DATA, 'feat_*_sup.npz'))})
    # a surrogate is complete once its diag json exists (written after both feature files)
    names = [n for n in names if n.startswith(('real', 'hyb')) or os.path.exists(os.path.join(DATA, f'diag_{n}.json'))]
    print('datasets:', names)
    res = {}
    for pk, per in PERIODS.items():
        key = f'real_{per[0]}_{per[1]}'
        res[key] = {}
        for tag in ('sup', 'sub'):
            F = load(REAL, tag, per)
            res[key].update(metrics(F, tag))
            res[key].update(matched(F, tag, res[key][f'N10_{SIGN[tag]}_ANN']))
    jobs = []
    for nm in names:
        if nm == REAL:
            continue
        ref = res['real_1948_1978' if '_1948_' in nm else MAIN]
        jobs.append((nm, {c: ref[f'N10_{c}_ANN'] for c in ('C', 'A')}))
    with Pool(int(os.environ.get('GN_WORKERS', 3))) as pool:
        for nm, M in pool.starmap(one_dataset, jobs):
            res[nm] = M
            print('  done', nm, flush=True)
    # C1: annual series
    c1 = {}
    for nm in names:
        c1[nm] = {}
        for tag in ('sup', 'sub'):
            F = load(nm, tag)
            c1[nm][f'TP2_{SIGN[tag]}'] = annual(F, 2, 1e4).tolist()
            c1[nm][f'TP1_{SIGN[tag]}'] = annual(F, 1, 100).tolist()
            c1[nm][f'max_{SIGN[tag]}'] = annual(F, 3, 100).tolist()
            c1[nm][f'N_{SIGN[tag]}'] = annual(F, 0, 1).tolist()
            dd = (F['peak'].astype(np.float64) - F['saddle']) / 100
            n10 = np.bincount((F['slot'][dd >= 10] - F['slot0']) // 365, minlength=F['T'] // 365) / 365
            c1[nm][f'N10_{SIGN[tag]}'] = n10.tolist()
    json.dump({'metrics': res, 'annual': c1}, open(os.path.join(OUT, 'metrics.json'), 'w'), default=float)
    print('wrote metrics.json')


if __name__ == '__main__':
    main()
