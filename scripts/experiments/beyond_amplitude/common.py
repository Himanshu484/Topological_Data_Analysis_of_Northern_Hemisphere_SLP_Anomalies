"""
Shared loading, per-extremum variables, detectors and tracking for the
beyond-amplitude experiment.  Units: hPa, km.

Every row is one strict 8-neighbour extremum (one H1 feature of any depth) from
data/processed_data/prototype/feat_{sup,sub}_1948_2023.npz, so every detector
below selects rows of the same table and tracks from different detectors can
be compared row by row.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
from tracking import pix2ll, hav, run  # noqa: E402,F401

PROTO = os.path.join(ROOT, 'data', 'processed_data', 'prototype')
DATA = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'beyond_amplitude')
OUTD = os.path.join(ROOT, 'output', 'experiments', 'beyond_amplitude')
Y0, Y1 = 1948, 2023
NSLOT = (Y1 - Y0 + 1) * 365
W = 172
NAME = {'sup': 'cyclones', 'sub': 'anticyclones'}


def _orog():
    import netCDF4 as nc
    h = nc.Dataset(os.path.join(DATA, 'hgt.sfc.nc'))['hgt'][0].filled(np.nan)
    l = nc.Dataset(os.path.join(DATA, 'land.nc'))['land'][0].filled(np.nan)
    return np.asarray(h, float), np.asarray(l, float)


def _grid_lookup(A, lat, lon):
    i = np.clip(np.round((90 - lat) / 2.5).astype(int), 0, 72)
    j = np.round((lon % 360) / 2.5).astype(int) % 144
    return A[i, j]


def table(t):
    """All extrema of one type with every per-row variable (cached)."""
    path = os.path.join(DATA, f'table_{t}.npz')
    if os.path.exists(path):
        Z = np.load(path)
        return {k: Z[k] for k in Z.files}
    F = np.load(os.path.join(PROTO, f'feat_{t}_{Y0}_{Y1}.npz'))
    D = np.load(os.path.join(DATA, f'desc_{t}_{Y0}_{Y1}.npz'))
    T = {k: F[k].astype(np.int64) for k in ['year', 'day', 'prow', 'pcol', 'srow', 'scol', 'parrow', 'parcol']}
    T['slot'] = (T['year'] - Y0) * 365 + T['day'] - 1
    T['amp'] = F['death'] / 100                      # extremum anomaly, sign-flipped for cyclones
    T['sad'] = F['birth'] / 100                      # saddle value
    T['depth'] = T['amp'] - T['sad']
    for k in ['m1', 'm2', 'm3', 'dip_e', 'dip_n', 'ccd', 'fut1', 'fut2', 'past1']:
        T[k] = D[k].astype(np.float64) / 100
    T['ann'] = T['amp'] - T['m2']                    # annulus (500-1000 km) depth
    T['lap'] = T['amp'] - T['m1']                    # 0-500 km depth, a discrete Laplacian proxy
    T['ann3'] = T['amp'] - T['m3']
    T['plat'], T['plon'] = pix2ll(T['prow'], T['pcol'])
    T['slat'], T['slon'] = pix2ll(T['srow'], T['scol'])
    T['sepkm'] = _gc(T['plat'], T['plon'], T['slat'], T['slon'])
    T['rim'] = (T['parrow'] < 0).astype(np.int64)
    h, l = _orog()
    T['hgt'] = _grid_lookup(h, T['plat'], T['plon'])
    T['land'] = _grid_lookup(l, T['plat'], T['plon'])
    # parent row
    key = T['slot'] * W * W + T['prow'] * W + T['pcol']
    o = np.argsort(key)
    pk = T['slot'] * W * W + T['parrow'] * W + T['parcol']
    pos = np.clip(np.searchsorted(key[o], pk), 0, len(key) - 1)
    par = np.where((T['parrow'] >= 0) & (key[o][pos] == pk), o[pos], -1)
    T['par'] = par
    ok = par >= 0
    T['par_depth'] = np.where(ok, T['depth'][np.maximum(par, 0)], np.nan)
    T['par_amp'] = np.where(ok, T['amp'][np.maximum(par, 0)], np.nan)
    T['par_km'] = np.where(ok, _gc(T['plat'], T['plon'], T['plat'][np.maximum(par, 0)],
                                   T['plon'][np.maximum(par, 0)]), np.nan)
    # same-day neighbourhood variables
    n = len(key)
    T['n_shallow'] = np.zeros(n)        # extrema with 1 hPa <= depth < own depth within 1000 km (topological)
    T['n_ext'] = np.zeros(n)            # extrema with amp >= 5 hPa within 1000 km (non-topological)
    T['d_stronger'] = np.full(n, 5000.)  # km to nearest extremum with larger amp (capped)
    T['amp_stronger'] = np.full(n, np.nan)
    T['d_deeper'] = np.full(n, 5000.)   # km to nearest extremum with larger depth (capped)
    T['nms'] = np.zeros(n, np.int64)    # amp is the largest within 1000 km
    starts = np.searchsorted(T['slot'], np.arange(NSLOT))
    ends = np.searchsorted(T['slot'], np.arange(NSLOT), 'right')
    for s in range(NSLOT):
        a, b = starts[s], ends[s]
        if b <= a:
            continue
        Dm = hav(T['plat'][a:b], T['plon'][a:b], T['plat'][a:b], T['plon'][a:b])
        np.fill_diagonal(Dm, np.inf)
        am, dp = T['amp'][a:b], T['depth'][a:b]
        near = Dm <= 1000
        T['n_shallow'][a:b] = (near & (dp[None, :] >= 1) & (dp[None, :] < dp[:, None])).sum(1)
        T['n_ext'][a:b] = (near & (am[None, :] >= 5)).sum(1)
        S = np.where(am[None, :] > am[:, None], Dm, np.inf)
        j = S.argmin(1)
        ds = S[np.arange(b - a), j]
        T['d_stronger'][a:b] = np.minimum(ds, 5000)
        T['amp_stronger'][a:b] = np.where(np.isfinite(ds), am[j], np.nan)
        T['nms'][a:b] = ~(near & (am[None, :] > am[:, None])).any(1)
        S = np.where(dp[None, :] > dp[:, None], Dm, np.inf)
        T['d_deeper'][a:b] = np.minimum(S.min(1), 5000)
    np.savez(path, **T)
    return T


def _gc(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la1 - la2) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


gc = _gc


def bearing(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.arctan2(np.sin(lo2 - lo1) * np.cos(la2),
                      np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1))


# ---------------------------------------------------------------- detectors
def scores(T):
    """Detector scores (larger = more detectable); -inf where undefined."""
    ccd = np.where(np.isnan(T['ccd']), -np.inf, T['ccd'])
    return {
        'P  prominence (topological depth)': T['depth'],
        'A  amplitude |extremum|': T['amp'],
        'AN amplitude + 1000 km suppression': np.where(T['nms'] == 1, T['amp'], -np.inf),
        'PN prominence + 1000 km suppression': np.where(T['d_deeper'] > 1000, T['depth'], -np.inf),
        'L  annulus depth (500-1000 km ring)': T['ann'],
        'C  closed-contour depth (R <= 1200 km)': ccd,
    }


def calibrate(score, n):
    """Threshold giving n detections (the n-th largest score)."""
    s = np.sort(score[np.isfinite(score)])[::-1]
    return s[n - 1]


# ---------------------------------------------------------------- tracking
def subF(T, rows):
    rows = np.sort(rows)
    F = {'slot': T['slot'][rows], 'plat': T['plat'][rows], 'plon': T['plon'][rows],
         'birth': T['sad'][rows], 'death': T['amp'][rows], 'nslots': NSLOT, 'rows': rows}
    F['starts'] = np.searchsorted(F['slot'], np.arange(NSLOT))
    F['ends'] = np.searchsorted(F['slot'], np.arange(NSLOT), 'right')
    return F


def nn_track(T, lo_mask, hi_mask=None, R=1000.0):
    """NN tracks on rows in lo_mask; keep tracks with a hi_mask row.  Returns global arrays."""
    F = subF(T, np.where(lo_mask)[0])
    r = run(F, 'nn', R=R)
    rows = F['rows']
    n = len(T['slot'])
    succ = -np.ones(n, np.int64)
    pred = -np.ones(n, np.int64)
    k = r['succ'] >= 0
    succ[rows[k]] = rows[r['succ'][k]]
    k = r['pred'] >= 0
    pred[rows[k]] = rows[r['pred'][k]]
    tid = -np.ones(n, np.int64)
    age = np.full(n, -1, np.int64)
    L = np.zeros(n, np.int64)
    trk = []
    for h in rows[pred[rows] < 0]:
        m = [h]
        while succ[m[-1]] >= 0:
            m.append(succ[m[-1]])
        m = np.array(m)
        if hi_mask is not None and not hi_mask[m].any():
            succ[m] = -1
            pred[m] = -1
            continue
        tid[m] = len(trk)
        age[m] = np.arange(len(m))
        L[m] = len(m)
        trk.append(m)
    return dict(succ=succ, pred=pred, tid=tid, age=age, L=L, tracks=trk)


def chance_links(T, mask, R=1000.0, lag_years=5):
    """Links obtained by NN-linking day t to day t+1 of a year `lag_years` later (chance level)."""
    F = subF(T, np.where(mask)[0])
    from scipy.optimize import linear_sum_assignment
    nl = 0
    for s in range(NSLOT - 1):
        a, b = F['starts'][s], F['ends'][s]
        s2 = (s + 1 + lag_years * 365) % NSLOT
        c, d = F['starts'][s2], F['ends'][s2]
        if b == a or d == c:
            continue
        D = hav(F['plat'][a:b], F['plon'][a:b], F['plat'][c:d], F['plon'][c:d])
        ri, ci = linear_sum_assignment(np.where(D <= R, D, 1e9))
        nl += int((D[ri, ci] <= R).sum())
    return nl


def track_metrics(T, tr, det_mask=None):
    """Coherence of tracks; det_mask = the high-level detections (fractions are of those)."""
    L = np.array([len(m) for m in tr['tracks']])
    if det_mask is None:
        det_mask = tr['tid'] >= 0
    dm = det_mask & (tr['tid'] >= 0)
    i = np.where(tr['succ'] >= 0)[0]
    step = gc(T['plat'][i], T['plon'][i], T['plat'][tr['succ'][i]], T['plon'][tr['succ'][i]])
    return dict(tracks=len(L), pct1=100 * np.mean(L == 1), meanL=L.mean(),
                ge3=100 * np.mean(tr['L'][dm] >= 3), ge5=int((L >= 5).sum()),
                step_med=float(np.median(step)) if len(step) else np.nan,
                linked=100 * np.mean((tr['succ'][dm] >= 0) | (tr['pred'][dm] >= 0)))
