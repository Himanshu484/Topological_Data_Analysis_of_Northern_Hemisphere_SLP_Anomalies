"""
Per-extremum field descriptors that need the discs (one pass over all days).

For every row of data/processed_data/prototype/feat_{sup,sub}_1948_2023.npz
(every strict 8-neighbour extremum), in the sign convention of that file
(g = f for `sub` / anticyclones, g = -f for `sup` / cyclones; units Pa):

    m1, m2, m3    mean of g over 0-500 km (centre excluded), 500-1000 km and
                  1000-1500 km rings around the extremum pixel
    dip_e, dip_n  first azimuthal moment of g on the 500-1000 km ring
                  (mean of (g - m2) * unit bearing), east and north components
    ccd           closed-contour depth limited to a 1200 km radius: v - max(s, W),
                  W = the highest level at which the region {g >= L} containing
                  the extremum reaches beyond 1200 km (or the rim).  Only for
                  topological depth >= 2 hPa (ccd <= depth), NaN otherwise.
    fut1, fut2    max of g within 800 km of the extremum pixel on day t+1, t+2
    past1         the same on day t-1
    (NaN when the day does not exist)

Only `sub` discs are read, since sup = -sub exactly.

    python descriptors.py [--jobs 2]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
from tracking import pix2ll, RE_KM  # noqa: E402

PROTO = os.path.join(ROOT, 'data', 'processed_data', 'prototype')
OUT = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'beyond_amplitude')
DISC = os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years')
W = 172
Y0, Y1 = 1948, 2023
RMAX_CC = 1200.0
S8 = np.ones((3, 3), int)
G = {}


def build_cache():
    """CSR list, for every valid pixel, of valid pixels within 1500 km: index, distance, bearing."""
    path = os.path.join(OUT, 'ring_cache.npz')
    if os.path.exists(path):
        return path
    g = np.load(os.path.join(DISC, '2000', 'slp_sub_2000_day_1.npy'))
    valid = (g != g[0, 0]).ravel()
    rr, cc = np.divmod(np.arange(W * W), W)
    lat, lon = pix2ll(rr, cc)
    la, lo = np.radians(lat), np.radians(lon)
    vid = np.where(valid)[0]
    ptr = [0]
    idx, dist, bear = [], [], []
    for p in range(W * W):
        if valid[p]:
            dla = la[vid] - la[p]
            dlo = lo[vid] - lo[p]
            a = np.sin(dla / 2) ** 2 + np.cos(la[p]) * np.cos(la[vid]) * np.sin(dlo / 2) ** 2
            d = 2 * RE_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            k = (d <= 1500) & (vid != p)
            q = vid[k]
            b = np.arctan2(np.sin(dlo[k]) * np.cos(la[q]),
                           np.cos(la[p]) * np.sin(la[q]) - np.sin(la[p]) * np.cos(la[q]) * np.cos(dlo[k]))
            idx.append(q.astype(np.int32))
            dist.append(d[k].astype(np.float32))
            bear.append(b.astype(np.float32))
            ptr.append(ptr[-1] + k.sum())
        else:
            ptr.append(ptr[-1])
    np.savez(path, ptr=np.array(ptr, np.int64), idx=np.concatenate(idx), dist=np.concatenate(dist),
             bear=np.concatenate(bear), valid=valid)
    return path


def _init(cache):
    Z = np.load(cache)
    G['ptr'], G['idx'], G['dist'], G['bear'] = Z['ptr'], Z['idx'], Z['dist'], Z['bear']
    G['valid'] = Z['valid']
    G['feat'] = {}
    for t in ['sup', 'sub']:
        F = np.load(os.path.join(PROTO, f'feat_{t}_{Y0}_{Y1}.npz'))
        G['feat'][t] = {k: F[k] for k in ['year', 'day', 'prow', 'pcol', 'birth', 'death']}


def _disc(y, d):
    if y < Y0 or y > Y1:
        return None
    f = np.load(os.path.join(DISC, str(y), f'slp_sub_{y}_day_{d}.npy')).astype(np.float64)
    return f


def _shift(y, d, k):
    s = (y - Y0) * 365 + d - 1 + k
    if s < 0 or s >= (Y1 - Y0 + 1) * 365:
        return None
    return Y0 + s // 365, s % 365 + 1


def _ccd(g, p, v, s):
    """Closed-contour depth within RMAX_CC of pixel p (flat), for the peak value v, saddle s."""
    a, b = G['ptr'][p], G['ptr'][p + 1]
    q, dq = G['idx'][a:b], G['dist'][a:b]
    rq, cq = np.divmod(q, W)
    r0, c0 = divmod(p, W)
    ra, rb = min(rq.min(), r0), max(rq.max(), r0) + 1
    ca, cb = min(cq.min(), c0), max(cq.max(), c0) + 1
    ra, rb, ca, cb = max(ra - 1, 0), min(rb + 1, W), max(ca - 1, 0), min(cb + 1, W)
    sub = g[ra:rb, ca:cb]
    inside = np.zeros(sub.shape, bool)
    kk = dq <= RMAX_CC
    inside[rq[kk] - ra, cq[kk] - ca] = True
    inside[r0 - ra, c0 - ca] = True
    outside = ~inside                                # beyond R, padding (rim), or not valid
    pr, pc = r0 - ra, c0 - ca
    # the region of the rim/padding is always "above" any level: include it in the set
    pad = ~G['valid'].reshape(W, W)[ra:rb, ca:cb]

    def opened(L):
        lab = ndimage.label((sub >= L) | pad, structure=S8)[0]
        return bool(outside[lab == lab[pr, pc]].any())

    lo, hi = s + 1.0, v
    if not opened(lo):
        return v - s
    # opened(lo) True; find the highest L with opened(L) True
    while hi - lo > 5.0:
        mid = 0.5 * (lo + hi)
        if opened(mid):
            lo = mid
        else:
            hi = mid
    return v - lo


def one_year(y):
    t0 = time.time()
    res = {}
    fields = {}

    def disc(yd):
        if yd is None:
            return None
        if yd not in fields:
            fields[yd] = _disc(*yd)
        return fields[yd]

    ptr, idx, dist, bear = G['ptr'], G['idx'], G['dist'], G['bear']
    for t in ['sup', 'sub']:
        F = G['feat'][t]
        rows = np.where(F['year'] == y)[0]
        n = len(rows)
        out = {k: np.full(n, np.nan, np.float32) for k in
               ['m1', 'm2', 'm3', 'dip_e', 'dip_n', 'ccd', 'fut1', 'fut2', 'past1']}
        res[t] = (rows, out)
    sgn = {'sup': -1.0, 'sub': 1.0}
    for d in range(1, 366):
        for k in list(fields):
            if (k[0] - Y0) * 365 + k[1] < (y - Y0) * 365 + d - 2:
                del fields[k]
        f0 = disc((y, d))
        fn = {k: disc(_shift(y, d, k)) for k in (-1, 1, 2)}
        for t in ['sup', 'sub']:
            F = G['feat'][t]
            rows, out = res[t]
            loc = np.where(F['day'][rows] == d)[0]
            if len(loc) == 0:
                continue
            g = sgn[t] * f0
            for j in loc:
                i = rows[j]
                p = int(F['prow'][i]) * W + int(F['pcol'][i])
                a, b = ptr[p], ptr[p + 1]
                q, dq, bq = idx[a:b], dist[a:b], bear[a:b]
                gq = g.ravel()[q]
                k1, k2, k3 = dq <= 500, (dq > 500) & (dq <= 1000), dq > 1000
                out['m1'][j] = gq[k1].mean() if k1.any() else np.nan
                if k2.any():
                    m2 = gq[k2].mean()
                    out['m2'][j] = m2
                    out['dip_e'][j] = np.mean((gq[k2] - m2) * np.sin(bq[k2]))
                    out['dip_n'][j] = np.mean((gq[k2] - m2) * np.cos(bq[k2]))
                out['m3'][j] = gq[k3].mean() if k3.any() else np.nan
                k8 = dq <= 800
                for key, kk in (('past1', -1), ('fut1', 1), ('fut2', 2)):
                    h = fn[kk]
                    if h is not None:
                        hv = sgn[t] * h.ravel()
                        out[key][j] = max(hv[p], hv[q[k8]].max())
                v, s = float(F['death'][i]), float(F['birth'][i])
                if v - s >= 200:
                    out['ccd'][j] = _ccd(g, p, v, s)
    print(f'{y} done in {time.time() - t0:.0f} s', flush=True)
    return y, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--years', nargs=2, type=int, default=[Y0, Y1])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cache = build_cache()
    print('cache ready', flush=True)
    years = list(range(a.years[0], a.years[1] + 1))
    N = {t: len(np.load(os.path.join(PROTO, f'feat_{t}_{Y0}_{Y1}.npz'))['year']) for t in ['sup', 'sub']}
    full = {t: {k: np.full(N[t], np.nan, np.float32) for k in
                ['m1', 'm2', 'm3', 'dip_e', 'dip_n', 'ccd', 'fut1', 'fut2', 'past1']} for t in N}
    with Pool(a.jobs, initializer=_init, initargs=(cache,)) as P:
        for y, res in P.imap_unordered(one_year, years):
            for t, (rows, out) in res.items():
                for k, v in out.items():
                    full[t][k][rows] = v
    tag = f'{a.years[0]}_{a.years[1]}'
    for t in full:
        np.savez_compressed(os.path.join(OUT, f'desc_{t}_{tag}.npz'), **full[t])
    print('saved', flush=True)


if __name__ == '__main__':
    main()
