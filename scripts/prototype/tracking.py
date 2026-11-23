"""
Day-to-day matchers for persistence features (units hPa and km).

Every matcher shares the same geodesic gate R on the death cells, so any
difference between them comes from the proposal step alone:

    wass      the manuscript's tracker: W1 with an L1 ground metric on (birth,
              death) only, diagonal allowed, then the gate
    nn        gated nearest neighbour: maximise the number of in-radius links,
              then minimise total distance
    lifted    lifted Wasserstein (Soler et al. 2018): |db| + |dd| + lam * dist,
              diagonal allowed, pairs beyond R forbidden
    lifted_s  as lifted, plus mu * (distance between the birth cells)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e12
RE_KM = 6371.0088
C_PIX = 86


def pix2ll(row, col, c=C_PIX):
    row = np.asarray(row, float)
    col = np.asarray(col, float)
    r = np.hypot(row - c, col - c)
    return 90 - 90 * r / c, (np.degrees(np.arctan2(row - c, col - c)) + 90) % 360


def ll2pix(lat, lon, c=C_PIX):
    r = c * (90 - lat) / 90
    th = np.radians(lon - 90)
    return int(round(c + r * np.sin(th))), int(round(c + r * np.cos(th)))


def hav(la1, lo1, la2, lo2):
    """Great-circle distance matrix (km)."""
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = (np.sin((la1[:, None] - la2[None, :]) / 2) ** 2
         + np.cos(la1)[:, None] * np.cos(la2)[None, :] * np.sin((lo1[:, None] - lo2[None, :]) / 2) ** 2)
    return 2 * RE_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load(path, tau):
    """Features of depth >= tau (hPa), sorted by day, with per-day index ranges."""
    Z = np.load(path)
    y0 = int(Z['y0'])
    keys = ['year', 'day', 'birth', 'death', 'prow', 'pcol', 'srow', 'scol', 'parrow', 'parcol']
    F = {k: Z[k] for k in keys}
    m = np.abs(F['death'] - F['birth']) >= tau * 100        # threshold in Pa, as the stored pipeline does
    F = {k: v[m].astype(np.float64 if k in ('birth', 'death') else np.int64) for k, v in F.items()}
    F['birth'] /= 100
    F['death'] /= 100
    F['slot'] = (F['year'] - y0) * 365 + F['day'] - 1
    o = np.lexsort((F['prow'], F['slot']))
    F = {k: v[o] for k, v in F.items()}
    F['plat'], F['plon'] = pix2ll(F['prow'], F['pcol'])
    F['slat'], F['slon'] = pix2ll(F['srow'], F['scol'])
    F['depth'] = np.abs(F['death'] - F['birth'])
    F['y0'] = y0
    F['nslots'] = (int(Z['y1']) - y0 + 1) * 365
    F['starts'] = np.searchsorted(F['slot'], np.arange(F['nslots']))
    F['ends'] = np.searchsorted(F['slot'], np.arange(F['nslots']), 'right')
    return F


def _partial(Cpp, pers1, pers2):
    M, N = Cpp.shape
    C = np.full((M + N, M + N), BIG)
    C[:M, :N] = Cpp
    C[np.arange(M), N + np.arange(M)] = pers1
    C[M + np.arange(N), np.arange(N)] = pers2
    C[M:, N:] = 0
    ri, ci = linear_sum_assignment(C)
    k = (ri < M) & (ci < N)
    return ri[k], ci[k]


def match(mode, F, a, b, c, d, R, lam=0.0, mu=0.0):
    D = hav(F['plat'][a:b], F['plon'][a:b], F['plat'][c:d], F['plon'][c:d])
    b1, d1, b2, d2 = F['birth'][a:b], F['death'][a:b], F['birth'][c:d], F['death'][c:d]
    Ctop = np.abs(b1[:, None] - b2[None, :]) + np.abs(d1[:, None] - d2[None, :])
    if mode == 'wass':
        ri, ci = _partial(Ctop, np.abs(d1 - b1), np.abs(d2 - b2))
    elif mode == 'nn':
        ri, ci = linear_sum_assignment(np.where(D <= R, D, 1e9))
    elif mode in ('lifted', 'lifted_s'):
        Cpp = Ctop + lam * D
        if mode == 'lifted_s':
            Cpp = Cpp + mu * hav(F['slat'][a:b], F['slon'][a:b], F['slat'][c:d], F['slon'][c:d])
        ri, ci = _partial(np.where(D <= R, Cpp, BIG), np.abs(d1 - b1), np.abs(d2 - b2))
    else:
        raise ValueError(mode)
    k = D[ri, ci] <= R
    return ri[k], ci[k], D


def run(F, mode, R=1000.0, lam=0.0, mu=0.0):
    n = len(F['slot'])
    succ = -np.ones(n, int)
    pred = -np.ones(n, int)
    step = np.full(n, np.nan)
    for t in range(F['nslots'] - 1):
        a, b = F['starts'][t], F['ends'][t]
        c, d = F['starts'][t + 1], F['ends'][t + 1]
        if b == a or d == c:
            continue
        ri, ci, D = match(mode, F, a, b, c, d, R, lam, mu)
        succ[a + ri] = c + ci
        pred[c + ci] = a + ri
        step[c + ci] = D[ri, ci]
    return {'succ': succ, 'pred': pred, 'step': step}


def tracks(res):
    succ, pred = res['succ'], res['pred']
    out = []
    for h in np.where(pred < 0)[0]:
        m = [h]
        while succ[m[-1]] >= 0:
            m.append(succ[m[-1]])
        out.append(m)
    return out


def summary(F, res, keep_mask=None):
    """Track statistics; with keep_mask, only tracks with at least one True snapshot."""
    T = tracks(res)
    if keep_mask is not None:
        T = [m for m in T if keep_mask[m].any()]
    L = np.array([len(m) for m in T])
    return dict(links=int((res['succ'] >= 0).sum()), N=len(L), pct1=100 * np.mean(L == 1),
                meanL=L.mean(), maxL=int(L.max()), n_ge5=int((L >= 5).sum()))


def ambiguity(F, R=1000.0):
    """Share of features with two or more candidate successors within R."""
    na = nf = 0
    for t in range(F['nslots'] - 1):
        a, b = F['starts'][t], F['ends'][t]
        c, d = F['starts'][t + 1], F['ends'][t + 1]
        if b == a or d == c:
            continue
        D = hav(F['plat'][a:b], F['plon'][a:b], F['plat'][c:d], F['plon'][c:d])
        nf += b - a
        na += int(((D <= R).sum(1) >= 2).sum())
    return na, nf
