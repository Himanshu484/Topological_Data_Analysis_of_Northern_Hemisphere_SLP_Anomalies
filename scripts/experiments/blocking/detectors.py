"""
Blocking detectors (topological and non-topological) and verification against a
reference index.  Every detector returns daily 'snapshots' (t, lat, lon, eid): the
position of a flagged blocking object on day t and the id of the event it belongs to.

Topological (tracked persistence features, NN tracker of scripts/prototype, R = 1000 km,
two-level: tracked down to tau_lo, flagged at thr):
  topo(depth)   a 5-day window of a track where every snapshot has depth >= thr,
                45 <= lat <= 80, and all five positions lie within Rs of their centroid
  topo(amp)     the same tracks and rule with the extremum VALUE in place of depth
                (ablation: what does prominence add over amplitude?)
  dipole        topo(depth) on Z500-anomaly maxima AND a Z500-anomaly minimum of depth
                >= thr_low on the same day, 5-30 deg equatorward, within 3000 km and
                40 deg of longitude (Rex / omega configuration)
Non-topological baselines:
  grid          anomaly >= thr at a gridpoint (45-80N) for >= 5 consecutive days; objects are
                space-time connected components, snapshot = daily anomaly maximum
  dg            Dole-Gordon / Schwierz-type anomaly-persistence index: daily regions of Z500
                anomaly >= the calendar-month 90th percentile (45-80N), area >= 1.5e6 km2,
                linked day to day by >= 50 % overlap, >= 5 days, centroid >= 45N

Verification (reference points = BE gridpoints of the reference index):
  sector-day  a sector-day is 'blocked' if it holds a reference point / a detector snapshot
  grid-day    a reference point is hit if a detector snapshot lies within D = 1000 km on the
              same day (POD, cos-lat weighted); a snapshot is correct if a reference point
              lies within D (success ratio SR); CSI = 1 / (1/POD + 1/SR - 1)
  event       a reference event is detected if it is hit on >= 50 % of its days; a detector
              event is correct if its snapshots are correct on >= 50 % of its days
"""
import os

import numpy as np
import pandas as pd
from scipy import ndimage

from common import (CACHE, LAT, LON, Y0, EVAL0, EVAL1, SEASON, SECTORS, in_sector, gc,
                    circmean_lon, load_feats, run, tracks, hav, MONTH)
from tm_index import PHI0, runs_time, label_periodic_3d

T0 = (EVAL0 - Y0) * 365
NT = (EVAL1 - EVAL0 + 1) * 365
YEAR = EVAL0 + np.arange(NT) // 365
SLOT = np.arange(NT) % 365 + 1
SEAS = SEASON[SLOT - 1]
SECS = list(SECTORS)
D_MATCH = 1000.0
ROWS_DET = np.where((LAT >= 45) & (LAT <= 80))[0]
CELL_KM2 = (np.radians(2.5) * 6371.0088) ** 2 * np.cos(np.radians(LAT))


def sector_idx(lon):
    out = np.zeros(len(lon), int)
    for k, s in enumerate(SECS):
        out[in_sector(np.asarray(lon), s)] = k
    return out


# ---------------------------------------------------------------- references
def ref_tm():
    p = np.load(os.path.join(CACHE, 'tm_points.npy'))
    k = (p[:, 0] >= T0) & (p[:, 0] < T0 + NT)
    p = p[k]
    lat = PHI0[p[:, 1]]
    return dict(t=p[:, 0] - T0, lat=lat, lon=LON[p[:, 2]], eid=p[:, 3], w=np.cos(np.radians(lat)))


def label2d_periodic(m):
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    L, R = lab[:, 0], lab[:, -1]
    H = len(L)
    for dh in (-1, 0, 1):
        a = L[max(0, -dh):H - max(0, dh)]
        b = R[max(0, dh):H + min(0, dh)]
        k = (a > 0) & (b > 0)
        for x, y in set(zip(a[k].tolist(), b[k].tolist())):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq, new = np.unique(roots, return_inverse=True)
    return new[lab], len(uniq) - 1


def dg_index(A, q=0.90, amin=1.5e6, overlap=0.5, L=5, return_points=False):
    """Anomaly-persistence (Dole-Gordon/Schwierz-type) index on Z500 anomalies A (NT, 37, 144)."""
    rows = ROWS_DET
    band = (LAT >= 50) & (LAT <= 75)
    thr = np.zeros(13)
    for m in range(1, 13):
        sel = MONTH[SLOT - 1] == m
        thr[m] = np.quantile(A[sel][:, band][:, :, ::2], q)
    cell = CELL_KM2[rows][:, None] * np.ones((1, 144))
    regions = []                         # per day: list of (flat index array, area)
    for t in range(NT):
        m = A[t, rows] >= thr[MONTH[SLOT[t] - 1]]
        lab, n = label2d_periodic(m)
        day = []
        if n:
            areas = ndimage.sum(cell, lab, np.arange(1, n + 1))
            for k in np.where(areas >= amin)[0]:
                day.append((np.flatnonzero(lab == k + 1), areas[k]))
        regions.append(day)
    # overlap linking (greedy by overlap area)
    nxt = {}
    prv = {}
    for t in range(NT - 1):
        for i, (ia, aa) in enumerate(regions[t]):
            best, bo = -1, 0.0
            for j, (ib, ab) in enumerate(regions[t + 1]):
                inter = np.intersect1d(ia, ib, assume_unique=True)
                if len(inter) == 0:
                    continue
                ov = cell.ravel()[inter].sum()
                if ov >= overlap * min(aa, ab) and ov > bo and (t + 1, j) not in prv:
                    best, bo = j, ov
            if best >= 0:
                nxt[(t, i)] = (t + 1, best)
                prv[(t + 1, best)] = (t, i)
    snaps = []
    pts = []
    eid = 0
    flat_lat = np.repeat(LAT[rows], 144)
    flat_lon = np.tile(LON, len(rows))
    for t in range(NT):
        for i in range(len(regions[t])):
            if (t, i) in prv:
                continue
            chain = [(t, i)]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            if len(chain) < L:
                continue
            rec = []
            for (tt, ii) in chain:
                idx, _ = regions[tt][ii]
                w = cell.ravel()[idx]
                la = float((flat_lat[idx] * w).sum() / w.sum())
                lo = float(circmean_lon(flat_lon[idx], w))
                rec.append((tt, la, lo))
            if np.mean([r[1] for r in rec]) < 45:
                continue
            for (tt, la, lo), (t2, ii) in zip(rec, chain):
                snaps.append((tt, la, lo, eid))
                if return_points:
                    idx, _ = regions[t2][ii]
                    for f in idx:
                        pts.append((t2, flat_lat[f], flat_lon[f], eid))
            eid += 1
    s = np.array(snaps)
    det = dict(t=s[:, 0].astype(int), lat=s[:, 1], lon=s[:, 2], eid=s[:, 3].astype(int))
    if return_points:
        p = np.array(pts)
        ref = dict(t=p[:, 0].astype(int), lat=p[:, 1], lon=p[:, 2], eid=p[:, 3].astype(int),
                   w=np.cos(np.radians(p[:, 1])))
        return det, ref, thr
    return det


# ---------------------------------------------------------------- baselines
def grid_detector(A, thr, scale=None, L=5):
    """A (NT, 37, 144) anomaly in detector units; scale (NT,) seasonal normaliser or None."""
    X = A[:, ROWS_DET]
    if scale is not None:
        X = X / scale[:, None, None]
    P = runs_time(X >= thr, L)
    lab, n = label_periodic_3d(P)
    t, r, c = np.nonzero(lab)
    if len(t) == 0:
        return dict(t=np.zeros(0, int), lat=np.zeros(0), lon=np.zeros(0), eid=np.zeros(0, int))
    l = lab[t, r, c]
    v = X[t, r, c]
    o = np.lexsort((-v, l, t))
    t, r, c, l = t[o], r[o], c[o], l[o]
    first = np.ones(len(t), bool)
    first[1:] = (t[1:] != t[:-1]) | (l[1:] != l[:-1])
    return dict(t=t[first], lat=LAT[ROWS_DET][r[first]], lon=LON[c[first]], eid=l[first] - 1)


# ---------------------------------------------------------------- topological
class TopoTracks:
    def __init__(self, path, tau_lo, scale_units):
        F = load_feats(path, tau_lo, scale_units, EVAL0, EVAL1)
        res = run(F, 'nn', R=1000.0)
        self.F = F
        self.T = [np.array(m) for m in tracks(res) if len(m) >= 5]
        self.ntracks = len(tracks(res))

    def detect(self, thr, use='depth', lat_min=45, lat_max=80, Rs=1500.0, L=5, extra=None, scale=None):
        F = self.F
        val = F['depth'] if use == 'depth' else F['death']
        if scale is not None:
            val = val / scale[F['slot']]
        cond = (val >= thr) & (F['plat'] >= lat_min) & (F['plat'] <= lat_max)
        if extra is not None:
            cond &= extra
        eid = -np.ones(len(val), int)
        ne = 0
        ker = np.ones(L, int)
        for m in self.T:
            c = cond[m]
            if c.sum() < L:
                continue
            idx = np.where(np.convolve(c.astype(int), ker, 'valid') == L)[0]
            if not len(idx):
                continue
            la, lo = F['plat'][m], F['plon'][m]
            fl = np.zeros(len(m), bool)
            for i in idx:
                wla, wlo = np.radians(la[i:i + L]), np.radians(lo[i:i + L])
                x, y, z = np.cos(wla) * np.cos(wlo), np.cos(wla) * np.sin(wlo), np.sin(wla)
                cla = np.degrees(np.arctan2(z.mean(), np.hypot(x.mean(), y.mean())))
                clo = np.degrees(np.arctan2(y.mean(), x.mean()))
                if Rs is None or gc(la[i:i + L], lo[i:i + L], cla, clo).max() <= Rs:
                    fl[i:i + L] = True
            lab, n = ndimage.label(fl)
            for k in range(1, n + 1):
                eid[m[lab == k]] = ne
                ne += 1
        k = eid >= 0
        return dict(t=F['slot'][k], lat=F['plat'][k], lon=F['plon'][k], eid=eid[k], val=val[k],
                    fidx=np.where(k)[0])


def dipole_mask(H, Lo, thr_low, dlat=(5, 30), dmax=3000.0, dlon=40.0, scale=None):
    """For each feature of H (highs), is there a feature of Lo (lows) of depth >= thr_low equatorward?"""
    out = np.zeros(len(H['slot']), bool)
    dL = Lo['depth'] if scale is None else Lo['depth'] / scale[Lo['slot']]
    for t in range(H['nslots']):
        a, b = H['starts'][t], H['ends'][t]
        c, d = Lo['starts'][t], Lo['ends'][t]
        if a == b or c == d:
            continue
        ok = dL[c:d] >= thr_low
        if not ok.any():
            continue
        la2, lo2 = Lo['plat'][c:d][ok], Lo['plon'][c:d][ok]
        la1, lo1 = H['plat'][a:b], H['plon'][a:b]
        D = hav(la1, lo1, la2, lo2)
        dla = la1[:, None] - la2[None, :]
        dlo = np.abs((lo1[:, None] - lo2[None, :] + 180) % 360 - 180)
        out[a:b] = ((D <= dmax) & (dla >= dlat[0]) & (dla <= dlat[1]) & (dlo <= dlon)).any(1)
    return out


# ---------------------------------------------------------------- verification
def match(det, ref, D=D_MATCH):
    rh = np.zeros(len(ref['t']), bool)
    ok = np.zeros(len(det['t']), bool)
    ro = np.argsort(ref['t'], kind='stable')
    do = np.argsort(det['t'], kind='stable')
    rs = np.searchsorted(ref['t'][ro], np.arange(NT + 1))
    ds = np.searchsorted(det['t'][do], np.arange(NT + 1))
    for t in range(NT):
        i = ro[rs[t]:rs[t + 1]]
        j = do[ds[t]:ds[t + 1]]
        if len(i) == 0 or len(j) == 0:
            continue
        M = hav(ref['lat'][i], ref['lon'][i], det['lat'][j], det['lon'][j]) <= D
        rh[i] = M.any(1)
        ok[j] = M.any(0)
    return rh, ok


def _events(obj, hit, weight=None):
    """Per-event table: onset t, sector, fraction of days hit/correct."""
    df = pd.DataFrame(dict(eid=obj['eid'], t=obj['t'], hit=hit, lat=obj['lat'], lon=obj['lon']))
    day = df.groupby(['eid', 't']).agg(hit=('hit', 'max'), lon=('lon', lambda x: circmean_lon(np.asarray(x))))
    day = day.reset_index()
    ev = day.groupby('eid').agg(t0=('t', 'min'), frac=('hit', 'mean'),
                                lon=('lon', lambda x: circmean_lon(np.asarray(x))), n=('t', 'size'))
    ev['sec'] = sector_idx(ev['lon'].values)
    ev['year'] = YEAR[ev['t0'].values]
    ev['seas'] = SEAS[ev['t0'].values]
    return ev


def counts(det, ref, rh=None, ok=None):
    """Per-year, per-season, per-sector count tables for the three verification levels."""
    if rh is None:
        rh, ok = match(det, ref)
    rsec = sector_idx(ref['lon'])
    dsec = sector_idx(det['lon'])
    Rf = np.zeros((NT, 4), bool)
    Df = np.zeros((NT, 4), bool)
    Rf[ref['t'], rsec] = True
    if len(det['t']):
        Df[det['t'], dsec] = True
    rows = []
    for k in range(4):
        rows.append(pd.DataFrame(dict(year=YEAR, seas=SEAS, sec=k, a=Rf[:, k] & Df[:, k],
                                      b=~Rf[:, k] & Df[:, k], c=Rf[:, k] & ~Df[:, k])))
    sd = pd.concat(rows).groupby(['year', 'seas', 'sec']).sum().reset_index()
    g1 = pd.DataFrame(dict(year=YEAR[ref['t']], seas=SEAS[ref['t']], sec=rsec, hw=ref['w'] * rh, w=ref['w']))
    g2 = pd.DataFrame(dict(year=YEAR[det['t']], seas=SEAS[det['t']], sec=dsec, ok=ok.astype(float), n=1.0))
    gd1 = g1.groupby(['year', 'seas', 'sec']).sum().reset_index()
    gd2 = g2.groupby(['year', 'seas', 'sec']).sum().reset_index()
    er = _events(ref, rh)
    ed = _events(det, ok) if len(det['t']) else None
    e1 = er.assign(det=(er['frac'] >= 0.5).astype(float), nref=1.0)[['year', 'seas', 'sec', 'det', 'nref']]
    e1 = e1.groupby(['year', 'seas', 'sec']).sum().reset_index()
    if ed is not None:
        e2 = ed.assign(cor=(ed['frac'] >= 0.5).astype(float), ndet=1.0)[['year', 'seas', 'sec', 'cor', 'ndet']]
        e2 = e2.groupby(['year', 'seas', 'sec']).sum().reset_index()
    else:
        e2 = pd.DataFrame(columns=['year', 'seas', 'sec', 'cor', 'ndet'])
    return dict(sd=sd, gd1=gd1, gd2=gd2, e1=e1, e2=e2)


def _csi(pod, sr):
    if pod <= 0 or sr <= 0:
        return 0.0
    return 1.0 / (1.0 / pod + 1.0 / sr - 1.0)


def scores(C, years=None, seas=None, sec=None):
    def f(df):
        m = np.ones(len(df), bool)
        if years is not None:
            m &= df['year'].isin(years).values
        if seas is not None:
            m &= (df['seas'] == seas).values
        if sec is not None:
            m &= (df['sec'] == sec).values
        return df[m]
    sd = f(C['sd'])
    a, b, c = sd['a'].sum(), sd['b'].sum(), sd['c'].sum()
    out = dict(sd_pod=a / max(a + c, 1), sd_far=b / max(a + b, 1), sd_csi=a / max(a + b + c, 1),
               sd_bias=(a + b) / max(a + c, 1))
    g1, g2 = f(C['gd1']), f(C['gd2'])
    pod = g1['hw'].sum() / max(g1['w'].sum(), 1e-9)
    sr = g2['ok'].sum() / max(g2['n'].sum(), 1e-9)
    out.update(gd_pod=pod, gd_far=1 - sr if g2['n'].sum() else np.nan, gd_csi=_csi(pod, sr))
    e1, e2 = f(C['e1']), f(C['e2'])
    pod = e1['det'].sum() / max(e1['nref'].sum(), 1e-9)
    sr = e2['cor'].sum() / max(e2['ndet'].sum(), 1e-9) if len(e2) else 0.0
    out.update(ev_pod=pod, ev_far=1 - sr if len(e2) else np.nan, ev_csi=_csi(pod, sr),
               ev_nref=e1['nref'].sum(), ev_ndet=e2['ndet'].sum() if len(e2) else 0)
    return out


def boot_diff(CA, CB, years, key, nb=1000, seed=0):
    """Year-block bootstrap of score(A) - score(B) over `years`; returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    years = np.array(list(years))

    base = scores(CA, years)[key] - scores(CB, years)[key]
    diffs = []
    for _ in range(nb):
        ys = rng.choice(years, len(years), replace=True)
        diffs.append(_score_resampled(CA, ys, key) - _score_resampled(CB, ys, key))
    return base, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _score_resampled(C, ys, key):
    """Score with years resampled (with multiplicity)."""
    cnt = pd.Series(ys).value_counts()

    def agg(df, cols):
        g = df.groupby('year')[cols].sum()
        g = g.reindex(cnt.index, fill_value=0)
        return (g.mul(cnt, axis=0)).sum()
    lvl = key.split('_')[0]
    if lvl == 'sd':
        s = agg(C['sd'], ['a', 'b', 'c'])
        return s['a'] / max(s['a'] + s['b'] + s['c'], 1)
    if lvl == 'gd':
        s1, s2 = agg(C['gd1'], ['hw', 'w']), agg(C['gd2'], ['ok', 'n'])
        return _csi(s1['hw'] / max(s1['w'], 1e-9), s2['ok'] / max(s2['n'], 1e-9))
    s1, s2 = agg(C['e1'], ['det', 'nref']), agg(C['e2'], ['cor', 'ndet'])
    return _csi(s1['det'] / max(s1['nref'], 1e-9), s2['cor'] / max(s2['ndet'], 1e-9))


_GLAT = np.repeat(LAT[ROWS_DET], 144)
_GLON = np.tile(LON, len(ROWS_DET))


def topo_loc(F, thr, use='depth', r=750.0, L=5, scale=None, extra=None, lat_min=45, lat_max=80):
    """
    Eulerian persistence of topological features: gridpoint g (45-80N) is 'on' at day t if a
    feature with depth (or amplitude) >= thr lies within r of g; blocked where 'on' for >= L
    consecutive days.  Objects are space-time components; the snapshot of a component-day is its
    deepest qualifying feature.  This is the gridpoint baseline with 'anomaly >= A at g' replaced
    by 'a prominence >= tau extremum within r of g'.
    """
    val = F['depth'] if use == 'depth' else F['death']
    if scale is not None:
        val = val / scale[F['slot']]
    q = (val >= thr) & (F['plat'] >= lat_min) & (F['plat'] <= lat_max)
    if extra is not None:
        q &= extra
    on = np.zeros((NT, len(ROWS_DET), 144), bool)
    qi = np.where(q)[0]
    st = np.searchsorted(F['slot'][qi], np.arange(NT + 1))
    for t in range(NT):
        j = qi[st[t]:st[t + 1]]
        if len(j) == 0:
            continue
        D = hav(_GLAT, _GLON, F['plat'][j], F['plon'][j])
        on[t] = (D <= r).any(1).reshape(len(ROWS_DET), 144)
    P = runs_time(on, L)
    lab, n = label_periodic_3d(P)
    rr = np.clip(np.round((90 - F['plat'][qi]) / 2.5).astype(int), 0, 36)
    cc = np.round(F['plon'][qi] / 2.5).astype(int) % 144
    rloc = np.searchsorted(ROWS_DET, rr)
    okr = (rloc < len(ROWS_DET)) & (ROWS_DET[np.clip(rloc, 0, len(ROWS_DET) - 1)] == rr)
    l = np.zeros(len(qi), int)
    l[okr] = lab[F['slot'][qi][okr], rloc[okr], cc[okr]]
    k = l > 0
    j, l = qi[k], l[k]
    o = np.lexsort((-val[j], l, F['slot'][j]))
    j, l = j[o], l[o]
    t = F['slot'][j]
    first = np.ones(len(j), bool)
    first[1:] = (t[1:] != t[:-1]) | (l[1:] != l[:-1])
    j, l = j[first], l[first]
    return dict(t=F['slot'][j], lat=F['plat'][j], lon=F['plon'][j], eid=l - 1, val=val[j], fidx=j)
