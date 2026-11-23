"""
Composite analysis: sector-restricted topological descriptors on index-defined blocks
against season- and location-matched non-blocked days.

Population: 2-D TM blocking events (phi0 >= 40N, >= 5 d), 1979-2023.  Each event-day has a
block centroid (cos-lat weighted centroid of that day's BE gridpoints).
Controls: for each event, K = 5 other years; each event day is replayed at the same
calendar slot and the same centroid; a control day is kept only if no BE gridpoint lies
within 1500 km of the centroid on that day.

Descriptors at the centroid (SLP 1-anticyclones from the manuscript's discs, and Z500-anomaly
maxima from Z500 discs built with the same projection):
  depth      depth (prominence) of the deepest maximum within 1000 km        [topological]
  amp        maximum anomaly on the lat-lon grid within 1000 km                [non-topological]
  basin      merge-tree basin area of that maximum: the 8-connected component of
             {f > saddle value} containing the peak, 10^6 km^2                [topological]
  nested     1 if its merge partner (parent) is another maximum, 0 if the disc rim
  nestdepth  number of ancestors in the merge tree before the rim
  tp_loc     sum of depths of maxima within 2500 km (local 'TP1')
  n_loc      number of maxima of depth >= 10 hPa / 100 gpm within 2500 km
  dw_loc     W1 (persim) between the local diagram (maxima within 2500 km of the day-t
             centroid, depth >= 1 hPa / 10 gpm) on day t and on day t+1   ['frozen' test]
  tp_hemi    hemispheric sum of depths (the manuscript's TP1, >= 1 hPa / 10 gpm)
  dw_hemi    hemispheric W1 between day t and t+1 (the manuscript's d_W)
  cutoff     prominence of the most prominent maximum of the FULL Z500 field within 1000 km
             (0 if none: no closed 500 hPa anticyclone)                         [Z500 only]

Statistics (event level, so autocorrelated days are not treated as independent):
  per event, mean descriptor over block days minus mean over its control days;
  Wilcoxon signed-rank p, median difference, and the day-level AUC (block vs control).
  Beyond amplitude: AUC of each descriptor within deciles of amp (pooled block + control days),
  and a logistic model  blocked ~ amp + descriptor  with an event-block bootstrap CI of the
  descriptor coefficient (standardised).

    python scripts/experiments/blocking/composites.py [--k 5]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import persim
from scipy import ndimage, stats

from common import (CACHE, OUT, SLP_DISCS, PROTO_FEAT, LAT, LON, Y0, EVAL0, EVAL1, SECTORS, SEASONS,
                    SEASON, disc, disc_pixel_area_km2, pix2ll, hav)
import detectors as dt

AREA = disc_pixel_area_km2()
R_NEAR, R_LOC, R_EXCL = 1000.0, 2500.0, 1500.0
W = 172


def feats_by_day(path, scale, min_depth, y0=EVAL0, y1=EVAL1):
    Z = np.load(path)
    m = (Z['year'] >= y0) & (Z['year'] <= y1) & (np.abs(Z['death'] - Z['birth']) >= min_depth * scale)
    t = (Z['year'][m].astype(int) - y0) * 365 + Z['day'][m].astype(int) - 1
    o = np.argsort(t, kind='stable')
    D = dict(t=t[o], b=Z['birth'][m][o] / scale, d=Z['death'][m][o] / scale,
             pr=Z['prow'][m][o].astype(int), pc=Z['pcol'][m][o].astype(int),
             sr=Z['srow'][m][o].astype(int), sc=Z['scol'][m][o].astype(int),
             ar=Z['parrow'][m][o].astype(int), ac=Z['parcol'][m][o].astype(int))
    D['lat'], D['lon'] = pix2ll(D['pr'], D['pc'])
    D['dep'] = np.abs(D['d'] - D['b'])
    D['st'] = np.searchsorted(D['t'], np.arange(dt.NT + 1))
    return D


def day_slice(D, t):
    return slice(D['st'][t], D['st'][t + 1])


def slp_disc(t):
    y, d = EVAL0 + t // 365, t % 365 + 1
    return np.load(os.path.join(SLP_DISCS, str(y), f'slp_sub_{y}_day_{d}.npy')).astype(np.float64) / 100.0


class ZDisc:
    def __init__(self):
        self.A = np.load(os.path.join(CACHE, 'z500_anom.npy'), mmap_mode='r')

    def __call__(self, t):
        return disc(np.asarray(self.A[EVAL0 - Y0 + t // 365, t % 365], np.float64), 'sub')


def basin_area(g, pr, pc, s):
    m = (g > s) & (g != g[0, 0])
    lab, _ = ndimage.label(m, structure=np.ones((3, 3), bool))
    k = lab[pr, pc]
    if k == 0:
        return np.nan
    return AREA[lab == k].sum() / 1e6


def nested_by_label(g, pr, pc, s):
    """1 if, at the saddle level, the peak's superlevel component does not reach the padding (rim)."""
    lab, _ = ndimage.label(g >= s, structure=np.ones((3, 3), bool))
    return float(not (lab == lab[pr, pc])[g == g[0, 0]].any())


def nest_depth(D, sl, i):
    """Number of merge-tree ancestors of feature i (index into the day slice) before the rim."""
    pk = {(r, c): j for j, (r, c) in enumerate(zip(D['pr'][sl], D['pc'][sl]))}
    n, j, seen = 0, i, set()
    while True:
        a = (D['ar'][sl][j], D['ac'][sl][j])
        if a[0] < 0 or a not in pk or j in seen:
            return n + (0 if a[0] < 0 else 1)
        seen.add(j)
        n += 1
        j = pk[a]


def w1(b1, d1, b2, d2):
    X = np.stack([b1, d1], 1) if len(b1) else np.zeros((0, 2))
    Y = np.stack([b2, d2], 1) if len(b2) else np.zeros((0, 2))
    if len(X) == 0 and len(Y) == 0:
        return 0.0
    return float(persim.wasserstein(X, Y))


def descriptors(t, lat, lon, D, getdisc, amp_field, n_thr, dmin, F_cut=None, cache=None):
    """All descriptors for location (lat, lon) on day t."""
    sl = day_slice(D, t)
    la, lo, dep = D['lat'][sl], D['lon'][sl], D['dep'][sl]
    out = {}
    if len(la):
        dist = hav(np.array([lat]), np.array([lon]), la, lo)[0]
    else:
        dist = np.zeros(0)
    near = np.where(dist <= R_NEAR)[0]
    if len(near):
        i = near[np.argmax(dep[near])]
        out['depth'] = dep[i]
        key = ('disc', t)
        if cache is not None and key in cache:
            g = cache[key]
        else:
            g = getdisc(t)
            if cache is not None:
                cache[key] = g
        out['basin'] = basin_area(g, D['pr'][sl][i], D['pc'][sl][i], D['b'][sl][i])
        if D['ar'][sl][i] == -2:           # cripser features: no stored parent
            out['nested'] = nested_by_label(g, D['pr'][sl][i], D['pc'][sl][i], D['b'][sl][i])
            out['nestdepth'] = np.nan
        else:
            out['nested'] = float(D['ar'][sl][i] >= 0)
            out['nestdepth'] = float(nest_depth(D, sl, i))
    else:
        out.update(depth=0.0, basin=0.0, nested=np.nan, nestdepth=np.nan)
    # amplitude on the lat-lon grid
    y, d = t // 365, t % 365
    f = amp_field[y * 365 + d]
    gd = hav(np.array([lat]), np.array([lon]), _GL, _GO)[0].reshape(37, 144)
    out['amp'] = float(f[gd <= R_NEAR].max())
    loc = dist <= R_LOC
    big = dep >= dmin
    out['tp_loc'] = float(dep[loc & big].sum())
    out['n_loc'] = float((loc & (dep >= n_thr)).sum())
    out['tp_hemi'] = float(dep[big].sum())
    # day-to-day change (same region, fixed at day t)
    if t + 1 < dt.NT:
        s2 = day_slice(D, t + 1)
        la2, lo2, dep2 = D['lat'][s2], D['lon'][s2], D['dep'][s2]
        dist2 = hav(np.array([lat]), np.array([lon]), la2, lo2)[0] if len(la2) else np.zeros(0)
        k1 = loc & big
        k2 = (dist2 <= R_LOC) & (dep2 >= dmin)
        out['dw_loc'] = w1(D['b'][sl][k1], D['d'][sl][k1], D['b'][s2][k2], D['d'][s2][k2])
        hk = ('dwh', t)
        if cache is not None and hk in cache:
            out['dw_hemi'] = cache[hk]
        else:
            out['dw_hemi'] = w1(D['b'][sl][big], D['d'][sl][big], D['b'][s2][dep2 >= dmin], D['d'][s2][dep2 >= dmin])
            if cache is not None:
                cache[hk] = out['dw_hemi']
    else:
        out['dw_loc'] = out['dw_hemi'] = np.nan
    if F_cut is not None:
        sc = day_slice(F_cut, t)
        if sc.stop > sc.start:
            dc = hav(np.array([lat]), np.array([lon]), F_cut['lat'][sc], F_cut['lon'][sc])[0]
            k = dc <= R_NEAR
            out['cutoff'] = float(F_cut['dep'][sc][k].max()) if k.any() else 0.0
        else:
            out['cutoff'] = 0.0
    return out


_GL = np.repeat(LAT, 144)
_GO = np.tile(LON, 37)


def build_table(K, seed=0):
    E = np.load(os.path.join(CACHE, 'tm_events.npz'), allow_pickle=True)
    BE = np.load(os.path.join(CACHE, 'tm_be.npy'), mmap_mode='r')
    from tm_index import PHI0
    rows40 = PHI0 >= 40
    ev_ok = (E['t0'] >= dt.T0) & (E['t1'] < dt.T0 + dt.NT)
    eids = E['eid'][ev_ok]
    sect = dict(zip(E['eid'], E['sector']))
    dmask = np.isin(E['d_eid'], eids)
    dd = pd.DataFrame(dict(eid=E['d_eid'][dmask], t=E['d_t'][dmask] - dt.T0, lat=E['d_lat'][dmask],
                           lon=E['d_lon'][dmask]))
    rng = np.random.default_rng(seed)
    years = np.arange(EVAL1 - EVAL0 + 1)
    recs = []
    # BE gridpoints (phi0 >= 40) per day for the exclusion test
    pts = np.load(os.path.join(CACHE, 'tm_points.npy'))
    pts = pts[(pts[:, 0] >= dt.T0) & (pts[:, 0] < dt.T0 + dt.NT)]
    pst = np.searchsorted(pts[:, 0] - dt.T0, np.arange(dt.NT + 1))
    plat, plon = PHI0[pts[:, 1]], LON[pts[:, 2]]

    def blocked_near(t, la, lo):
        s = slice(pst[t], pst[t + 1])
        if s.stop == s.start:
            return False
        return bool((hav(np.array([la]), np.array([lo]), plat[s], plon[s])[0] <= R_EXCL).any())

    for eid, g in dd.groupby('eid'):
        yb = g['t'].iloc[0] // 365
        for _, r in g.iterrows():
            recs.append((eid, 1, int(r.t), r.lat, r.lon, yb, int(r.t)))
        cy = rng.choice(years[years != yb], K, replace=False)
        for y in cy:
            for _, r in g.iterrows():
                tc = int(r.t) + (y - yb) * 365
                if 0 <= tc < dt.NT - 1 and not blocked_near(tc, r.lat, r.lon):
                    recs.append((eid, 0, tc, r.lat, r.lon, y, int(r.t)))
    T = pd.DataFrame(recs, columns=['eid', 'blk', 't', 'lat', 'lon', 'yidx', 't_event'])
    T['sector'] = T['eid'].map(sect)
    T['season'] = SEASON[(T['t_event'] % 365).values]
    return T


def run_descriptors(T, which):
    if which == 'slp':
        D = feats_by_day(os.path.join(PROTO_FEAT, 'feat_sub_1948_2023.npz'), 100.0, 0.5)
        getdisc = slp_disc
        amp = np.load(os.path.join(CACHE, 'slp_anom.npy'), mmap_mode='r')
        amp = np.asarray(amp[EVAL0 - Y0:EVAL1 - Y0 + 1]).reshape(dt.NT, 37, 144) / 100.0
        n_thr, dmin, Fc = 10.0, 1.0, None
    else:
        D = feats_by_day(os.path.join(CACHE, f'feat_za_sub_{EVAL0}_{EVAL1}.npz'), 1.0, 5.0)
        getdisc = ZDisc()
        amp = np.load(os.path.join(CACHE, 'z500_anom.npy'), mmap_mode='r')
        amp = np.asarray(amp[EVAL0 - Y0:EVAL1 - Y0 + 1]).reshape(dt.NT, 37, 144)
        n_thr, dmin = 100.0, 10.0
        Fc = feats_by_day(os.path.join(CACHE, f'feat_zf_sub_{EVAL0}_{EVAL1}.npz'), 1.0, 1.0)
    out = []
    T = T.sort_values('t')
    cache = {}
    last_t = -1
    t0 = time.time()
    for n, r in enumerate(T.itertuples()):
        if r.t != last_t:
            cache = {k: v for k, v in cache.items() if k[1] >= r.t}
            last_t = r.t
        o = descriptors(r.t, r.lat, r.lon, D, getdisc, amp, n_thr, dmin, Fc, cache)
        o['idx'] = r.Index
        out.append(o)
        if n % 20000 == 0:
            print(f'   {which}: {n}/{len(T)} [{time.time() - t0:.0f}s]', flush=True)
    return pd.DataFrame(out).set_index('idx').sort_index()


def add_pkval(T):
    """Non-topological twin of 'depth': extremum VALUE of the same (deepest-near) maximum; 0 if none."""
    for which, path, scale, dmin in (('slp', os.path.join(PROTO_FEAT, 'feat_sub_1948_2023.npz'), 100.0, 0.5),
                                     ('z500', os.path.join(CACHE, f'feat_za_sub_{EVAL0}_{EVAL1}.npz'), 1.0, 5.0)):
        D = feats_by_day(path, scale, dmin)
        pk = np.zeros(len(T))
        for n, (t, la, lo) in enumerate(zip(T['t'].values, T['lat'].values, T['lon'].values)):
            sl = day_slice(D, t)
            if sl.stop == sl.start:
                continue
            dist = hav(np.array([la]), np.array([lo]), D['lat'][sl], D['lon'][sl])[0]
            k = np.where(dist <= R_NEAR)[0]
            if len(k):
                i = k[np.argmax(D['dep'][sl][k])]
                pk[n] = D['d'][sl][i]
        T[f'{which}_pkval'] = pk
    T['z500_has_cut'] = (T['z500_cutoff'] > 0).astype(float)
    return T


def auc(x1, x0):
    x1, x0 = x1[np.isfinite(x1)], x0[np.isfinite(x0)]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan
    u = stats.mannwhitneyu(x1, x0).statistic
    return u / (len(x1) * len(x0))


def analyse(T, cols, tag):
    L = []
    L.append(f'\n=== {tag}: {T[T.blk == 1].eid.nunique()} events, {int((T.blk == 1).sum())} block days, '
             f'{int((T.blk == 0).sum())} control days ===')
    L.append(f'{"stratum":10} {"descriptor":10} {"block med":>10} {"ctrl med":>10} {"ev dmed":>9} '
             f'{"Wilcoxon p":>11} {"AUC":>6} {"AUC|amp":>8} {"beta|amp [95% CI]":>24}')
    strata = [('all', T)] + [(s, T[T.sector == s]) for s in ('EA', 'US', 'PAC')] + \
             [(s, T[T.season == s]) for s in SEASONS]
    rng = np.random.default_rng(1)
    for sname, S in strata:
        if S.blk.sum() < 50:
            continue
        ev = S.groupby(['eid', 'blk'])[cols].mean().unstack('blk')
        dec = pd.qcut(S['amp'].rank(method='first'), 10, labels=False)
        for c in cols:
            x1, x0 = S.loc[S.blk == 1, c].values, S.loc[S.blk == 0, c].values
            dif = (ev[(c, 1)] - ev[(c, 0)]).dropna()
            p = stats.wilcoxon(dif).pvalue if len(dif) > 10 and (dif != 0).any() else np.nan
            a = auc(x1, x0)
            # within-amplitude-decile AUC (weighted by stratum size)
            aa, ww = [], []
            for q in range(10):
                m = dec == q
                s1, s0 = S.loc[m & (S.blk == 1), c].values, S.loc[m & (S.blk == 0), c].values
                if len(s1) > 20 and len(s0) > 20:
                    aa.append(auc(s1, s0))
                    ww.append(min(len(s1), len(s0)))
            a_amp = np.average(aa, weights=ww) if aa else np.nan
            beta, lo, hi = logit_beta(S, c, rng) if c != 'amp' else (np.nan, np.nan, np.nan)
            L.append(f'{sname:10} {c:10} {np.nanmedian(x1):10.3g} {np.nanmedian(x0):10.3g} '
                     f'{np.median(dif):+9.3g} {p:11.2e} {a:6.3f} {a_amp:8.3f} '
                     f'{beta:+8.3f} [{lo:+.3f},{hi:+.3f}]')
    return L


def logit_beta(S, c, rng, nb=200):
    """Standardised coefficient of c in logit(blk) ~ amp + c; event-block bootstrap CI."""
    from scipy.optimize import minimize
    S = S[['eid', 'blk', 'amp', c]].dropna()
    if S[c].std() == 0 or len(S) < 200:
        return np.nan, np.nan, np.nan
    Xa = (S['amp'] - S['amp'].mean()) / S['amp'].std()
    Xc = (S[c] - S[c].mean()) / S[c].std()
    X = np.stack([np.ones(len(S)), Xa.values, Xc.values], 1)
    y = S['blk'].values.astype(float)

    def fit(X, y):
        def nll(b):
            z = X @ b
            return np.sum(np.logaddexp(0, z) - y * z)

        def grad(b):
            p = 1 / (1 + np.exp(-(X @ b)))
            return X.T @ (p - y)
        return minimize(nll, np.zeros(X.shape[1]), jac=grad, method='BFGS').x
    b = fit(X, y)[2]
    eids = S['eid'].values
    ue = np.unique(eids)
    idx = {e: np.where(eids == e)[0] for e in ue}
    bs = []
    for _ in range(nb):
        pick = rng.choice(ue, len(ue), replace=True)
        ii = np.concatenate([idx[e] for e in pick])
        bs.append(fit(X[ii], y[ii])[2])
    return b, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    path = os.path.join(CACHE, f'composite_table_k{a.k}.pkl')
    if os.path.exists(path):
        T = pd.read_pickle(path)
        if 'z500_pkval' not in T:
            T = add_pkval(T)
            T.to_pickle(path)
    else:
        T = build_table(a.k)
        print(f'table: {len(T)} rows [{time.time() - t0:.0f}s]', flush=True)
        for which in ('slp', 'z500'):
            Dx = run_descriptors(T, which)
            T = T.join(Dx.add_prefix(f'{which}_'))
        T = add_pkval(T)
        T.to_pickle(path)
    cols_s = [f'slp_{c}' for c in ('depth', 'pkval', 'amp', 'basin', 'nested', 'nestdepth', 'tp_loc', 'n_loc',
                                   'dw_loc', 'tp_hemi', 'dw_hemi')]
    cols_z = [f'z500_{c}' for c in ('depth', 'pkval', 'amp', 'basin', 'nested', 'tp_loc', 'n_loc',
                                    'dw_loc', 'tp_hemi', 'dw_hemi', 'cutoff', 'has_cut')]
    L = []
    for which, cols in (('SLP', cols_s), ('Z500', cols_z)):
        S = T.rename(columns={c: c.split('_', 1)[1] for c in cols})
        L += analyse(S, [c.split('_', 1)[1] for c in cols], which)
    txt = '\n'.join(L)
    print(txt)
    open(os.path.join(OUT, 'composites.txt'), 'w').write(txt + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [('slp_depth', 'SLP depth (hPa)'), ('slp_amp', 'SLP amp (hPa)'), ('slp_basin', 'SLP basin (1e6 km2)'),
            ('slp_dw_loc', 'SLP local d_W'), ('slp_tp_hemi', 'SLP hemispheric TP1'),
            ('slp_dw_hemi', 'SLP hemispheric d_W'),
            ('z500_depth', 'Z500a depth (gpm)'), ('z500_basin', 'Z500a basin (1e6 km2)'),
            ('z500_cutoff', 'Z500 cut-off prominence (gpm)'), ('z500_dw_loc', 'Z500a local d_W'),
            ('z500_nested', 'Z500a nested (0/1)'), ('z500_dw_hemi', 'Z500a hemispheric d_W')]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, (c, lab) in zip(axes.ravel(), show):
        for s, col in (('EA', 'C0'), ('US', 'C1'), ('PAC', 'C2')):
            S = T[T.sector == s]
            x1, x0 = S.loc[S.blk == 1, c].dropna(), S.loc[S.blk == 0, c].dropna()
            hi = np.nanpercentile(T[c], 99)
            bins = np.linspace(np.nanmin(T[c]), hi, 40)
            ax.hist(x1, bins, histtype='step', color=col, density=True, lw=1.5, label=f'{s} block')
            ax.hist(x0, bins, histtype='step', color=col, density=True, ls='--', lw=1, label=f'{s} ctrl')
        ax.set_title(lab, fontsize=9)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('TM-defined block days (solid) vs season/location-matched non-blocked days (dashed), 1979-2023')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'composites.png'), dpi=110)
    print(f'done [{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
