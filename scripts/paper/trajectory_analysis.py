"""
Trajectory and climatology statistics of the manuscript, computed with the lifted Wasserstein
tracker of src/lifted_tracker.py from the feature files of the Gaussian-null
pipeline (scripts/experiments/gaussian_null/run_surrogates.py).

    python scripts/paper/trajectory_analysis.py                    # NCEP-NCAR R1, 1979-2023
    GN_SOURCE=era5 python scripts/paper/trajectory_analysis.py     # ERA5, 1979-2022

Writes
    output/paper/analysis_{R1,ERA5}.json     every number used in the text and tables
    data/processed_data/paper/tracks_{R1,ERA5}_{C,A}.npz   one row per tracked snapshot

Reference tracker: features of depth >= 5 hPa are linked, tracks reaching
10 hPa are kept, linking radius 1000 km, lam = 0.01 hPa/km (a displacement of
the whole radius costs as much as the reporting depth).
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'experiments', 'gaussian_null'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'prototype'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from analyse import load, W  # noqa: E402
from common import MAIN_PERIOD, REAL, SOURCE, START_YEAR  # noqa: E402
from tracking import pix2ll  # noqa: E402
import lifted_tracker as lt  # noqa: E402

SRC = 'R1' if SOURCE == 'r1' else 'ERA5'
OUT = os.path.join(ROOT, 'output', 'paper')
DAT = os.path.join(ROOT, 'data', 'processed_data', 'paper')
os.makedirs(OUT, exist_ok=True)
os.makedirs(DAT, exist_ok=True)
TAG = {'C': 'sup', 'A': 'sub'}
REF = dict(tau_lo=5.0, tau_hi=10.0, R=1000.0, lam=0.01)
CONFIGS = {'reference': REF,
           'R = 500 km': dict(REF, R=500.0), 'R = 750 km': dict(REF, R=750.0),
           'R = 1250 km': dict(REF, R=1250.0), 'R = 1500 km': dict(REF, R=1500.0),
           'lam = 0.005': dict(REF, lam=0.005), 'lam = 0.02': dict(REF, lam=0.02),
           'thresholds 10/10': dict(REF, tau_lo=10.0), 'thresholds 3/10': dict(REF, tau_lo=3.0),
           'thresholds 7.5/15': dict(REF, tau_lo=7.5, tau_hi=15.0),
           'thresholds 10/20': dict(REF, tau_lo=10.0, tau_hi=20.0)}
MONTH_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_OF_DOY = np.repeat(np.arange(12), MONTH_LEN)
DOM_OF_DOY = np.concatenate([np.arange(1, n + 1) for n in MONTH_LEN])
NBOOT = 1000


def date(slot):
    y, d = START_YEAR + int(slot) // 365, int(slot) % 365
    return f'{y}-{MONTH_OF_DOY[d] + 1:02d}-{DOM_OF_DOY[d]:02d}'


def features(name, c):
    """Features of one data set and sign, sorted by day, in tracker units (hPa, degrees)."""
    F = load(REAL, TAG[c], MAIN_PERIOD) if name == 'real' else load(name, TAG[c])
    o = np.lexsort((F['peak_idx'], F['slot']))
    lat, lon = pix2ll(*np.divmod(F['peak_idx'][o], W))
    sgn = -1.0 if c == 'C' else 1.0                      # the stored field is g = -a for lows
    return dict(day=(F['slot'][o] - F['slot0']).astype(int), slot=F['slot'][o].astype(int),
                birth=F['saddle'][o].astype(float) / 100, death=F['peak'][o].astype(float) / 100,
                lat=lat, lon=lon, depth=F['depth'][o], half=F['half_km2'][o].astype(float),
                anom=sgn * F['peak'][o].astype(float) / 100, nested=F['parent_idx'][o] >= 0,
                n_days=int(F['T']), agg=F['agg'])


def run(F, mode='lifted', **cfg):
    cfg = dict(REF, **cfg)
    tracks, keep, links = lt.track(F['day'], F['birth'], F['death'], F['lat'], F['lon'], F['n_days'], mode=mode, **cfg)
    s = lt.summary(tracks, links)
    s['tracks_per_year'] = s['n_tracks'] / (F['n_days'] / 365)
    return tracks, keep, links, s


def surrogate(name):
    out = {}
    for c in 'CA':
        F = features(name, c)
        out[c] = {'reference': {k: v for k, v in run(F)[3].items() if k != 'survival'}}
        out[c]['reference']['survival'] = run(F)[3]['survival']
        r = int(name.split('_r')[-1])
        if r < 5:
            for lab, cfg in CONFIGS.items():
                if lab != 'reference':
                    out[c][lab] = {k: v for k, v in run(F, **cfg)[3].items() if k != 'survival'}
    return name, out


def boot_years(per_year, fn, rng):
    """95 % interval of fn(sum over resampled years) for an array (n_years, ...)."""
    n = per_year.shape[0]
    v = np.array([fn(per_year[rng.integers(0, n, n)].sum(0)) for _ in range(NBOOT)])
    return np.nanpercentile(v, [2.5, 97.5], axis=0)


def real_analysis():
    rng = np.random.default_rng(20261123)
    res, tab = {}, {}
    ny = (MAIN_PERIOD[1] - MAIN_PERIOD[0] + 1)
    for c in 'CA':
        F = features('real', c)
        tracks, keep, links, s = run(F)
        r = {'reference': s}
        # agreement with nearest-neighbour assignment
        _, _, lnn, snn = run(F, mode='nn')
        a, b = links['succ'], lnn['succ']
        r['nn'] = dict(snn, lifted_links_that_are_nn=float(np.mean(a[a >= 0] == b[a >= 0])),
                       nn_links_kept_by_lifted=float(np.mean(a[b >= 0] == b[b >= 0])))
        for lab, cfg in CONFIGS.items():
            if lab != 'reference':
                r[lab] = {k: v for k, v in run(F, **cfg)[3].items() if k != 'survival'}
        # one row per track
        n = np.array([len(m) for m in tracks])
        first = np.array([m[0] for m in tracks])
        yr = F['day'][first] // 365
        mon = MONTH_OF_DOY[F['day'][first] % 365]
        mdepth = np.array([F['depth'][m].mean() for m in tracks])
        xdepth = np.array([F['depth'][m].max() for m in tracks])
        marea = np.array([F['half'][m].mean() for m in tracks]) / 1e6
        idx_kept = np.where(keep)[0]
        pos = {int(j): i for i, j in enumerate(idx_kept)}
        path = np.array([np.nansum(links['step'][[pos[int(j)] for j in m[1:]]]) if len(m) > 1 else 0.0 for m in tracks])
        r['per_track'] = dict(mean_depth=float(mdepth.mean()), mean_max_depth=float(xdepth.mean()),
                              mean_area=float(marea.mean()), median_area=float(np.median(marea)),
                              req_of_median_area_km=float(np.sqrt(np.median(marea) * 1e6 / np.pi)),
                              mean_path_km=float(path.mean()), max_path_km=float(path.max()),
                              mean_start_lat=float(F['lat'][first].mean()))
        # snapshots of depth >= 10 hPa: share that belongs to multi-day tracks
        in_multi = np.zeros(len(F['day']), bool)
        for m in tracks:
            if len(m) > 1:
                in_multi[m] = True
        deep = F['depth'] >= 10
        r['share_of_deep_snapshots_in_multiday_tracks'] = float(in_multi[deep].mean())
        r['n_deep_snapshots'] = int(deep.sum())
        # the longest tracks
        longest = []
        for k in np.argsort(-n)[:5]:
            m = tracks[k]
            longest.append(dict(length=int(n[k]), start=date(F['slot'][m[0]]), end=date(F['slot'][m[-1]]),
                                mean_lat=float(F['lat'][m].mean()),
                                mean_lon=float(np.degrees(np.arctan2(np.sin(np.radians(F['lon'][m])).mean(),
                                                                      np.cos(np.radians(F['lon'][m])).mean())) % 360),
                                max_depth=float(F['depth'][m].max())))
        r['longest'] = longest
        # monthly summaries by starting month, with year-bootstrap intervals
        S = np.zeros((ny, 12, 4))                         # count, sum length, sum mean depth, sum mean area
        np.add.at(S, (yr, mon, 0), 1.0)
        np.add.at(S, (yr, mon, 1), n)
        np.add.at(S, (yr, mon, 2), mdepth)
        np.add.at(S, (yr, mon, 3), marea)
        fns = {'count_per_year': lambda t: t[:, 0] / ny, 'mean_length': lambda t: t[:, 1] / t[:, 0],
               'mean_depth': lambda t: t[:, 2] / t[:, 0], 'mean_area': lambda t: t[:, 3] / t[:, 0]}
        r['monthly'] = {}
        for k, fn in fns.items():
            lo, hi = boot_years(S, fn, rng)
            r['monthly'][k] = dict(value=fn(S.sum(0)).tolist(), lo=lo.tolist(), hi=hi.tolist())
        # overall intervals (year bootstrap) of the headline track statistics
        Y = np.zeros((ny, 6))                             # count, length, 1 d, >=5 d, >=10 d, area
        np.add.at(Y, (yr, 0), 1.0); np.add.at(Y, (yr, 1), n); np.add.at(Y, (yr, 2), n == 1)
        np.add.at(Y, (yr, 3), n >= 5); np.add.at(Y, (yr, 4), n >= 10); np.add.at(Y, (yr, 5), marea)
        hf = {'tracks_per_year': lambda t: t[0] / ny, 'mean_length': lambda t: t[1] / t[0], 'pct_1d': lambda t: 100 * t[2] / t[0],
              'pct_ge5d': lambda t: 100 * t[3] / t[0], 'pct_ge10d': lambda t: 100 * t[4] / t[0], 'mean_area': lambda t: t[5] / t[0]}
        r['ci95'] = {k: boot_years(Y, fn, rng).tolist() for k, fn in hf.items()}
        # daily aggregates: total persistence climatology
        agg = F['agg'].reshape(ny, 365, 4)
        tp = {'N': agg[..., 0], 'TP1': agg[..., 1] / 100, 'TP2': agg[..., 2] / 1e4, 'max': agg[..., 3] / 100}
        r['climatology'] = {}
        for k, v in tp.items():
            mm = np.stack([v[:, MONTH_OF_DOY == mth].mean(1) for mth in range(12)], 1)          # (ny, 12)
            bs = np.array([mm[rng.integers(0, ny, ny)].mean(0) for _ in range(NBOOT)])
            r['climatology'][k] = dict(annual_mean=float(v.mean()), monthly_mean=mm.mean(0).tolist(),
                                       monthly_lo=np.percentile(bs, 2.5, 0).tolist(),
                                       monthly_hi=np.percentile(bs, 97.5, 0).tolist(),
                                       month_of_max_bootstrap_counts=np.bincount(bs.argmax(1), minlength=12).tolist(),
                                       month_of_min_bootstrap_counts=np.bincount(bs.argmin(1), minlength=12).tolist())
        res[c] = r
        tab[c] = dict(yr=yr, n=n, marea=marea)
        # snapshot table for the figures
        tid = -np.ones(len(F['day']), int)
        for i, m in enumerate(tracks):
            tid[m] = i
        k = tid >= 0
        np.savez_compressed(os.path.join(DAT, f'tracks_{SRC}_{c}.npz'), track=tid[k], slot=F['slot'][k], lat=F['lat'][k],
                            lon=F['lon'][k], depth=F['depth'][k], anom=F['anom'][k], saddle=np.where(c == 'C', -1, 1) * F['birth'][k],
                            half_km2=F['half'][k], nested=F['nested'][k])
    # highs against lows: probability that a track of a high has the larger mean footprint
    from scipy.stats import rankdata

    def auc(a, b):
        rk = rankdata(np.concatenate([a, b]))
        return (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    a0, c0 = tab['A'], tab['C']
    p = auc(a0['marea'], c0['marea'])
    bs = []
    for _ in range(300):
        ys = rng.integers(0, ny, ny)
        aa = np.concatenate([a0['marea'][a0['yr'] == y] for y in ys])
        cc = np.concatenate([c0['marea'][c0['yr'] == y] for y in ys])
        bs.append(auc(aa, cc))
    res['P_high_footprint_larger'] = dict(value=float(p), ci95=np.percentile(bs, [2.5, 97.5]).tolist())
    return res


def event_2012(res):
    """The high of late January - February 2012 over north-western Russia (R1 only)."""
    Z = np.load(os.path.join(DAT, f'tracks_{SRC}_A.npz'))
    s0, s1 = (2012 - START_YEAR) * 365 + 14, (2012 - START_YEAR) * 365 + 51       # 15 Jan - 20 Feb
    k = (Z['slot'] >= s0) & (Z['slot'] <= s1) & (Z['lat'] >= 50) & (Z['lat'] <= 80) & (Z['lon'] >= 0) & (Z['lon'] <= 90)
    out = []
    for t in np.unique(Z['track'][k]):
        m = Z['track'] == t
        if m.sum() < 3:
            continue
        o = np.argsort(Z['slot'][m])
        out.append(dict(track=int(t), length=int(m.sum()),
                        days=[dict(date=date(s), lat=float(a), lon=float(b), depth=float(d), anom=float(e), saddle=float(f),
                                   area_1e6km2=float(h / 1e6), nested=bool(g))
                              for s, a, b, d, e, f, h, g in zip(Z['slot'][m][o], Z['lat'][m][o], Z['lon'][m][o], Z['depth'][m][o],
                                                                Z['anom'][m][o], Z['saddle'][m][o], Z['half_km2'][m][o],
                                                                Z['nested'][m][o])]))
    res['event_2012'] = sorted(out, key=lambda d: d['days'][0]['date'])


def main():
    res = {'source': SRC, 'period': list(MAIN_PERIOD), 'reference': REF}
    res.update(real_analysis())
    if SRC == 'R1':
        event_2012(res)
    import glob
    from common import DATA
    names = sorted({os.path.basename(p)[5:-8] for p in glob.glob(os.path.join(DATA, f'feat_*_{MAIN_PERIOD[0]}_{MAIN_PERIOD[1]}_r*_sup.npz'))})
    names = [n for n in names if n.startswith(('ft_', 'iaaft_')) and os.path.exists(os.path.join(DATA, f'diag_{n}.json'))]
    with Pool(int(os.environ.get('GN_WORKERS', 4))) as pool:
        sur = dict(pool.imap_unordered(surrogate, names))
    res['surrogates'] = {}
    for kind in ('ft', 'iaaft'):
        ks = sorted(n for n in sur if n.startswith(kind + '_'))
        res['surrogates'][kind] = {'n': len(ks)}
        for c in 'CA':
            E = {}
            for lab in CONFIGS:
                use = [sur[n][c][lab] for n in ks if lab in sur[n][c]]
                E[lab] = {k: dict(mean=float(np.mean([u[k] for u in use])), min=float(np.min([u[k] for u in use])),
                                  max=float(np.max([u[k] for u in use])), n=len(use))
                          for k in use[0] if k != 'survival'}
                if lab == 'reference':
                    sv = np.array([u['survival'] for u in use])
                    E[lab]['survival'] = dict(mean=sv.mean(0).tolist(), min=sv.min(0).tolist(), max=sv.max(0).tolist())
            res['surrogates'][kind][c] = E
    json.dump(res, open(os.path.join(OUT, f'analysis_{SRC}.json'), 'w'), indent=1, default=float)
    for c in 'CA':
        r = res[c]['reference']
        print(SRC, c, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'survival'})
        print('   nn agreement', round(res[c]['nn']['lifted_links_that_are_nn'], 4), round(res[c]['nn']['nn_links_kept_by_lifted'], 4))
        print('   per track', {k: round(v, 2) for k, v in res[c]['per_track'].items()})
        print('   longest', res[c]['longest'][:3])
        for kind in ('ft', 'iaaft'):
            e = res['surrogates'][kind][c]['reference']
            print('   ', kind, {k: round(e[k]['mean'], 2) for k in ('mean_length', 'pct_1d', 'pct_ge5d', 'pct_ge10d', 'median_step_km', 'tracks_per_year')})
    print('P(high footprint larger)', res['P_high_footprint_larger'])
    if 'event_2012' in res:
        for t in res['event_2012']:
            print('2012 track', t['track'], t['length'], [(d['date'][5:], round(d['lat']), round(d['lon']), round(d['depth'], 1)) for d in t['days']])


if __name__ == '__main__':
    main()
