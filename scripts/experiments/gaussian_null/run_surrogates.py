"""
Build surrogates for one period, project them through the pipeline and store
their persistence features (depth >= 2 hPa) plus daily aggregates over all
features (count, TP1, TP2, max depth).  Also stores field diagnostics.

    python run_surrogates.py --real                       # real field, 1948-2023
    python run_surrogates.py --period 1979 2023 --kinds ft iaaft --reals 0 1 2 3
    python run_surrogates.py --period 1948 1978 --kinds ft iaaft --reals 0 1 2
    python run_surrogates.py --period 1979 2023 --kinds ftg --reals 0   # no seasonal windows (check)

Seeds: realisation r of period (y0, y1) uses seed 1000*y0 + r (same phases for
'ft' and the 'iaaft' start).  Memory ~1.5 GB for 1979-2023; one worker.
"""
import argparse
import json
import os
import time

import numpy as np

from common import DATA, PERIODS, REAL, START_YEAR, build_anomalies, field_features
from surrogates import Period, MONTH_OF_DOY

LAT = np.arange(90, -0.1, -2.5)


def diagnostics(field):
    """Cheap second-order and marginal diagnostics of a (T, 37, 144) field (Pa)."""
    T = field.shape[0]
    doy = np.tile(np.arange(365), T // 365)
    w = np.cos(np.radians(LAT))[:, None] * np.ones((1, 144))
    nh = LAT >= 20
    wn = w[nh] / w[nh].sum()
    f = field[:, nh, :].astype(np.float64)
    mu = f.mean(0)
    d = f - mu
    sd = np.sqrt((d ** 2).mean(0))
    skew = (d ** 3).mean(0) / sd ** 3
    djf = np.isin(MONTH_OF_DOY[doy], [11, 0, 1])
    jja = np.isin(MONTH_OF_DOY[doy], [5, 6, 7])
    r1 = (d[1:] * d[:-1]).mean(0) / sd ** 2
    r3 = (d[3:] * d[:-3]).mean(0) / sd ** 2
    i50 = np.where(LAT[nh] == 50)[0][0]
    zc = d[:, i50] / sd[i50]
    cz = [(zc * np.roll(zc, -k, axis=1)).mean() for k in (2, 4, 8)]
    return dict(skew_20_90=float((wn * skew).sum()), frac_neg_skew=float((wn * (skew < 0)).sum()),
                sd_djf_hPa=float(np.sqrt((wn * (d[djf] ** 2).mean(0)).sum()) / 100),
                sd_jja_hPa=float(np.sqrt((wn * (d[jja] ** 2).mean(0)).sum()) / 100),
                acf1=float((wn * r1).sum()), acf3=float((wn * r3).sum()),
                zonal_corr_50N_5_10_20deg=[float(c) for c in cz])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--period', nargs=2, type=int, default=[1979, 2023])
    ap.add_argument('--kinds', nargs='*', default=['ft', 'iaaft'])
    ap.add_argument('--reals', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--iters', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    anom, _, _ = build_anomalies()
    if a.real:
        path = os.path.join(DATA, f'feat_{REAL}_' + '{tag}.npz')
        field_features(anom, 0, path)
        diag = {}
        for y0, y1 in PERIODS:
            diag[f'{y0}_{y1}'] = diagnostics(anom[(y0 - START_YEAR) * 365:(y1 - START_YEAR + 1) * 365])
        json.dump(diag, open(os.path.join(DATA, 'diag_real.json'), 'w'), indent=1)
        log(f'real done in {time.time() - t0:.0f}s: {diag}')
        return
    y0, y1 = a.period
    s0, s1 = (y0 - START_YEAR) * 365, (y1 - START_YEAR + 1) * 365
    seg = anom[s0:s1].copy()
    del anom
    P = Period(seg, windows='ftg' not in a.kinds)
    del seg
    log(f'period {y0}-{y1} prepared in {time.time() - t0:.0f}s')
    for r in a.reals:
        for kind in a.kinds:
            if os.path.exists(os.path.join(DATA, f'diag_{kind}_{y0}_{y1}_r{r}.json')):
                log(f'  {kind} r{r} exists, skipped')
                continue
            t1 = time.time()
            rng = np.random.default_rng(1000 * y0 + r)
            info = {}
            if kind in ('ft', 'ftg'):          # ftg: whole-record standardisation (check)
                S = P.ft(rng)
            else:
                S, errs = P.iaaft(rng, n_iter=a.iters, log=log)
                info['spectral_error'] = errs
            info.update(diagnostics(S))
            tag = f'{kind}_{y0}_{y1}_r{r}'
            if r == 0 and y0 == 1979:
                np.save(os.path.join(DATA, f'sample_{tag}.npy'), S[(2000 - y0) * 365:(2000 - y0) * 365 + 31])
            t2 = time.time()
            field_features(S, s0, os.path.join(DATA, f'feat_{tag}_' + '{tag}.npz'))
            del S
            json.dump(info, open(os.path.join(DATA, f'diag_{tag}.json'), 'w'), indent=1)
            log(f'  {tag}: surrogate {t2 - t1:.0f}s, features {time.time() - t2:.0f}s; '
                f'skew {info["skew_20_90"]:+.3f} sd DJF {info["sd_djf_hPa"]:.2f} JJA {info["sd_jja_hPa"]:.2f} '
                f'acf1 {info["acf1"]:.3f}')


if __name__ == '__main__':
    main()
