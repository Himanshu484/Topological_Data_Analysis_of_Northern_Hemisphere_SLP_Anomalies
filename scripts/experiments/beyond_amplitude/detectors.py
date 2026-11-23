"""
Test 1: detector comparison at matched detection density.

Detectors (all select rows of the same strict-extremum table):
    P   topological depth (prominence)            -- the paper's
    A   amplitude |extremum anomaly|
    AN  amplitude, keeping only extrema that are the strongest within 1000 km
    L   annulus depth: extremum minus the mean on the 500-1000 km ring
    C   closed-contour depth limited to R <= 1200 km (Wernli-Schwierz-like)

Each detector's threshold is set so that it gives the same number of
detections as P at 10 hPa (and at 5 hPa for the lower level of two-level
tracking).  All tracking is gated nearest-neighbour, R = 1000 km (and 500 km).

    python detectors.py [--type sup|sub]
"""
import argparse
import json
import os
import time

import numpy as np

from common import (table, scores, calibrate, nn_track, chance_links, track_metrics, gc, hav,
                    NAME, OUTD, DATA, NSLOT)

HEAT = {'Indus/Thar 24-34N 62-78E': (24, 34, 62, 78), 'Tarim 36-43N 75-92E': (36, 43, 75, 92),
        'Iran/Afghan 28-38N 50-68E': (28, 38, 50, 68), 'Sahara 18-32N 10W-30E': (18, 32, -10, 30),
        'Arabia 16-30N 40-58E': (16, 30, 40, 58), 'SW US/Mexico 25-37N 120-100W': (25, 37, -120, -100)}


def in_box(lat, lon, box):
    la0, la1, lo0, lo1 = box
    lo = ((lon + 180) % 360) - 180
    return (lat >= la0) & (lat <= la1) & (lo >= lo0) & (lo <= lo1)


def month(T):
    import datetime
    base = datetime.date(2001, 1, 1)
    m = np.array([(base + datetime.timedelta(int(d) - 1)).month for d in range(1, 366)])
    return m[T['day'] - 1]


def redundancy(T, mask, r=500.0):
    rows = np.where(mask)[0]
    s = T['slot'][rows]
    st = np.searchsorted(s, np.arange(NSLOT))
    en = np.searchsorted(s, np.arange(NSLOT), 'right')
    k = 0
    for t in range(NSLOT):
        a, b = st[t], en[t]
        if b - a < 2:
            continue
        rr = rows[a:b]
        D = hav(T['plat'][rr], T['plon'][rr], T['plat'][rr], T['plon'][rr])
        np.fill_diagonal(D, np.inf)
        k += int((D.min(1) <= r).sum())
    return 100 * k / len(rows)


def stationary(T, tr, mon, Lmin=10, dmax=1000.0):
    """Long quasi-stationary tracks: >= Lmin days, never > dmax from the first position."""
    out = []
    for m in tr['tracks']:
        if len(m) < Lmin:
            continue
        d = gc(T['plat'][m[0]], T['plon'][m[0]], T['plat'][m], T['plon'][m])
        if d.max() <= dmax:
            out.append(m)
    n = len(out)
    summer = sum(4 <= mon[m[0]] <= 8 for m in out)
    lat = np.array([T['plat'][m].mean() for m in out]) if n else np.zeros(0)
    return dict(n=n, pct_MJJA_start=100 * summer / max(n, 1),
                pct_lat_lt45=100 * np.mean(lat < 45) if n else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='sup')
    ap.add_argument('--curve', action='store_true', help='also coherence-vs-density curve')
    a = ap.parse_args()
    t = a.type
    T = table(t)
    mon = month(T)
    S = scores(T)
    N10 = int((T['depth'] >= 10).sum())
    N5 = int((T['depth'] >= 5).sum())
    res = {}
    saved = {}
    print(f'== {NAME[t]}: matched densities N(10 hPa) = {N10} ({N10 / NSLOT:.2f}/day), '
          f'N(5 hPa) = {N5} ({N5 / NSLOT:.2f}/day)', flush=True)
    P10 = T['depth'] >= 10
    summer_heat = np.zeros(len(P10), bool)
    for box in HEAT.values():
        summer_heat |= in_box(T['plat'], T['plon'], box)
    summer_heat &= (mon >= 5) & (mon <= 8)
    for name, sc in S.items():
        t0 = time.time()
        key = name.split()[0]
        c10, c5 = calibrate(sc, N10), calibrate(sc, N5)
        hi, lo = sc >= c10, sc >= c5
        r = dict(thr10=float(c10), thr5=float(c5), n=int(hi.sum()),
                 jaccard_P=float((hi & P10).sum() / (hi | P10).sum()),
                 depth_med=float(np.median(T['depth'][hi])), amp_med=float(np.median(T['amp'][hi])),
                 pct_nested=100 * float(np.mean(T['rim'][hi] == 0)),
                 redund500=redundancy(T, hi))
        for R in (1000.0, 500.0):
            tr = nn_track(T, hi, R=R)
            m = track_metrics(T, tr)
            nl = int((tr['succ'] >= 0).sum())
            ch = chance_links(T, hi, R=R)
            if R == 1000.0:
                saved[key + '_L1'] = tr['L'].astype(np.int32)
            r[f'single_R{int(R)}'] = m
            r[f'single_R{int(R)}']['links'] = nl
            r[f'single_R{int(R)}']['chance_links'] = ch
            r[f'single_R{int(R)}']['excess_link_rate'] = 100 * (nl - ch) / hi.sum()
        tr2 = nn_track(T, lo, hi_mask=hi)
        r['two_level'] = track_metrics(T, tr2, det_mask=hi)
        r['two_level']['pct_hi_kept_in_ge2'] = 100 * float(np.mean(tr2['L'][hi] >= 2))
        saved[key + '_L'] = tr2['L'].astype(np.int32)
        saved[key + '_tid'] = tr2['tid'].astype(np.int32)
        saved[key + '_succ'] = tr2['succ'].astype(np.int32)
        saved[key + '_pred'] = tr2['pred'].astype(np.int32)
        saved[key + '_age'] = tr2['age'].astype(np.int32)
        saved[key + '_hi'] = hi
        # spatial realism
        r['pct_hgt_gt1500'] = 100 * float(np.mean(T['hgt'][hi] > 1500))
        r['pct_hgt_gt1500_in_ge5d'] = 100 * float(np.mean(T['hgt'][hi & (tr2['L'] >= 5)] > 1500))
        r['pct_summer_heatlow_boxes'] = 100 * float(np.mean(summer_heat[hi]))
        r['pct_summer_heatlow_in_ge5d'] = 100 * float(np.mean(summer_heat[hi & (tr2['L'] >= 5)]))
        r['pct_lat_lt25'] = 100 * float(np.mean(T['plat'][hi] < 25))
        r['stationary_ge10d'] = stationary(T, tr2, mon)
        djf = np.isin(mon, [12, 1, 2])
        jja = np.isin(mon, [6, 7, 8])
        r['DJF_over_JJA'] = float((hi & djf).sum() / max((hi & jja).sum(), 1))
        res[key] = r
        s1, s2 = r['single_R1000'], r['two_level']
        print(f'{name:42s} thr10 {c10:6.2f} thr5 {c5:6.2f} | J(P) {r["jaccard_P"]:.2f} | '
              f'1-level: 1-day {s1["pct1"]:.1f}% mean {s1["meanL"]:.2f} d, in>=3d {s1["ge3"]:.1f}%, '
              f'excess link {s1["excess_link_rate"]:.1f}% | 2-level: 1-day {s2["pct1"]:.1f}% '
              f'mean {s2["meanL"]:.2f} in>=3d {s2["ge3"]:.1f}% >=5d {s2["ge5"]} | redund {r["redund500"]:.1f}% | '
              f'hgt>1.5km {r["pct_hgt_gt1500"]:.1f}% heat-low {r["pct_summer_heatlow_boxes"]:.1f}% | '
              f'stat>=10d {r["stationary_ge10d"]["n"]} ({time.time() - t0:.0f}s)', flush=True)
    # year-block bootstrap of the share of detections in >= 3-day tracks, difference to P
    rng = np.random.default_rng(0)
    yr = T['year'] - T['year'].min()
    ny = yr.max() + 1
    boot = {}
    for lev in ['_L1', '_L']:
        cnt = {}
        for key in res:
            hi = saved[key + '_hi']
            cnt[key] = (np.bincount(yr[hi], minlength=ny), np.bincount(yr[hi & (saved[key + lev] >= 3)], minlength=ny))
        W = rng.multinomial(ny, np.ones(ny) / ny, size=1000)
        fP = (W @ cnt['P'][1]) / (W @ cnt['P'][0])
        for key in res:
            f = (W @ cnt[key][1]) / (W @ cnt[key][0])
            d = 100 * (f - fP)
            boot[f'{key}{lev}'] = [float(100 * cnt[key][1].sum() / cnt[key][0].sum() - 100 * cnt['P'][1].sum() / cnt['P'][0].sum()),
                                   float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            print(f'   {"single" if lev == "_L1" else "two-level"} {key:3s} in>=3d minus P: {boot[f"{key}{lev}"][0]:+.2f} '
                  f'[{boot[f"{key}{lev}"][1]:+.2f}, {boot[f"{key}{lev}"][2]:+.2f}] pp', flush=True)
    res['boot_ge3_minus_P'] = boot
    # by season (single level)
    for key in list(res):
        if key in ('boot_ge3_minus_P',):
            continue
        hi = saved[key + '_hi']
        res[key]['ge3_DJF'] = 100 * float(np.mean(saved[key + '_L1'][hi & np.isin(mon, [12, 1, 2])] >= 3))
        res[key]['ge3_JJA'] = 100 * float(np.mean(saved[key + '_L1'][hi & np.isin(mon, [6, 7, 8])] >= 3))
        print(f'   {key:3s} in>=3d DJF {res[key]["ge3_DJF"]:.1f}% JJA {res[key]["ge3_JJA"]:.1f}%', flush=True)
    if a.curve:
        curve = {}
        for tau in [3, 5, 7, 10, 15, 20]:
            n = int((T['depth'] >= tau).sum())
            for name, sc in S.items():
                key = name.split()[0]
                hi = sc >= calibrate(sc, n)
                tr = nn_track(T, hi)
                m = track_metrics(T, tr)
                m['links'] = int((tr['succ'] >= 0).sum())
                m['chance_links'] = chance_links(T, hi)
                m['excess_link_rate'] = 100 * (m['links'] - m['chance_links']) / n
                curve[f'{key}_{tau}'] = m
                print(f'   curve tau {tau:2d} {key:3s} n {n} 1-day {m["pct1"]:.1f}% in>=3d {m["ge3"]:.1f}% '
                      f'excess {m["excess_link_rate"]:.1f}%', flush=True)
        res['curve'] = curve
    with open(os.path.join(OUTD, f'detectors_{t}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    np.savez_compressed(os.path.join(DATA, f'tracks_{t}.npz'), **saved)


if __name__ == '__main__':
    main()
