"""
Test 1b: as detectors.py, but each detector's threshold is calibrated separately for
every calendar month, so that all detectors have the same number of detections as
prominence >= 10 hPa (and >= 5 hPa for the lower level) in every month.  This removes
the seasonal-density difference between amplitude and prominence thresholds.

    python detectors_monthly.py --type sup
"""
import argparse
import json
import os

import numpy as np

from common import table, scores, calibrate, nn_track, track_metrics, NAME, OUTD
from detectors import month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='sup')
    t = ap.parse_args().type
    T = table(t)
    mon = month(T)
    S = scores(T)
    res = {}
    yr = T['year'] - T['year'].min()
    ny = yr.max() + 1
    rng = np.random.default_rng(0)
    Wb = rng.multinomial(ny, np.ones(ny) / ny, size=1000)
    cnt = {}
    print(f'== {NAME[t]}: month-by-month matched density')
    for name, sc in S.items():
        key = name.split()[0]
        hi = np.zeros(len(sc), bool)
        lo = np.zeros(len(sc), bool)
        for m in range(1, 13):
            k = mon == m
            n10 = int((T['depth'][k] >= 10).sum())
            n5 = int((T['depth'][k] >= 5).sum())
            hi[k] = sc[k] >= calibrate(sc[k], n10)
            lo[k] = sc[k] >= calibrate(sc[k], n5)
        tr1 = nn_track(T, hi)
        tr2 = nn_track(T, lo, hi_mask=hi)
        m1, m2 = track_metrics(T, tr1), track_metrics(T, tr2, det_mask=hi)
        cnt[key] = [(np.bincount(yr[hi], minlength=ny), np.bincount(yr[hi & (tr[ 'L'] >= 3)], minlength=ny))
                    for tr in (tr1, tr2)]
        res[key] = dict(n=int(hi.sum()), single=m1, two_level=m2,
                        pct_hgt_gt1500=100 * float(np.mean(T['hgt'][hi] > 1500)),
                        ge3_DJF=100 * float(np.mean(tr1['L'][hi & np.isin(mon, [12, 1, 2])] >= 3)),
                        ge3_JJA=100 * float(np.mean(tr1['L'][hi & np.isin(mon, [6, 7, 8])] >= 3)))
    for key in res:
        for n, lev in enumerate(['single', 'two_level']):
            fP = (Wb @ cnt['P'][n][1]) / (Wb @ cnt['P'][n][0])
            f = (Wb @ cnt[key][n][1]) / (Wb @ cnt[key][n][0])
            d = 100 * (f - fP)
            res[key][lev]['ge3_minus_P'] = [res[key][lev]['ge3'] - res['P'][lev]['ge3'],
                                            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        r = res[key]
        print(f'  {key:3s} single: 1-day {r["single"]["pct1"]:.1f}% in>=3d {r["single"]["ge3"]:.1f}% '
              f'(vs P {r["single"]["ge3_minus_P"][0]:+.2f} [{r["single"]["ge3_minus_P"][1]:+.2f},{r["single"]["ge3_minus_P"][2]:+.2f}]) '
              f'DJF {r["ge3_DJF"]:.1f} JJA {r["ge3_JJA"]:.1f} | two-level: 1-day {r["two_level"]["pct1"]:.1f}% '
              f'in>=3d {r["two_level"]["ge3"]:.1f}% (vs P {r["two_level"]["ge3_minus_P"][0]:+.2f} '
              f'[{r["two_level"]["ge3_minus_P"][1]:+.2f},{r["two_level"]["ge3_minus_P"][2]:+.2f}]) '
              f'>=5d {r["two_level"]["ge5"]} | hgt>1.5 km {r["pct_hgt_gt1500"]:.1f}%', flush=True)
    with open(os.path.join(OUTD, f'detectors_monthly_{t}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
