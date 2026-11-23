"""
Test 3: lifecycle phenomenology of prominence (P) and amplitude (A) tracks.

Both are two-level NN tracks at matched density (P: 5/10 hPa depth; A: amplitude
thresholds with the same numbers of detections), from detectors.py.

  (a) How do tracks end?  For the last feature i of a track (day s), look on day
      s+1 at the strongest same-type extremum within 800 km of i ("remnant"):
        P ends:  remnant still P-detectable (depth >= 5)       -> detected, not linked
                 depth < 5 but amplitude >= a5                -> prominence lost, amplitude kept
                 otherwise                                    -> both decayed
                 plus: remnant belongs to a continuing P track -> absorbed by a neighbour
        A ends:  symmetric (amplitude lost, prominence kept, ...)
      Chance: the same test at the same latitude, longitude rotated by U(60, 300) deg.
  (b) Composites of depth, amplitude and depth/amplitude from genesis and to lysis.
  (c) Lifetimes of the same systems under the two detectors (tracks sharing rows).

    python lifecycle.py --type sup
"""
import argparse
import json
import os

import numpy as np

from common import table, gc, NAME, OUTD, DATA, NSLOT


def remnant(T, s2, lat, lon, starts, ends, r=800.0):
    a, b = starts[s2], ends[s2]
    if b <= a:
        return -1
    d = gc(lat, lon, T['plat'][a:b], T['plon'][a:b])
    k = np.where(d <= r)[0]
    if len(k) == 0:
        return -1
    return a + k[np.argmax(T['amp'][a + k])]


G = {}


def classify_ends(T, S, key, a5, rng, nrot=3):
    tid, succ, pred = S[key + '_tid'], S[key + '_succ'], S[key + '_pred']
    starts = np.searchsorted(T['slot'], np.arange(NSLOT))
    ends = np.searchsorted(T['slot'], np.arange(NSLOT), 'right')
    last = np.where((tid >= 0) & (succ < 0) & (T['slot'] < NSLOT - 1))[0]
    cont = (tid >= 0) & (pred >= 0)          # a feature of a track that already existed the day before

    def flags(i, lon):
        j = remnant(T, T['slot'][i] + 1, T['plat'][i], lon, starts, ends)
        if j < 0:
            return (0, 0, 0, 1, 0)
        detP, detA = T['depth'][j] >= 5, T['amp'][j] >= a5
        if key == 'P':
            det, lost = detP, (not detP) and detA
        else:
            det, lost = detA, (not detA) and detP
        return (int(det), int(lost), int((not det) and (not lost)), 0, int(cont[j]))
    act = np.array([flags(i, T['plon'][i]) for i in last])
    # character of the remnants whose own criterion was lost while the other one was kept
    rem = np.array([remnant(T, T['slot'][i] + 1, T['plat'][i], T['plon'][i], starts, ends) for i in last])
    lost = (act[:, 1] == 1) & (rem >= 0)
    j = rem[lost]
    G['lost_remnant'] = dict(n=int(lost.sum()), pct_nested=100 * float(np.mean(T['rim'][j] == 0)),
                             median_depth=float(np.median(T['depth'][j])), median_amp=float(np.median(T['amp'][j])),
                             median_parent_km=float(np.nanmedian(T['par_km'][j])),
                             median_parent_depth=float(np.nanmedian(T['par_depth'][j])),
                             median_dist_to_last_km=float(np.median(gc(T['plat'][last[lost]], T['plon'][last[lost]],
                                                                       T['plat'][j], T['plon'][j]))))
    rot = np.array([flags(i, (T['plon'][i] + rng.uniform(60, 300)) % 360) for i in last for _ in range(nrot)])
    labs = ['detected, not linked', 'lost own criterion, other kept', 'both below', 'no extremum within 800 km',
            'remnant is in a continuing track (absorbed)']
    return {l: [100 * float(act[:, n].mean()), 100 * float(rot[:, n].mean())] for n, l in enumerate(labs)}, last


def composites(T, S, key, Lmin=5):
    tr = S[key + '_tid']
    age, L = S[key + '_age'], S[key + '_L']
    rows = np.where((tr >= 0) & (L >= Lmin))[0]
    out = {}
    for lab, rel in [('from genesis', age[rows]), ('to lysis', age[rows] - L[rows] + 1)]:
        out[lab] = {}
        for k in (range(0, 4) if lab == 'from genesis' else range(-3, 1)):
            m = rows[rel == k]
            out[lab][int(k)] = dict(depth=float(np.median(T['depth'][m])), amp=float(np.median(T['amp'][m])),
                                    ratio=float(np.median(T['depth'][m] / np.maximum(T['amp'][m], 0.1))),
                                    nested=100 * float(np.mean(T['rim'][m] == 0)))
    m = rows[(age[rows] >= 2) & (age[rows] <= L[rows] - 3)]
    out['mid-life'] = dict(depth=float(np.median(T['depth'][m])), amp=float(np.median(T['amp'][m])),
                           ratio=float(np.median(T['depth'][m] / np.maximum(T['amp'][m], 0.1))),
                           nested=100 * float(np.mean(T['rim'][m] == 0)))
    return out


def lifetimes(T, S):
    tP, tA = S['P_tid'], S['A_tid']
    LP, LA = S['P_L'], S['A_L']
    both = np.where((tP >= 0) & (tA >= 0))[0]
    # for each P track, the longest overlapping A track
    order = np.lexsort((-LA[both], tP[both]))
    first = np.unique(tP[both][order], return_index=True)[1]
    pairs = both[order][first]
    lp, la = LP[pairs], LA[pairs]
    nP = tP.max() + 1
    nA = tA.max() + 1
    trackP_hasA = len(np.unique(tP[both])) / nP
    trackA_hasP = len(np.unique(tA[both])) / nA
    return dict(nP=int(nP), nA=int(nA), pct_P_tracks_with_A=100 * trackP_hasA, pct_A_tracks_with_P=100 * trackA_hasP,
                matched=int(len(pairs)), medP=float(np.median(lp)), medA=float(np.median(la)),
                meanP=float(lp.mean()), meanA=float(la.mean()),
                pct_P_longer=100 * float(np.mean(lp > la)), pct_A_longer=100 * float(np.mean(la > lp)),
                pct_equal=100 * float(np.mean(la == lp)),
                corr_log=float(np.corrcoef(np.log(lp), np.log(la))[0, 1]),
                pct_detections_P_in_A=100 * float(np.mean(tA[tP >= 0] >= 0)),
                pct_detections_A_in_P=100 * float(np.mean(tP[tA >= 0] >= 0)),
                meanL_Ponly_tracks=float(np.mean([LP[tP == k][0] for k in np.setdiff1d(np.arange(nP), tP[both])[:5000]])),
                meanL_Aonly_tracks=float(np.mean([LA[tA == k][0] for k in np.setdiff1d(np.arange(nA), tA[both])[:5000]])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--type', default='sup')
    a = ap.parse_args()
    t = a.type
    T = table(t)
    S = np.load(os.path.join(DATA, f'tracks_{t}.npz'))
    D = json.load(open(os.path.join(OUTD, f'detectors_{t}.json')))
    a5 = D['A']['thr5']
    rng = np.random.default_rng(2)
    res = {'a5': a5}
    print(f'== {NAME[t]}: amplitude low threshold matched to depth 5 hPa: {a5:.2f} hPa')
    for key in ['P', 'A']:
        c, last = classify_ends(T, S, key, a5, rng)
        res[f'ends_{key}'] = c
        res[f'lost_remnant_{key}'] = G['lost_remnant']
        print('     remnants that lost the own criterion:', {k: round(v, 1) for k, v in G['lost_remnant'].items()})
        print(f'  {key} track ends: {len(last)}')
        for l, (x, y) in c.items():
            print(f'     {l:45s} actual {x:5.1f}%  chance {y:5.1f}%')
        # the state on the last day
        res[f'last_{key}'] = dict(depth=float(np.median(T['depth'][last])), amp=float(np.median(T['amp'][last])),
                                  ratio=float(np.median(T['depth'][last] / np.maximum(T['amp'][last], 0.1))),
                                  nested=100 * float(np.mean(T['rim'][last] == 0)))
        print(f'     last day: median depth {res[f"last_{key}"]["depth"]:.1f}, amp {res[f"last_{key}"]["amp"]:.1f}, '
              f'depth/amp {res[f"last_{key}"]["ratio"]:.2f}, nested {res[f"last_{key}"]["nested"]:.1f}%')
        res[f'comp_{key}'] = comp = composites(T, S, key)
        for lab in ['from genesis', 'to lysis']:
            print('     ' + lab + ': ' + '  '.join(f'{k}: d {v["depth"]:.1f} a {v["amp"]:.1f} r {v["ratio"]:.2f} '
                                                  f'n {v["nested"]:.0f}%' for k, v in comp[lab].items()))
        m = comp['mid-life']
        print(f'     mid-life: d {m["depth"]:.1f} a {m["amp"]:.1f} r {m["ratio"]:.2f} n {m["nested"]:.0f}%', flush=True)
    res['lifetimes'] = lt = lifetimes(T, S)
    print('  lifetimes of the same systems:', json.dumps({k: round(v, 3) for k, v in lt.items()}))
    with open(os.path.join(OUTD, f'lifecycle_{t}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
