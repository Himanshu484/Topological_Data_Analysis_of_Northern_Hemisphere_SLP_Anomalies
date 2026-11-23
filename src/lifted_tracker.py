"""
Day-to-day tracking of persistence features with a lifted Wasserstein matching
and two depth thresholds.

An earlier version of this work matched consecutive persistence diagrams on
(birth, death) alone and applied a distance gate afterwards; on its own
features that tracker missed 28-53 % of the links a nearest-neighbour
assignment finds.  Here the
position of the extremum enters the cost itself, as in the lifted Wasserstein
matcher of Soler et al. (2018, IEEE LDAV):

    cost(p, q) = |b_p - b_q| + |d_p - d_q| + lam * dist(p, q)      if dist <= R
               = forbidden                                         otherwise
    cost(p, diagonal) = d_p - b_p                                  (the depth of p)

with (b, d) the birth and death values in hPa, dist the great-circle distance
between the extrema in km, and lam in hPa per km.  The optimal partial matching
is one-to-one.  A feature is left unmatched when linking it would cost more
than sending both partners to the diagonal, i.e. when the change of (b, d) plus
the weighted displacement exceeds the sum of the two depths.

Two thresholds (hysteresis): features are followed down to `tau_lo`, and a
track is kept if at least one of its members reaches `tau_hi`.  This stops a
track from breaking each time its depth dips below the reporting threshold.

With lam = 0 and no diagonal the scheme reduces to a gated nearest-neighbour
assignment (`mode='nn'`), kept for comparison.

Units: hPa, km, degrees.  Features of one record are passed as flat arrays
sorted by day.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

EARTH_RADIUS_KM = 6371.0088
_BIG = 1e12


def great_circle(lat1, lon1, lat2, lon2):
    """Great-circle distance matrix in km between two sets of points (degrees)."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((la1[:, None] - la2[None, :]) / 2) ** 2
         + np.cos(la1)[:, None] * np.cos(la2)[None, :] * np.sin((lo1[:, None] - lo2[None, :]) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_day(b1, d1, lat1, lon1, b2, d2, lat2, lon2, R=1000.0, lam=0.01, mode='lifted'):
    """Match the features of one day to those of the next.

    Returns (i, j, dist): index arrays of the linked pairs and their distances (km).
    """
    D = great_circle(lat1, lon1, lat2, lon2)
    if mode == 'nn':
        ri, ci = linear_sum_assignment(np.where(D <= R, D, 1e9))
    elif mode == 'lifted':
        b1, d1, b2, d2 = (np.asarray(v, float) for v in (b1, d1, b2, d2))
        cost = np.abs(b1[:, None] - b2[None, :]) + np.abs(d1[:, None] - d2[None, :]) + lam * D
        M, N = cost.shape
        C = np.full((M + N, M + N), _BIG)
        C[:M, :N] = np.where(D <= R, cost, _BIG)
        C[np.arange(M), N + np.arange(M)] = np.abs(d1 - b1)        # feature of day 1 -> diagonal
        C[M + np.arange(N), np.arange(N)] = np.abs(d2 - b2)        # diagonal -> feature of day 2
        C[M:, N:] = 0.0
        ri, ci = linear_sum_assignment(C)
        k = (ri < M) & (ci < N)
        ri, ci = ri[k], ci[k]
    else:
        raise ValueError(mode)
    k = D[ri, ci] <= R
    return ri[k], ci[k], D[ri[k], ci[k]]


def link(day, birth, death, lat, lon, n_days, R=1000.0, lam=0.01, mode='lifted'):
    """Link all consecutive days of a record.

    day    integer day index of each feature (0 .. n_days-1), sorted ascending
    Returns dict(succ, pred, step): successor and predecessor index of each
    feature (-1 if none) and the distance to the predecessor (nan if none).
    """
    day = np.asarray(day)
    assert (np.diff(day) >= 0).all(), 'features must be sorted by day'
    n = len(day)
    succ = -np.ones(n, int)
    pred = -np.ones(n, int)
    step = np.full(n, np.nan)
    starts = np.searchsorted(day, np.arange(n_days))
    ends = np.searchsorted(day, np.arange(n_days), 'right')
    for t in range(n_days - 1):
        a, b, c, d = starts[t], ends[t], starts[t + 1], ends[t + 1]
        if b == a or d == c:
            continue
        i, j, dist = match_day(birth[a:b], death[a:b], lat[a:b], lon[a:b],
                               birth[c:d], death[c:d], lat[c:d], lon[c:d], R=R, lam=lam, mode=mode)
        succ[a + i] = c + j
        pred[c + j] = a + i
        step[c + j] = dist
    return dict(succ=succ, pred=pred, step=step)


def chains(links):
    """Maximal chains of links, each a list of feature indices in time order."""
    succ, pred = links['succ'], links['pred']
    out = []
    for h in np.where(pred < 0)[0]:
        m = [int(h)]
        while succ[m[-1]] >= 0:
            m.append(int(succ[m[-1]]))
        out.append(m)
    return out


def track(day, birth, death, lat, lon, n_days, tau_lo=5.0, tau_hi=10.0, R=1000.0, lam=0.01, mode='lifted'):
    """Tracks with two thresholds.

    Features with depth >= tau_lo are linked; a track is kept if it reaches tau_hi.
    Returns (tracks, keep, links): `tracks` is a list of index arrays into the
    arrays passed in, `keep` the boolean mask of features with depth >= tau_lo
    and `links` the output of `link` on the kept features.
    """
    depth = np.abs(np.asarray(death, float) - np.asarray(birth, float))
    keep = depth >= tau_lo
    idx = np.where(keep)[0]
    L = link(np.asarray(day)[idx], np.asarray(birth, float)[idx], np.asarray(death, float)[idx],
             np.asarray(lat, float)[idx], np.asarray(lon, float)[idx], n_days, R=R, lam=lam, mode=mode)
    tracks = [idx[m] for m in chains(L) if (depth[idx[m]] >= tau_hi).any()]
    return tracks, keep, L


def summary(tracks, links=None):
    """Length statistics of a set of tracks (and the median daily step if `links` is given)."""
    n = np.array([len(m) for m in tracks])
    out = dict(n_tracks=int(len(n)), mean_length=float(n.mean()), max_length=int(n.max()),
               pct_1d=float(100 * np.mean(n == 1)), pct_2_4d=float(100 * np.mean((n >= 2) & (n <= 4))),
               pct_5_9d=float(100 * np.mean((n >= 5) & (n <= 9))), pct_ge5d=float(100 * np.mean(n >= 5)),
               pct_ge10d=float(100 * np.mean(n >= 10)), n_ge10d=int((n >= 10).sum()),
               survival=[float(np.mean(n >= j)) for j in range(1, 31)])
    if links is not None:
        s = links['step'][~np.isnan(links['step'])]
        out['median_step_km'] = float(np.median(s))
        out['n_links'] = int(len(s))
    return out
