"""
Persistence features of one polar disc, with birth cell (saddle) and parent.

In 2-D, H1 of the V-construction sublevel filtration of a padded image is dual to
H0 of the superlevel filtration with 8-connectivity.  A union-find over pixels in
decreasing order, with the elder rule and the padding (plus the image exterior)
treated as the eldest region, therefore gives exactly the diagram Ripserer.jl
stores in `data/processed_data/representative_data`, and in addition:

    death cell   the peak pixel of the component       (= stored death vertex)
    birth cell   the pixel at which it merges            (= the saddle; an endpoint
                                                           of the birth edge on the
                                                           representative cycle)
    parent       the peak it merges into, or -1 if it merges into the padding
                 (i.e. the disc rim)

Apply it to `slp_sub_*` discs for 1-anticyclones and to `slp_sup_*` discs (the
negated field) for 1-cyclones.  Values are returned in the disc's units (Pa).
"""
import numpy as np

_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def features(g, sea_mask):
    """
    Parameters
    ----------
    g : (H, W) array, the disc
    sea_mask : (H, W) bool, True on the padding

    Returns
    -------
    list of (peak_value, saddle_value, peak_idx, saddle_idx, parent_peak_idx)
    with flat pixel indices; parent_peak_idx is -1 for the rim.
    """
    H, W = g.shape
    N = H * W
    SEA = N
    parent = np.arange(N + 1)
    peak = np.full(N + 1, -1)
    peak[SEA] = SEA
    val = g.ravel().astype(np.float64)
    sea = sea_mask.ravel()
    added = sea.copy()
    parent[np.where(sea)[0]] = SEA

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    def elder(pa, pb):
        if pa == SEA:
            return True
        if pb == SEA:
            return False
        return (val[pa], -pa) > (val[pb], -pb)

    order = np.lexsort((np.arange(N), -val))
    order = order[~sea[order]]
    out = []
    for p in order:
        p = int(p)
        r, c = divmod(p, W)
        roots = []
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                x = SEA
            else:
                q = rr * W + cc
                if not added[q]:
                    continue
                x = find(q)
            if x not in roots:
                roots.append(x)
        added[p] = True
        if not roots:
            peak[p] = p
            continue
        el = roots[0]
        for x in roots[1:]:
            if elder(peak[x], peak[el]):
                el = x
        for x in roots:
            if x != el:
                pp = int(peak[el])
                if val[peak[x]] != val[p]:      # Ripserer drops zero-length intervals (plateau ties)
                    out.append((val[peak[x]], val[p], int(peak[x]), p, -1 if pp == SEA else pp))
                parent[x] = el
        parent[p] = el
    return out
