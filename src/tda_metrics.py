"""
Wasserstein distance between persistence diagrams, with the ground metric
stated explicitly.

Why this module exists
----------------------
The manuscript's Def. 2.7-2.8 define the matching cost as

    cost(M) = sum_{(p,q) in M} ( |b_p - b_q| + |d_p - d_q| )

i.e. **W_1 with an L^1 ground metric**.  An earlier version of the code called
`persim.wasserstein`, which builds its cost matrix with
`sklearn.metrics.pairwise.pairwise_distances(S, T)` -- whose default is
**Euclidean** -- and after the 45 deg rotation charges `(d - b)/sqrt(2)` to
send a point to the diagonal.  That is W_1 with an **L^2** ground metric,
not the distance the paper defines.

This matters beyond bookkeeping: the Cohen-Steiner stability theorem the paper
leans on is stated for a particular ground metric, and with `alpha = 0` the
matching produced here is the *proposal step* of the tracker -- change the
ground metric and a different set of candidate pairs is proposed.

`wasserstein()` below implements the definition the paper states (`L1`, q=1) as
its default, and can reproduce the previous behaviour exactly with
`ground_metric='L2'`, so the change is auditable rather than silent.

Diagonal costs
--------------
For an unmatched point p = (b, d) the cost of sending it to the diagonal is the
ground distance from p to the nearest diagonal point (x, x):

    L^1   : min_x |b - x| + |d - x| = d - b          (the persistence)
    L^2   : (d - b) / sqrt(2)
    L^inf : (d - b) / 2

The L^inf convention is the one under which the classical bottleneck/Wasserstein
stability results are usually stated; it is offered here for completeness.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['wasserstein', 'cost_matrix', 'GROUND_METRICS', 'DIAGONAL_FACTOR']

GROUND_METRICS = ('L1', 'L2', 'Linf')

# multiply by persistence (d - b) to get the cost of matching a point to the diagonal
DIAGONAL_FACTOR = {
    'L1':   1.0,
    'L2':   1.0 / np.sqrt(2.0),
    'Linf': 0.5,
}


def _pairwise(S, T, ground_metric):
    """Ground distance between every point of S and every point of T."""
    diff = np.abs(S[:, None, :] - T[None, :, :])
    if ground_metric == 'L1':
        return diff.sum(axis=2)
    if ground_metric == 'L2':
        return np.sqrt((diff ** 2).sum(axis=2))
    if ground_metric == 'Linf':
        return diff.max(axis=2)
    raise ValueError(f"ground_metric must be one of {GROUND_METRICS}, got {ground_metric!r}")


def cost_matrix(dgm1, dgm2, ground_metric='L1'):
    """
    Build the (M+N) x (M+N) augmented cost matrix of the optimal-partial-matching
    problem, in the standard block form:

        [ point-to-point   point-to-diagonal ]
        [ diagonal-to-point  0 (diag-to-diag) ]

    Returns (C, M, N) where M = len(dgm1), N = len(dgm2).
    """
    S = np.asarray(dgm1, dtype=float)[:, :2] if len(dgm1) else np.empty((0, 2))
    T = np.asarray(dgm2, dtype=float)[:, :2] if len(dgm2) else np.empty((0, 2))
    M, N = len(S), len(T)
    fac = DIAGONAL_FACTOR[ground_metric] if ground_metric in DIAGONAL_FACTOR else None
    if fac is None:
        raise ValueError(f"ground_metric must be one of {GROUND_METRICS}, got {ground_metric!r}")

    C = np.zeros((M + N, M + N), dtype=float)
    if M and N:
        C[:M, :N] = _pairwise(S, T, ground_metric)
    if M:
        # |d - b|: correct for superlevel diagrams stored with death < birth too
        pers_S = fac * np.abs(S[:, 1] - S[:, 0])
        C[:M, N:] = np.inf
        C[np.arange(M), N + np.arange(M)] = pers_S
    if N:
        pers_T = fac * np.abs(T[:, 1] - T[:, 0])
        C[M:, :N] = np.inf
        C[M + np.arange(N), np.arange(N)] = pers_T
    # the diagonal-to-diagonal block is free
    return C, M, N


def wasserstein(dgm1, dgm2, q=1, ground_metric='L1', matching=False):
    """
    Optimal partial matching between two persistence diagrams.

    Parameters
    ----------
    dgm1, dgm2 : (n, >=2) array
        Birth/death pairs.  Extra columns are ignored, so a diagram carrying
        spatial coordinates in columns 2-3 can be passed straight in.
    q : int or float
        Order.  q=1 (the default, and what the manuscript defines) returns the
        SUM of matched ground distances.  q=2 returns
        (sum of squared ground distances)^(1/2).  q=np.inf gives the bottleneck
        distance under the chosen ground metric.
    ground_metric : {'L1', 'L2', 'Linf'}
        Ground metric on R^2.  'L1' is the manuscript's Def. 2.7-2.8.
        'L2' reproduces `persim.wasserstein` exactly (see module docstring).
    matching : bool
        If True also return the list of matched index pairs.  A pair
        (i, j) with i >= 0 and j >= 0 is a genuine point-to-point match;
        j = -1 means point i of dgm1 went to the diagonal, and i = -1 means
        point j of dgm2 came from the diagonal.

    Returns
    -------
    d               if matching is False
    (d, pairs)      if matching is True

    Notes
    -----
    q = np.inf is solved by bisection on the point-to-point costs, not by
    linear_sum_assignment, because the bottleneck objective is a min-max.
    """
    if ground_metric not in GROUND_METRICS:
        raise ValueError(f"ground_metric must be one of {GROUND_METRICS}, got {ground_metric!r}")

    S = np.asarray(dgm1, dtype=float) if len(dgm1) else np.empty((0, 2))
    T = np.asarray(dgm2, dtype=float) if len(dgm2) else np.empty((0, 2))
    # drop non-finite deaths, as persim does
    if len(S):
        S = S[np.isfinite(S[:, 1])]
    if len(T):
        T = T[np.isfinite(T[:, 1])]

    M, N = len(S), len(T)
    if M == 0 and N == 0:
        return (0.0, []) if matching else 0.0

    C, M, N = cost_matrix(S, T, ground_metric)

    if np.isinf(q):
        d, pairs = _bottleneck(C, M, N)
    else:
        # linear_sum_assignment minimises a SUM, so raise the entries to the
        # power q, solve, and take the q-th root of the total.
        W = C.copy()
        finite = np.isfinite(W)
        if q != 1:
            W[finite] = W[finite] ** q
        W[~finite] = 1e18                      # linear_sum_assignment rejects inf
        ri, ci = linear_sum_assignment(W)
        total = C[ri, ci][np.isfinite(C[ri, ci])]
        if q == 1:
            d = float(total.sum())
        else:
            d = float((total ** q).sum() ** (1.0 / q))
        pairs = _decode(ri, ci, M, N)

    return (d, pairs) if matching else d


def _decode(ri, ci, M, N):
    """Turn an assignment on the augmented matrix into (i, j) pairs."""
    pairs = []
    for r, c in zip(ri, ci):
        if r < M and c < N:
            pairs.append((int(r), int(c)))
        elif r < M:                     # dgm1 point r -> diagonal
            pairs.append((int(r), -1))
        elif c < N:                     # diagonal -> dgm2 point c
            pairs.append((-1, int(c)))
        # diagonal-to-diagonal: nothing to record
    return pairs


def _bottleneck(C, M, N):
    """Bottleneck distance by bisection over the distinct entries of C."""
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.sparse import csr_matrix

    vals = np.unique(C[np.isfinite(C)])
    lo, hi = 0, len(vals) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        eps = vals[mid]
        adj = csr_matrix(np.isfinite(C) & (C <= eps + 1e-12))
        m = maximum_bipartite_matching(adj, perm_type='column')
        if (m >= 0).sum() == C.shape[0]:
            best = (eps, m)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        eps = vals[-1]
        adj = csr_matrix(np.isfinite(C) & (C <= eps + 1e-12))
        best = (eps, maximum_bipartite_matching(adj, perm_type='column'))
    eps, m = best
    ri = np.arange(C.shape[0])
    return float(eps), _decode(ri, m, M, N)
