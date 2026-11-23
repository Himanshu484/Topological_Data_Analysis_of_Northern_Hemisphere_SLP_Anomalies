"""
Regression tests for the Wasserstein distance and for the stored
representative cycles.

Covers:
  * `src/tda_metrics.py` -- Wasserstein distance with an explicit ground metric
    (the manuscript defines it with the L1 ground metric)
  * `disc_utils.resolve_death_vertex` -- the vertex of the death 2-cube that
    carries the death value
  * the death value is the extremum of the region its representative encloses

Run:  python -m pytest tests/test_metrics_and_geometry.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import tda_metrics as tm              # noqa: E402
import disc_utils as ft         # noqa: E402

MATRIX_SIZE = 172
CENTRE = MATRIX_SIZE // 2


# ── tda_metrics ───────────────────────────────────────────────────────────────

def test_L2_reproduces_persim_exactly():
    """The old behaviour must remain reachable, bit-for-bit, for auditing."""
    persim = pytest.importorskip('persim')
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(50):
        m, n = int(rng.integers(1, 10)), int(rng.integers(1, 10))
        b1 = rng.normal(0, 5, m); D1 = np.c_[b1, b1 + np.abs(rng.normal(0, 3, m))]
        b2 = rng.normal(0, 5, n); D2 = np.c_[b2, b2 + np.abs(rng.normal(0, 3, n))]
        worst = max(worst, abs(tm.wasserstein(D1, D2, q=1, ground_metric='L2')
                               - persim.wasserstein(D1, D2)))
    assert worst < 1e-9, worst


def test_L1_is_the_manuscript_definition():
    """
    Def. 2.7-2.8: cost = sum(|b_p - b_q| + |d_p - d_q|), and an unmatched point
    goes to the diagonal at its own persistence.
    """
    D1 = np.array([[0.0, 10.0]])
    D2 = np.array([[1.0, 12.0]])
    # matching the two points costs |0-1| + |10-12| = 3;
    # sending both to the diagonal costs 10 + 11 = 21.  The optimum is 3.
    assert tm.wasserstein(D1, D2, q=1, ground_metric='L1') == pytest.approx(3.0)

    # a lone point must go to the diagonal at exactly its persistence
    assert tm.wasserstein(D1, np.empty((0, 2)), q=1,
                          ground_metric='L1') == pytest.approx(10.0)


def test_diagonal_factors():
    D = np.array([[0.0, 8.0]])
    empty = np.empty((0, 2))
    assert tm.wasserstein(D, empty, ground_metric='L1') == pytest.approx(8.0)
    assert tm.wasserstein(D, empty, ground_metric='L2') == pytest.approx(8.0 / np.sqrt(2))
    assert tm.wasserstein(D, empty, ground_metric='Linf') == pytest.approx(4.0)


def test_metric_axioms():
    rng = np.random.default_rng(7)
    b = rng.normal(0, 5, 6); D1 = np.c_[b, b + np.abs(rng.normal(0, 3, 6))]
    b = rng.normal(0, 5, 5); D2 = np.c_[b, b + np.abs(rng.normal(0, 3, 5))]
    for gm in tm.GROUND_METRICS:
        assert tm.wasserstein(D1, D1, ground_metric=gm) == pytest.approx(0.0, abs=1e-9)
        assert tm.wasserstein(D1, D2, ground_metric=gm) == pytest.approx(
            tm.wasserstein(D2, D1, ground_metric=gm))
        assert tm.wasserstein(D1, D2, ground_metric=gm) >= 0


def test_matching_indices_are_wellformed():
    rng = np.random.default_rng(3)
    b = rng.normal(0, 5, 4); D1 = np.c_[b, b + np.abs(rng.normal(0, 3, 4))]
    b = rng.normal(0, 5, 6); D2 = np.c_[b, b + np.abs(rng.normal(0, 3, 6))]
    _, pairs = tm.wasserstein(D1, D2, matching=True)
    seen_i = [i for i, j in pairs if i >= 0]
    seen_j = [j for i, j in pairs if j >= 0]
    assert len(seen_i) == len(set(seen_i))          # one-to-one on the left
    assert len(seen_j) == len(set(seen_j))          # one-to-one on the right
    assert set(seen_i) == set(range(len(D1)))       # every point accounted for
    assert set(seen_j) == set(range(len(D2)))


def test_bottleneck_is_a_max_not_a_sum():
    D1 = np.array([[0.0, 10.0], [0.0, 10.0]])
    D2 = np.array([[1.0, 11.0], [2.0, 12.0]])
    w1 = tm.wasserstein(D1, D2, q=1, ground_metric='Linf')
    wb = tm.wasserstein(D1, D2, q=np.inf, ground_metric='Linf')
    assert wb <= w1 + 1e-9
    assert wb == pytest.approx(2.0)


# ── death vertex ──────────────────────────────────────────────────────────────

def test_resolve_death_vertex_picks_the_value_carrying_vertex():
    """
    Measured over 10 763 features in this repository, the vertex carrying the
    death value is the LAST of the four in 100 % of cases and the FIRST in 0 %.
    The resolver identifies it BY VALUE, so it does not depend on that
    regularity holding.
    """
    disc = np.zeros((10, 10))
    disc[3, 4] = 7.0                      # the critical vertex
    vs = [[2, 3], [3, 3], [2, 4], [3, 4]]
    assert ft.resolve_death_vertex(vs, 7.0, disc=disc) == (3, 4)
    # the diagonally opposite corner is what the old code returned
    assert tuple(vs[0]) == (2, 3)


def test_resolve_death_vertex_falls_back_to_the_convention():
    vs = [[2, 3], [3, 3], [2, 4], [3, 4]]
    assert ft.resolve_death_vertex(vs, None, disc=None) == (3, 4)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'data', 'processed_data',
                                    'representative_data', '2012',
                                    'slp_sub_2012_day_35.json')),
    reason='processed data not present')
def test_death_vertex_rule_holds_on_real_data():
    import json
    d = json.load(open(os.path.join(ROOT, 'data', 'processed_data',
                                    'representative_data', '2012',
                                    'slp_sub_2012_day_35.json')))
    M = np.load(os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years',
                             '2012', 'slp_sub_2012_day_35.npy'))
    n_first = 0
    for vs, dv in zip(d['death_simplex_vertices'], d['deaths']):
        r, c = ft.resolve_death_vertex(vs, dv, disc=M)
        assert abs(M[r, c] - dv) < 1e-3
        n_first += (r, c) == (vs[0][0], vs[0][1])
    assert n_first == 0, 'death_simplex_vertices[0] should never be the critical vertex'


# ── the Sec. 2.4 / Sec. 3.1 contradiction ─────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'data', 'processed_data',
                                    'representative_data', '2012',
                                    'slp_sub_2012_day_35.json')),
    reason='processed data not present')
def test_death_value_is_the_extremum_of_the_enclosed_region():
    """
    Sec. 2.4 says the death square "does not necessarily coincide with the
    absolute intensity extremum of the corresponding region"; Sec. 3.1 says the
    death value IS that extremum.  They cannot both be right.

    For a 2-D full cubical complex the class dies exactly when the enclosed disc
    is filled, i.e. at the extremum of the enclosed region -- so Sec. 3.1 is
    right and Sec. 2.4 should go.  Verified here on real features (and on 1 569
    features across three years and both filtrations during the remediation:
    100 % agreement).
    """
    import json
    from matplotlib.path import Path
    from shapely.geometry import MultiPoint

    d = json.load(open(os.path.join(ROOT, 'data', 'processed_data',
                                    'representative_data', '2012',
                                    'slp_sub_2012_day_35.json')))
    M = np.load(os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years',
                             '2012', 'slp_sub_2012_day_35.npy'))
    checked = 0
    for rep, death in zip(d['list_data'], d['deaths']):
        if len(rep) < 4:
            continue
        pts = set()
        for e in rep:
            pts.add((e[0][0], e[0][1]))
            pts.add((e[1][0], e[1][1]))
        hull = MultiPoint([(c, r) for r, c in sorted(pts)]).convex_hull
        if hull.geom_type != 'Polygon':
            continue
        xs, ys = hull.exterior.xy
        r0, r1 = int(min(ys)), int(max(ys))
        c0, c1 = int(min(xs)), int(max(xs))
        rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        inside = Path(np.c_[xs, ys]).contains_points(
            np.c_[cc.ravel(), rr.ravel()], radius=0.5).reshape(rr.shape)
        if not inside.any():
            continue
        assert M[rr[inside], cc[inside]].max() == pytest.approx(death, abs=1e-3)
        checked += 1
    assert checked > 20, f'only {checked} features exercised'
