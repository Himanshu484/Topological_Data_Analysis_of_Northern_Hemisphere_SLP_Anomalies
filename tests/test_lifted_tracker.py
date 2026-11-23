"""Tests of src/lifted_tracker.py on small synthetic records."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import lifted_tracker as lt  # noqa: E402


def _record(rows):
    """rows: (day, birth, death, lat, lon) -> arrays sorted by day."""
    a = np.array(sorted(rows), float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def test_great_circle_quarter_meridian():
    d = lt.great_circle([0.0], [0.0], [90.0], [0.0])[0, 0]
    assert abs(d - np.pi / 2 * lt.EARTH_RADIUS_KM) < 1e-6


def test_two_features_keep_their_identity():
    # two highs 40 degrees of longitude apart, each drifting 3 degrees east per day
    rows = []
    for t in range(5):
        rows.append((t, 0.0, 20.0, 50.0, 10.0 + 3 * t))
        rows.append((t, 0.0, 12.0, 50.0, 50.0 + 3 * t))
    day, b, d, lat, lon = _record(rows)
    tracks, keep, links = lt.track(day, b, d, lat, lon, n_days=5)
    assert len(tracks) == 2 and all(len(m) == 5 for m in tracks)
    for m in tracks:                       # a track never mixes the two depths
        assert len(set(np.round(d[m] - b[m]))) == 1
    assert np.nanmax(links['step']) < 250


def test_gate_breaks_a_track():
    rows = [(0, 0.0, 15.0, 50.0, 0.0), (1, 0.0, 15.0, 50.0, 30.0)]        # about 2100 km apart
    day, b, d, lat, lon = _record(rows)
    tracks, _, _ = lt.track(day, b, d, lat, lon, n_days=2, R=1000.0)
    assert sorted(len(m) for m in tracks) == [1, 1]


def test_lifted_vetoes_a_large_change_of_depth_but_nn_does_not():
    # same place and same depth (6 hPa), but birth and death both shifted by 10 hPa: the change of
    # (birth, death) costs 20, more than sending both features to the diagonal (6 + 6), so no link
    rows = [(0, 0.0, 6.0, 50.0, 0.0), (1, 10.0, 16.0, 50.0, 0.0)]
    day, b, d, lat, lon = _record(rows)
    t_l, _, _ = lt.track(day, b, d, lat, lon, n_days=2, tau_lo=5.0, tau_hi=5.0, mode='lifted')
    t_n, _, _ = lt.track(day, b, d, lat, lon, n_days=2, tau_lo=5.0, tau_hi=5.0, mode='nn')
    assert sorted(len(m) for m in t_l) == [1, 1]
    assert [len(m) for m in t_n] == [2]


def test_two_thresholds_bridge_a_dip():
    depths = [12.0, 7.0, 12.0]             # dips below 10 hPa on the middle day
    rows = [(t, 0.0, x, 50.0, 2.0 * t) for t, x in enumerate(depths)]
    day, b, d, lat, lon = _record(rows)
    one, _, _ = lt.track(day, b, d, lat, lon, n_days=3, tau_lo=10.0, tau_hi=10.0)
    two, _, _ = lt.track(day, b, d, lat, lon, n_days=3, tau_lo=5.0, tau_hi=10.0)
    assert sorted(len(m) for m in one) == [1, 1]
    assert [len(m) for m in two] == [3]


def test_tracks_that_never_reach_the_upper_threshold_are_dropped():
    rows = [(t, 0.0, 7.0, 50.0, 2.0 * t) for t in range(4)]
    day, b, d, lat, lon = _record(rows)
    tracks, _, _ = lt.track(day, b, d, lat, lon, n_days=4, tau_lo=5.0, tau_hi=10.0)
    assert tracks == []


def test_summary_counts():
    tracks = [np.arange(1), np.arange(3), np.arange(12)]
    s = lt.summary(tracks)
    assert s['n_tracks'] == 3 and s['max_length'] == 12 and s['n_ge10d'] == 1
    assert abs(s['pct_1d'] - 100 / 3) < 1e-9 and abs(s['mean_length'] - 16 / 3) < 1e-9
