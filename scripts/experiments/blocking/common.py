"""Shared paths, grid, calendar, sector and season helpers for the blocking experiment."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PROTO = os.path.join(ROOT, 'scripts', 'prototype')
sys.path.insert(0, PROTO)
sys.path.insert(0, os.path.join(ROOT, 'src'))
from tracking import pix2ll, hav, run, tracks, RE_KM, C_PIX  # noqa: E402,F401

CACHE = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'blocking')
OUT = os.path.join(ROOT, 'output', 'experiments', 'blocking')
SLP_DISCS = os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years')
PROTO_FEAT = os.path.join(ROOT, 'data', 'processed_data', 'prototype')
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

LAT = np.arange(90, -0.1, -2.5)          # 37, descending, NCEP order
LON = np.arange(0, 360, 2.5)             # 144
Y0, Y1 = 1948, 2023
EVAL0, EVAL1 = 1979, 2023                # satellite era used for all evaluation
TRAIN = (1979, 2000)                     # thresholds tuned here
TEST = (2001, 2023)                      # scores reported here

# no-leap calendar: month of each slot 1..365
_MLEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH = np.concatenate([np.full(n, m + 1) for m, n in enumerate(_MLEN)])      # index slot-1
DOM = np.concatenate([np.arange(1, n + 1) for n in _MLEN])
SEASON_OF_MONTH = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM',
                   6: 'JJA', 7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON'}
SEASONS = ['DJF', 'MAM', 'JJA', 'SON']
SEASON = np.array([SEASON_OF_MONTH[m] for m in MONTH])

# longitude sectors (deg E, half-open [a, b) on the circle)
SECTORS = {'EA': (300.0, 60.0),     # Euro-Atlantic 60W-60E
           'US': (60.0, 150.0),     # Ural / Siberia 60E-150E
           'PAC': (150.0, 240.0),   # Pacific 150E-120W
           'NAM': (240.0, 300.0)}   # North America 120W-60W
SECTOR_NAMES = {'EA': 'Euro-Atlantic (60W-60E)', 'US': 'Ural-Siberia (60E-150E)',
                'PAC': 'Pacific (150E-120W)', 'NAM': 'N. America (120W-60W)'}


def in_sector(lon, s):
    a, b = SECTORS[s]
    lon = np.mod(lon, 360)
    return (lon >= a) & (lon < b) if a < b else (lon >= a) | (lon < b)


def sector_of(lon):
    lon = np.atleast_1d(np.mod(lon, 360))
    out = np.empty(lon.shape, object)
    for s in SECTORS:
        out[in_sector(lon, s)] = s
    return out


def slot_date(year, slot):
    return f'{year}-{MONTH[slot - 1]:02d}-{DOM[slot - 1]:02d}'


def date_slot(month, day):
    return int(np.where((MONTH == month) & (DOM == day))[0][0]) + 1


def gc(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la1 - la2) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo1 - lo2) / 2) ** 2
    return 2 * RE_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def disc_pixel_area_km2(n=172):
    """Area on the sphere of each disc pixel (lat = 90 - 90 r / 86, i.e. azimuthal equidistant)."""
    c = n // 2
    y, x = np.mgrid[:n, :n]
    r = np.hypot(y - c, x - c)
    colat = np.radians(90 * r / c)
    step = RE_KM * np.radians(90.0 / c)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.where(colat > 0, np.sin(colat) / colat, 1.0)
    return step ** 2 * f


def project(field2d, fill='largest'):
    """NCEP (37 lat desc x 144 lon) field -> 172x172 disc with the manuscript's projection code."""
    import globe_visualization as gv
    return gv.reshape_to_polar_matrix(np.asarray(field2d, np.float64).ravel(), list(range(37)),
                                      list(range(144)), default_value=fill,
                                      lat_values=LAT, lon_values=LON)


def disc(field2d, kind):
    """kind 'sub': maxima of field are features; 'sup': minima (disc = -field)."""
    if kind == 'sub':
        return project(field2d, 'largest')
    return -project(field2d, 'smallest')


def circmean_lon(lon, w=None):
    th = np.radians(lon)
    w = np.ones_like(th) if w is None else w
    return np.degrees(np.arctan2((w * np.sin(th)).sum(), (w * np.cos(th)).sum())) % 360


def load_feats(path, tau, scale=100.0, y0=None, y1=None):
    """Feature table (prototype npz format) with depth >= tau (in units value/scale), per-slot ranges."""
    Z = np.load(path)
    keys = ['year', 'day', 'birth', 'death', 'prow', 'pcol', 'srow', 'scol', 'parrow', 'parcol']
    F = {k: Z[k] for k in keys}
    m = np.abs(F['death'] - F['birth']) >= tau * scale
    ya, yb = int(Z['y0']), int(Z['y1'])
    if y0 is not None:
        m &= (F['year'] >= y0) & (F['year'] <= y1)
        ya, yb = y0, y1
    F = {k: v[m].astype(np.float64 if k in ('birth', 'death') else np.int64) for k, v in F.items()}
    F['birth'] /= scale
    F['death'] /= scale
    F['slot'] = (F['year'] - ya) * 365 + F['day'] - 1
    o = np.lexsort((F['prow'], F['slot']))
    F = {k: v[o] for k, v in F.items()}
    F['plat'], F['plon'] = pix2ll(F['prow'], F['pcol'])
    F['depth'] = np.abs(F['death'] - F['birth'])
    F['y0'] = ya
    F['nslots'] = (yb - ya + 1) * 365
    F['starts'] = np.searchsorted(F['slot'], np.arange(F['nslots']))
    F['ends'] = np.searchsorted(F['slot'], np.arange(F['nslots']), 'right')
    return F
