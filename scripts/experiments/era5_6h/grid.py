"""
Shared pieces of the 6-hourly ERA5 experiment: calendar, paths, the disc
projection for an arbitrary source grid and disc size, and per-field feature
extraction.

Conventions (identical to the daily pipeline unless stated):
  * no-leap calendar, 29 February dropped; a "slot" is a 6-hour step counted
    from 1979-01-01 00 UTC, so one year is 1460 slots and slot % 4 is the
    synoptic hour (0, 6, 12, 18 UTC);
  * azimuthal-equidistant disc with the pole at pixel (c, c) and the equator
    at radius c pixels; latitude = 90 - 90 r / c, longitude = atan2(row - c,
    col - c) + 90 degrees, exactly as scripts/prototype/tracking.py::pix2ll;
  * the image is (2c + 1) x (2c + 1) and a pixel is inside the disc if r <= c
    (the daily pipeline used a 172 x 172 image, i.e. 2c wide; the pixels the
    two share are at the same positions and get the same values, see
    validate_projection in run_checks.py);
  * bilinear interpolation in (lat, lon) with longitude periodic.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(HERE, '..', 'gaussian_null'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from basins import Disc  # noqa: E402

Y0, Y1 = 1979, 2022
NY = Y1 - Y0 + 1
SPD = 4                                   # steps per day
SPY = 365 * SPD                           # slots per year
T = NY * SPY
DATA = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'era5_6h')      # feature files (kept)
OUT = os.path.join(ROOT, 'output', 'experiments', 'era5_6h')
# large, re-creatable arrays (fields, climatologies, surrogate scratch); keep them out of synced folders
CACHE = os.environ.get('ERA6H_CACHE', os.path.join(DATA, 'cache'))
for d in (DATA, OUT, CACHE):
    os.makedirs(d, exist_ok=True)

STORES = {'0p25': 'weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
          '1p5': 'weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'}
# NH grids, latitude descending from 90 N to 0, longitude from 0 eastward
GRIDS = {'0p25': dict(dlat=0.25, nlat=361, nlon=1440, c=360),
         '1p5': dict(dlat=1.5, nlat=61, nlon=240, c=120),
         'r1': dict(dlat=2.5, nlat=37, nlon=144, c=86)}          # NCEP-NCAR Reanalysis 1 (its own grid)
R1_URL = 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/slp.{year}.nc'   # 4-times-daily SLP
MIN_DEPTH_STORE = 200.0                   # Pa: shallower features are kept only in the per-field aggregates
MONTH_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_OF_DOY = np.repeat(np.arange(12), MONTH_LEN)


def noleap_time_index():
    """Indices into the store's 6-hourly time axis (hours since 1959-01-01) for Y0..Y1 without 29 February."""
    base = np.datetime64('1959-01-01T00', 'h')
    t = np.arange(np.datetime64(f'{Y0}-01-01T00', 'h'), np.datetime64(f'{Y1 + 1}-01-01T00', 'h'), np.timedelta64(6, 'h'))
    md = np.array([str(x)[5:10] for x in t.astype('datetime64[D]')])
    t = t[md != '02-29']
    assert len(t) == T
    return ((t - base).astype(np.int64) // 6).astype(np.int64)


def slot_date(slot):
    y, s = Y0 + int(slot) // SPY, int(slot) % SPY
    d, h = s // SPD, 6 * (s % SPD)
    m = int(MONTH_OF_DOY[d])
    return f'{y}-{m + 1:02d}-{d - sum(MONTH_LEN[:m]) + 1:02d}T{h:02d}'


def pix2ll(row, col, c):
    row, col = np.asarray(row, float), np.asarray(col, float)
    r = np.hypot(row - c, col - c)
    return 90 - 90 * r / c, (np.degrees(np.arctan2(row - c, col - c)) + 90) % 360


def projection_operator(res):
    """Sparse bilinear operator (n_valid x nlat*nlon), valid flat pixel indices, image shape."""
    g = GRIDS[res]
    c, nlat, nlon, dlat = g['c'], g['nlat'], g['nlon'], g['dlat']
    path = os.path.join(CACHE, f'projection_{res}_c{c}.npz')
    if os.path.exists(path):
        Z = np.load(path)
        return sp.csr_matrix((Z['data'], Z['indices'], Z['indptr']), shape=tuple(Z['shape'])), Z['valid'], tuple(Z['img'])
    n = 2 * c + 1
    rr, cc = np.mgrid[:n, :n]
    r = np.hypot(rr - c, cc - c)
    valid = np.where((r <= c).ravel())[0]
    lat, lon = pix2ll(rr.ravel()[valid], cc.ravel()[valid], c)
    fi = (90 - lat) / dlat                                   # fractional row, 0 at the pole
    i0 = np.clip(np.floor(fi).astype(int), 0, nlat - 2)
    wi = np.clip(fi - i0, 0, 1)
    fj = lon / (360.0 / nlon)
    j0 = np.floor(fj).astype(int) % nlon
    wj = fj - np.floor(fj)
    j1 = (j0 + 1) % nlon
    rows = np.tile(np.arange(len(valid)), 4)
    cols = np.concatenate([i0 * nlon + j0, i0 * nlon + j1, (i0 + 1) * nlon + j0, (i0 + 1) * nlon + j1])
    vals = np.concatenate([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj])
    P = sp.csr_matrix((vals, (rows, cols)), shape=(len(valid), nlat * nlon))
    P.sum_duplicates()
    np.savez(path, data=P.data, indices=P.indices, indptr=P.indptr, shape=P.shape, valid=valid, img=(n, n))
    return P, valid, (n, n)


def field_features(f_flat, P, valid, img):
    """Features of both signs of one anomaly field (Pa, flat lat-lon).

    Returns {sign: (dict of arrays for features deeper than MIN_DEPTH_STORE, agg)} with
    agg = [number of extrema, TP1, TP2, largest depth] over all features (Pa units).
    """
    f = np.asarray(f_flat, np.float64)
    inner = P @ f
    step = max(1.0, (f.max() - f.min()) * 0.01)
    out = {}
    for sign, g_in, pad in (('C', -inner, -(f.min() - step)), ('A', inner, f.max() + step)):
        g = np.full(img[0] * img[1], pad)
        g[valid] = g_in
        D = Disc(g.reshape(img))
        F = D.features()
        depth = F['peak'] - F['saddle']
        agg = np.array([len(depth), depth.sum(), (depth ** 2).sum(), depth.max() if len(depth) else 0.0])
        k = depth >= MIN_DEPTH_STORE
        half = D.level_area(F['peak_idx'][k], F['peak'][k] - 0.5 * depth[k])
        out[sign] = (dict(peak=F['peak'][k].astype(np.float32), saddle=F['saddle'][k].astype(np.float32),
                          peak_idx=F['peak_idx'][k].astype(np.int32), parent_idx=F['parent_idx'][k].astype(np.int32),
                          half_km2=half.astype(np.float32)), agg)
    return out


def save_features(path_fmt, slots, per_field, n_slots, slot0=0, meta=None):
    """per_field: list (aligned with slots) of field_features outputs -> one npz per sign."""
    for sign in 'CA':
        keys = ('peak', 'saddle', 'peak_idx', 'parent_idx', 'half_km2')
        cat = {k: np.concatenate([pf[sign][0][k] for pf in per_field]) for k in keys}
        cat['slot'] = np.concatenate([np.full(len(pf[sign][0]['peak']), s, np.int32) for s, pf in zip(slots, per_field)])
        agg = np.zeros((n_slots, 4), np.float64)
        agg[np.asarray(slots) - slot0] = np.array([pf[sign][1] for pf in per_field])
        np.savez_compressed(path_fmt.format(sign=sign), slot0=slot0, nslots=n_slots, agg=agg.astype(np.float32),
                            **cat, **(meta or {}))
