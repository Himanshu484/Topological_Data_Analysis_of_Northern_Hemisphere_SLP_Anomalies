"""
Shared pieces: the anomaly field exactly as scripts/regenerate_polar_discs.py
builds it, the disc projection as a sparse linear operator, and the per-day
feature extraction used identically for real and surrogate fields.

Anomaly (identical to regenerate_polar_discs.build_anomalies): 0-90 N rows of
slp.daily.nc, 29 Feb deleted, 1948-2023 on a 365-day calendar, minus the raw
76-sample per-calendar-day mean (float64), stored as float32.  Only the NH rows
are read, so memory is 0.6 GB instead of 2.3 GB.

Projection: gv.reshape_to_polar_matrix is linear in the data on the valid disc
(bilinear interpolation, all valid points in bounds), so it is probed once with
the 5328 unit vectors to give a sparse (n_valid x 5328) matrix.  The padding
value is reproduced from the lat-lon field exactly as the pipeline does
('largest' for sub, 'smallest' then negated for sup); only the mask matters.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, HERE)

# GN_SOURCE=era5 switches every script to ERA5 1979-2022 (download_era5.py), with
# its own data and output folders; the default is NCEP R1 1948-2023.
SOURCE = os.environ.get('GN_SOURCE', 'r1')
assert SOURCE in ('r1', 'era5')
SUFFIX = '' if SOURCE == 'r1' else '_era5'
DATA_R1 = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'gaussian_null')
DATA = DATA_R1 + SUFFIX
OUT = os.path.join(ROOT, 'output', 'experiments', 'gaussian_null' + SUFFIX)
START_YEAR, END_YEAR = (1948, 2023) if SOURCE == 'r1' else (1979, 2022)
MAIN_PERIOD = (1979, END_YEAR)
PERIODS = [(1948, 1978), (1979, 2023)] if SOURCE == 'r1' else [(1979, 2022)]
REAL = f'real_{START_YEAR}_{END_YEAR}'
N_YEARS = END_YEAR - START_YEAR + 1
NLAT, NLON = 37, 144
MIN_DEPTH_STORE = 200.0          # Pa; features shallower than 2 hPa are only kept in daily aggregates


def build_anomalies():
    """(N_YEARS*365, 37, 144) float32 anomaly, lat (descending), lon."""
    if SOURCE == 'era5':
        arr = np.load(os.path.join(DATA, f'era5_msl_daily_r1grid_{START_YEAR}_{END_YEAR}.npy'))
        arr = arr.reshape(N_YEARS, 365, NLAT, NLON)
        clim = arr.mean(axis=0, dtype=np.float64)
        anom = (arr - clim[None]).astype(np.float32)
        return anom.reshape(N_YEARS * 365, NLAT, NLON), np.arange(90, -0.1, -2.5), np.arange(0, 360, 2.5)
    import netCDF4 as nc
    ds = nc.Dataset(os.path.join(ROOT, 'data', 'raw', 'slp.daily.nc'))
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])
    li = [i for i in range(len(lat)) if 0 <= lat[i] <= 90]
    assert li == list(range(li[0], li[-1] + 1))
    v = ds.variables['slp']
    v.set_auto_mask(False)
    slp = np.asarray(v[:, li[0]:li[-1] + 1, :], dtype=np.float32)
    ds.close()
    keep = []
    d0 = np.datetime64(f'{START_YEAR}-01-01')
    for t in range(len(slp)):
        d = (d0 + np.timedelta64(t, 'D')).astype('datetime64[D]').astype(object)
        if d.year > END_YEAR:
            break
        if not (d.month == 2 and d.day == 29):
            keep.append(t)
    slp = slp[keep][:N_YEARS * 365]
    arr = slp.reshape(N_YEARS, 365, NLAT, NLON)
    clim = arr.mean(axis=0, dtype=np.float64)
    anom = np.empty_like(arr)
    for y in range(N_YEARS):
        anom[y] = (arr[y].astype(np.float64) - clim).astype(np.float32)
    return anom.reshape(N_YEARS * 365, NLAT, NLON), lat[li], lon


def projection_operator():
    """Sparse P (n_valid x 5328), valid-pixel flat indices, disc shape."""
    path = os.path.join(DATA_R1, 'projection_operator.npz')   # same grid for both sources
    if os.path.exists(path):
        Z = np.load(path)
        P = sp.csr_matrix((Z['data'], Z['indices'], Z['indptr']), shape=tuple(Z['shape']))
        return P, Z['valid'], tuple(Z['disc_shape'])
    import globe_visualization as gv
    lat = np.arange(90, -0.1, -2.5)
    lon = np.arange(0, 360, 2.5)
    li, lo = list(range(NLAT)), list(range(NLON))
    ref = gv.reshape_to_polar_matrix(np.zeros(NLAT * NLON), li, lo, default_value=-9e9,
                                     lat_values=lat, lon_values=lon)
    valid = np.where(ref.ravel() != -9e9)[0]
    rows, cols, vals = [], [], []
    for j in range(NLAT * NLON):
        e = np.zeros(NLAT * NLON)
        e[j] = 1.0
        m = gv.reshape_to_polar_matrix(e, li, lo, default_value=-9e9, lat_values=lat, lon_values=lon).ravel()[valid]
        nz = np.nonzero(m)[0]
        rows.append(nz); cols.append(np.full(len(nz), j)); vals.append(m[nz])
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(valid), NLAT * NLON))
    os.makedirs(DATA_R1, exist_ok=True)
    np.savez(path, data=P.data, indices=P.indices, indptr=P.indptr, shape=P.shape, valid=valid,
             disc_shape=ref.shape)
    return P, valid, ref.shape


def discs(field_flat, P, valid, shape):
    """Return (sub, sup) discs for one day's lat-lon field (flat, 5328)."""
    f = field_flat.astype(np.float64)
    inner = P @ f
    rng = f.max() - f.min()
    step = max(1, rng * 0.01)
    sub = np.full(shape[0] * shape[1], f.max() + step)
    sub[valid] = inner
    sup = np.full(shape[0] * shape[1], -(f.min() - step))
    sup[valid] = -inner
    return sub.reshape(shape), sup.reshape(shape)


def day_features(field_flat, P, valid, shape):
    """Features of both filtrations of one day. Returns {tag: (dict, agg)}."""
    from basins import Disc
    out = {}
    sub, sup = discs(field_flat, P, valid, shape)
    for tag, g in (('sup', sup), ('sub', sub)):
        D = Disc(g)
        F = D.features()
        depth = F['peak'] - F['saddle']
        agg = np.array([len(depth), depth.sum(), (depth ** 2).sum(), depth.max() if len(depth) else 0.0])
        k = depth >= MIN_DEPTH_STORE
        F = {kk: v[k] for kk, v in F.items()}
        F['half_km2'] = D.level_area(F['peak_idx'], F['peak'] - 0.5 * (F['peak'] - F['saddle']))
        out[tag] = (F, agg)
    return out


def field_features(field, slot0, path):
    """Run day_features over a (T, 37, 144) field and save one npz per filtration."""
    P, valid, shape = projection_operator()
    T = field.shape[0]
    acc = {t: {k: [] for k in ('slot', 'peak', 'saddle', 'peak_idx', 'saddle_idx', 'parent_idx',
                               'basin_km2', 'half_km2', 'pole')} for t in ('sup', 'sub')}
    aggs = {t: np.zeros((T, 4)) for t in ('sup', 'sub')}
    for i in range(T):
        res = day_features(field[i].ravel(), P, valid, shape)
        for t, (F, agg) in res.items():
            aggs[t][i] = agg
            for k in acc[t]:
                acc[t][k].append(np.full(len(F['peak']), slot0 + i, np.int32) if k == 'slot' else F[k])
    for t in acc:
        A = {k: np.concatenate(v) for k, v in acc[t].items()}
        A['peak'] = A['peak'].astype(np.float32); A['saddle'] = A['saddle'].astype(np.float32)
        A['basin_km2'] = A['basin_km2'].astype(np.float32); A['half_km2'] = A['half_km2'].astype(np.float32)
        for k in ('peak_idx', 'saddle_idx', 'parent_idx'):
            A[k] = A[k].astype(np.int32)
        np.savez_compressed(path.format(tag=t), slot0=slot0, nslots=T, agg=aggs[t].astype(np.float32), **A)
