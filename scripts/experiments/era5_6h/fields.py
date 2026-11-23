"""
ERA5 6-hourly mean sea-level pressure from the public WeatherBench 2 archive,
its climatology, and persistence features.

    PY=<python of a venv with zarr<3 and gcsfs>   (see README.md)
    export ERA6H_CACHE=<a scratch folder with ~15 GB free>

    $PY fields.py download-1p5            # 1.5 degree fields, climatology, anomaly memmap   (~3 min, 7.5 GB)
    $PY fields.py features-1p5            # features of the 1.5 degree reanalysis            (~5 min)
    $PY fields.py clim-0p25               # pass 1 over the 0.25 degree archive: climatology (~45 min, 3 GB)
    $PY fields.py features-0p25           # pass 2: anomalies and features, streamed         (~45 min)
    $PY fields.py download-r1             # NCEP-NCAR R1 six-hourly SLP from NOAA PSL, 2.5 degrees (~2 min, 2.3 GB)
    $PY fields.py features-r1             # features of the six-hourly R1 reanalysis          (~2 min)

The 0.25 degree fields are never stored: 64,240 fields x 361 x 1440 values is
133 GB.  Pass 1 accumulates the climatology, pass 2 downloads again and keeps
only the features.

Climatology: the 44-year mean for each calendar day and synoptic hour,
smoothed with a 31-day circular running mean over days, separately for each
hour (so the diurnal cycle is kept and removed with it).  The daily pipeline
used the unsmoothed calendar-day mean; with one field per slot the sampling
noise of an unsmoothed 44-sample mean is about 1 hPa in the storm tracks.

Feature files: data/processed_data/experiments/era5_6h/feat_<name>_<res>_{C,A}.npz
with, per feature of depth >= 2 hPa: slot, peak, saddle (Pa, of g = -a for C
and +a for A), peak_idx, parent_idx (flat index in the (2c+1)^2 image, -1 =
rim), half_km2; and agg (T, 4) = extrema, TP1, TP2, largest depth per field.
"""
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import uniform_filter1d

import grid as G

OFFSET = 101325.0


def open_store(res):
    import gcsfs
    import zarr
    fs = gcsfs.GCSFileSystem(token='anon')
    g = zarr.open_consolidated(fs.get_mapper(G.STORES[res]), mode='r')
    assert g['time'].attrs['units'] == 'hours since 1959-01-01'
    return g


def smooth_clim(c):
    """31-day circular running mean over days, per synoptic hour.  c: (1460, X)."""
    out = np.empty_like(c)
    c4 = c.reshape(365, G.SPD, -1)
    o4 = out.reshape(365, G.SPD, -1)
    for h in range(G.SPD):
        o4[:, h] = uniform_filter1d(c4[:, h].astype(np.float64), 31, axis=0, mode='wrap').astype(c.dtype)
    return out


def read_retry(fn, tries=6):
    for a in range(tries):
        try:
            return fn()
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))


# ------------------------------------------------------------------ 1.5 degree
def download_1p5():
    g = open_store('1p5')
    v = g['mean_sea_level_pressure']                         # (time, lon, lat), lat ascending -90..90
    lat = np.asarray(g['latitude'][:])
    j = np.where(lat >= -1e-6)[0]
    assert len(j) == 61 and abs(lat[j[0]]) < 1e-6
    idx = G.noleap_time_index()
    raw = np.lib.format.open_memmap(os.path.join(G.CACHE, 'msl_1p5.npy'), mode='w+', dtype=np.float32, shape=(G.T, 61, 240))
    edges = list(range(0, G.T, 1024)) + [G.T]

    def block(k):
        a, b = edges[k], edges[k + 1]
        ii = idx[a:b]
        x = read_retry(lambda: np.asarray(v[int(ii[0]):int(ii[-1]) + 1, :, j[0]:j[-1] + 1]))      # contiguous in store time
        x = x[ii - ii[0]]                                    # drop 29 February
        return a, b, np.ascontiguousarray(x.transpose(0, 2, 1)[:, ::-1, :])                       # (t, lat desc, lon)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for a, b, x in ex.map(block, range(len(edges) - 1)):
            raw[a:b] = x
    raw.flush()
    print(f'1.5 deg fields downloaded in {time.time() - t0:.0f}s', flush=True)
    A = raw.reshape(G.NY, G.SPY, -1)
    clim = smooth_clim(A.mean(axis=0, dtype=np.float64).astype(np.float32))
    np.save(os.path.join(G.CACHE, 'clim_1p5.npy'), clim)
    anom = np.lib.format.open_memmap(os.path.join(G.CACHE, 'anom_1p5.npy'), mode='w+', dtype=np.float32, shape=(G.T, 61 * 240))
    for y in range(G.NY):
        anom[y * G.SPY:(y + 1) * G.SPY] = A[y] - clim
    anom.flush()
    del raw
    os.remove(os.path.join(G.CACHE, 'msl_1p5.npy'))
    print('wrote anom_1p5.npy', anom.shape, 'SD %.2f hPa' % (anom[::50].std() / 100), flush=True)


def _array_worker(args):
    path, res, a, b = args
    P, valid, img = G.projection_operator(res)
    X = np.load(path, mmap_mode='r')
    return a, [G.field_features(X[s], P, valid, img) for s in range(a, b)]


def features_from_array(path, res, name, workers=8, block=365):
    """Features of every field of a (T, X) anomaly array -> feat_<name>_<res>_{C,A}.npz"""
    G.projection_operator(res)                 # build the cached operator once, before the workers race for it
    jobs = [(path, res, a, min(a + block, G.T)) for a in range(0, G.T, block)]
    per = [None] * G.T
    t0 = time.time()
    with Pool(workers) as pool:
        for a, out in pool.imap_unordered(_array_worker, jobs):
            per[a:a + len(out)] = out
    G.save_features(os.path.join(G.DATA, f'feat_{name}_{res}_' + '{sign}.npz'), np.arange(G.T), per, G.T,
                    meta=dict(c=G.GRIDS[res]['c']))
    print(f'features {name} {res}: {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------ NCEP-NCAR R1, 2.5 degree
def download_r1():
    """Six-hourly R1 SLP (NOAA PSL, one file per year) -> anom_r1.npy, same calendar and climatology as ERA5."""
    import urllib.request
    import netCDF4 as nc
    d = os.path.join(G.CACHE, 'r1_6h')
    os.makedirs(d, exist_ok=True)
    raw = np.empty((G.T, 37, 144), np.float32)
    for y in range(G.Y0, G.Y1 + 1):
        f = os.path.join(d, f'slp.{y}.nc')
        if not os.path.exists(f):
            read_retry(lambda: urllib.request.urlretrieve(G.R1_URL.format(year=y), f))
        ds = nc.Dataset(f)
        lat = np.asarray(ds['lat'][:])
        assert lat[0] == 90 and lat[36] == 0 and ds['slp'].shape[1:] == (73, 144) and ds['slp'].units.lower().startswith('pa')
        t = nc.num2date(ds['time'][:], ds['time'].units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        keep = np.array([not (x.month == 2 and x.day == 29) for x in t])
        assert keep.sum() == G.SPY and t[0].hour == 0 and (t[1] - t[0]).total_seconds() == 21600
        x = np.asarray(ds['slp'][:, :37, :].filled(np.nan), np.float32)[keep]
        assert np.isfinite(x).all()
        raw[(y - G.Y0) * G.SPY:(y - G.Y0 + 1) * G.SPY] = x
        ds.close()
    A = raw.reshape(G.NY, G.SPY, -1)
    clim = smooth_clim(A.mean(axis=0, dtype=np.float64).astype(np.float32))
    np.save(os.path.join(G.CACHE, 'clim_r1.npy'), clim)
    anom = np.lib.format.open_memmap(os.path.join(G.CACHE, 'anom_r1.npy'), mode='w+', dtype=np.float32, shape=(G.T, 37 * 144))
    for y in range(G.NY):
        anom[y * G.SPY:(y + 1) * G.SPY] = A[y] - clim
    anom.flush()
    print('wrote anom_r1.npy', anom.shape, 'SD %.2f hPa' % (anom[::50].std() / 100), flush=True)


# ------------------------------------------------------------------ 0.25 degree
def clim_0p25():
    g = open_store('0p25')
    v = g['mean_sea_level_pressure']                         # (time, lat desc 90..-90, lon)
    idx = G.noleap_time_index()
    acc = np.zeros((G.SPY, 361, 1440), np.float32)
    t0 = time.time()
    B = 256
    with ThreadPoolExecutor(32) as ex:
        for a in range(0, G.T, B):
            ii = idx[a:a + B]
            xs = ex.map(lambda i: read_retry(lambda: np.asarray(v[int(i), :361, :])), ii)
            for s, x in zip(range(a, a + len(ii)), xs):
                acc[s % G.SPY] += x - OFFSET
            if (a // B) % 20 == 0:
                print(f'  clim pass {a}/{G.T}  {time.time() - t0:.0f}s', flush=True)
    acc /= G.NY
    clim = smooth_clim(acc.reshape(G.SPY, -1))
    np.save(os.path.join(G.CACHE, 'clim_0p25.npy'), clim)
    print(f'wrote clim_0p25.npy in {time.time() - t0:.0f}s', flush=True)


def _stream_worker(year):
    part = os.path.join(G.DATA, 'parts', f'feat_real_0p25_{year}_' + '{sign}.npz')
    if all(os.path.exists(part.format(sign=s)) for s in 'CA'):
        return year, 0.0
    t0 = time.time()
    g = open_store('0p25')
    v = g['mean_sea_level_pressure']
    idx = G.noleap_time_index()
    clim = np.load(os.path.join(G.CACHE, 'clim_0p25.npy'), mmap_mode='r')
    P, valid, img = G.projection_operator('0p25')
    s0 = (year - G.Y0) * G.SPY
    per = []
    with ThreadPoolExecutor(6) as ex:
        for a in range(s0, s0 + G.SPY, 48):
            ss = range(a, min(a + 48, s0 + G.SPY))
            xs = list(ex.map(lambda s: read_retry(lambda: np.asarray(v[int(idx[s]), :361, :])), ss))
            for s, x in zip(ss, xs):
                f = (x.reshape(-1).astype(np.float64) - OFFSET) - clim[s % G.SPY]
                per.append(G.field_features(f, P, valid, img))
    G.save_features(part, np.arange(s0, s0 + G.SPY), per, G.SPY, slot0=s0)
    return year, time.time() - t0


def features_0p25(workers=7):
    os.makedirs(os.path.join(G.DATA, 'parts'), exist_ok=True)
    G.projection_operator('0p25')
    t0 = time.time()
    with Pool(workers) as pool:
        for year, dt in pool.imap_unordered(_stream_worker, range(G.Y0, G.Y1 + 1)):
            print(f'  {year} done ({dt:.0f}s; elapsed {time.time() - t0:.0f}s)', flush=True)
    for sign in 'CA':
        parts = [np.load(p) for p in sorted(glob.glob(os.path.join(G.DATA, 'parts', f'feat_real_0p25_*_{sign}.npz')))]
        assert len(parts) == G.NY
        cat = {k: np.concatenate([p[k] for p in parts]) for k in ('slot', 'peak', 'saddle', 'peak_idx', 'parent_idx', 'half_km2', 'agg')}
        np.savez_compressed(os.path.join(G.DATA, f'feat_real_0p25_{sign}.npz'), slot0=0, nslots=G.T, c=G.GRIDS['0p25']['c'], **cat)
    print(f'features real 0p25: {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'download-1p5':
        download_1p5()
    elif cmd == 'features-1p5':
        features_from_array(os.path.join(G.CACHE, 'anom_1p5.npy'), '1p5', 'real')
    elif cmd == 'download-r1':
        download_r1()
    elif cmd == 'features-r1':
        features_from_array(os.path.join(G.CACHE, 'anom_r1.npy'), 'r1', 'real')
    elif cmd == 'clim-0p25':
        clim_0p25()
    elif cmd == 'features-0p25':
        features_0p25(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    else:
        raise SystemExit(__doc__)
