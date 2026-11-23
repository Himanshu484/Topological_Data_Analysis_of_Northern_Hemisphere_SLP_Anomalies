"""
ERA5 daily-mean mean sea level pressure, 1979-2022, on the NCEP R1 2.5 deg grid
(0-90 N, lat descending, lon 0..357.5), 29 Feb removed: the same layout as
common.build_anomalies reads from slp.daily.nc.

Source: WeatherBench2's public copy of ERA5 (no credentials needed),
gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr
i.e. ERA5 msl conservatively averaged from 0.25 deg to 1 deg, 6-hourly.  The store
ends 2023-01-10, hence 1979-2022.

Daily mean = mean of 00/06/12/18 UTC (as R1's daily means).  1 deg -> 2.5 deg is
first-order conservative: overlap weights in sin(lat) and in lon, applied
separably, rows normalised.  Target cells are centre +- 1.25 deg, clipped at 90 N.

Needs zarr<3 and gcsfs; run it from a virtual environment layered on the main one:
    python -m venv --system-site-packages <venv>; <venv>/bin/pip install "zarr<3" gcsfs
    <venv>/bin/python download_era5.py
Writes data/processed_data/experiments/gaussian_null_era5/era5_msl_daily_r1grid_1979_2022.npy
(float32, (44*365, 37, 144), Pa).  About 4 GB is read; ~20-40 min.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DATA = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'gaussian_null_era5')
STORE = 'weatherbench2/datasets/era5/1959-2023_01_10-6h-360x181_equiangular_with_poles_conservative.zarr'
Y0, Y1 = 1979, 2022


def overlap_matrix(src_lo, src_hi, dst_lo, dst_hi, period=None):
    """W[i, j] = measure of dst cell i intersected with src cell j, rows normalised."""
    W = np.zeros((len(dst_lo), len(src_lo)))
    shifts = [0.0] if period is None else [-period, 0.0, period]
    for i in range(len(dst_lo)):
        for s in shifts:
            ov = np.minimum(dst_hi[i], src_hi + s) - np.maximum(dst_lo[i], src_lo + s)
            W[i] += np.clip(ov, 0, None)
    return W / W.sum(1, keepdims=True)


def regrid_matrices(src_lat, src_lon):
    tlat = np.arange(90, -0.1, -2.5)
    tlon = np.arange(0, 360, 2.5)
    d = 0.5 * (src_lat[1] - src_lat[0])
    s_lo = np.sin(np.radians(np.clip(src_lat - d, -90, 90)))
    s_hi = np.sin(np.radians(np.clip(src_lat + d, -90, 90)))
    t_lo = np.sin(np.radians(np.clip(tlat - 1.25, -90, 90)))
    t_hi = np.sin(np.radians(np.clip(tlat + 1.25, -90, 90)))
    Wlat = overlap_matrix(s_lo, s_hi, t_lo, t_hi)
    dl = 0.5 * (src_lon[1] - src_lon[0])
    Wlon = overlap_matrix(src_lon - dl, src_lon + dl, tlon - 1.25, tlon + 1.25, period=360.0)
    return Wlat, Wlon


def main():
    import gcsfs
    import zarr
    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, f'era5_msl_daily_r1grid_{Y0}_{Y1}.npy')
    fs = gcsfs.GCSFileSystem(token='anon')
    g = zarr.open_consolidated(fs.get_mapper(STORE), mode='r')
    v = g['mean_sea_level_pressure']                       # (time, lon, lat)
    lat = np.asarray(g['latitude'][:], float)
    lon = np.asarray(g['longitude'][:], float)
    hours = np.asarray(g['time'][:], np.int64)
    assert g['time'].attrs['units'] == 'hours since 1959-01-01'
    assert (np.diff(hours) == 6).all() and hours[0] == 0
    Wlat, Wlon = regrid_matrices(lat, lon)
    jl = np.nonzero(Wlat.sum(0) > 0)[0]
    Wlat_s, j0, j1 = Wlat[:, jl], jl[0], jl[-1] + 1
    assert (jl == np.arange(j0, j1)).all()
    base = np.datetime64('1959-01-01T00', 'h')

    def year(y):
        t0 = int((np.datetime64(f'{y}-01-01T00', 'h') - base).astype(int)) // 6
        t1 = int((np.datetime64(f'{y + 1}-01-01T00', 'h') - base).astype(int)) // 6
        for attempt in range(5):
            try:
                x = np.asarray(v[t0:t1, :, j0:j1], np.float64)   # (4*ndays, lon, lat)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        nd = (t1 - t0) // 4
        daily = x.reshape(nd, 4, x.shape[1], x.shape[2]).mean(1)       # (nd, lon, lat_sub)
        days = np.arange(np.datetime64(f'{y}-01-01'), np.datetime64(f'{y + 1}-01-01'))
        keep = np.array([not (str(d)[5:] == '02-29') for d in days])
        daily = daily[keep]
        assert len(daily) == 365
        r = np.einsum('ij,djk,lk->dli', Wlon, daily, Wlat_s)           # (365, 37 lat, 144 lon)
        return y, r.astype(np.float32)

    years = list(range(Y0, Y1 + 1))
    res = np.empty((len(years) * 365, 37, 144), np.float32)
    t = time.time()
    with ThreadPoolExecutor(4) as ex:
        for y, r in ex.map(year, years):
            i = (y - Y0) * 365
            res[i:i + 365] = r
            print(f'{y} done ({time.time() - t:.0f}s)  mean {r.mean() / 100:.1f} hPa', flush=True)
    np.save(out_path, res)
    print('saved', out_path, res.shape)


if __name__ == '__main__':
    main()
