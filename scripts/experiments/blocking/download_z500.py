"""
Download NCEP-NCAR R1 daily-mean 500 hPa geopotential height, 0-90N, from PSL
THREDDS (OPeNDAP), one year per request, sequentially, with retries.

Cache: data/processed_data/experiments/blocking/z500_{year}.npz
       (z: float32 (ndays, 37, 144) in gpm, lat descending 90..0, lon 0..357.5,
        dates: int yyyymmdd)

    python scripts/experiments/blocking/download_z500.py [--years 1948 2023]
"""
import argparse
import datetime as dt
import os
import time

import netCDF4 as nc
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CACHE = os.path.join(ROOT, 'data', 'processed_data', 'experiments', 'blocking')
URL = 'https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis.dailyavgs/pressure/hgt.{y}.nc'


def fetch(year):
    ds = nc.Dataset(URL.format(y=year))
    lev = np.array(ds.variables['level'][:])
    lat = np.array(ds.variables['lat'][:])
    k = int(np.where(lev == 500)[0][0])
    j = np.where(lat >= 0)[0]
    z = np.array(ds.variables['hgt'][:, k, j[0]:j[-1] + 1, :], dtype=np.float32)
    t = np.array(ds.variables['time'][:])
    ds.close()
    base = dt.datetime(1800, 1, 1)
    dates = np.array([int((base + dt.timedelta(hours=float(h))).strftime('%Y%m%d')) for h in t])
    return z, dates, lat[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs=2, type=int, default=[1948, 2023])
    y0, y1 = ap.parse_args().years
    os.makedirs(CACHE, exist_ok=True)
    for y in range(y0, y1 + 1):
        out = os.path.join(CACHE, f'z500_{y}.npz')
        if os.path.exists(out):
            continue
        for attempt in range(6):
            try:
                z, dates, lat = fetch(y)
                assert z.shape[1:] == (37, 144) and lat[0] == 90 and lat[-1] == 0
                np.savez(out, z=z, dates=dates)
                print(f'{y}: {z.shape} ok', flush=True)
                break
            except Exception as e:  # noqa: BLE001
                wait = 5 * 2 ** attempt
                print(f'{y}: attempt {attempt + 1} failed ({e}); retry in {wait}s', flush=True)
                time.sleep(wait)
        else:
            raise SystemExit(f'giving up on {y}')
        time.sleep(1)


if __name__ == '__main__':
    main()
