"""
Download NCEP-NCAR Reanalysis 1 daily-mean sea-level pressure, 1948-2023, from
the NOAA Physical Sciences Laboratory and write it as one file,
data/raw/slp.daily.nc (time x 73 x 144, float32, Pa; about 0.6 GB), which is
what every daily script of this repository reads.

    python scripts/download_r1_daily.py                # 76 yearly files, ~8 MB each
    python scripts/download_r1_daily.py --out other.nc # e.g. to compare with an existing copy

The yearly files are kept in data/raw/r1_daily/ and are not downloaded again.
Source: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
"""
import argparse
import os
import time
import urllib.request

import numpy as np
import netCDF4 as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = 'https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/slp.{year}.nc'
Y0, Y1 = 1948, 2023


def fetch(year, folder, tries=5):
    path = os.path.join(folder, f'slp.{year}.nc')
    if os.path.exists(path):
        return path
    for a in range(tries):
        try:
            urllib.request.urlretrieve(URL.format(year=year), path + '.part')
            os.replace(path + '.part', path)
            return path
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(5 * (a + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'raw', 'slp.daily.nc'))
    a = ap.parse_args()
    folder = os.path.join(ROOT, 'data', 'raw', 'r1_daily')
    os.makedirs(folder, exist_ok=True)
    out = nc.Dataset(a.out, 'w', format='NETCDF4')
    n = 0
    for year in range(Y0, Y1 + 1):
        ds = nc.Dataset(fetch(year, folder))
        v = ds['slp']
        if n == 0:
            out.createDimension('time', None)
            out.createDimension('lat', len(ds['lat']))
            out.createDimension('lon', len(ds['lon']))
            for name in ('lat', 'lon'):
                x = out.createVariable(name, 'f4', (name,))
                x[:] = ds[name][:]
                x.units = ds[name].units
            t = out.createVariable('time', 'f8', ('time',))
            t.units = ds['time'].units
            s = out.createVariable('slp', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=1)
            s.units, s.long_name = v.units, v.long_name
            out.title = 'NCEP-NCAR Reanalysis 1 daily-mean sea-level pressure, 1948-2023, concatenated from the PSL yearly files'
            out.source = URL
            assert ds['lat'][0] == 90 and ds['time'].units == 'hours since 1800-01-01 00:00:0.0', 'unexpected grid or calendar'
        k = len(ds['time'])
        out['time'][n:n + k] = ds['time'][:]
        out['slp'][n:n + k] = np.asarray(v[:].filled(np.nan), np.float32)
        n += k
        ds.close()
        print(year, k, flush=True)
    out.close()
    print('wrote', a.out, n, 'days')


if __name__ == '__main__':
    main()
