"""
Stack the downloaded Z500 onto the 365-day no-leap calendar (29 Feb dropped, as for
the SLP discs), and build anomaly fields on the 2.5 deg lat-lon grid.

  z500_full.npy   (76, 365, 37, 144) float32 gpm, 1948-2023
  z500_anom.npy   same, anomaly from a 1979-2023 calendar-day climatology smoothed
                  to its mean + first 3 annual harmonics
  slp_anom.npy    (76, 365, 37, 144) float32 Pa, anomaly from the RAW 76-year
                  calendar-day mean, i.e. exactly the field the SLP discs project

    python scripts/experiments/blocking/build_fields.py
"""
import os

import netCDF4 as nc
import numpy as np

from common import CACHE, ROOT, Y0, Y1, EVAL0, EVAL1


def harmonic_smooth(clim, nh=3):
    F = np.fft.rfft(clim, axis=0)
    F[nh + 1:] = 0
    return np.fft.irfft(F, n=clim.shape[0], axis=0)


def main():
    ny = Y1 - Y0 + 1
    Z = np.lib.format.open_memmap(os.path.join(CACHE, 'z500_full.npy'), 'w+', np.float32, (ny, 365, 37, 144))
    for y in range(Y0, Y1 + 1):
        d = np.load(os.path.join(CACHE, f'z500_{y}.npz'))
        z, dates = d['z'], d['dates']
        keep = ~(((dates // 100) % 100 == 2) & (dates % 100 == 29))
        z = z[keep]
        assert z.shape[0] == 365, (y, z.shape)
        Z[y - Y0] = z
    Z.flush()
    clim = Z[EVAL0 - Y0:EVAL1 - Y0 + 1].astype(np.float64).mean(0)
    clim = harmonic_smooth(clim).astype(np.float32)
    A = np.lib.format.open_memmap(os.path.join(CACHE, 'z500_anom.npy'), 'w+', np.float32, Z.shape)
    for j in range(ny):
        A[j] = Z[j] - clim
    A.flush()
    np.save(os.path.join(CACHE, 'z500_clim.npy'), clim)
    print('Z500 done; 4 Feb 2012 anomaly max (gpm):', float(A[2012 - Y0, 34].max()))

    ds = nc.Dataset(os.path.join(ROOT, 'data', 'raw', 'slp.daily.nc'))
    lat = np.array(ds.variables['lat'][:])
    j = np.where(lat >= 0)[0]
    slp = ds.variables['slp']
    S = np.lib.format.open_memmap(os.path.join(CACHE, 'slp_anom.npy'), 'w+', np.float32, (ny, 365, 37, 144))
    d0 = np.datetime64(f'{Y0}-01-01')
    t = 0
    for y in range(Y0, Y1 + 1):
        n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        x = np.array(slp[t:t + n, j[0]:j[-1] + 1, :], dtype=np.float64)
        if n == 366:
            x = np.delete(x, 59, axis=0)
        S[y - Y0] = x.astype(np.float32)
        t += n
    ds.close()
    assert (d0 + np.timedelta64(t - 1, 'D')).astype(str) == f'{Y1}-12-31'
    acc = np.zeros((365, 37, 144))
    for yy in range(ny):
        acc += S[yy]
    m = (acc / ny).astype(np.float32)
    for yy in range(ny):
        S[yy] = S[yy] - m
    S.flush()
    print('SLP done; 4 Feb 2012 anomaly max (hPa):', float(S[2012 - Y0, 34].max()) / 100)


if __name__ == '__main__':
    main()
