"""
Build every polar disc of the daily R1 anomaly, 1948-2023 (two files per day:
the anomaly for the sublevel filtration and its negative for the superlevel one).

Two coordinate defects of an earlier version of this step are fixed here:

  A  the latitude axis handed to RegularGridInterpolator was declared
     ascending while NCEP stores it descending, mirroring the field about the
     equator (true lat L reported at 90 - L, equator at the centre of the disc)
  B  the 21-column roll (52.5 deg) was never undone, and
     np.linspace(0, 360, 144) put the grid on a 2.51748 deg step

Both live in `globe_visualization.reshape_to_polar_matrix`, which now takes the
grid's own coordinate vectors; this script passes them explicitly so nothing is
inferred.

The climatology is unchanged in definition: a raw 76-sample per-calendar-day
mean on the 365-day no-leap calendar (29 February deleted).  It is not smoothed.

    python scripts/regenerate_polar_discs.py [--dry-run] [--years 2012 2003]
"""
import argparse
import os
import sys
import time

import numpy as np
import netCDF4 as nc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import globe_visualization as gv          # noqa: E402
import disc_utils as ft             # noqa: E402

START_YEAR, END_YEAR = 1948, 2023
N_YEARS = END_YEAR - START_YEAR + 1
OUT_BASE = os.path.join(ROOT, 'data', 'processed_data', 'SLP_data_years')


def build_anomalies():
    """Return (anom, lat_nh, lon, lat_index, lon_index) with anom (76, 365, 37, 144)."""
    ds = nc.Dataset(os.path.join(ROOT, 'data', 'raw', 'slp.daily.nc'))
    lat = np.array(ds.variables['lat'][:])
    lon = np.array(ds.variables['lon'][:])

    lat_index = [i for i in range(len(lat)) if 0 <= lat[i] <= 90]
    lon_index = list(range(len(lon)))
    lat_nh = lat[lat_index]

    print(f'  grid: {len(lat_index)} lat x {len(lon_index)} lon, '
          f'lat[0]={lat_nh[0]} lat[-1]={lat_nh[-1]}, lon step {lon[1]-lon[0]}')
    assert lat_nh[0] > lat_nh[-1], 'expected a descending latitude axis'

    # NO 21-column roll: the grid keeps its own longitudes.
    slp = np.array(ds.variables['slp'][:], dtype=np.float64)[:, lat_index, :]
    ds.close()

    # Delete 29 February to give 365 slots/year.
    keep = []
    d0 = np.datetime64(f'{START_YEAR}-01-01')
    for t in range(len(slp)):
        d = (d0 + np.timedelta64(t, 'D')).astype('datetime64[D]').astype(object)
        if d.year > END_YEAR:
            break
        if not (d.month == 2 and d.day == 29):
            keep.append(t)
    slp = slp[keep][:N_YEARS * 365]
    assert slp.shape[0] == N_YEARS * 365, slp.shape

    arr = slp.reshape(N_YEARS, 365, len(lat_index), len(lon_index))
    # Accumulate the per-calendar-day climatology in float64, then store the
    # anomaly in float32 — the dtype of the original files, so the regenerated
    # discs are byte-for-byte comparable in size and precision to the published
    # ones and the only difference is the coordinate fix.
    anom = (arr - arr.mean(axis=0, keepdims=True)).astype(np.float32)
    return anom, lat_nh, lon, lat_index, lon_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--years', type=int, nargs='*', default=None)
    args = ap.parse_args()

    print('Building anomalies (no roll, no-leap calendar) …')
    anom, lat_nh, lon, lat_index, lon_index = build_anomalies()

    # Sanity check on the paper's own case study before writing 6 GB.
    y, d = 2012 - START_YEAR, ft.date_to_noleap_slot(__import__('datetime').date(2012, 2, 4)) - 1
    field = anom[y, d]
    r, c = np.unravel_index(np.argmax(field), field.shape)
    print(f'  4 Feb 2012 maximum: {lat_nh[r]:.1f} N, {lon[c]:.1f} E, {field[r,c]/100:+.2f} hPa')
    assert abs(lat_nh[r] - 65.0) < 1e-6 and abs(lon[c] - 45.0) < 1e-6, \
        'the 4 Feb 2012 maximum is not where it should be — check the anomaly build'

    polar = gv.reshape_to_polar_matrix(field.flatten(), lat_index, lon_index,
                                       default_value='smallest',
                                       lat_values=lat_nh, lon_values=lon)
    pr, pc = np.unravel_index(np.nanargmax(polar), polar.shape)
    la, lo = ft._rowcol_to_latlon(pr, pc, polar.shape[0], (0, 90), (0, 360))
    print(f'  round trip through the disc: {la:.1f} N, {lo:.1f} E')
    assert abs(la - 65.0) < 2.5 and abs((lo - 45.0 + 180) % 360 - 180) < 2.5, \
        'projection round trip failed — refusing to regenerate'

    if args.dry_run:
        print('dry run — nothing written')
        return

    years = args.years or list(range(START_YEAR, END_YEAR + 1))
    t0 = time.time()
    for yi, year in enumerate(years):
        j = year - START_YEAR
        year_dir = os.path.join(OUT_BASE, str(year))
        os.makedirs(year_dir, exist_ok=True)
        for i in range(365):
            flat = anom[j, i].flatten()
            for tag, fill, sign in (('sup', 'smallest', -1.0), ('sub', 'largest', 1.0)):
                m = gv.reshape_to_polar_matrix(flat, lat_index, lon_index,
                                               default_value=fill,
                                               lat_values=lat_nh, lon_values=lon)
                np.save(os.path.join(year_dir, f'slp_{tag}_{year}_day_{i+1}.npy'), sign * m)
        el = time.time() - t0
        print(f'\r  {year}  ({yi+1}/{len(years)})  {el:6.1f}s elapsed, '
              f'{el/(yi+1)*(len(years)-yi-1):6.1f}s left', end='', flush=True)
    print(f'\nDone: {len(years)} years x 365 days x 2 filtrations in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
