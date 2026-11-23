"""
Round-trip regression tests for the polar reprojection.  They guard against two
defects of an earlier version of the code.

Defect A: the latitude axis handed to RegularGridInterpolator was declared
          ascending while the NCEP grid stores latitude descending, mirroring
          the field about the equator (reported lat = 90 - true lat) and
          putting equatorial data at the centre of the disc.
Defect B: `np.linspace(0, 360, 144)` gives a 2.51748 deg step on a 2.5 deg
          grid (a 0.7 % zonal stretch), and the grid was rolled by 21
          columns (52.5 deg) without ever undoing it.

Run:  python -m pytest tests/test_projection.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import globe_visualization as gv          # noqa: E402
import disc_utils as ft             # noqa: E402

N_LAT, N_LON = 37, 144                    # 0-90 N, 0-357.5 E on the 2.5 deg NCEP grid
MATRIX_SIZE = 172
LAT_RANGE, LON_RANGE = (0, 90), (0, 360)

# NCEP `slp.daily.nc` axes, in file order
NCEP_LAT = np.linspace(90.0, 0.0, N_LAT)          # descending: row 0 = 90 N
NCEP_LON = np.arange(N_LON) * 2.5                 # ascending:  col 0 = 0 E


def _delta_field(lat_deg, lon_deg):
    """A field that is 0 everywhere except 1.0 at (lat_deg, lon_deg)."""
    field = np.zeros((N_LAT, N_LON))
    row = int(np.argmin(np.abs(NCEP_LAT - lat_deg)))
    col = int(np.argmin(np.abs(NCEP_LON - lon_deg)))
    field[row, col] = 1.0
    return field


def _project_and_locate(lat_deg, lon_deg):
    """Project a delta at (lat, lon) onto the disc and read its position back."""
    field = _delta_field(lat_deg, lon_deg)
    polar = gv.reshape_to_polar_matrix(
        field.flatten(), range(N_LAT), range(N_LON),
        lat_range=LAT_RANGE, lon_range=LON_RANGE,
        matrix_size=MATRIX_SIZE, default_value=0.0,
    )
    row, col = np.unravel_index(np.nanargmax(polar), polar.shape)
    return ft._rowcol_to_latlon(row, col, MATRIX_SIZE, LAT_RANGE, LON_RANGE)


# Half a grid box meridionally; zonally the disc resolution degrades towards
# the pole, so the tolerance is one grid box there.
LAT_TOL, LON_TOL = 2.5, 2.5


@pytest.mark.parametrize("true_lat,true_lon", [
    (65.0, 45.0),     # the 4 Feb 2012 blocking high — the paper's own case study
    (50.0, 0.0),      # on the 0/360 seam
    (50.0, 357.5),    # the other side of the seam
    (40.0, 180.0),    # antimeridian
    (30.0, 270.0),    # far end of the axis, where the linspace stretch peaked
    (75.0, 120.0),
])
def test_projection_roundtrip(true_lat, true_lon):
    lat, lon = _project_and_locate(true_lat, true_lon)
    dlon = abs((lon - true_lon + 180) % 360 - 180)
    assert abs(lat - true_lat) < LAT_TOL, f"latitude {lat:.2f} != {true_lat}"
    assert dlon < LON_TOL, f"longitude {lon:.2f} != {true_lon}"


def test_pole_is_at_the_centre_of_the_disc():
    """A zonally uniform ring at 90 N must land at the centre, not the rim."""
    field = np.zeros((N_LAT, N_LON))
    field[0, :] = 1.0                       # row 0 = 90 N
    polar = gv.reshape_to_polar_matrix(
        field.flatten(), range(N_LAT), range(N_LON),
        lat_range=LAT_RANGE, lon_range=LON_RANGE,
        matrix_size=MATRIX_SIZE, default_value=0.0,
    )
    c = MATRIX_SIZE // 2
    assert polar[c, c] > 0.9, "the North Pole is not at the centre of the disc"
    assert polar[c, 2] < 0.1, "polar data is smeared around the rim"


def test_equator_is_at_the_rim_of_the_disc():
    field = np.zeros((N_LAT, N_LON))
    field[-1, :] = 1.0                      # last row = 0 N
    polar = gv.reshape_to_polar_matrix(
        field.flatten(), range(N_LAT), range(N_LON),
        lat_range=LAT_RANGE, lon_range=LON_RANGE,
        matrix_size=MATRIX_SIZE, default_value=0.0,
    )
    c = MATRIX_SIZE // 2
    assert polar[c, c] < 0.1, "equatorial data sits at the centre of the disc"
    # column 0 of row c is exactly on the rim (r = max_radius, lat = 0 N)
    assert polar[c, 0] > 0.9, "the equator is not at the rim of the disc"


def test_longitude_step_is_the_grid_step_not_the_linspace_step():
    """
    The 0.7 % zonal stretch of linspace(0, 360, 144) accumulates to a full grid
    box at the far end of the axis; a two-point check pins the step down.
    """
    _, lon_a = _project_and_locate(45.0, 90.0)
    _, lon_b = _project_and_locate(45.0, 270.0)
    assert abs((lon_b - lon_a) - 180.0) < 2.5


def test_no_azimuthal_seam_of_padding():
    """
    Longitude is periodic: the sector between the last column and the first
    must be interpolated, never filled with the extremal padding value.
    """
    field = np.zeros((N_LAT, N_LON))
    polar = gv.reshape_to_polar_matrix(
        field.flatten(), range(N_LAT), range(N_LON),
        lat_range=LAT_RANGE, lon_range=LON_RANGE,
        matrix_size=MATRIX_SIZE, default_value=-999.0,
    )
    c = MATRIX_SIZE // 2
    y, x = np.ogrid[:MATRIX_SIZE, :MATRIX_SIZE]
    inside = np.sqrt((x - c) ** 2 + (y - c) ** 2) <= c
    assert not (polar[inside] == -999.0).any(), "padding leaked inside the disc"


def test_explicit_coordinate_vectors_win():
    """Passing the grid's own axes must reproduce the same round trip."""
    field = _delta_field(65.0, 45.0)
    polar = gv.reshape_to_polar_matrix(
        field.flatten(), range(N_LAT), range(N_LON),
        lat_range=LAT_RANGE, lon_range=LON_RANGE, matrix_size=MATRIX_SIZE,
        default_value=0.0, lat_values=NCEP_LAT, lon_values=NCEP_LON,
    )
    row, col = np.unravel_index(np.nanargmax(polar), polar.shape)
    lat, lon = ft._rowcol_to_latlon(row, col, MATRIX_SIZE, LAT_RANGE, LON_RANGE)
    assert abs(lat - 65.0) < LAT_TOL and abs((lon - 45.0 + 180) % 360 - 180) < LON_TOL
