import numpy as np
import warnings
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")


def reshape_to_polar_matrix(data_1d, lat_index, lon_index, lat_range=(0, 90), lon_range=(0, 360), 
                           matrix_size=None, default_value="smallest",
                           lat_values=None, lon_values=None, lat_descending=True):
    """
    Reshape 1D data array into circular polar matrix for North Polar Stereo view.
    
    Parameters:
    -----------
    data_1d : array_like
        1D input data array (like slp_2023[n])
    lat_index : array_like
        Latitude indices for reshaping
    lon_index : array_like  
        Longitude indices for reshaping
    matrix_size : int, optional
        Size of output matrix. If None, calculated automatically from input data.
    default_value : float or str
        Value to fill areas outside data coverage. 
        If "smallest", uses one value smaller than data minimum.
        If "largest", uses one value larger than data maximum.
    lat_values, lon_values : array_like, optional
        The actual coordinate values of the input grid rows/columns, in the
        order the rows/columns appear in `data_1d`.  Pass these whenever they
        are available; they remove every assumption below.
    lat_descending : bool
        Only used when `lat_values` is None.  True (the default) means row 0 of
        the input grid is the NORTHERNMOST row, which is how NCEP-NCAR
        `slp.daily.nc` stores latitude (lat[0] = 90 N).

    Notes
    -----
    The disc is parameterised with the North Pole at the centre and
    `lat_range[0]` at the rim.  Two conventions must therefore hold, and both
    were previously violated:

    * the latitude axis handed to the interpolator must run in the same
      direction as the data rows.  Declaring a descending grid as ascending
      does not raise (the vector is still monotonic) but mirrors the field
      about the equator, so a feature at true latitude L is reported at
      90 - L, and the disc centre ends up holding equatorial data.
    * longitude is a PERIODIC axis: `n_lon` cells span the full circle, so the
      grid step is span/n_lon (2.5 deg for 144 columns), not span/(n_lon - 1)
      as `np.linspace(0, 360, 144)` gives.  The linspace form stretches the
      field zonally by 0.7 %, accumulating to a full grid box by the far end
      of the axis.  The circle is closed explicitly below so that no azimuthal
      seam of padding is left between the last column and the first.
    """
    # Reshape 1D data to 2D
    data = data_1d.reshape((len(lat_index), len(lon_index)))
    n_lat, n_lon = data.shape
    
    # Handle special default_value cases
    if isinstance(default_value, str):
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_range = data_max - data_min
        
        if default_value.lower() == "smallest":
            # One step smaller than minimum (use 1% of data range or 1 if range is small)
            step = max(1, data_range * 0.01)
            default_value = data_min - step
            #print(f"Default value set to 'smallest': {default_value:.2e} (data min: {data_min:.2e})")
            
        elif default_value.lower() == "largest":
            # One step larger than maximum (use 1% of data range or 1 if range is small)
            step = max(1, data_range * 0.01)
            default_value = data_max + step
            #print(f"Default value set to 'largest': {default_value:.2e} (data max: {data_max:.2e})")
            
        else:
            raise ValueError(f"Invalid string for default_value: '{default_value}'. Use 'smallest', 'largest', or a numeric value.")
    
    # Calculate matrix size automatically if not provided
    if matrix_size is None:
        # Use the maximum dimension to ensure good coverage
        # Multiply by 1.2 to ensure we don't lose resolution at edges
        matrix_size = int(max(n_lat, n_lon) * 1.2)
        # Make it even for better centering
        if matrix_size % 2 == 1:
            matrix_size += 1
    
    # print(f"Input data shape: ({n_lat}, {n_lon})")
    # print(f"Output matrix size: {matrix_size} x {matrix_size}")
    # print(f"Data range: {np.nanmin(data):.2e} to {np.nanmax(data):.2e}")
    # print(f"Default value: {default_value:.2e}")
    
    # Original lat/lon coordinates — see the Notes section of the docstring.
    if lat_values is not None:
        orig_lat = np.asarray(lat_values, dtype=float)
    elif lat_descending:
        orig_lat = np.linspace(lat_range[1], lat_range[0], n_lat)
    else:
        orig_lat = np.linspace(lat_range[0], lat_range[1], n_lat)

    lon_span = float(lon_range[1] - lon_range[0])
    if lon_values is not None:
        orig_lon = np.asarray(lon_values, dtype=float)
    else:
        orig_lon = lon_range[0] + np.arange(n_lon) * (lon_span / n_lon)

    # RegularGridInterpolator wants ascending axes; flip the rows with the axis
    # so the pairing between a row and its latitude is preserved.
    if orig_lat[0] > orig_lat[-1]:
        orig_lat = orig_lat[::-1]
        data = data[::-1, :]

    # Close the longitude circle: repeat column 0 at lon_0 + span so the sector
    # between the last column and the first is interpolated rather than filled
    # with the (extremal) padding value.
    orig_lon_closed = np.concatenate([orig_lon, [orig_lon[0] + lon_span]])
    data_closed = np.concatenate([data, data[:, :1]], axis=1)

    # Create interpolator for original data
    interpolator = RegularGridInterpolator((orig_lat, orig_lon_closed), data_closed,
                                         bounds_error=False, fill_value=default_value)
    
    # Create circular polar matrix
    center = matrix_size // 2
    polar_matrix = np.full((matrix_size, matrix_size), default_value)
    
    # Create coordinate arrays for the circular matrix
    y, x = np.ogrid[:matrix_size, :matrix_size]
    x_centered = x - center
    y_centered = y - center
    
    # Convert to polar coordinates
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Convert angle to longitude, wrapped into [lon_range[0], lon_range[0]+span)
    lon_polar = lon_range[0] + (theta * 180 / np.pi + 90 - lon_range[0]) % lon_span
    
    # Convert radius to latitude (center=90°N, edge=0°N)
    max_radius = center
    lat_polar = lat_range[1] - (r / max_radius) * (lat_range[1] - lat_range[0])
    
    # Create mask for valid regions (within the circle and latitude range)
    valid_mask = (r <= max_radius) & (lat_polar >= lat_range[0]) & (lat_polar <= lat_range[1])
    
    # Interpolate data for valid points
    valid_points = np.column_stack([lat_polar[valid_mask], lon_polar[valid_mask]])
    interpolated_values = interpolator(valid_points)
    
    # Fill the polar matrix
    polar_matrix[valid_mask] = interpolated_values
    
    return polar_matrix


import numpy as np


