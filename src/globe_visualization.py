import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")

def sort_by_d_minus_b(arr):
    """
    Sort a 2D array by the difference y - x (second column minus first column)
    
    Parameters:
    arr: numpy array with shape (n, 2) where each row is [x, y]
    
    Returns:
    numpy array sorted by y - x values
    """
    differences = arr[:, 1] - arr[:, 0]  # y - x
    sorted_indices = np.argsort(-differences)
    return arr[sorted_indices]

def create_standalone_colorbar(reading_list, lat_index, lon_index, std_scaling=False, std_scale=2.0, 
                              label='Hectopascal (hPa)', label_fontsize=14, tick_fontsize=12,
                              cmap='RdBu_r', figsize=(2, 4)):
    """
    Create a standalone colorbar figure with no plot.
    
    Parameters:
    reading_list: array of data values
    std_scaling: if True, use standard deviation scaling; if False, use absolute max
    std_scale: number of standard deviations for color scaling when std_scaling=True
    label: Label for the colorbar
    label_fontsize: Font size for the colorbar label
    tick_fontsize: Font size for the colorbar tick labels
    cmap: Colormap to use
    figsize: Figure size (width, height)
    
    Returns:
    fig: The figure object containing only the colorbar
    """
    # Get the data for scaling calculations
    data = reading_list.reshape((len(lat_index), len(lon_index)))
    
    # Determine color scale
    if std_scaling:
        std_dev = np.std(data)
        vmax = std_scale * std_dev
    else:
        vmax = np.max(np.abs(data))
    
    # Create figure and axis for colorbar only
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.3, 0.8])  # [left, bottom, width, height]
    
    # Create a ScalarMappable for the colorbar
    norm = Normalize(vmin=-vmax, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    
    # Create the colorbar
    cbar = plt.colorbar(sm, cax=ax)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(label, size=label_fontsize)
    
    return fig

def plot_on_globe(reading_list,lat_left, lat_right, lat_index, lon_index, std_scaling=False, std_scale=2.0):
    """
    Plot data on a global map using PlateCarree projection for better stability.
    """
    # Generate latitude and longitude arrays
    lat = np.linspace(lat_left, lat_right, reading_list.reshape((len(lat_index), len(lon_index))).shape[0])
    lon = np.linspace(-180, 180, reading_list.reshape((len(lat_index), len(lon_index))).shape[1])
    lon, lat = np.meshgrid(lon, lat)
    
    # Use PlateCarree projection instead of Robinson for better stability
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9], projection=ccrs.PlateCarree())
    
    # Set extent with safe bounds
    lat_min = max(lat_left, -85)
    lat_max = min(lat_right, 85)
    ax.set_extent([-180, 180, lat_min, lat_max], ccrs.PlateCarree())
    
    # Add simple features only
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.LAND, alpha=0.2, color='lightgray')
    
    # Get the data for plotting
    data = reading_list.reshape((len(lat_index), len(lon_index)))
    
    # Determine color scale
    if std_scaling:
        std_dev = np.std(data)
        vmax = std_scale * std_dev
    else:
        vmax = np.max(np.abs(data))
    
    # Plot the data
    cs = ax.pcolormesh(lon, lat, data, 
                       cmap='RdBu_r',
                       vmin=-vmax, 
                       vmax=vmax,
                       transform=ccrs.PlateCarree(),
                       shading='auto')
    
    # Add colorbar
    cax = fig.add_axes([0.82, 0.25, 0.03, 0.5])
    cbar = plt.colorbar(cs, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Pressure (Pa)', size=14)
    
    return plt

def plot_north_polar_stereo(reading_list, lat_index, lon_index, std_scaling=False, std_scale=2.0, 
                           lat_left=0, lat_right=90, lon_left=0, lon_right=360):
    """
    Plot data on North Polar Stereo projection with coastlines only.
    """
    # Generate coordinate arrays
    lat = np.linspace(lat_left, lat_right, reading_list.reshape((len(lat_index), len(lon_index))).shape[0])
    lon = np.linspace(lon_left, lon_right, reading_list.reshape((len(lat_index), len(lon_index))).shape[1])
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Get the data
    data = reading_list.reshape((len(lat_index), len(lon_index)))
    
    # Create mask for regions outside desired latitude range
    mask = (lat_mesh < lat_left) | (lat_mesh > lat_right)
    data_masked = np.ma.masked_where(mask, data)
    
    # Determine color scale
    if std_scaling:
        std_dev = np.std(data_masked.compressed())
        vmax = std_scale * std_dev
    else:
        vmax = np.max(np.abs(data_masked.compressed()))
    
    # Create figure with North Polar Stereo projection
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo())
    
    # Set extent to show the Arctic region
    ax.set_extent([-180, 180, lat_left-5, 90], ccrs.PlateCarree())
    
    # Set white background
    ax.set_facecolor('white')
    
    # Plot the data (masked data will appear white)
    cs = ax.pcolormesh(lon_mesh, lat_mesh, data_masked,
                      transform=ccrs.PlateCarree(),
                      cmap='RdBu_r',
                      vmin=-vmax,
                      vmax=vmax,
                      shading='auto',
                      zorder=1)
    
    # Add only coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=1, alpha=0.7, color='black', zorder=3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', 
                       shrink=0.6, pad=0.05, aspect=20)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Sea Level Pressure (Hectopascal hPa)', size=14, weight='bold')
    
    # # Add title
    # plt.title(f'North Polar Stereo View\nData Coverage: {lat_left}°N - {lat_right}°N', 
    #           size=16, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_simple_heatmap(polar_matrix, title="Polar Matrix Heatmap"):
    """
    Simple heatmap plot of the polar matrix.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Simple heatmap
    im = ax.imshow(polar_matrix, cmap='RdBu_r', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pressure (Pa)', size=12)
    
    # Simple title
    ax.set_title(title, size=14, weight='bold', pad=20)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig, ax

def reshape_to_polar_matrix(data_1d, lat_index, lon_index, lat_range=(0, 90), lon_range=(0, 360), 
                           matrix_size=None, default_value="smallest"):
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
    
    # Original lat/lon coordinates
    orig_lat = np.linspace(lat_range[0], lat_range[1], n_lat)
    orig_lon = np.linspace(lon_range[0], lon_range[1], n_lon)
    
    # Create interpolator for original data
    interpolator = RegularGridInterpolator((orig_lat, orig_lon), data, 
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
    
    # Convert angle to longitude (0-360)
    lon_polar = (theta * 180 / np.pi + 90) % 360
    
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


from datetime import date, timedelta
import numpy as np
import os

def plot_north_polar_stereo_from_matrix(polar_matrix, matrix_size=None, lat_range=(0, 90), lon_range=(0, 360),
                                       std_scaling=False, std_scale=2.0, title=None, figsize=(12, 12)):
    """
    Plot polar matrix data on North Polar Stereo projection with coastlines only.
    
    Parameters:
    -----------
    polar_matrix : np.array
        2D polar matrix (e.g., from reshape_to_polar_matrix)
    matrix_size : int, optional
        Size of the matrix. If None, inferred from polar_matrix.shape
    lat_range : tuple
        Latitude range (min_lat, max_lat)
    lon_range : tuple  
        Longitude range (min_lon, max_lon)
    std_scaling : bool
        Whether to use standard deviation scaling
    std_scale : float
        Standard deviation scaling factor
    title : str, optional
        Custom title for the plot
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    
    # Get matrix size
    if matrix_size is None:
        matrix_size = polar_matrix.shape[0]  # Assume square matrix
    
    height, width = polar_matrix.shape
    center_y, center_x = height // 2, width // 2
    
    # Create coordinate arrays for the entire matrix
    y_indices, x_indices = np.ogrid[:height, :width]
    x_centered = x_indices - center_x
    y_centered = y_indices - center_y
    
    # Convert matrix indices to polar coordinates
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Convert polar coordinates to geographic coordinates
    # Longitude: angle in radians to degrees (0-360)
    lon_matrix = (theta * 180 / np.pi + 90) % 360
    
    # Latitude: radius to latitude (center = max_lat, edge = min_lat)
    max_radius = min(center_x, center_y)
    lat_matrix = lat_range[1] - (r / max_radius) * (lat_range[1] - lat_range[0])
    
    # Handle longitude range conversion if needed
    if lon_range[0] == -180 and lon_range[1] == 180:
        lon_matrix = lon_matrix - 180  # Convert to -180 to 180 range
    
    # Create mask for areas outside the valid circle and latitude range
    circle_mask = r <= max_radius
    lat_mask = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1])
    valid_mask = circle_mask & lat_mask
    
    # Mask the data
    data_masked = np.ma.masked_where(~valid_mask, polar_matrix)
    
    # Determine color scale
    if std_scaling:
        valid_data = data_masked.compressed()
        if len(valid_data) > 0:
            std_dev = np.std(valid_data)
            vmax = std_scale * std_dev
        else:
            vmax = 1.0
    else:
        valid_data = data_masked.compressed()
        if len(valid_data) > 0:
            vmax = np.max(np.abs(valid_data))
        else:
            vmax = 1.0
    
    # Create figure with North Polar Stereo projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo())
    
    # Set extent to show the Arctic region
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0]-5, lat_range[1]], ccrs.PlateCarree())
    
    # Set white background
    ax.set_facecolor('white')
    
    # Plot the data using pcolormesh with geographic coordinates
    cs = ax.pcolormesh(lon_matrix, lat_matrix, data_masked,
                      transform=ccrs.PlateCarree(),
                      cmap='RdBu_r',
                      vmin=-vmax,
                      vmax=vmax,
                      shading='auto',
                      zorder=1)
    
    # Add only coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black', zorder=2)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=1, alpha=0.7, color='black', zorder=3)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', 
                       shrink=0.6, pad=0.05, aspect=20)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Values', size=14, weight='bold')
    
    # Add title
    if title is None:
        title = f'North Polar Stereo View\nData Coverage: {lat_range[0]}°N - {lat_range[1]}°N'
    
    plt.title(title, size=16, weight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def date_to_day_index(date_obj):
    """Convert date to day index (1-based, no leap years)"""
    year = date_obj.year
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = date_obj.month
    day = date_obj.day
    day_index = sum(days_in_month[:month-1]) + day  # 1-based indexing
    return day_index

def plot_date_polar_stereo(target_date, data_type='sub', notebook_dir=None, 
                          std_scaling=False, std_scale=2.0, figsize=(12, 12),
                          title=None, matrix_size=172, lat_range=(0, 90), lon_range=(0, 360)):
    """
    Plot North Polar Stereo projection for a specific date
    
    Parameters:
    -----------
    target_date : datetime.date or str
        Date to plot (e.g., date(2023, 5, 15) or "2023-05-15")
    data_type : str
        'sub' for sublevel, 'sup' for superlevel
    notebook_dir : str
        Directory path for data loading
    std_scaling : bool
        Whether to use standard deviation scaling
    std_scale : float
        Standard deviation scaling factor
    figsize : tuple
        Figure size
    title : str, optional
        Custom title. If None, auto-generated from date
    matrix_size : int
        Size of polar matrix (if creating from raw data)
    lat_range, lon_range : tuple
        Geographic coordinate ranges
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    
    if notebook_dir is None:
        raise ValueError("notebook_dir parameter is required")
    
    # Convert string date to date object if needed
    if isinstance(target_date, str):
        from datetime import datetime
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    
    year = target_date.year
    day = date_to_day_index(target_date)
    
    print(f"Loading data for {target_date} (Year: {year}, Day: {day})")
    
    # Load polar matrix
    try:
        polar_data_path = os.path.join(notebook_dir, "data", "processed_data", "SLP_data_years", 
                                      str(year), f"slp_{data_type}_{year}_day_{day}.npy")
        
        if os.path.exists(polar_data_path):
            print(f"Loading polar matrix from: {polar_data_path}")
            polar_matrix = np.load(polar_data_path)
            print(f"Loaded polar matrix with shape: {polar_matrix.shape}")
        else:
            print(f"Polar matrix file not found: {polar_data_path}")
            return None
            
    except Exception as e:
        print(f"Error loading polar matrix: {e}")
        return None
    
    # Generate title if not provided
    if title is None:
        title = f'{data_type.upper()} Level Data\n{target_date.strftime("%B %d, %Y")}'
    
    # Plot using the polar stereo function
    try:
        fig = plot_north_polar_stereo_from_matrix(
            polar_matrix=polar_matrix,
            matrix_size=matrix_size,
            lat_range=lat_range,
            lon_range=lon_range,
            std_scaling=std_scaling,
            std_scale=std_scale,
            title=title,
            figsize=figsize
        )
        
        print(f"Successfully plotted {target_date}")
        return fig
        
    except Exception as e:
        print(f"Error plotting: {e}")
        return None