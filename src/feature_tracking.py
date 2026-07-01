import warnings
warnings.filterwarnings('ignore', message='invalid value encountered', category=RuntimeWarning)

import matplotlib.colors as colors
import matplotlib.cm as cm
import persim
import numpy as np
from collections import defaultdict
from datetime import date, timedelta
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from glob import glob
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import date, timedelta
import importlib.util
import sys

def _rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range):
    """Convert matrix row-col coordinates to lat-lon."""
    center = matrix_size // 2
    x_centered = col - center
    y_centered = row - center
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    lon = (theta * 180 / np.pi + 90) % 360
    max_radius = center
    lat = lat_range[1] - (r / max_radius) * (lat_range[1] - lat_range[0])
    if lon_range[0] == -180 and lon_range[1] == 180:
        lon = lon - 180
    return lat, lon


def _geodesic_distance_km(row1, col1, row2, col2, matrix_size, lat_range, lon_range):
    """Return geodesic distance in km between two grid points."""
    lat1, lon1 = _rowcol_to_latlon(row1, col1, matrix_size, lat_range, lon_range)
    lat2, lon2 = _rowcol_to_latlon(row2, col2, matrix_size, lat_range, lon_range)
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return np.nan
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except Exception:
        return np.nan


def compute_representative_area_km2(representative, matrix_size=172,
                                     lat_range=(0, 90), lon_range=(0, 360)):
    """
    Compute the geographic area (km²) enclosed by a representative 1-cycle.

    Extracts unique grid vertices from the representative edges, converts them
    to (lat, lon) via the polar-matrix projection, takes the convex hull, and
    returns the geodetic area using the WGS-84 ellipsoid.

    Parameters
    ----------
    representative : list of [[r1,c1],[r2,c2]] edge pairs
    matrix_size : int  (default 172)
    lat_range, lon_range : tuples matching the polar-matrix parameterisation

    Returns
    -------
    float : area in km², or 0.0 if the representative is too small
    """
    import warnings
    import numpy as np
    from pyproj import Geod
    from shapely.geometry import MultiPoint

    if not representative or len(representative) < 3:
        return 0.0

    vertices = set()
    for edge in representative:
        vertices.add((edge[0][0], edge[0][1]))
        vertices.add((edge[1][0], edge[1][1]))

    if len(vertices) < 3:
        return 0.0

    # Filter to valid, finite coordinates only
    coords = []
    for row, col in vertices:
        lat, lon = _rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range)
        if np.isfinite(lat) and np.isfinite(lon) and 0 <= lat <= 90:
            coords.append((float(lon), float(lat)))

    if len(coords) < 3:
        return 0.0

    # np.errstate suppresses C-level NumPy/GEOS floating-point warnings;
    # warnings.catch_warnings suppresses any Python-level RuntimeWarnings.
    # Both layers are needed because Shapely 2.x fires through both paths.
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        hull = MultiPoint(coords).convex_hull

    if hull.geom_type not in ('Polygon', 'MultiPolygon') or hull.area < 1e-10:
        return 0.0

    exterior = list(hull.exterior.coords)
    lons_hull = [c[0] for c in exterior]
    lats_hull = [c[1] for c in exterior]
    geod = Geod(ellps='WGS84')
    area_m2, _ = geod.polygon_area_perimeter(lons_hull, lats_hull)
    return abs(area_m2) / 1e6   # m² → km²


def get_area_by_year_day(data_type, notebook_dir, start_year, end_year,
                         matrix_size=172, lat_range=(0, 90), lon_range=(0, 360)):
    """
    Compute total representative area (km²) per day, organised by year.

    For each day the areas of all features are summed, mirroring the structure
    returned by get_persistence_by_year_day.

    Returns
    -------
    dict : {year: [total_area_day1, ..., total_area_day365]}  (365 floats per year)
    """
    from tqdm import tqdm

    years = list(range(start_year, end_year + 1))
    print(f"Computing {data_type} area data for years {start_year}-{end_year}")

    area_dict = {}
    for year in tqdm(years, desc=f"Computing {data_type} area"):
        year_df = load_all_representative_data_to_df(
            years=year, data_types=[data_type], notebook_dir=notebook_dir
        )

        year_data = [0.0] * 365
        if not year_df.empty:
            for day in range(1, 366):
                day_df = year_df[year_df['day'] == day]
                if not day_df.empty:
                    year_data[day - 1] = sum(
                        compute_representative_area_km2(rep, matrix_size, lat_range, lon_range)
                        for rep in day_df['representative']
                    )

        area_dict[year] = year_data

    print(f"Area dictionary created for {len(area_dict)} years")
    return area_dict


def load_all_representative_data_to_df(years=None, days=None, data_types=["sub", "sup"], base_folder=None, notebook_dir=None):
    """
    Load representative data files into a pandas DataFrame for specific years and days
    
    Parameters:
    -----------
    years : int, list, or None
        Specific year(s) to load. If None, loads all years.
    days : int, list, or None
        Specific day(s) to load. If None, loads all days.
    data_types : list
        Data types to load (["sub", "sup"])
    base_folder : str, optional
        Base folder path
    notebook_dir : str, optional
        Notebook directory path
    """
    
    if notebook_dir is None:
        notebook_dir = os.getcwd()
    
    if base_folder is None:
        base_folder = os.path.join(notebook_dir, "data", "processed_data", "representative_data")
    
    # Convert single values to lists
    if years is not None:
        if isinstance(years, int):
            years = [years]
    
    if days is not None:
        if isinstance(days, int):
            days = [days]
    
    all_data = []
    
    # Get all year directories
    year_dirs = [d for d in os.listdir(base_folder) 
                 if os.path.isdir(os.path.join(base_folder, d)) and d.isdigit()]
    
    # Filter years if specified
    if years is not None:
        year_dirs = [d for d in year_dirs if int(d) in years]
    
    for year_dir in year_dirs:
        year = int(year_dir)
        year_path = os.path.join(base_folder, year_dir)
        
        # Get all JSON files for specified data types
        json_files = []
        for data_type in data_types:
            pattern = os.path.join(year_path, f"slp_{data_type}_*.json")
            files = glob(pattern)
            json_files.extend(files)
        
        for filepath in json_files:
            try:
                filename = os.path.basename(filepath)
                
                # Extract info from filename
                parts = filename.replace('.json', '').split('_')
                if len(parts) >= 5:
                    file_data_type = parts[1]
                    file_year = int(parts[2])
                    file_day = int(parts[4])
                else:
                    continue
                
                # Filter days if specified
                if days is not None and file_day not in days:
                    continue
                
                # Load JSON data
                with open(filepath, 'r') as f:
                    data_dict = json.load(f)
                
                births = data_dict.get("births", [])
                deaths = data_dict.get("deaths", [])
                representatives = data_dict.get("list_data", [])
                death_simplex_vertices = data_dict.get("death_simplex_vertices", [])
                
                # Create one row per persistence feature
                for i, (birth, death) in enumerate(zip(births, deaths)):
                    row_data = {
                        'year': file_year,
                        'day': file_day,
                        'data_type': file_data_type,
                        'feature_index': i,
                        'birth': birth,
                        'death': death,
                        'persistence': death - birth,
                        'representative': representatives[i] if i < len(representatives) else [],
                        'death_simplex_vertices': death_simplex_vertices[i] if i < len(death_simplex_vertices) else []
                    }
                    
                    all_data.append(row_data)
                    
            except Exception as e:
                continue
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        df = df.sort_values(['year', 'day', 'data_type', 'persistence'], 
                           ascending=[True, True, True, False])
        df = df.reset_index(drop=True)
    
    return df

def get_persistence_by_year_day(data_type, notebook_dir, start_year, end_year):
    """
    Get persistence values organized by year and day for a range of years
    
    Parameters:
    -----------
    data_type : str
        'sub' for sublevel, 'sup' for superlevel
    notebook_dir : str
        Directory path for data loading
    start_year : int
        Starting year (inclusive)
    end_year : int
        Ending year (inclusive)
    
    Returns:
    --------
    dict : Dictionary with structure:
        {
            year: [
                array([persistence_values...]),  # day 1
                array([persistence_values...]),  # day 2
                ...
                array([persistence_values...])   # day 365
            ]
        }
    """
    import numpy as np
    
    years = list(range(start_year, end_year + 1))
    print(f"Loading {data_type} persistence data for years {start_year}-{end_year}")
    
    # Load all data for the specified years and data type
    df = load_all_representative_data_to_df(
        years=years,
        data_types=[data_type],
        notebook_dir=notebook_dir
    )
    
    if df.empty:
        print("No data found!")
        return {}
    
    # Initialize result dictionary
    persistence_dict = {}
    
    for year in years:
        # Initialize list for 365 days
        year_data = [np.array([]) for _ in range(365)]
        
        # Get data for this year
        year_df = df[df['year'] == year]
        
        if not year_df.empty:
            # Process each day
            for day in range(1, 366):  # days 1-365
                day_df = year_df[year_df['day'] == day]
                
                if not day_df.empty:
                    # Extract persistence values for this day
                    persistence_values = day_df['persistence'].values
                    year_data[day - 1] = np.array(persistence_values)  # day-1 for 0-based indexing
        
        persistence_dict[year] = year_data
    
    print(f"Dictionary created for {len(persistence_dict)} years")
    return persistence_dict


def plot_representative_on_heatmap(year, day, data_type, feature_index, 
                                 notebook_dir=None, base_folder=None, figsize=(5, 5),
                                 data_loader_path="data_loader.py"):
    """
    Plot representative and death simplex vertices on heatmap for a specific feature
    """
    
    if notebook_dir is None:
        notebook_dir = os.getcwd()
    
    if base_folder is None:
        base_folder = os.path.join(notebook_dir, "data", "processed_data", "SLP_data_years")
    
    df = load_all_representative_data_to_df(
        years=year, 
        days=day, 
        data_types=[data_type], 
        notebook_dir=notebook_dir
    )
    
    if df.empty:
        print(f"No data found for year={year}, day={day}, data_type={data_type}")
        return None
    
    # Filter DataFrame to get the specific feature
    mask = (df['year'] == year) & (df['day'] == day) & \
           (df['data_type'] == data_type) & (df['feature_index'] == feature_index)
    
    feature_data = df[mask]
    
    if feature_data.empty:
        print(f"No feature found for year={year}, day={day}, data_type={data_type}, feature_index={feature_index}")
        return None
    
    if len(feature_data) > 1:
        print(f"Warning: Multiple features found, using first one")
    
    # Get the feature data
    feature_row = feature_data.iloc[0]
    representative = feature_row['representative']
    death_simplex_vertices = feature_row['death_simplex_vertices']
    
    # Convert day to actual date
    start_date = date(year, 1, 1)
    actual_date = start_date + timedelta(days=day-1)  # day is 1-based
    
    # Load the original data matrix
    data_filename = f"slp_{data_type}_{year}_day_{day}.npy"
    data_filepath = os.path.join(base_folder, str(year), data_filename)
    
    if not os.path.exists(data_filepath):
        print(f"Data file not found: {data_filepath}")
        return None
    
    try:
        data_matrix = np.load(data_filepath)
        
        # For superlevel, use negative of the data
        if data_type == 'sup':
            data_matrix = -data_matrix
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create circular mask to hide areas outside the globe
    height, width = data_matrix.shape
    center_y, center_x = height // 2, width // 2
    radius = min(center_x, center_y)
    
    # Create coordinate arrays
    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Mask the data - set areas outside circle to NaN
    data_masked = data_matrix.copy().astype(float)
    data_masked[~mask_circle] = np.nan
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap (NaN values will be transparent)
    im = ax.imshow(data_masked, cmap='RdBu_r', origin='lower', aspect='equal')
    
    # Plot representative points
    if representative:
        rep_coords = []
        for edge in representative:
            if len(edge) == 2:
                start_coords = edge[0]  # [row, col] in 0-based indexing
                end_coords = edge[1]    # [row, col] in 0-based indexing
                rep_coords.extend([start_coords, end_coords])
        
        if rep_coords:
            # Extract x (col) and y (row) coordinates
            rep_rows = [coord[0] for coord in rep_coords]
            rep_cols = [coord[1] for coord in rep_coords]
            
            # Plot representative points
            ax.scatter(rep_cols, rep_rows, 
                      color='red', s=20, alpha=0.7, 
                      label=f'Representative ({len(rep_coords)} points)', 
                      edgecolors='white', linewidths=0.5)
    
    # Plot death simplex as transparent box
    if death_simplex_vertices and len(death_simplex_vertices) == 4:
        # Extract coordinates for all 4 vertices
        death_rows = [vertex[0] for vertex in death_simplex_vertices]  # row coordinates
        death_cols = [vertex[1] for vertex in death_simplex_vertices]  # col coordinates
        
        # Draw transparent rectangle
        min_row, max_row = min(death_rows), max(death_rows)
        min_col, max_col = min(death_cols), max(death_cols)
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((min_col-0.5, min_row-0.5), 
                        max_col-min_col+1, max_row-min_row+1,
                        linewidth=2, edgecolor='yellow', facecolor='yellow', 
                        alpha=1, label='Death Simplex')
        ax.add_patch(rect)
    
    # Simple title with just the date
    ax.set_title(f'{actual_date.strftime("%Y-%m-%d")}', fontsize=16, pad=20)
    
    # Remove axis labels and ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    #ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    
    plt.tight_layout()
    
    return fig, ax

def plot_multiple_features(year, day, data_type, feature_indices=None, 
                          max_features=5, notebook_dir=None, base_folder=None,
                          data_loader_path="data_loader.py"):
    """
    Plot multiple features from the same day on one heatmap
    """
    
    if notebook_dir is None:
        notebook_dir = os.getcwd()
    
    if base_folder is None:
        base_folder = os.path.join(notebook_dir, "data", "processed_data", "SLP_data_years")
    
    # Load data using the data_loader function
    df = load_all_representative_data_to_df(
        years=year, 
        days=day, 
        data_types=[data_type], 
        notebook_dir=notebook_dir
    )
    
    if df.empty:
        print(f"No data found for year={year}, day={day}, data_type={data_type}")
        return None
    
    # Get all features for this day
    mask = (df['year'] == year) & (df['day'] == day) & (df['data_type'] == data_type)
    day_features = df[mask].copy()
    
    if day_features.empty:
        print(f"No features found for year={year}, day={day}, data_type={data_type}")
        return None
    
    # Select features to plot
    if feature_indices is None:
        # Plot top persistent features
        day_features = day_features.nlargest(max_features, 'persistence')
    else:
        # Plot specific features
        day_features = day_features[day_features['feature_index'].isin(feature_indices)]
    
    if day_features.empty:
        print("No features selected for plotting")
        return None
    
    # Convert day to actual date
    start_date = date(year, 1, 1)
    actual_date = start_date + timedelta(days=day-1)
    
    # Load the data matrix
    data_filename = f"slp_{data_type}_{year}_day_{day}.npy"
    data_filepath = os.path.join(base_folder, str(year), data_filename)
    
    if not os.path.exists(data_filepath):
        print(f"Data file not found: {data_filepath}")
        return None
    
    data_matrix = np.load(data_filepath)
    
    # For superlevel, use negative of the data
    if data_type == 'sup':
        data_matrix = -data_matrix
    
    # Create circular mask to hide areas outside the globe
    height, width = data_matrix.shape
    center_y, center_x = height // 2, width // 2
    radius = min(center_x, center_y)
    
    # Create coordinate arrays
    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # Mask the data
    data_masked = data_matrix.copy().astype(float)
    data_masked[~mask_circle] = np.nan
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the heatmap
    im = ax.imshow(data_masked, cmap='RdBu_r', origin='lower', aspect='equal')
    
    # Define colors for different features
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Plot each feature
    for idx, (_, feature_row) in enumerate(day_features.iterrows()):
        color = colors[idx % len(colors)]
        feature_index = feature_row['feature_index']
        representative = feature_row['representative']
        death_simplex_vertices = feature_row['death_simplex_vertices']
        persistence = feature_row['persistence']
        
        # Plot representative
        if representative:
            rep_coords = []
            for edge in representative:
                if len(edge) == 2:
                    rep_coords.extend([edge[0], edge[1]])
            
            if rep_coords:
                rep_rows = [coord[0] for coord in rep_coords]
                rep_cols = [coord[1] for coord in rep_coords]
                
                ax.scatter(rep_cols, rep_rows, 
                          color=color, s=15, alpha=0.8,
                          label=f'Feature #{feature_index} (pers={persistence:.3f})')
        
        # Plot death simplex as transparent box
        if death_simplex_vertices and len(death_simplex_vertices) == 4:
            death_rows = [vertex[0] for vertex in death_simplex_vertices]
            death_cols = [vertex[1] for vertex in death_simplex_vertices]
            
            # Draw transparent rectangle
            min_row, max_row = min(death_rows), max(death_rows)
            min_col, max_col = min(death_cols), max(death_cols)
            
            from matplotlib.patches import Rectangle
            rect = Rectangle((min_col-0.5, min_row-0.5), 
                           max_col-min_col+1, max_row-min_row+1,
                           linewidth=1.5, edgecolor=color, facecolor='none', 
                           alpha=1)
            ax.add_patch(rect)
    
    # Simple title with just the date
    ax.set_title(f'{actual_date.strftime("%Y-%m-%d")}', fontsize=16, pad=20)
    
    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    return fig, ax

def analyze_persistence_trajectories_filtered_wasserstein_geodesic_incremental(
    start_year, end_year, data_type='sub', notebook_dir=None, 
    persistence_threshold=0.01, alpha=1.0, beta=1.0, distance_threshold=np.inf,
    matrix_size=172, lat_range=(0, 90), lon_range=(0, 360)):
    """
    Analyze persistence trajectories using Wasserstein matching with geodesic distance filtering
    Loads data incrementally to avoid memory issues
    
    Parameters:
    -----------
    start_year, end_year : int
        Year range to analyze
    data_type : str
        'sub' for sublevel, 'sup' for superlevel
    notebook_dir : str
        Directory path for data loading
    persistence_threshold : float
        Minimum persistence to consider
    alpha, beta : float
        Weights for topological and spatial distances
    distance_threshold : float
        Maximum combined distance threshold
    matrix_size : int
        Size of the polar matrix
    lat_range, lon_range : tuple
        Geographic coordinate ranges
    """
    
    from datetime import date, timedelta
    import numpy as np
    from geopy.distance import geodesic
    import persim
    
    if notebook_dir is None:
        raise ValueError("notebook_dir parameter is required")
    
    def rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range):
        """Convert matrix row-col coordinates to lat-lon"""
        center = matrix_size // 2
        x_centered = col - center
        y_centered = row - center
        r = np.sqrt(x_centered**2 + y_centered**2)
        theta = np.arctan2(y_centered, x_centered)
        lon = (theta * 180 / np.pi + 90) % 360
        max_radius = center
        lat = lat_range[1] - (r / max_radius) * (lat_range[1] - lat_range[0])
        
        if lon_range[0] == 0 and lon_range[1] == 360:
            pass
        elif lon_range[0] == -180 and lon_range[1] == 180:
            lon = lon - 180
            
        return lat, lon
    
    def geodesic_distance_km(row1, col1, row2, col2, matrix_size, lat_range, lon_range):
        """Calculate geodesic distance between two points in kilometers"""
        lat1, lon1 = rowcol_to_latlon(row1, col1, matrix_size, lat_range, lon_range)
        lat2, lon2 = rowcol_to_latlon(row2, col2, matrix_size, lat_range, lon_range)
        
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            return np.inf
        if not (lon_range[0] <= lon1 <= lon_range[1] and lon_range[0] <= lon2 <= lon_range[1]):
            return np.inf
            
        try:
            distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
            return distance
        except:
            return np.inf
    
    def df_to_diagram_with_indices(df, year, day, data_type):
        """Convert DataFrame to persistence diagram format [birth, death, row, col] with feature indices"""
        mask = (df['year'] == year) & (df['day'] == day) & (df['data_type'] == data_type)
        day_data = df[mask].copy()
        
        if day_data.empty:
            return np.array([]).reshape(0, 4), []
        
        # Sort by feature_index to maintain order
        day_data = day_data.sort_values('feature_index')
        
        # Extract data with feature indices
        diagrams = []
        feature_indices = []
        
        for _, row in day_data.iterrows():
            birth = row['birth']
            death = row['death']
            feature_index = row['feature_index']
            
            # Get location from death simplex vertices (use first vertex)
            death_vertices = row['death_simplex_vertices']
            if death_vertices and len(death_vertices) > 0:
                # Use first vertex coordinates
                vertex_row, vertex_col = death_vertices[0]
            else:
                # Fallback: use center if no death vertices
                vertex_row, vertex_col = matrix_size//2, matrix_size//2
            
            diagrams.append([birth, death, vertex_row, vertex_col])
            feature_indices.append(feature_index)
        
        return np.array(diagrams), feature_indices
    
    def filter_by_persistence_with_indices(diagram, feature_indices, threshold):
        """Filter diagram to keep only points with persistence >= threshold, maintaining indices"""
        if len(diagram) == 0:
            return diagram, feature_indices
        
        persistence = diagram[:, 1] - diagram[:, 0]
        keep_mask = persistence >= threshold
        
        filtered_diagram = diagram[keep_mask]
        filtered_indices = [feature_indices[i] for i in range(len(feature_indices)) if keep_mask[i]]
        
        return filtered_diagram, filtered_indices
    
    def wasserstein_with_geodesic_filter(D1, D2, indices1, indices2, alpha, beta, threshold, matrix_size, lat_range, lon_range):
        """Use Wasserstein matching, then filter by combined distance"""
        if len(D1) == 0 or len(D2) == 0:
            return []
        
        bd_coords_1 = D1[:, 0:2]
        bd_coords_2 = D2[:, 0:2]
        
        d_wasserstein, wasserstein_matches = persim.wasserstein(bd_coords_1, bd_coords_2, matching=True)
        
        spatial_coords_1 = D1[:, 2:4]
        spatial_coords_2 = D2[:, 2:4]
        
        filtered_matches = []
        
        for match in wasserstein_matches:
            i, j = int(match[0]), int(match[1])
            
            if i < 0 or j < 0 or i >= len(D1) or j >= len(D2):
                continue
                
            d_topological = np.linalg.norm(bd_coords_1[i] - bd_coords_2[j])
            
            row1, col1 = spatial_coords_1[i]
            row2, col2 = spatial_coords_2[j]
            d_spatial = geodesic_distance_km(row1, col1, row2, col2, matrix_size, lat_range, lon_range)
            
            if d_spatial == np.inf:
                continue
            
            combined_distance = alpha * d_topological + beta * d_spatial
            
            if combined_distance <= distance_threshold:
                lat1, lon1 = rowcol_to_latlon(row1, col1, matrix_size, lat_range, lon_range)
                lat2, lon2 = rowcol_to_latlon(row2, col2, matrix_size, lat_range, lon_range)
                
                filtered_matches.append({
                    'i': i, 'j': j,
                    'combined_distance': combined_distance,
                    'topological_distance': d_topological,
                    'spatial_distance_km': d_spatial,
                    'wasserstein_distance': d_wasserstein,
                    'coords1': {'lat': lat1, 'lon': lon1, 'row': row1, 'col': col1},
                    'coords2': {'lat': lat2, 'lon': lon2, 'row': row2, 'col': col2},
                    'feature_index1': indices1[i],
                    'feature_index2': indices2[j]
                })
        
        return filtered_matches
    
    # Setup
    years = list(range(start_year, end_year + 1))
    print(f"Analyzing trajectories for years {start_year}-{end_year}")
    print(f"Data type: {data_type}")
    print(f"Using Wasserstein matching with geodesic distance filter:")
    print(f"C = {alpha} * d_topological + {beta} * d_spatial_km <= {distance_threshold}")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    
    # Initialize tracking variables
    trajectories = []
    active_trajectories = {}
    next_traj_id = 0
    matching_stats = {
        'total_wasserstein_matches': 0,
        'filtered_matches': 0,
        'avg_topological': 0,
        'avg_spatial_km': 0,
        'avg_combined': 0,
        'max_spatial_km': 0,
        'min_spatial_km': np.inf
    }
    
    # Generate all dates (all days for each year)
    all_dates = []
    for year in years:
        for day in range(1, 366):  # 1-365
            try:
                current_date = date(year, 1, 1) + timedelta(days=day-1)
                if not (current_date.month == 2 and current_date.day == 29):  # Skip leap days
                    all_dates.append((year, day, current_date))
            except:
                continue
    
    print(f"Processing {len(all_dates)} time points...")
    
    # Process each day incrementally
    prev_diagram = None
    prev_indices = None
    
    for time_index, (year, day, current_date) in enumerate(all_dates):
        
        if time_index % 100 == 0:
            print(f"Processing time step {time_index+1}/{len(all_dates)} - {current_date}")
        
        # Load data for current day
        try:
            df = load_all_representative_data_to_df(
                years=year,
                days=day,
                data_types=[data_type],
                notebook_dir=notebook_dir
            )
            
            if df.empty:
                current_diagram = np.array([]).reshape(0, 4)
                current_indices = []
            else:
                current_diagram, current_indices = df_to_diagram_with_indices(df, year, day, data_type)
                current_diagram, current_indices = filter_by_persistence_with_indices(
                    current_diagram, current_indices, persistence_threshold
                )
                
        except Exception as e:
            print(f"Error loading data for {year}-{day}: {e}")
            current_diagram = np.array([]).reshape(0, 4)
            current_indices = []
        
        # Initialize trajectories for first day
        if time_index == 0:
            if len(current_diagram) > 0:
                for point_idx, (point, feature_index) in enumerate(zip(current_diagram, current_indices)):
                    birth, death, row, col = point
                    persistence = death - birth
                    lat, lon = rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range)
                    
                    trajectory = [{
                        'time_index': time_index,
                        'date': current_date,
                        'year': year,
                        'day': day,
                        'birth': birth,
                        'death': death,
                        'persistence': persistence,
                        'row': row,
                        'col': col,
                        'lat': lat,
                        'lon': lon,
                        'feature_index': feature_index
                    }]
                    active_trajectories[point_idx] = {
                        'id': next_traj_id,
                        'trajectory': trajectory,
                        'last_point': point,
                        'last_feature_index': feature_index
                    }
                    next_traj_id += 1
            
            prev_diagram = current_diagram
            prev_indices = current_indices
            continue
        
        # Match with previous day
        if len(prev_diagram) == 0 or len(current_diagram) == 0:
            # End all active trajectories
            for traj_info in active_trajectories.values():
                trajectories.append(traj_info['trajectory'])
            active_trajectories = {}
            
            # Start new trajectories
            if len(current_diagram) > 0:
                for point_idx, (point, feature_index) in enumerate(zip(current_diagram, current_indices)):
                    birth, death, row, col = point
                    persistence = death - birth
                    lat, lon = rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range)
                    
                    trajectory = [{
                        'time_index': time_index,
                        'date': current_date,
                        'year': year,
                        'day': day,
                        'birth': birth,
                        'death': death,
                        'persistence': persistence,
                        'row': row,
                        'col': col,
                        'lat': lat,
                        'lon': lon,
                        'feature_index': feature_index
                    }]
                    active_trajectories[point_idx] = {
                        'id': next_traj_id,
                        'trajectory': trajectory,
                        'last_point': point,
                        'last_feature_index': feature_index
                    }
                    next_traj_id += 1
        else:
            # Get matches
            matches = wasserstein_with_geodesic_filter(
                prev_diagram, current_diagram, prev_indices, current_indices,
                alpha, beta, distance_threshold, matrix_size, lat_range, lon_range
            )
            
            # Update statistics
            bd_coords_1 = prev_diagram[:, 0:2]
            bd_coords_2 = current_diagram[:, 0:2]
            _, all_wasserstein_matches = persim.wasserstein(bd_coords_1, bd_coords_2, matching=True)
            valid_wasserstein = sum(1 for m in all_wasserstein_matches 
                                  if int(m[0]) >= 0 and int(m[1]) >= 0 
                                  and int(m[0]) < len(prev_diagram) and int(m[1]) < len(current_diagram))
            
            matching_stats['total_wasserstein_matches'] += valid_wasserstein
            matching_stats['filtered_matches'] += len(matches)
            
            if matches:
                spatial_distances = [m['spatial_distance_km'] for m in matches]
                matching_stats['avg_topological'] += sum(m['topological_distance'] for m in matches)
                matching_stats['avg_spatial_km'] += sum(spatial_distances)
                matching_stats['avg_combined'] += sum(m['combined_distance'] for m in matches)
                matching_stats['max_spatial_km'] = max(matching_stats['max_spatial_km'], max(spatial_distances))
                matching_stats['min_spatial_km'] = min(matching_stats['min_spatial_km'], min(spatial_distances))
            
            new_active_trajectories = {}
            matched_indices_d2 = set()
            
            # Process matches
            for match in matches:
                m0, m1 = match['i'], match['j']
                matched_indices_d2.add(m1)
                
                if m0 in active_trajectories:
                    point = current_diagram[m1]
                    feature_index = current_indices[m1]
                    birth, death, row, col = point
                    persistence = death - birth
                    
                    traj_info = active_trajectories[m0]
                    traj_info['trajectory'].append({
                        'time_index': time_index,
                        'date': current_date,
                        'year': year,
                        'day': day,
                        'birth': birth,
                        'death': death,
                        'persistence': persistence,
                        'row': row,
                        'col': col,
                        'lat': match['coords2']['lat'],
                        'lon': match['coords2']['lon'],
                        'feature_index': feature_index,
                        'match_info': match
                    })
                    traj_info['last_point'] = point
                    traj_info['last_feature_index'] = feature_index
                    new_active_trajectories[m1] = traj_info
            
            # End unmatched trajectories
            for old_idx, traj_info in active_trajectories.items():
                if not any(match['i'] == old_idx for match in matches):
                    trajectories.append(traj_info['trajectory'])
            
            # Start new trajectories
            for point_idx, (point, feature_index) in enumerate(zip(current_diagram, current_indices)):
                if point_idx not in matched_indices_d2:
                    birth, death, row, col = point
                    persistence = death - birth
                    lat, lon = rowcol_to_latlon(row, col, matrix_size, lat_range, lon_range)
                    
                    trajectory = [{
                        'time_index': time_index,
                        'date': current_date,
                        'year': year,
                        'day': day,
                        'birth': birth,
                        'death': death,
                        'persistence': persistence,
                        'row': row,
                        'col': col,
                        'lat': lat,
                        'lon': lon,
                        'feature_index': feature_index
                    }]
                    new_active_trajectories[point_idx] = {
                        'id': next_traj_id,
                        'trajectory': trajectory,
                        'last_point': point,
                        'last_feature_index': feature_index
                    }
                    next_traj_id += 1
            
            active_trajectories = new_active_trajectories
        
        prev_diagram = current_diagram
        prev_indices = current_indices
    
    # End remaining trajectories
    for traj_info in active_trajectories.values():
        trajectories.append(traj_info['trajectory'])
    
    # Calculate final statistics
    if matching_stats['filtered_matches'] > 0:
        matching_stats['avg_topological'] /= matching_stats['filtered_matches']
        matching_stats['avg_spatial_km'] /= matching_stats['filtered_matches']
        matching_stats['avg_combined'] /= matching_stats['filtered_matches']
    else:
        matching_stats['min_spatial_km'] = 0
    
    matching_stats['filter_rate'] = (matching_stats['filtered_matches'] / 
                                   max(matching_stats['total_wasserstein_matches'], 1))
    
    # Calculate trajectory statistics
    trajectory_lengths = [len(traj) for traj in trajectories]
    trajectory_lifespans = []
    trajectory_distances = []
    
    for traj in trajectories:
        if len(traj) > 1:
            start_date = traj[0]['date']
            end_date = traj[-1]['date']
            lifespan = (end_date - start_date).days
            
            total_distance = 0
            for j in range(1, len(traj)):
                if 'match_info' in traj[j]:
                    total_distance += traj[j]['match_info']['spatial_distance_km']
            trajectory_distances.append(total_distance)
        else:
            lifespan = 0
            trajectory_distances.append(0)
        trajectory_lifespans.append(lifespan)
    
    results = {
        'trajectories': trajectories,
        'trajectory_lengths': trajectory_lengths,
        'trajectory_lifespans': trajectory_lifespans,
        'trajectory_distances': trajectory_distances,
        'dates': all_dates,
        'matching_stats': matching_stats,
        'parameters': {
            'start_year': start_year,
            'end_year': end_year,
            'data_type': data_type,
            'persistence_threshold': persistence_threshold,
            'alpha': alpha,
            'beta': beta,
            'distance_threshold': distance_threshold,
            'total_trajectories': len(trajectories),
            'matrix_size': matrix_size,
            'lat_range': lat_range,
            'lon_range': lon_range
        }
    }
    
    print(f"Analysis complete: {len(trajectories)} trajectories found")
    print(f"Wasserstein matches: {matching_stats['total_wasserstein_matches']}")
    print(f"Filtered matches: {matching_stats['filtered_matches']}")
    print(f"Filter rate: {matching_stats['filter_rate']:.3f}")
    if matching_stats['filtered_matches'] > 0:
        print(f"Average spatial distance: {matching_stats['avg_spatial_km']:.2f} km")
        print(f"Spatial distance range: {matching_stats['min_spatial_km']:.2f} - {matching_stats['max_spatial_km']:.2f} km")
    
    return results

import pickle
import os
import json
from datetime import datetime

def analyze_persistence_trajectories_cached(
    start_year, end_year, data_type='sub', notebook_dir=None, 
    persistence_threshold=0.01, alpha=1.0, beta=1.0, distance_threshold=np.inf,
    matrix_size=172, lat_range=(0, 90), lon_range=(0, 360), force_recompute=False):
    """
    Cached version of trajectory analysis - saves/loads results to avoid recomputation
    """
    
    if notebook_dir is None:
        raise ValueError("notebook_dir parameter is required")
    
    # Create cache directory
    cache_dir = os.path.join(notebook_dir, "data", "processed_data", "feature_tracking")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create unique identifier based on parameters
    params_hash = {
        'start_year': start_year,
        'end_year': end_year,
        'data_type': data_type,
        'persistence_threshold': persistence_threshold,
        'alpha': alpha,
        'beta': beta,
        'distance_threshold': distance_threshold,
        'matrix_size': matrix_size,
        'lat_range': list(lat_range),  # Convert to list for JSON compatibility
        'lon_range': list(lon_range)   # Convert to list for JSON compatibility
    }
    
    # Create filename based on parameters
    param_str = f"{data_type}_{start_year}-{end_year}_pers{persistence_threshold}_a{alpha}_b{beta}_dist{distance_threshold}_size{matrix_size}"
    # Replace problematic characters
    param_str = param_str.replace('.', 'p').replace('inf', 'inf')
    
    results_file = os.path.join(cache_dir, f"trajectory_results_{param_str}.pkl")
    metadata_file = os.path.join(cache_dir, f"trajectory_metadata_{param_str}.json")
    
    def normalize_param_for_comparison(value):
        """Normalize parameters for comparison (handle tuple/list conversion)"""
        if isinstance(value, (tuple, list)):
            return list(value)
        return value
    
    # Check if cached results exist
    if os.path.exists(results_file) and os.path.exists(metadata_file) and not force_recompute:
        print("="*60)
        print("LOADING CACHED RESULTS")
        print("="*60)
        
        # Load metadata to verify parameters
        try:
            with open(metadata_file, 'r') as f:
                cached_metadata = json.load(f)
            
            # Check if parameters match (with normalization for tuple/list)
            params_match = True
            for key, value in params_hash.items():
                cached_value = cached_metadata['parameters'].get(key)
                normalized_value = normalize_param_for_comparison(value)
                normalized_cached_value = normalize_param_for_comparison(cached_value)
                
                if normalized_cached_value != normalized_value:
                    params_match = False
                    print(f"Parameter mismatch: {key} = {value} (requested) vs {cached_value} (cached)")
                    break
            
            if params_match:
                print(f"Loading cached results from: {results_file}")
                print(f"Cache created: {cached_metadata['created_at']}")
                print(f"Total trajectories: {cached_metadata['total_trajectories']}")
                
                # Load results
                with open(results_file, 'rb') as f:
                    results = pickle.load(f)
                
                print("Cached results loaded successfully!")
                return results
            else:
                print("Parameters don't match cached version. Recomputing...")
                
        except Exception as e:
            print(f"Error loading cached results: {e}")
            print("Recomputing...")
    
    # If we reach here, we need to compute
    print("="*60)
    print("COMPUTING TRAJECTORY ANALYSIS")
    print("="*60)
    print(f"Parameters: {params_hash}")
    print(f"Results will be cached to: {results_file}")
    
    # Run original analysis function (convert lists back to tuples for the actual function)
    results = analyze_persistence_trajectories_filtered_wasserstein_geodesic_incremental(
        start_year=start_year,
        end_year=end_year,
        data_type=data_type,
        notebook_dir=notebook_dir,
        persistence_threshold=persistence_threshold,
        alpha=alpha,
        beta=beta,
        distance_threshold=distance_threshold,
        matrix_size=matrix_size,
        lat_range=tuple(lat_range) if isinstance(lat_range, list) else lat_range,
        lon_range=tuple(lon_range) if isinstance(lon_range, list) else lon_range
    )
    
    # Save results and metadata
    try:
        print("="*60)
        print("SAVING RESULTS TO CACHE")
        print("="*60)
        
        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Create metadata
        metadata = {
            'parameters': params_hash,  # Already converted to lists above
            'created_at': datetime.now().isoformat(),
            'total_trajectories': len(results['trajectories']),
            'total_time_points': len(results['dates']),
            'filename': os.path.basename(results_file),
            'file_size_mb': os.path.getsize(results_file) / (1024 * 1024)
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")
        print(f"Metadata saved to: {metadata_file}")
        print(f"File size: {metadata['file_size_mb']:.2f} MB")
        print(f"Total trajectories: {metadata['total_trajectories']}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        print("Continuing with results in memory...")
    
    return results

# Alternative: Force load existing cache if you know it's the right one
def force_load_cached_result(notebook_dir, pattern):
    """
    Force load a cached result by pattern matching (bypasses parameter check)
    
    Parameters:
    -----------
    notebook_dir : str
        Directory path
    pattern : str
        Pattern to match in filename (e.g., "sub_1948-2023")
    """
    
    cache_dir = os.path.join(notebook_dir, "data", "processed_data", "feature_tracking")
    
    # Find matching results file
    results_files = [f for f in os.listdir(cache_dir) 
                    if f.startswith('trajectory_results_') and pattern in f and f.endswith('.pkl')]
    
    if not results_files:
        print(f"No cached results found matching pattern: {pattern}")
        return None
    
    if len(results_files) > 1:
        print(f"Multiple files found matching pattern '{pattern}':")
        for f in results_files:
            print(f"  {f}")
        print("Using first match...")
    
    results_file = os.path.join(cache_dir, results_files[0])
    
    print(f"Force loading cached results from: {results_file}")
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print("Cached results loaded successfully!")
        return results
    except Exception as e:
        print(f"Error loading cached results: {e}")
        return None

# Helper function to plot trajectory
def plot_trajectory(results, trajectory_id, notebook_dir,figsize=(10, 10)):
    """
    Plot all points of a specific trajectory using the plotting function
    
    Parameters:
    -----------
    results : dict
        Results from trajectory analysis
    trajectory_id : int
        ID of trajectory to plot (0 to len(results['trajectories'])-1)
    notebook_dir : str
        Directory for data loading
    """
    
    if trajectory_id >= len(results['trajectories']):
        print(f"Trajectory {trajectory_id} not found. Available: 0-{len(results['trajectories'])-1}")
        return
    
    trajectory = results['trajectories'][trajectory_id]
    data_type = results['parameters']['data_type']
    
    print(f"Plotting trajectory {trajectory_id} with {len(trajectory)} points")
    print(f"Duration: {trajectory[0]['date']} to {trajectory[-1]['date']}")
    
    for i, point in enumerate(trajectory):
        year = point['year']
        day = point['day']
        feature_index = point['feature_index']
        date = point['date']
        
        #print(f"Point {i+1}: {date} - Feature #{feature_index}")
        
        try:
            fig, ax = plot_representative_on_heatmap(
                year=year,
                day=day, 
                data_type=data_type,
                feature_index=feature_index,
                notebook_dir=notebook_dir,
                figsize=figsize
            )
            plt.show()
        except Exception as e:
            print(f"Error plotting {date}: {e}")

from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt

def plot_trajectory_slider(results, trajectory_id, notebook_dir, figsize=(6, 5)):
    """
    Interactive slider to navigate through trajectory points
    
    Parameters:
    -----------
    results : dict
        Results from trajectory analysis
    trajectory_id : int
        ID of trajectory to plot
    notebook_dir : str
        Directory for data loading
    figsize : tuple
        Figure size
    """
    
    if trajectory_id >= len(results['trajectories']):
        print(f"Trajectory {trajectory_id} not found. Available: 0-{len(results['trajectories'])-1}")
        return
    
    trajectory = results['trajectories'][trajectory_id]
    data_type = results['parameters']['data_type']
    
    print(f"Interactive trajectory viewer for trajectory {trajectory_id}")
    print(f"Duration: {trajectory[0]['date']} to {trajectory[-1]['date']}")
    print(f"Total points: {len(trajectory)}")
    
    def show_trajectory_frame(frame_index):
        """Display specific frame of trajectory"""
        
        if frame_index >= len(trajectory):
            print(f"Frame {frame_index} not available")
            return
        
        point = trajectory[frame_index]
        year = point['year']
        day = point['day']
        feature_index = point['feature_index']
        date = point['date']
        
        # Clear previous plots
        plt.close('all')
        
        try:
            fig, ax = plot_representative_on_heatmap(
                year=year,
                day=day, 
                data_type=data_type,
                feature_index=feature_index,
                notebook_dir=notebook_dir,
                figsize=figsize
            )
            
            # Enhanced title with trajectory info
            persistence = point['persistence']
            lat = point.get('lat', 'N/A')
            lon = point.get('lon', 'N/A')
            
            ax.set_title(f'{date.strftime("%Y-%m-%d")} - Feature #{feature_index}\n'
                        f'Trajectory {trajectory_id} - Frame {frame_index+1}/{len(trajectory)}\n'
                        f'Persistence: {persistence:.3f}, Location: ({lat:.2f}, {lon:.2f})', 
                        fontsize=12, pad=20)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting frame {frame_index} ({date}): {e}")
    
    # Create slider widget
    slider = IntSlider(
        value=0,
        min=0,
        max=len(trajectory) - 1,
        step=1,
        description=f'Traj {trajectory_id}:',
        style={'description_width': 'initial'},
        layout={'width': '500px'}
    )
    
    # Create interactive widget
    interact(show_trajectory_frame, frame_index=slider)

def plot_multiple_trajectories_slider(results, notebook_dir, figsize=(6, 5)):
    """
    Interactive slider to navigate through multiple trajectories and their frames
    
    Parameters:
    -----------
    results : dict
        Results from trajectory analysis
    notebook_dir : str
        Directory for data loading
    figsize : tuple
        Figure size
    """
    
    trajectories = results['trajectories']
    data_type = results['parameters']['data_type']
    
    print(f"Interactive viewer for {len(trajectories)} trajectories")
    
    def show_trajectory_and_frame(trajectory_id, frame_index):
        """Display specific trajectory and frame"""
        
        if trajectory_id >= len(trajectories):
            print(f"Trajectory {trajectory_id} not available")
            return
            
        trajectory = trajectories[trajectory_id]
        
        if frame_index >= len(trajectory):
            print(f"Frame {frame_index} not available for trajectory {trajectory_id}")
            return
        
        point = trajectory[frame_index]
        year = point['year']
        day = point['day']
        feature_index = point['feature_index']
        date = point['date']
        
        # Clear previous plots
        plt.close('all')
        
        try:
            fig, ax = plot_representative_on_heatmap(
                year=year,
                day=day, 
                data_type=data_type,
                feature_index=feature_index,
                notebook_dir=notebook_dir,
                figsize=figsize
            )
            
            # Enhanced title with trajectory info
            persistence = point['persistence']
            lat = point.get('lat', 'N/A')
            lon = point.get('lon', 'N/A')
            
            ax.set_title(f'{date.strftime("%Y-%m-%d")} - Feature #{feature_index}\n'
                        f'Trajectory {trajectory_id} ({len(trajectory)} points) - Frame {frame_index+1}\n'
                        f'Persistence: {persistence:.3f}, Location: ({lat:.2f}, {lon:.2f})', 
                        fontsize=12, pad=20)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting trajectory {trajectory_id}, frame {frame_index}: {e}")
    
    # Get max trajectory length for frame slider
    max_frames = max(len(traj) for traj in trajectories) if trajectories else 1
    
    # Create sliders
    traj_slider = IntSlider(
        value=0,
        min=0,
        max=len(trajectories) - 1,
        step=1,
        description='Trajectory:',
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    frame_slider = IntSlider(
        value=0,
        min=0,
        max=max_frames - 1,
        step=1,
        description='Frame:',
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    # Create interactive widget
    interact(show_trajectory_and_frame, 
             trajectory_id=traj_slider, 
             frame_index=frame_slider)

def plot_top_trajectories_slider(results, notebook_dir, top_n=10, figsize=(6, 5)):
    """
    Interactive slider for top N longest trajectories
    
    Parameters:
    -----------
    results : dict
        Results from trajectory analysis
    notebook_dir : str
        Directory for data loading
    top_n : int
        Number of top trajectories to include
    figsize : tuple
        Figure size
    """
    
    # Get top N longest trajectories
    trajectory_lengths = results['trajectory_lengths']
    top_indices = sorted(range(len(trajectory_lengths)), 
                        key=lambda i: trajectory_lengths[i], reverse=True)[:top_n]
    
    trajectories = results['trajectories']
    data_type = results['parameters']['data_type']
    
    print(f"Interactive viewer for top {len(top_indices)} longest trajectories")
    
    # Create mapping for display
    traj_info = []
    for idx, traj_id in enumerate(top_indices):
        traj = trajectories[traj_id]
        traj_info.append({
            'display_id': idx,
            'original_id': traj_id,
            'length': len(traj),
            'trajectory': traj,
            'start_date': traj[0]['date'],
            'end_date': traj[-1]['date']
        })
    
    for info in traj_info:
        print(f"  {info['display_id']}: Trajectory {info['original_id']} - "
              f"{info['length']} points ({info['start_date']} to {info['end_date']})")
    
    def show_top_trajectory_frame(trajectory_index, frame_index):
        """Display specific frame from top trajectories"""
        
        if trajectory_index >= len(traj_info):
            print(f"Trajectory index {trajectory_index} not available")
            return
            
        info = traj_info[trajectory_index]
        trajectory = info['trajectory']
        
        if frame_index >= len(trajectory):
            print(f"Frame {frame_index} not available for trajectory {trajectory_index}")
            return
        
        point = trajectory[frame_index]
        year = point['year']
        day = point['day']
        feature_index = point['feature_index']
        date = point['date']
        
        # Clear previous plots
        plt.close('all')
        
        try:
            fig, ax = plot_representative_on_heatmap(
                year=year,
                day=day, 
                data_type=data_type,
                feature_index=feature_index,
                notebook_dir=notebook_dir,
                figsize=figsize
            )
            
            # Enhanced title
            persistence = point['persistence']
            lat = point.get('lat', 'N/A')
            lon = point.get('lon', 'N/A')
            
            ax.set_title(f'{date.strftime("%Y-%m-%d")} - Feature #{feature_index}\n'
                        f'Trajectory {info["original_id"]} (Rank #{trajectory_index+1}, {info["length"]} points)\n'
                        f'Frame {frame_index+1}/{info["length"]} - Persistence: {persistence:.3f}', 
                        fontsize=12, pad=20)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting trajectory {trajectory_index}, frame {frame_index}: {e}")
    
    # Get max trajectory length for frame slider
    max_frames = max(info['length'] for info in traj_info)
    
    # Create sliders
    traj_slider = IntSlider(
        value=0,
        min=0,
        max=len(traj_info) - 1,
        step=1,
        description='Top Trajectory:',
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    frame_slider = IntSlider(
        value=0,
        min=0,
        max=max_frames - 1,
        step=1,
        description='Frame:',
        style={'description_width': 'initial'},
        layout={'width': '400px'}
    )
    
    # Create interactive widget
    interact(show_top_trajectory_frame, 
             trajectory_index=traj_slider, 
             frame_index=frame_slider)


def plot_trajectory_slider_with_persistence_fixed(results, trajectory_id, notebook_dir,
                                                   figsize=(18, 8), lat_range=(0, 90),
                                                   lon_range=(0, 360), fontsize=11,
                                                   death_square_scale=1.0, color='orange',
                                                   star_size=160):
    """
    Interactive slider: North Polar Stereo globe (left) + persistence diagram (right)
    for a single trajectory.

    Parameters
    ----------
    results : dict   Results from trajectory analysis.
    trajectory_id : int   ID of trajectory to plot.
    notebook_dir : str   Directory for data loading.
    figsize : tuple   Figure size. Default (18, 8).
    lat_range : tuple   (min_lat, max_lat). Default (0, 90).
    lon_range : tuple   (min_lon, max_lon). Default (0, 360).
    fontsize : int   Base font size. Default 11.
    death_square_scale : float   Scale factor for death-simplex box. Default 1.0.
    color : str   Representative marker color. Default 'orange'.
    star_size : int   Star marker size on persistence diagram. Default 160.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.gridspec as gridspec

    if trajectory_id >= len(results['trajectories']):
        print(f"Trajectory {trajectory_id} not found. Available: 0-{len(results['trajectories'])-1}")
        return

    trajectory = results['trajectories'][trajectory_id]
    data_type = results['parameters']['data_type']
    _pd_scale = -1.0 / 100.0 if data_type == 'sup' else 1.0 / 100.0

    print(f"Trajectory {trajectory_id}: {trajectory[0]['date']} → {trajectory[-1]['date']}  ({len(trajectory)} days)")
    print("Pre-computing persistence bounds...")

    all_births, all_deaths = [], []
    for p in trajectory:
        try:
            df_b = load_all_representative_data_to_df(
                years=p['year'], days=p['day'], data_types=[data_type], notebook_dir=notebook_dir
            )
            if not df_b.empty:
                d = df_b[(df_b['year'] == p['year']) & (df_b['day'] == p['day']) &
                         (df_b['data_type'] == data_type)]
                if not d.empty:
                    all_births.extend(d['birth'].values * _pd_scale)
                    all_deaths.extend(d['death'].values * _pd_scale)
        except Exception:
            continue

    if all_births and all_deaths:
        mn = min(min(all_births), min(all_deaths))
        mx = max(max(all_births), max(all_deaths))
        margin = (mx - mn) * 0.1
        pd_xlim = (mn - margin, mx + margin)
        pd_ylim = (mn - margin, mx + margin)
    else:
        pd_xlim = pd_ylim = (-1, 1)
    print(f"Persistence bounds: {pd_xlim}")

    def matrix_to_latlon(row, col, height, width):
        cy, cx = height // 2, width // 2
        max_r = min(cx, cy)
        xc = col - cx
        yc = row - cy
        r = np.sqrt(xc**2 + yc**2)
        theta = np.arctan2(yc, xc)
        lon = (theta * 180 / np.pi + 90) % 360
        lat = lat_range[1] - (r / max_r) * (lat_range[1] - lat_range[0])
        return float(lat), float(lon)

    def show_trajectory_frame(frame_index):
        from IPython.display import display as _display
        point = trajectory[frame_index]
        year, day = point['year'], point['day']

        plt.close('all')
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1])

        # ===== LEFT: Globe =====
        ax_globe = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
        ax_globe.set_extent([lon_range[0], lon_range[1],
                             lat_range[0] - 5, lat_range[1]], ccrs.PlateCarree())
        ax_globe.set_facecolor('white')

        data_path = os.path.join(
            notebook_dir, "data", "processed_data", "SLP_data_years",
            str(year), f"slp_{data_type}_{year}_day_{day}.npy"
        )
        polar_matrix = None
        h = w = None
        if os.path.exists(data_path):
            polar_matrix = np.load(data_path)
            if data_type == 'sup':
                polar_matrix = -polar_matrix

            h, w = polar_matrix.shape
            cy, cx = h // 2, w // 2
            max_r = min(cx, cy)
            y_idx, x_idx = np.ogrid[:h, :w]
            xc = x_idx - cx
            yc = y_idx - cy
            r_mat = np.sqrt(xc**2 + yc**2)
            theta_mat = np.arctan2(yc, xc)
            lon_mat = (theta_mat * 180 / np.pi + 90) % 360
            lat_mat = lat_range[1] - (r_mat / max_r) * (lat_range[1] - lat_range[0])

            valid = (r_mat <= max_r) & (lat_mat >= lat_range[0]) & (lat_mat <= lat_range[1])
            data_masked = np.ma.masked_where(~valid, polar_matrix)

            vmax = np.max(np.abs(data_masked.compressed())) if data_masked.count() > 0 else 1.0
            cs = ax_globe.pcolormesh(
                lon_mat, lat_mat, data_masked / 100.0,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r', vmin=-vmax / 100.0, vmax=vmax / 100.0,
                shading='auto', zorder=1
            )
            cbar = fig.colorbar(cs, ax=ax_globe, orientation='vertical',
                                shrink=0.5, pad=0.05, aspect=20)
            cbar.set_label('hPa', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

        ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax_globe.gridlines(linewidth=0.5, alpha=0.5, color='gray', zorder=3)

        if polar_matrix is not None:
            try:
                df_rep = load_all_representative_data_to_df(
                    years=year, days=day, data_types=[data_type], notebook_dir=notebook_dir
                )
                if not df_rep.empty:
                    feat = df_rep[
                        (df_rep['year'] == year) & (df_rep['day'] == day) &
                        (df_rep['data_type'] == data_type) &
                        (df_rep['feature_index'] == point['feature_index'])
                    ]
                    if not feat.empty:
                        row0 = feat.iloc[0]
                        rep = row0['representative']
                        dsv = row0['death_simplex_vertices']

                        if rep:
                            coords = [v for edge in rep if len(edge) == 2 for v in edge]
                            if coords:
                                lats, lons = [], []
                                for (r, c) in coords:
                                    la, lo = matrix_to_latlon(r, c, h, w)
                                    if lat_range[0] <= la <= lat_range[1]:
                                        lats.append(la); lons.append(lo)
                                if lats:
                                    ax_globe.scatter(
                                        lons, lats, color=color, s=20, alpha=0.85,
                                        transform=ccrs.PlateCarree(), zorder=5,
                                        edgecolors='white', linewidths=0.4
                                    )

                        if dsv and len(dsv) == 4:
                            dr = [v[0] for v in dsv]
                            dc = [v[1] for v in dsv]
                            min_r_d, max_r_d = min(dr), max(dr)
                            min_c_d, max_c_d = min(dc), max(dc)
                            cr = (min_r_d + max_r_d) / 2
                            cc = (min_c_d + max_c_d) / 2
                            half_r = (max_r_d - min_r_d) / 2 * death_square_scale
                            half_c = (max_c_d - min_c_d) / 2 * death_square_scale
                            if max(half_r, half_c) <= min(h, w) * 0.15:
                                _n = 50
                                _pts = (
                                    [(cr - half_r, cc - half_c + t * 2 * half_c / _n) for t in range(_n)] +
                                    [(cr - half_r + t * 2 * half_r / _n, cc + half_c) for t in range(_n)] +
                                    [(cr + half_r, cc + half_c - t * 2 * half_c / _n) for t in range(_n)] +
                                    [(cr + half_r - t * 2 * half_r / _n, cc - half_c) for t in range(_n)] +
                                    [(cr - half_r, cc - half_c)]
                                )
                                box_lats, box_lons = [], []
                                for (r, c) in _pts:
                                    la, lo = matrix_to_latlon(r, c, h, w)
                                    box_lats.append(la); box_lons.append(lo)
                                # break line where longitude wraps 0°/360°
                                blons_plot, blats_plot = [], []
                                for i, (lo, la) in enumerate(zip(box_lons, box_lats)):
                                    if i > 0 and abs(lo - box_lons[i-1]) > 180:
                                        blons_plot.append(float('nan'))
                                        blats_plot.append(float('nan'))
                                    blons_plot.append(lo); blats_plot.append(la)
                                ax_globe.plot(
                                    blons_plot, blats_plot,
                                    color='yellow', linewidth=2, alpha=0.9,
                                    transform=ccrs.PlateCarree(), zorder=6
                                )
            except Exception:
                pass

        ax_globe.set_title(point['date'].strftime("%Y-%m-%d"), fontsize=fontsize, pad=6)

        # ===== RIGHT: Persistence diagram =====
        ax_pd = fig.add_subplot(gs[1])

        star_birth = point['birth'] * _pd_scale
        star_death = point['death'] * _pd_scale
        ax_pd.scatter(
            [star_birth], [star_death],
            c=color, s=star_size, marker='*',
            edgecolors=color, linewidths=0, zorder=10
        )
        # annotate birth/death pair and depth of the highlighted point
        star_depth = star_death - star_birth
        ax_pd.annotate(
            f"({star_birth:.2f}, {star_death:.2f})\ndepth={star_depth:.2f} hPa",
            xy=(star_birth, star_death),
            xytext=(8, 8), textcoords='offset points',
            fontsize=fontsize * 0.7, color='black', zorder=11,
            ha='left', va='bottom'
        )

        try:
            df_bg = load_all_representative_data_to_df(
                years=year, days=day, data_types=[data_type], notebook_dir=notebook_dir
            )
            if not df_bg.empty:
                bg = df_bg[(df_bg['year'] == year) & (df_bg['day'] == day) &
                           (df_bg['data_type'] == data_type)]
                if not bg.empty:
                    ax_pd.scatter(bg['birth'] * _pd_scale, bg['death'] * _pd_scale,
                                  c='black', s=15, alpha=0.8, zorder=1)
        except Exception:
            pass

        ax_pd.set_xlim(pd_xlim); ax_pd.set_ylim(pd_ylim)
        ax_pd.plot(pd_xlim, pd_ylim, 'k--', alpha=0.3, linewidth=1)
        # x=0 and y=0 reference lines
        ax_pd.axvline(0, color='gray', linewidth=1, alpha=0.6, zorder=0)
        ax_pd.axhline(0, color='gray', linewidth=1, alpha=0.6, zorder=0)
        ax_pd.set_xlabel('Birth (hPa)', fontsize=fontsize)
        ax_pd.set_ylabel('Death (hPa)', fontsize=fontsize)
        ax_pd.tick_params(labelsize=fontsize - 1)
        ax_pd.grid(False)
        ax_pd.set_aspect('equal')

        plt.tight_layout()
        _display(fig)
        plt.close(fig)

    from ipywidgets import IntSlider, Output
    from IPython.display import display

    out = Output()
    slider = IntSlider(
        value=0, min=0, max=len(trajectory) - 1, step=1,
        description=f'Traj {trajectory_id}:',
        style={'description_width': 'initial'},
        layout={'width': '500px'}
    )

    def on_slider_change(change):
        out.clear_output(wait=True)
        with out:
            show_trajectory_frame(change['new'])

    slider.observe(on_slider_change, names='value')
    display(slider, out)
    with out:
        show_trajectory_frame(0)




def plot_two_trajectories_globe_slider(results, trajectory_id_1, trajectory_id_2, notebook_dir,
                                       figsize=(18, 8), lat_range=(0, 90), lon_range=(0, 360),
                                       fontsize=10, death_square_scale=1.0, star_size=160):
    """
    Interactive slider: one globe (North Polar Stereo) + one persistence diagram,
    both showing trajectory_id_1 and trajectory_id_2 overlaid.

    Layout (1 row × 2 cols):
        Left : Globe for the current date with both trajectories' representatives marked.
        Right: Persistence diagram with both trajectories' history up to current date.

    The slider spans the union of both trajectories' date ranges.
    If a trajectory has no point on the current date its markers are simply absent.

    Parameters
    ----------
    results : dict
        Results from analyze_persistence_trajectories_cached.
    trajectory_id_1 : int
        ID of the first trajectory (red).
    trajectory_id_2 : int
        ID of the second trajectory (blue).
    notebook_dir : str
        Directory used for data loading.
    figsize : tuple, optional
        Figure size (width, height). Default (18, 8).
    lat_range : tuple, optional
        Latitude range for globe. Default (0, 90).
    lon_range : tuple, optional
        Longitude range for globe. Default (0, 360).
    fontsize : int, optional
        Base font size for all text in the figure. Default 10.
    death_square_scale : float, optional
        Scaling factor for the death simplex bounding box. 1.0 = original size,
        >1.0 = larger, <1.0 = smaller. Default 1.0.
    star_size : int, optional
        Size of the star marker on the persistence diagram. Default 160.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.gridspec as gridspec

    n_traj = len(results['trajectories'])
    for tid in (trajectory_id_1, trajectory_id_2):
        if tid >= n_traj:
            print(f"Trajectory {tid} not found. Available: 0-{n_traj - 1}")
            return

    traj1 = results['trajectories'][trajectory_id_1]
    traj2 = results['trajectories'][trajectory_id_2]
    data_type = results['parameters']['data_type']
    _pd_scale = -1.0 / 100.0 if data_type == 'sup' else 1.0 / 100.0

    date_to_point1 = {p['date']: p for p in traj1}
    date_to_point2 = {p['date']: p for p in traj2}
    all_dates = sorted(set(date_to_point1) | set(date_to_point2))

    print(f"Trajectory {trajectory_id_1}: {traj1[0]['date']} → {traj1[-1]['date']}  ({len(traj1)} days)")
    print(f"Trajectory {trajectory_id_2}: {traj2[0]['date']} → {traj2[-1]['date']}  ({len(traj2)} days)")
    print(f"Combined timeline: {all_dates[0]} → {all_dates[-1]}  ({len(all_dates)} dates)")

    # ---------- pre-compute fixed persistence bounds ----------
    print("Pre-computing persistence bounds...")
    all_births, all_deaths = [], []
    combined_year_days = set((p['year'], p['day']) for p in traj1 + traj2)
    for year, day in combined_year_days:
        try:
            df_b = load_all_representative_data_to_df(
                years=year, days=day, data_types=[data_type], notebook_dir=notebook_dir
            )
            if not df_b.empty:
                d = df_b[(df_b['year'] == year) & (df_b['day'] == day) & (df_b['data_type'] == data_type)]
                if not d.empty:
                    all_births.extend(d['birth'].values * _pd_scale)
                    all_deaths.extend(d['death'].values * _pd_scale)
        except Exception:
            continue

    if all_births and all_deaths:
        mn = min(min(all_births), min(all_deaths))
        mx = max(max(all_births), max(all_deaths))
        margin = (mx - mn) * 0.1
        pd_xlim = (mn - margin, mx + margin)
        pd_ylim = (mn - margin, mx + margin)
    else:
        pd_xlim = pd_ylim = (-1, 1)
    print(f"Persistence bounds: {pd_xlim}")

    # ---------- helper: matrix (row, col) → (lat, lon) ----------
    def matrix_to_latlon(row, col, height, width):
        cy, cx = height // 2, width // 2
        max_r = min(cx, cy)
        xc = col - cx
        yc = row - cy
        r = np.sqrt(xc**2 + yc**2)
        theta = np.arctan2(yc, xc)
        lon = (theta * 180 / np.pi + 90) % 360
        lat = lat_range[1] - (r / max_r) * (lat_range[1] - lat_range[0])
        return float(lat), float(lon)

    # ---------- frame renderer ----------
    COLORS = {trajectory_id_1: 'red', trajectory_id_2: 'blue'}

    def show_frame(date_index):
        date = all_dates[date_index]
        point1 = date_to_point1.get(date)
        point2 = date_to_point2.get(date)

        # Use whichever trajectory is active for the background data
        active_point = point1 or point2
        year, day = active_point['year'], active_point['day']

        plt.close('all')
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.1, 1])

        # ===== LEFT: Globe =====
        ax_globe = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
        ax_globe.set_extent([lon_range[0], lon_range[1],
                             lat_range[0] - 5, lat_range[1]], ccrs.PlateCarree())
        ax_globe.set_facecolor('white')

        # Load and plot polar matrix background
        data_path = os.path.join(
            notebook_dir, "data", "processed_data", "SLP_data_years",
            str(year), f"slp_{data_type}_{year}_day_{day}.npy"
        )
        polar_matrix = None
        if os.path.exists(data_path):
            polar_matrix = np.load(data_path)
            if data_type == 'sup':
                polar_matrix = -polar_matrix

            h, w = polar_matrix.shape
            cy, cx = h // 2, w // 2
            max_r = min(cx, cy)
            y_idx, x_idx = np.ogrid[:h, :w]
            xc = x_idx - cx
            yc = y_idx - cy
            r_mat = np.sqrt(xc**2 + yc**2)
            theta_mat = np.arctan2(yc, xc)
            lon_mat = (theta_mat * 180 / np.pi + 90) % 360
            lat_mat = lat_range[1] - (r_mat / max_r) * (lat_range[1] - lat_range[0])

            valid = (r_mat <= max_r) & (lat_mat >= lat_range[0]) & (lat_mat <= lat_range[1])
            data_masked = np.ma.masked_where(~valid, polar_matrix)

            vmax = np.max(np.abs(data_masked.compressed())) if data_masked.count() > 0 else 1.0
            cs = ax_globe.pcolormesh(
                lon_mat, lat_mat, data_masked / 100.0,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r', vmin=-vmax / 100.0, vmax=vmax / 100.0,
                shading='auto', zorder=1
            )
            cbar = fig.colorbar(cs, ax=ax_globe, orientation='vertical',
                                shrink=0.5, pad=0.05, aspect=20)
            cbar.set_label('hPa', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

        ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax_globe.gridlines(linewidth=0.5, alpha=0.5, color='gray', zorder=3)

        # Mark representatives for each active trajectory on the globe
        for point, color in [(point1, COLORS[trajectory_id_1]),
                             (point2, COLORS[trajectory_id_2])]:
            if point is None:
                continue
            p_year, p_day = point['year'], point['day']
            try:
                df_rep = load_all_representative_data_to_df(
                    years=p_year, days=p_day, data_types=[data_type], notebook_dir=notebook_dir
                )
                if not df_rep.empty and polar_matrix is not None:
                    feat = df_rep[
                        (df_rep['year'] == p_year) & (df_rep['day'] == p_day) &
                        (df_rep['data_type'] == data_type) &
                        (df_rep['feature_index'] == point['feature_index'])
                    ]
                    if not feat.empty:
                        row0 = feat.iloc[0]
                        rep = row0['representative']
                        dsv = row0['death_simplex_vertices']

                        # Plot representative points
                        if rep:
                            coords = [v for edge in rep if len(edge) == 2 for v in edge]
                            if coords:
                                lats = []; lons = []
                                for (r, c) in coords:
                                    la, lo = matrix_to_latlon(r, c, h, w)
                                    if lat_range[0] <= la <= lat_range[1]:
                                        lats.append(la); lons.append(lo)
                                if lats:
                                    ax_globe.scatter(
                                        lons, lats, color=color, s=20, alpha=0.85,
                                        transform=ccrs.PlateCarree(), zorder=5,
                                        edgecolors='white', linewidths=0.4
                                    )

                        # Plot death simplex as a closed polygon on the globe
                        if dsv and len(dsv) == 4:
                            dr = [v[0] for v in dsv]
                            dc = [v[1] for v in dsv]
                            min_r, max_r = min(dr), max(dr)
                            min_c, max_c = min(dc), max(dc)
                            cr = (min_r + max_r) / 2
                            cc = (min_c + max_c) / 2
                            half_r = (max_r - min_r) / 2 * death_square_scale
                            half_c = (max_c - min_c) / 2 * death_square_scale
                            if max(half_r, half_c) <= min(h, w) * 0.15:
                                _n = 50
                                _pts = (
                                    [(cr - half_r, cc - half_c + t * 2 * half_c / _n) for t in range(_n)] +
                                    [(cr - half_r + t * 2 * half_r / _n, cc + half_c) for t in range(_n)] +
                                    [(cr + half_r, cc + half_c - t * 2 * half_c / _n) for t in range(_n)] +
                                    [(cr + half_r - t * 2 * half_r / _n, cc - half_c) for t in range(_n)] +
                                    [(cr - half_r, cc - half_c)]
                                )
                                box_lats = []; box_lons = []
                                for (r, c) in _pts:
                                    la, lo = matrix_to_latlon(r, c, h, w)
                                    box_lats.append(la); box_lons.append(lo)
                                blons_plot, blats_plot = [], []
                                for i, (lo, la) in enumerate(zip(box_lons, box_lats)):
                                    if i > 0 and abs(lo - box_lons[i-1]) > 180:
                                        blons_plot.append(float('nan'))
                                        blats_plot.append(float('nan'))
                                    blons_plot.append(lo); blats_plot.append(la)
                                ax_globe.plot(
                                    blons_plot, blats_plot,
                                    color='yellow', linewidth=2, alpha=0.9,
                                    transform=ccrs.PlateCarree(), zorder=6
                                )
            except Exception:
                pass

        ax_globe.set_title(date.strftime("%Y-%m-%d"), fontsize=fontsize, pad=6)

        # ===== RIGHT: Persistence diagram =====
        ax_pd = fig.add_subplot(gs[1])

        for date_to_pt, color in [
            (date_to_point1, COLORS[trajectory_id_1]),
            (date_to_point2, COLORS[trajectory_id_2]),
        ]:
            point = date_to_pt.get(date)

            # Current point only
            if point is not None:
                star_birth = point['birth'] * _pd_scale
                star_death = point['death'] * _pd_scale
                ax_pd.scatter(
                    [star_birth], [star_death],
                    c=color, s=star_size, marker='*',
                    edgecolors=color, linewidths=0, zorder=10
                )
                # annotate birth/death pair and depth of the highlighted point
                star_depth = star_death - star_birth
                ax_pd.annotate(
                    f"({star_birth:.2f}, {star_death:.2f})\ndepth={star_depth:.2f} hPa",
                    xy=(star_birth, star_death),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=fontsize * 0.7, color='black', zorder=11,
                    ha='left', va='bottom'
                )

        # Background features — no label
        try:
            df_bg = load_all_representative_data_to_df(
                years=year, days=day, data_types=[data_type], notebook_dir=notebook_dir
            )
            if not df_bg.empty:
                bg = df_bg[(df_bg['year'] == year) & (df_bg['day'] == day) &
                           (df_bg['data_type'] == data_type)]
                if not bg.empty:
                    ax_pd.scatter(bg['birth'] * _pd_scale, bg['death'] * _pd_scale,
                                  c='black', s=15, alpha=0.8, zorder=1)
        except Exception:
            pass

        ax_pd.set_xlim(pd_xlim); ax_pd.set_ylim(pd_ylim)
        ax_pd.plot(pd_xlim, pd_ylim, 'k--', alpha=0.3, linewidth=1)
        # x=0 and y=0 reference lines
        ax_pd.axvline(0, color='gray', linewidth=1, alpha=0.6, zorder=0)
        ax_pd.axhline(0, color='gray', linewidth=1, alpha=0.6, zorder=0)
        ax_pd.set_xlabel('Birth (hPa)', fontsize=fontsize)
        ax_pd.set_ylabel('Death (hPa)', fontsize=fontsize)
        #ax_pd.set_title('Persistence Diagram', fontsize=fontsize, pad=6)
        ax_pd.tick_params(labelsize=fontsize - 1)
        ax_pd.grid(False)
        ax_pd.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    # ---------- slider ----------
    slider = IntSlider(
        value=0, min=0, max=len(all_dates) - 1, step=1,
        description='Date:', style={'description_width': 'initial'},
        layout={'width': '600px'}
    )
    interact(show_frame, date_index=slider)


def plot_two_trajectories_globe_only(results1, trajectory_id_1, results2, trajectory_id_2,
                                     notebook_dir, figsize=(10, 10),
                                     lat_range=(0, 90), lon_range=(0, 360),
                                     fontsize=11, death_square_scale=1.0, star_size=160):
    """
    Interactive slider showing two trajectories (from different results dicts) overlaid
    on a single North Polar Stereographic globe — no persistence diagram.

    Parameters
    ----------
    results1 : dict   Results dict containing trajectory_id_1.
    trajectory_id_1 : int   Trajectory ID in results1 (plotted in red).
    results2 : dict   Results dict containing trajectory_id_2.
    trajectory_id_2 : int   Trajectory ID in results2 (plotted in blue).
    notebook_dir : str   Root directory used for data loading.
    figsize : tuple   Figure size. Default (10, 10).
    lat_range : tuple   (min_lat, max_lat) for globe. Default (0, 90).
    lon_range : tuple   (min_lon, max_lon) for globe. Default (0, 360).
    fontsize : int   Base font size. Default 11.
    death_square_scale : float  Scale factor for death-simplex bounding box. Default 1.0.
    star_size : int   Marker size for current-day feature centre. Default 160.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    n1 = len(results1['trajectories'])
    n2 = len(results2['trajectories'])
    if trajectory_id_1 >= n1:
        print(f"Trajectory {trajectory_id_1} not found in results1. Available: 0-{n1-1}")
        return
    if trajectory_id_2 >= n2:
        print(f"Trajectory {trajectory_id_2} not found in results2. Available: 0-{n2-1}")
        return

    traj1 = results1['trajectories'][trajectory_id_1]
    traj2 = results2['trajectories'][trajectory_id_2]
    data_type1 = results1['parameters']['data_type']
    data_type2 = results2['parameters']['data_type']

    date_to_point1 = {p['date']: p for p in traj1}
    date_to_point2 = {p['date']: p for p in traj2}
    all_dates = sorted(set(date_to_point1) | set(date_to_point2))

    print(f"Trajectory {trajectory_id_1} (red):  {traj1[0]['date']} → {traj1[-1]['date']}  ({len(traj1)} days)")
    print(f"Trajectory {trajectory_id_2} (blue): {traj2[0]['date']} → {traj2[-1]['date']}  ({len(traj2)} days)")
    print(f"Combined timeline: {all_dates[0]} → {all_dates[-1]}  ({len(all_dates)} dates)")

    def matrix_to_latlon(row, col, height, width):
        cy, cx = height // 2, width // 2
        max_r = min(cx, cy)
        xc = col - cx
        yc = row - cy
        r = np.sqrt(xc**2 + yc**2)
        theta = np.arctan2(yc, xc)
        lon = (theta * 180 / np.pi + 90) % 360
        lat = lat_range[1] - (r / max_r) * (lat_range[1] - lat_range[0])
        return float(lat), float(lon)

    def show_frame(date_index):
        date = all_dates[date_index]
        point1 = date_to_point1.get(date)
        point2 = date_to_point2.get(date)

        active_point = point1 or point2
        year, day = active_point['year'], active_point['day']
        # prefer data_type of whichever point is active (both usually match)
        active_data_type = data_type1 if point1 else data_type2

        plt.close('all')
        fig, ax_globe = plt.subplots(
            1, 1, figsize=figsize,
            subplot_kw={'projection': ccrs.NorthPolarStereo()}
        )

        ax_globe.set_extent([lon_range[0], lon_range[1],
                             lat_range[0] - 5, lat_range[1]], ccrs.PlateCarree())
        ax_globe.set_facecolor('white')

        # Background SLP field
        data_path = os.path.join(
            notebook_dir, "data", "processed_data", "SLP_data_years",
            str(year), f"slp_{active_data_type}_{year}_day_{day}.npy"
        )
        polar_matrix = None
        h = w = None
        if os.path.exists(data_path):
            polar_matrix = np.load(data_path)
            if active_data_type == 'sup':
                polar_matrix = -polar_matrix

            h, w = polar_matrix.shape
            cy, cx = h // 2, w // 2
            max_r = min(cx, cy)
            y_idx, x_idx = np.ogrid[:h, :w]
            xc = x_idx - cx
            yc = y_idx - cy
            r_mat = np.sqrt(xc**2 + yc**2)
            theta_mat = np.arctan2(yc, xc)
            lon_mat = (theta_mat * 180 / np.pi + 90) % 360
            lat_mat = lat_range[1] - (r_mat / max_r) * (lat_range[1] - lat_range[0])

            valid = (r_mat <= max_r) & (lat_mat >= lat_range[0]) & (lat_mat <= lat_range[1])
            data_masked = np.ma.masked_where(~valid, polar_matrix)

            vmax = np.max(np.abs(data_masked.compressed())) if data_masked.count() > 0 else 1.0
            cs = ax_globe.pcolormesh(
                lon_mat, lat_mat, data_masked / 100.0,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r', vmin=-vmax / 100.0, vmax=vmax / 100.0,
                shading='auto', zorder=1
            )
            cbar = fig.colorbar(cs, ax=ax_globe, orientation='vertical',
                                shrink=0.5, pad=0.05, aspect=20)
            cbar.set_label('hPa', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

        ax_globe.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
        ax_globe.gridlines(linewidth=0.5, alpha=0.5, color='gray', zorder=3)

        # Mark representatives for both trajectories
        for point, color, data_type in [
            (point1, 'orange', data_type1),
            (point2, 'cyan',   data_type2),
        ]:
            if point is None or polar_matrix is None:
                continue
            p_year, p_day = point['year'], point['day']
            try:
                df_rep = load_all_representative_data_to_df(
                    years=p_year, days=p_day, data_types=[data_type], notebook_dir=notebook_dir
                )
                if df_rep.empty:
                    continue
                feat = df_rep[
                    (df_rep['year'] == p_year) & (df_rep['day'] == p_day) &
                    (df_rep['data_type'] == data_type) &
                    (df_rep['feature_index'] == point['feature_index'])
                ]
                if feat.empty:
                    continue
                row0 = feat.iloc[0]
                rep = row0['representative']
                dsv = row0['death_simplex_vertices']

                # Representative scatter
                if rep:
                    coords = [v for edge in rep if len(edge) == 2 for v in edge]
                    if coords:
                        lats, lons = [], []
                        for (r, c) in coords:
                            la, lo = matrix_to_latlon(r, c, h, w)
                            if lat_range[0] <= la <= lat_range[1]:
                                lats.append(la); lons.append(lo)
                        if lats:
                            ax_globe.scatter(
                                lons, lats, color=color, s=20, alpha=0.85,
                                transform=ccrs.PlateCarree(), zorder=5,
                                edgecolors='white', linewidths=0.4
                            )

                # Death simplex bounding box
                if dsv and len(dsv) == 4:
                    dr = [v[0] for v in dsv]
                    dc = [v[1] for v in dsv]
                    min_r_d, max_r_d = min(dr), max(dr)
                    min_c_d, max_c_d = min(dc), max(dc)
                    cr = (min_r_d + max_r_d) / 2
                    cc = (min_c_d + max_c_d) / 2
                    half_r = (max_r_d - min_r_d) / 2 * death_square_scale
                    half_c = (max_c_d - min_c_d) / 2 * death_square_scale
                    if max(half_r, half_c) <= min(h, w) * 0.15:
                        _n = 50
                        _pts = (
                            [(cr - half_r, cc - half_c + t * 2 * half_c / _n) for t in range(_n)] +
                            [(cr - half_r + t * 2 * half_r / _n, cc + half_c) for t in range(_n)] +
                            [(cr + half_r, cc + half_c - t * 2 * half_c / _n) for t in range(_n)] +
                            [(cr + half_r - t * 2 * half_r / _n, cc - half_c) for t in range(_n)] +
                            [(cr - half_r, cc - half_c)]
                        )
                        box_lats = []; box_lons = []
                        for (r, c) in _pts:
                            la, lo = matrix_to_latlon(r, c, h, w)
                            box_lats.append(la); box_lons.append(lo)
                        blons_plot, blats_plot = [], []
                        for i, (lo, la) in enumerate(zip(box_lons, box_lats)):
                            if i > 0 and abs(lo - box_lons[i-1]) > 180:
                                blons_plot.append(float('nan'))
                                blats_plot.append(float('nan'))
                            blons_plot.append(lo); blats_plot.append(la)
                        ax_globe.plot(
                            blons_plot, blats_plot,
                            color='yellow', linewidth=2, alpha=0.9,
                            transform=ccrs.PlateCarree(), zorder=6
                        )
            except Exception:
                pass

        ax_globe.set_title(date.strftime("%Y-%m-%d"), fontsize=fontsize, pad=6)

        plt.tight_layout()
        plt.show()

    slider = IntSlider(
        value=0, min=0, max=len(all_dates) - 1, step=1,
        description='Date:', style={'description_width': 'initial'},
        layout={'width': '600px'}
    )
    interact(show_frame, date_index=slider)


def print_death_square_edges(results, trajectory_id, frame_index, notebook_dir,
                              lat_range=(0, 90)):
    """
    Print the death simplex bounding box corners and edges (matrix coords + lat/lon)
    for a given trajectory frame. Useful for debugging the death square size.

    Parameters
    ----------
    results : dict
    trajectory_id : int
    frame_index : int   Frame (day) within the trajectory.
    notebook_dir : str
    lat_range : tuple
    """
    traj = results['trajectories'][trajectory_id]
    data_type = results['parameters']['data_type']
    point = traj[frame_index]
    year, day = point['year'], point['day']

    print(f"Trajectory {trajectory_id}, frame {frame_index}, date {point['date']}")
    print(f"  feature_index={point['feature_index']}, year={year}, day={day}")

    data_path = os.path.join(notebook_dir, "data", "processed_data", "SLP_data_years",
                             str(year), f"slp_{data_type}_{year}_day_{day}.npy")
    if not os.path.exists(data_path):
        print("  Data file not found:", data_path)
        return

    pm = np.load(data_path)
    h, w = pm.shape
    cy, cx = h // 2, w // 2
    max_r = min(cx, cy)

    def _to_latlon(r, c):
        xc, yc = c - cx, r - cy
        lon = (np.arctan2(yc, xc) * 180 / np.pi + 90) % 360
        lat = lat_range[1] - (np.sqrt(xc**2 + yc**2) / max_r) * (lat_range[1] - lat_range[0])
        return round(float(lat), 2), round(float(lon), 2)

    df = load_all_representative_data_to_df(
        years=year, days=day, data_types=[data_type], notebook_dir=notebook_dir)
    feat = df[(df['year'] == year) & (df['day'] == day) &
              (df['data_type'] == data_type) &
              (df['feature_index'] == point['feature_index'])]

    print(f"  Matching rows in df: {len(feat)}")
    if feat.empty:
        print("  No feature data found.")
        return

    dsv = feat.iloc[0]['death_simplex_vertices']
    if not dsv or len(dsv) != 4:
        print(f"  death_simplex_vertices: {dsv}  (unexpected format)")
        return

    dr = [v[0] for v in dsv]
    dc = [v[1] for v in dsv]
    min_r, max_r2 = min(dr), max(dr)
    min_c, max_c  = min(dc), max(dc)
    cr = (min_r + max_r2) / 2
    cc = (min_c + max_c)  / 2
    half_r = (max_r2 - min_r) / 2
    half_c = (max_c  - min_c) / 2

    print(f"\n  Matrix coords:")
    print(f"    row range : {min_r:.1f} → {max_r2:.1f}  (half_r = {half_r:.1f})")
    print(f"    col range : {min_c:.1f} → {max_c:.1f}  (half_c = {half_c:.1f})")
    print(f"    centre    : ({cr:.1f}, {cc:.1f})")
    print(f"    matrix size: h={h}, w={w}  →  threshold = {min(h,w)*0.15:.1f}")
    print(f"    box passes size check (<=0.15): {max(half_r, half_c) <= min(h, w) * 0.15}")

    corners = [
        ('top-left',  min_r, min_c),
        ('top-right', min_r, max_c),
        ('bot-right', max_r2, max_c),
        ('bot-left',  max_r2, min_c),
    ]
    print(f"\n  Corner lat/lon:")
    for name, r, c in corners:
        lat, lon = _to_latlon(r, c)
        print(f"    {name:12s}  matrix=({r:.1f},{c:.1f})  lat={lat}°  lon={lon}°")


def get_trajectory_summary(results):
    """
    Get a summary DataFrame with one row per trajectory (sorted by length)
    
    Returns:
    --------
    pd.DataFrame with trajectory-level statistics including original trajectory IDs
    """
    import pandas as pd
    import numpy as np
    
    trajectories = results['trajectories']
    trajectory_lengths = results['trajectory_lengths']
    trajectory_lifespans = results['trajectory_lifespans']
    params = results.get('parameters', {})
    matrix_size = params.get('matrix_size', 172)
    lat_range = tuple(params.get('lat_range', (0, 90)))
    lon_range = tuple(params.get('lon_range', (0, 360)))

    if len(trajectories) == 0:
        return pd.DataFrame()
    
    # Get trajectory indices sorted by length (descending)
    sorted_indices = np.argsort(trajectory_lengths)[::-1]
    
    summary_data = []
    
    for rank, traj_idx in enumerate(sorted_indices, 1):
        trajectory = trajectories[traj_idx]
        traj_length = trajectory_lengths[traj_idx]
        traj_lifespan = trajectory_lifespans[traj_idx]
        
        # Extract trajectory statistics (adjust indices based on trajectory format)
        if isinstance(trajectory[0], dict):
            # Dictionary format
            dates = [point['date'] for point in trajectory]
            births = [point['birth'] for point in trajectory]
            deaths = [point['death'] for point in trajectory]
            persistences = [point['persistence'] for point in trajectory]
            rows = [point['row'] for point in trajectory]
            cols = [point['col'] for point in trajectory]
        else:
            # Tuple/list format: (time_index, date, birth, death, persistence, row, col)
            dates = [point[1] for point in trajectory]
            births = [point[2] for point in trajectory]
            deaths = [point[3] for point in trajectory]
            persistences = [point[4] for point in trajectory]
            rows = [point[5] for point in trajectory] if len(trajectory[0]) > 5 else [0] * len(trajectory)
            cols = [point[6] for point in trajectory] if len(trajectory[0]) > 6 else [0] * len(trajectory)
        
        # Calculate spatial movement in km using geodesic distances
        spatial_distances = []
        for i in range(1, len(rows)):
            dist = _geodesic_distance_km(
                rows[i-1], cols[i-1], rows[i], cols[i],
                matrix_size, lat_range, lon_range
            )
            if not np.isnan(dist):
                spatial_distances.append(dist)
        
        summary_data.append({
            'trajectory_id': traj_idx,  # Original trajectory index
            'trajectory_rank': rank,
            'trajectory_length': traj_length,
            'trajectory_lifespan': traj_lifespan,
            'start_date': dates[0],
            'end_date': dates[-1],
            'mean_persistence': np.mean(persistences),
            'max_persistence': np.max(persistences),
            'min_persistence': np.min(persistences),
            'std_persistence': np.std(persistences),
            'mean_birth': np.mean(births),
            'mean_death': np.mean(deaths),
            'start_row': rows[0],
            'start_col': cols[0],
            'end_row': rows[-1],
            'end_col': cols[-1],
            'total_spatial_movement': sum(spatial_distances) if spatial_distances else 0,
            'avg_spatial_movement_per_day': np.mean(spatial_distances) if spatial_distances else 0,
            'max_spatial_movement': max(spatial_distances) if spatial_distances else 0,
            'dates_list': dates,
            'birth_death_pairs': [(b, d) for b, d in zip(births, deaths)],
            'persistence_values': persistences,
            'spatial_path': [(r, c) for r, c in zip(rows, cols)]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('trajectory_rank')
    
    print(f"Created summary DataFrame with {len(summary_df)} trajectories")

    return summary_df

def find_overlapping_trajectories(summary_df, min_overlap_days=3, min_trajectory_length=2):
    """
    Find all pairs of trajectories whose date ranges overlap by at least min_overlap_days.

    Two trajectories A and B overlap if:
        start_A <= end_B  AND  start_B <= end_A

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of get_trajectory_summary(). Must contain columns
        'trajectory_id', 'start_date', 'end_date', 'trajectory_length'.
    min_overlap_days : int, optional (default=3)
        Minimum number of shared days required to count as an overlap.
    min_trajectory_length : int, optional (default=2)
        Trajectories with length strictly less than this value are excluded
        before overlap detection.

    Returns
    -------
    overlap_df : pd.DataFrame
        Each row is one overlapping pair with columns:
            traj_id_1, start_1, end_1,
            traj_id_2, start_2, end_2,
            overlap_start, overlap_end, overlap_days
        Sorted by overlap_days descending.
    groups : dict
        Maps each trajectory_id to the set of trajectory_ids it overlaps with.
    """
    import pandas as pd

    filtered = summary_df[summary_df['trajectory_length'] >= min_trajectory_length]
    print(f"Trajectories after length filter (>= {min_trajectory_length}): {len(filtered)} / {len(summary_df)}")

    df = filtered[['trajectory_id', 'start_date', 'end_date']].copy()
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    records = df.to_dict('records')

    rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            overlap_start = max(a['start_date'], b['start_date'])
            overlap_end   = min(a['end_date'],   b['end_date'])
            if overlap_start <= overlap_end:
                days = (overlap_end - overlap_start).days + 1
                if days >= min_overlap_days:
                    rows.append({
                        'traj_id_1':     a['trajectory_id'],
                        'start_1':       a['start_date'],
                        'end_1':         a['end_date'],
                        'traj_id_2':     b['trajectory_id'],
                        'start_2':       b['start_date'],
                        'end_2':         b['end_date'],
                        'overlap_start': overlap_start,
                        'overlap_end':   overlap_end,
                        'overlap_days':  days,
                    })

    overlap_df = pd.DataFrame(rows).sort_values('overlap_days', ascending=False).reset_index(drop=True)

    # Build a lookup: trajectory_id -> set of overlapping trajectory_ids
    groups = {}
    for _, row in overlap_df.iterrows():
        groups.setdefault(row['traj_id_1'], set()).add(row['traj_id_2'])
        groups.setdefault(row['traj_id_2'], set()).add(row['traj_id_1'])

    print(f"Found {len(overlap_df)} overlapping pairs among {len(groups)} trajectories.")
    return overlap_df, groups

def get_overall_summary_stats(results, summary_df):
    """
    Get overall summary statistics across all trajectories
    """
    import pandas as pd
    import numpy as np
    
    print("=== TRAJECTORY ANALYSIS SUMMARY ===")
    print(f"Data type: {results['parameters'].get('data_type', 'N/A')}")
    print(f"Start year: {results['parameters'].get('start_year', 'N/A')}")
    print(f"End year: {results['parameters'].get('end_year', 'N/A')}")
    print(f"Persistence threshold: {results['parameters']['persistence_threshold']}")
    print(f"Alpha (topological weight): {results['parameters'].get('alpha', 'N/A')}")
    print(f"Beta (spatial weight): {results['parameters'].get('beta', 'N/A')}")
    print(f"Distance threshold: {results['parameters'].get('distance_threshold', 'N/A')}")
    print(f"Matrix size: {results['parameters'].get('matrix_size', 'N/A')}")
    print()
    
    # Basic trajectory statistics
    print("=== TRAJECTORY LENGTH STATISTICS ===")
    print(f"Total trajectories: {len(summary_df)}")
    print(f"Mean length: {summary_df['trajectory_length'].mean():.2f} days")
    print(f"Median length: {summary_df['trajectory_length'].median():.2f} days")
    print(f"Max length: {summary_df['trajectory_length'].max()} days")
    print(f"Min length: {summary_df['trajectory_length'].min()} days")
    print(f"Std length: {summary_df['trajectory_length'].std():.2f} days")
    print()
    
    # Lifespan statistics
    print("=== TRAJECTORY LIFESPAN STATISTICS ===")
    print(f"Mean lifespan: {summary_df['trajectory_lifespan'].mean():.2f} days")
    print(f"Median lifespan: {summary_df['trajectory_lifespan'].median():.2f} days")
    print(f"Max lifespan: {summary_df['trajectory_lifespan'].max()} days")
    print()
    
    # Persistence statistics
    print("=== PERSISTENCE STATISTICS ===")
    print(f"Mean persistence: {summary_df['mean_persistence'].mean():.4f}")
    print(f"Max persistence (across all): {summary_df['max_persistence'].max():.4f}")
    print(f"Min persistence (across all): {summary_df['min_persistence'].min():.4f}")
    print(f"Std of mean persistence: {summary_df['mean_persistence'].std():.4f}")
    print()
    
    # Birth-Death statistics
    print("=== BIRTH-DEATH STATISTICS ===")
    print(f"Mean birth value: {summary_df['mean_birth'].mean():.4f}")
    print(f"Mean death value: {summary_df['mean_death'].mean():.4f}")
    print()
    
    # Spatial movement statistics
    if 'total_spatial_movement' in summary_df.columns:
        print("=== SPATIAL MOVEMENT STATISTICS ===")
        print(f"Mean total movement: {summary_df['total_spatial_movement'].mean():.2f} km")
        print(f"Max total movement: {summary_df['total_spatial_movement'].max():.2f} km")
        print(f"Mean avg movement per day: {summary_df['avg_spatial_movement_per_day'].mean():.2f} km/day")
        print(f"Max single-day movement: {summary_df['max_spatial_movement'].max():.2f} km")
        print()
    
    # Duration bins
    print("=== TRAJECTORY DURATION DISTRIBUTION ===")
    duration_bins = [1, 2, 5, 10, 20, 50, 100, np.inf]
    duration_labels = ['1 day', '2-4 days', '5-9 days', '10-19 days', 
                      '20-49 days', '50-99 days', '100+ days']
    
    duration_counts = pd.cut(summary_df['trajectory_length'], 
                           bins=duration_bins, 
                           labels=duration_labels, 
                           right=False).value_counts()
    
    for duration, count in duration_counts.items():
        percentage = (count / len(summary_df)) * 100
        print(f"{duration}: {count} trajectories ({percentage:.1f}%)")
    print()
    
    # Top longest trajectories with trajectory IDs
    print("=== TOP 10 LONGEST TRAJECTORIES ===")
    top_10 = summary_df.head(10)[['trajectory_id', 'trajectory_length', 'trajectory_lifespan', 
                                 'start_date', 'end_date', 'mean_persistence']]
    print(top_10.to_string())
    print()
    
    # Top by persistence
    print("=== TOP 10 MOST PERSISTENT TRAJECTORIES ===")
    top_persistent = summary_df.nlargest(10, 'mean_persistence')[['trajectory_id', 'trajectory_length', 
                                                                 'mean_persistence', 'max_persistence', 
                                                                 'start_date', 'end_date']]
    print(top_persistent.to_string())
    print()
    
    # Most mobile trajectories
    if 'total_spatial_movement' in summary_df.columns:
        print("=== TOP 10 MOST MOBILE TRAJECTORIES ===")
        top_mobile = summary_df.nlargest(10, 'total_spatial_movement')[['trajectory_id', 'trajectory_length',
                                                                       'total_spatial_movement', 
                                                                       'avg_spatial_movement_per_day',
                                                                       'start_date', 'end_date']]
        print(top_mobile.to_string())
        print()
    
    # Matching statistics if available
    if 'matching_stats' in results:
        stats = results['matching_stats']
        print("=== MATCHING STATISTICS ===")
        print(f"Total Wasserstein matches: {stats.get('total_wasserstein_matches', 'N/A')}")
        print(f"Filtered matches kept: {stats.get('filtered_matches', 'N/A')}")
        print(f"Filter rate: {stats.get('filter_rate', 0):.3f}")
        print(f"Avg topological distance: {stats.get('avg_topological', 0):.4f}")
        
        # Handle both possible column names for spatial distance
        spatial_key = 'avg_spatial_km' if 'avg_spatial_km' in stats else 'avg_spatial'
        spatial_val = stats.get(spatial_key, 0)
        spatial_unit = 'km' if 'avg_spatial_km' in stats else 'grid cells'
        print(f"Avg spatial distance: {spatial_val:.4f} {spatial_unit}")
        
        print(f"Avg combined distance: {stats.get('avg_combined', 0):.4f}")
        
        if 'min_spatial_km' in stats:
            print(f"Min spatial distance: {stats.get('min_spatial_km', 0):.4f} km")
            print(f"Max spatial distance: {stats.get('max_spatial_km', 0):.4f} km")
        print()
    
    # Trajectory ID ranges
    print("=== TRAJECTORY ID INFORMATION ===")
    print(f"Original trajectory IDs range: {summary_df['trajectory_id'].min()} - {summary_df['trajectory_id'].max()}")
    print(f"Longest trajectory (rank 1) has original ID: {summary_df.loc[1, 'trajectory_id']}")
    if len(summary_df) >= 10:
        print(f"10th longest trajectory (rank 10) has original ID: {summary_df.loc[10, 'trajectory_id']}")
    print()
    
    # Date range analysis
    print("=== TEMPORAL COVERAGE ===")
    all_start_dates = summary_df['start_date'].tolist()
    all_end_dates = summary_df['end_date'].tolist()
    earliest_start = min(all_start_dates)
    latest_end = max(all_end_dates)
    print(f"Trajectory period spans: {earliest_start} to {latest_end}")
    
    # Monthly distribution of trajectory starts
    start_months = pd.to_datetime(summary_df['start_date']).dt.month.value_counts().sort_index()
    print(f"\nTrajectory starts by month:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, count in start_months.items():
        print(f"  {month_names[month-1]}: {count} trajectories")
    print()
    
    return {
        'trajectory_count': len(summary_df),
        'mean_length': summary_df['trajectory_length'].mean(),
        'mean_lifespan': summary_df['trajectory_lifespan'].mean(),
        'mean_persistence': summary_df['mean_persistence'].mean(),
        'duration_distribution': duration_counts,
        'top_trajectories_by_length': top_10,
        'top_trajectories_by_persistence': top_persistent,
        'longest_trajectory_id': summary_df.loc[1, 'trajectory_id'],
        'temporal_span': (earliest_start, latest_end),
        'monthly_starts': start_months,
        'matching_stats': results.get('matching_stats', {})
    }

# Enhanced function to get specific trajectory recommendations
def get_trajectory_recommendations(results, summary_df, notebook_dir):
    """
    Get recommendations for interesting trajectories to analyze
    """
    print("=== TRAJECTORY RECOMMENDATIONS ===")
    
    recommendations = {}
    
    # Longest trajectory
    longest_id = summary_df.loc[1, 'trajectory_id']
    longest_info = summary_df.loc[1]
    recommendations['longest'] = {
        'trajectory_id': longest_id,
        'rank': 1,
        'reason': 'Longest duration trajectory',
        'stats': f"{longest_info['trajectory_length']} days, {longest_info['trajectory_lifespan']} day lifespan"
    }
    
    # Most persistent
    most_persistent_rank = summary_df['mean_persistence'].idxmax()
    most_persistent_id = summary_df.loc[most_persistent_rank, 'trajectory_id']
    most_persistent_info = summary_df.loc[most_persistent_rank]
    recommendations['most_persistent'] = {
        'trajectory_id': most_persistent_id,
        'rank': most_persistent_rank,
        'reason': 'Highest average persistence',
        'stats': f"Persistence: {most_persistent_info['mean_persistence']:.4f}, {most_persistent_info['trajectory_length']} days"
    }
    
    # Most mobile (if spatial data available)
    if 'total_spatial_movement' in summary_df.columns:
        most_mobile_rank = summary_df['total_spatial_movement'].idxmax()
        most_mobile_id = summary_df.loc[most_mobile_rank, 'trajectory_id']
        most_mobile_info = summary_df.loc[most_mobile_rank]
        recommendations['most_mobile'] = {
            'trajectory_id': most_mobile_id,
            'rank': most_mobile_rank,
            'reason': 'Greatest spatial movement',
            'stats': f"Movement: {most_mobile_info['total_spatial_movement']:.2f} km, {most_mobile_info['trajectory_length']} days"
        }
    
    # Print recommendations
    for category, info in recommendations.items():
        print(f"{category.upper()}:")
        print(f"  Trajectory ID: {info['trajectory_id']} (Rank {info['rank']})")
        print(f"  Reason: {info['reason']}")
        print(f"  Stats: {info['stats']}")
        print(f"  Plot command: plot_trajectory_slider(results, {info['trajectory_id']}, notebook_dir)")
        print()
    
    return recommendations