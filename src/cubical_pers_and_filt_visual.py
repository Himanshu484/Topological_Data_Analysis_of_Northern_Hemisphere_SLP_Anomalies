import matplotlib.pyplot as plt
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ipywidgets import interact, IntSlider

def get_diagram(P, d):
  return P[(P[:, 2]<1e100) & (P[:, 0]==d)][:, 1:3]

def plot_cubical_ripser_diagram(P,dim_list = [0,1]):
  for d in dim_list:
    PD = get_diagram(P, d)
    plt.scatter(*PD.T, label=d, alpha=1)
    plt.axis('equal') 
  plt.legend()
  return PD

def plot_points(points_array, fontsize=12, annotate_top_n=False):
    """
    Plot an array of 2D points with the x=y line

    Parameters:
    points_array : numpy array or list of [x, y] coordinates
    fontsize : int, optional, default=12, controls the size of text elements
    annotate_top_n : int or False, optional, default=False
        If an integer n, annotates the n most persistent points (largest death-birth)
        with their (birth, death) values.
    """
    # Convert to numpy array if not already
    points = np.array(points_array)

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Find min and max across both axes, ensuring 0 is always included
    min_val = min(np.min(x), np.min(y), 0)
    max_val = max(np.max(x), np.max(y), 0)

    # Add a small margin for better visualization
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Plot x=y line
    diagonal = np.linspace(plot_min, plot_max, 100)
    ax.plot(diagonal, diagonal, 'k-', linewidth=1.5, alpha=0.7)

    # Add x=0 and y=0 lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Plot points
    ax.scatter(x, y, color='black', s=100)

    # Annotate top-n most persistent points with (birth, death) labels
    if annotate_top_n:
        persistence = np.abs(y - x)
        top_indices = np.argsort(persistence)[::-1][:int(annotate_top_n)]
        for idx in top_indices:
            ax.annotate(
                f'({x[idx]:.2f}, {y[idx]:.2f})',
                xy=(x[idx], y[idx]),
                xytext=(-8, -12),
                textcoords='offset points',
                fontsize=fontsize * 0.75,
                color='black',
                ha='right',
            )

    # Set equal limits for both axes
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

    # # Create equal tick positions for both axes
    # ticks = np.linspace(plot_min, plot_max, 5)  # 5 ticks from min to max
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)

    # Add labels
    ax.set_xlabel('Birth', fontsize=fontsize)
    ax.set_ylabel('Death', fontsize=fontsize)

    # Increase font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')

    plt.tight_layout()
    return plt

def get_fun(X, show_degrees=False):
    def show_thr(t):
        # Suppress shapely warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
            
            # Clear any existing plots
            plt.close('all')
            
            plt.figure(figsize=(5, 5))
            
            # Create a mask based on the threshold
            mask = X <= t
            
            # X is already a polar matrix (172x172), so we work with it directly
            matrix_size = X.shape[0]  # 172
            center = matrix_size // 2
            
            # Create coordinate arrays for the polar matrix
            y, x = np.ogrid[:matrix_size, :matrix_size]
            x_centered = x - center
            y_centered = y - center
            
            # Convert to polar coordinates
            r = np.sqrt(x_centered**2 + y_centered**2)
            theta = np.arctan2(y_centered, x_centered)
            
            # Convert to lat/lon for plotting
            lon_polar = (theta * 180 / np.pi + 90) % 360
            lat_polar = 90 - (r / center) * 90  # North Pole at center, 0° at edge
            
            # Create North Polar Stereo projection
            ax = plt.axes(projection=ccrs.NorthPolarStereo())
            
            # Set extent for polar view
            ax.set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
            
            # Add only coastlines
            try:
                ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
            except:
                pass
            
            # Use a custom colormap
            colors = ['white', 'gray']
            custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            # Plot the data using pcolormesh with polar coordinates
            cs = ax.pcolormesh(lon_polar, lat_polar, mask.astype(float), 
                               cmap=custom_cmap,
                               transform=ccrs.PlateCarree(),
                               shading='auto',
                               alpha=0.7)
            
            # Add gridlines with conditional degree labels
            ax.gridlines(draw_labels=show_degrees, alpha=0.5)
            
            plt.title(f'North Polar View - Threshold: {t}', size=14, weight='bold', pad=20)
            plt.tight_layout()
            plt.show()
        
    return show_thr

def vis(X, v=0):
    # Limit the slider range to prevent performance issues
    slider = IntSlider(
        value=v, 
        min=int(X.min()), 
        max=int(X.max()), 
        step=max(1, int((X.max() - X.min()) / 100)),  # Limit to ~100 steps
        description="thr"
    )
    interact(get_fun(X), t=slider)