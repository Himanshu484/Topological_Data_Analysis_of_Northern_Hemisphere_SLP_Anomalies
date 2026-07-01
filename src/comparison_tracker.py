import warnings
warnings.filterwarnings('ignore', message='invalid value encountered', category=RuntimeWarning)

"""
Simplified Murray-Simmonds (1991) cyclone detection and tracking,
implemented on the same NCEP SLP data used by the TDA method.

Implemented to support algorithm intercomparison following the framework of:
    Bourdin, S. et al. (2022). Intercomparison of eight tracking algorithms for
    extratropical cyclones. Weather and Climate Dynamics, 3, 923-945.

Detection metrics follow Bourdin et al. (2022) definitions:
    Hit Rate  (HR)  = a / (a + b)       a=hit, b=miss, c=false alarm
    False Alarm Rate (FAR) = c / (a + c)
    Critical Success Index (CSI) = a / (a + b + c)
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import date
from pyproj import Geod
from scipy.ndimage import minimum_filter, uniform_filter

_GEOD = Geod(ellps='WGS84')

def _geodist_km(lat1, lon1, lat2, lon2):
    """Geodesic distance in km between two (lat, lon) points."""
    _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0
import netCDF4 as nc
from tqdm import tqdm


# ── 1. Data loading ──────────────────────────────────────────────────────────

def load_slp_daily(notebook_dir):
    """
    Load NCEP daily SLP.  Returns (dates, lats, lons, slp).
    - lats : (73,)  descending, 90→-90, degrees_north
    - lons : (144,) ascending,  0→357.5, degrees_east
    - slp  : (T, 73, 144) Pascals, NaN where fill
    - dates: list of datetime.date objects
    """
    path = os.path.join(notebook_dir, 'data', 'raw', 'slp.daily.nc')
    ds = nc.Dataset(path)

    lats = np.array(ds.variables['lat'][:])
    lons = np.array(ds.variables['lon'][:])

    time_var = ds.variables['time']
    raw_times = nc.num2date(time_var[:], time_var.units)
    dates = [date(int(t.year), int(t.month), int(t.day)) for t in raw_times]

    slp = np.array(ds.variables['slp'][:], dtype=np.float32)
    fill = ds.variables['slp']._FillValue
    slp[np.abs(slp) > 1e30] = np.nan
    slp[slp == fill] = np.nan
    ds.close()

    return dates, lats, lons, slp


# ── 2. Local-minimum detection (M&S step 1) ──────────────────────────────────

def _local_minima_2d(slp_2d):
    """
    Return boolean mask of 8-connected local SLP minima.
    Longitude axis wraps; latitude axis uses nearest-edge reflection.
    """
    min_filt = minimum_filter(slp_2d, size=3, mode=['reflect', 'wrap'])
    return slp_2d == min_filt


def _depth(slp_2d, r, c, lat, lon, ring_deg=5.0):
    """
    Cyclone depth = mean SLP in a surrounding ring - centre SLP (Pa).
    Positive value means the centre is lower than surroundings.
    """
    clat, clon = lat[r], lon[c]
    dlat = np.abs(lat[:, None] - clat)
    dlon = np.abs(lon[None, :] - clon)
    dlon = np.minimum(dlon, 360.0 - dlon)
    dist_deg = np.sqrt(dlat**2 + dlon**2)
    ring = (dist_deg >= ring_deg - 1.25) & (dist_deg <= ring_deg + 1.25)
    if ring.sum() == 0:
        return 0.0
    return float(np.nanmean(slp_2d[ring]) - slp_2d[r, c])


def detect_centers(slp_2d, lat, lon,
                   depth_threshold_pa=500.0,
                   lat_min=20.0, lat_max=88.0):
    """
    Detect cyclone centres for one daily NH SLP field.

    Parameters
    ----------
    slp_2d            : (nlat, nlon) Pa
    lat, lon          : 1-D coordinate arrays for this slice
    depth_threshold_pa: minimum depth in Pa (default 500 Pa = 5 hPa)
    lat_min/max       : latitudinal band

    Returns
    -------
    list of dicts {lat, lon, slp_min_pa, depth_pa}
    """
    is_min = _local_minima_2d(slp_2d)
    rows, cols = np.where(is_min)

    centres = []
    for r, c in zip(rows, cols):
        clat = float(lat[r])
        if not (lat_min <= clat <= lat_max):
            continue
        if np.isnan(slp_2d[r, c]):
            continue
        d = _depth(slp_2d, r, c, lat, lon)
        if d >= depth_threshold_pa:
            centres.append({
                'lat': clat,
                'lon': float(lon[c]),
                'slp_min_pa': float(slp_2d[r, c]),
                'depth_pa': d,
            })
    return centres


# ── 3. Track linking (M&S step 2) ────────────────────────────────────────────

def link_tracks(detections_by_date, max_dist_km=1000.0, min_track_days=2):
    """
    Greedy nearest-neighbour track linking across consecutive days.

    Parameters
    ----------
    detections_by_date : dict {date_str: [centre_dict, ...]}
    max_dist_km        : maximum allowable linking distance
    min_track_days     : discard tracks shorter than this

    Returns
    -------
    list of tracks; each track is a list of centre_dicts with 'date' added.
    """
    dates_sorted = sorted(detections_by_date.keys())
    active = []      # list of in-progress track lists
    finished = []

    for date_str in dates_sorted:
        today = detections_by_date.get(date_str, [])

        if not active:
            for c in today:
                active.append([dict(c, date=date_str)])
            continue

        tail_lats = np.array([t[-1]['lat'] for t in active])
        tail_lons = np.array([t[-1]['lon'] for t in active])
        today_lats = np.array([c['lat'] for c in today]) if today else np.array([])
        today_lons = np.array([c['lon'] for c in today]) if today else np.array([])

        matched_a, matched_b = set(), set()

        if len(today) > 0:
            dists = np.full((len(active), len(today)), np.inf)
            for i, (tlat, tlon) in enumerate(zip(tail_lats, tail_lons)):
                for j, (clat, clon) in enumerate(zip(today_lats, today_lons)):
                    try:
                        dists[i, j] = _geodist_km(tlat, tlon, clat, clon)
                    except Exception:
                        pass

            for idx in np.argsort(dists.ravel()):
                i, j = divmod(int(idx), len(today))
                if dists[i, j] > max_dist_km:
                    break
                if i in matched_a or j in matched_b:
                    continue
                active[i].append(dict(today[j], date=date_str))
                matched_a.add(i)
                matched_b.add(j)

        new_active = []
        for i, track in enumerate(active):
            if i in matched_a:
                new_active.append(track)
            else:
                finished.append(track)
        active = new_active

        for j, c in enumerate(today):
            if j not in matched_b:
                active.append([dict(c, date=date_str)])

    finished.extend(active)
    return [t for t in finished if len(t) >= min_track_days]


# ── 4. Full M&S pipeline with caching ────────────────────────────────────────

def run_ms_tracker_cached(notebook_dir, start_year=1948, end_year=2023,
                           depth_threshold_pa=500.0, max_dist_km=1000.0,
                           min_track_days=2, lat_min=0.0, lat_max=90.0,
                           force_recompute=False):
    """
    Run the M&S tracker (or load from cache) and return a list of tracks.

    Defaults match the TDA domain exactly: lat 0-90 N, same NCEP dataset.
    depth_threshold_pa=500 Pa (5 hPa) is roughly equivalent to TDA
    persistence_threshold=1000 Pa because persistence measures the full
    SLP-anomaly range across a feature (~2x the depth for a symmetric cyclone).

    Cache stored at:
      data/processed_data/ms_tracker/ms_tracks_{start}-{end}_depth{depth}_dist{dist}_lat{min}-{max}.pkl
    """
    cache_dir = os.path.join(notebook_dir, 'data', 'processed_data', 'ms_tracker')
    os.makedirs(cache_dir, exist_ok=True)

    fname = (f"ms_tracks_{start_year}-{end_year}"
             f"_depth{int(depth_threshold_pa)}"
             f"_dist{int(max_dist_km)}"
             f"_mindays{min_track_days}"
             f"_lat{int(lat_min)}-{int(lat_max)}.pkl")
    cache_path = os.path.join(cache_dir, fname)

    if not force_recompute and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            tracks = pickle.load(f)
        print(f"Loaded {len(tracks)} M&S tracks from cache: {fname}")
        return tracks

    print("Running M&S tracker on NCEP SLP data …")
    dates, lats, lons, slp = load_slp_daily(notebook_dir)

    # Restrict to NH
    nh_mask = lats >= lat_min
    lats_nh = lats[nh_mask]
    slp_nh = slp[:, nh_mask, :]

    detections_by_date = {}
    for t_idx in tqdm(range(len(dates)), desc="Detecting centres"):
        d = dates[t_idx]
        if not (start_year <= d.year <= end_year):
            continue
        slp_day = slp_nh[t_idx]
        if np.all(np.isnan(slp_day)):
            continue
        centres = detect_centers(slp_day, lats_nh, lons,
                                  depth_threshold_pa=depth_threshold_pa,
                                  lat_min=lat_min, lat_max=lat_max)
        detections_by_date[d.isoformat()] = centres

    print(f"Linking tracks (max_dist={max_dist_km} km, min_days={min_track_days}) …")
    tracks = link_tracks(detections_by_date, max_dist_km=max_dist_km,
                          min_track_days=min_track_days)

    with open(cache_path, 'wb') as f:
        pickle.dump(tracks, f)
    print(f"M&S tracker: {len(tracks)} tracks — saved to {fname}")
    return tracks


# ── 5. TDA track extraction ───────────────────────────────────────────────────

def extract_tda_tracks(results):
    """
    Convert TDA trajectory results (from analyze_persistence_trajectories_cached)
    into a list of tracks matching the M&S format.

    Each track: list of dicts {date, lat, lon, persistence}.
    """
    tracks = []
    for traj in results['trajectories']:
        track = []
        for step in traj:
            lat = step.get('lat')
            lon = step.get('lon')
            d = step.get('date')
            pers = step.get('persistence', 0.0)
            if lat is None or lon is None or d is None:
                continue
            if not (0 <= lat <= 90):
                continue
            # Normalise date to ISO string so it matches M&S date keys
            d = d.isoformat() if hasattr(d, 'isoformat') else str(d)
            track.append({'date': d, 'lat': lat, 'lon': lon,
                          'persistence': pers})
        if track:
            tracks.append(track)
    return tracks


# ── 6. Matching & metrics (Bourdin et al. 2022) ───────────────────────────────

def _match_two_sets(centres_a, centres_b, threshold_km=500.0):
    """
    Greedy spatial matching of two (lat, lon) lists.
    Returns (matched_a, matched_b, unmatched_a, unmatched_b) as index sets.
    """
    if not centres_a or not centres_b:
        return set(), set(), set(range(len(centres_a))), set(range(len(centres_b)))

    dists = np.full((len(centres_a), len(centres_b)), np.inf)
    for i, (la, loa) in enumerate(centres_a):
        for j, (lb, lob) in enumerate(centres_b):
            try:
                dists[i, j] = _geodist_km(la, loa, lb, lob)
            except Exception:
                pass

    ma, mb = set(), set()
    for idx in np.argsort(dists.ravel()):
        i, j = divmod(int(idx), len(centres_b))
        if dists[i, j] > threshold_km:
            break
        if i not in ma and j not in mb:
            ma.add(i); mb.add(j)
    return ma, mb, set(range(len(centres_a))) - ma, set(range(len(centres_b))) - mb


def compute_daily_match_stats(ms_tracks, tda_tracks, threshold_km=500.0):
    """
    Per-date detection matching.  M&S is the reference.

    Returns a DataFrame with columns:
        date, n_ms, n_tda, hits, misses, false_alarms, hit_rate, far, csi
    """
    def _to_str(d):
        return d.isoformat() if hasattr(d, 'isoformat') else str(d)

    ms_by_date = {}
    for track in ms_tracks:
        for s in track:
            ms_by_date.setdefault(_to_str(s['date']), []).append((s['lat'], s['lon']))

    tda_by_date = {}
    for track in tda_tracks:
        for s in track:
            tda_by_date.setdefault(_to_str(s['date']), []).append((s['lat'], s['lon']))

    all_dates = sorted(set(ms_by_date) | set(tda_by_date))
    rows = []
    for d in all_dates:
        ms_c  = ms_by_date.get(d, [])
        tda_c = tda_by_date.get(d, [])
        ma, mb, ua, ub = _match_two_sets(ms_c, tda_c, threshold_km)
        a, b, c = len(ma), len(ua), len(ub)
        rows.append({
            'date': pd.Timestamp(d),
            'n_ms': len(ms_c), 'n_tda': len(tda_c),
            'hits': a, 'misses': b, 'false_alarms': c,
            'hit_rate': a / (a + b) if (a + b) else np.nan,
            'far':      c / (a + c) if (a + c) else np.nan,
            'csi':      a / (a + b + c) if (a + b + c) else np.nan,
        })
    return pd.DataFrame(rows)


# ── 7. Summary DataFrames ─────────────────────────────────────────────────────

def tracks_to_summary(tracks, label='ms'):
    """
    Convert a list of tracks to a flat summary DataFrame.
    Columns: method, track_id, start_date, end_date, lifetime_days,
             genesis_lat, genesis_lon, mean_lat, total_displacement_km
    """
    rows = []
    for i, track in enumerate(tracks):
        if not track:
            continue
        dates_ts = [pd.Timestamp(s['date']) for s in track]
        lats = [s['lat'] for s in track]
        lons = [s['lon'] for s in track]

        km = 0.0
        for k in range(1, len(track)):
            try:
                km += _geodist_km(lats[k-1], lons[k-1], lats[k], lons[k])
            except Exception:
                pass

        rows.append({
            'method': label,
            'track_id': i,
            'start_date': dates_ts[0],
            'end_date': dates_ts[-1],
            'lifetime_days': len(track),
            'genesis_lat': lats[0],
            'genesis_lon': lons[0],
            'lysis_lat': lats[-1],
            'lysis_lon': lons[-1],
            'mean_lat': float(np.mean(lats)),
            'total_displacement_km': km,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        df['month'] = df['start_date'].dt.month
        df['year']  = df['start_date'].dt.year
    return df


def combined_summary_table(ms_summary, tda_summary, match_stats):
    """
    Print a paper-ready comparison table.
    """
    rows = {}
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    for label, df in [('M&S (1991)', ms_summary), ('TDA (ours)', tda_summary)]:
        rows[label] = {
            'Total tracks': len(df),
            'Mean lifetime (days)': f"{df['lifetime_days'].mean():.1f}",
            'Median lifetime (days)': f"{df['lifetime_days'].median():.1f}",
            'Mean genesis lat (°N)': f"{df['genesis_lat'].mean():.1f}",
            'Mean displacement (km)': f"{df['total_displacement_km'].mean():.0f}",
            'Tracks ≥ 3 days (%)': f"{100*(df['lifetime_days']>=3).mean():.1f}",
        }

    ms_ref = match_stats.dropna(subset=['hit_rate','far','csi'])
    rows['Match statistics'] = {
        'Mean Hit Rate (HR)':         f"{ms_ref['hit_rate'].mean():.3f}",
        'Mean False Alarm Rate (FAR)':f"{ms_ref['far'].mean():.3f}",
        'Mean CSI':                   f"{ms_ref['csi'].mean():.3f}",
    }

    print("=" * 60)
    print("ALGORITHM INTERCOMPARISON SUMMARY")
    print("=" * 60)
    for section, vals in rows.items():
        print(f"\n{section}")
        print("-" * 40)
        for k, v in vals.items():
            print(f"  {k:<35} {v}")
    print("=" * 60)


# ── 8. Plotting ───────────────────────────────────────────────────────────────

def plot_monthly_frequency(ms_summary, tda_summary, figsize=(10, 5)):
    """
    Bar chart of mean monthly cyclone count, M&S vs TDA side by side.
    """
    import matplotlib.pyplot as plt

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    x = np.arange(1, 13)

    def monthly_avg(df):
        return (df.groupby(['year','month']).size()
                  .reset_index(name='count')
                  .groupby('month')['count'].mean()
                  .reindex(x, fill_value=0))

    ms_m  = monthly_avg(ms_summary)
    tda_m = monthly_avg(tda_summary)

    w = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w/2, ms_m,  w, color='steelblue', label='M&S (1991)', alpha=0.85)
    ax.bar(x + w/2, tda_m, w, color='tomato',    label='TDA (ours)', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Mean monthly count', fontsize=13)
    ax.set_title('Monthly cyclone frequency: M&S vs TDA', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_lifetime_distribution(ms_summary, tda_summary, max_days=20, figsize=(8, 5)):
    """
    Overlaid histograms of track lifetime (days).
    """
    import matplotlib.pyplot as plt

    bins = np.arange(1, max_days + 2) - 0.5
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(ms_summary['lifetime_days'].clip(upper=max_days), bins=bins,
            density=True, histtype='step', lw=2, color='steelblue', label='M&S (1991)')
    ax.hist(tda_summary['lifetime_days'].clip(upper=max_days), bins=bins,
            density=True, histtype='step', lw=2, color='tomato', label='TDA (ours)')
    ax.set_xlabel('Lifetime (days)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Cyclone track lifetime distribution', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spatial_density(ms_summary, tda_summary,
                          lat_bins=18, lon_bins=36, figsize=(14, 5)):
    """
    Side-by-side genesis density maps (cyclones per grid box per year).
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    nyears = ms_summary['year'].nunique()

    lat_edges = np.linspace(20, 90, lat_bins + 1)
    lon_edges = np.linspace(0, 360, lon_bins + 1)

    def density(df):
        h, _, _ = np.histogram2d(df['genesis_lat'], df['genesis_lon'],
                                  bins=[lat_edges, lon_edges])
        return h / nyears

    ms_d  = density(ms_summary)
    tda_d = density(tda_summary)
    vmax  = max(ms_d.max(), tda_d.max())

    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    LON, LAT = np.meshgrid(lon_c, lat_c)

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              subplot_kw={'projection': ccrs.NorthPolarStereo()})
    titles = ['M&S (1991) genesis density', 'TDA (ours) genesis density']
    for ax, D, title in zip(axes, [ms_d, tda_d], titles):
        ax.set_extent([0, 360, 20, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, lw=0.6)
        cs = ax.pcolormesh(LON, LAT, D,
                           transform=ccrs.PlateCarree(),
                           cmap='YlOrRd', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12)
        plt.colorbar(cs, ax=ax, shrink=0.7, label='tracks / year')
    plt.suptitle('Cyclone genesis density (20–90°N)', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_match_metrics_monthly(match_stats, figsize=(10, 4)):
    """
    Monthly mean HR, FAR, CSI as grouped bar chart.
    """
    import matplotlib.pyplot as plt

    ms = match_stats.copy()
    ms['month'] = ms['date'].dt.month
    monthly = ms.groupby('month')[['hit_rate', 'far', 'csi']].mean()

    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']
    x = np.arange(1, 13)
    w = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w,   monthly['hit_rate'].reindex(x, fill_value=0), w,
           color='steelblue', label='Hit Rate (HR)', alpha=0.85)
    ax.bar(x,       monthly['csi'].reindex(x, fill_value=0), w,
           color='seagreen', label='CSI', alpha=0.85)
    ax.bar(x + w,   monthly['far'].reindex(x, fill_value=0), w,
           color='tomato', label='False Alarm Rate (FAR)', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detection match metrics vs M&S (1991) — Bourdin et al. (2022) framework',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig
