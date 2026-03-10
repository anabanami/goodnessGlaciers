import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import the local tools
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, get_rema_cache

BASE_PATH = 'shortcut_to_culled-data'
DEM_PATH = os.path.join(BASE_PATH, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

# Module-level transformer (created once, reused)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


def extract_rema_subset(dem_path, bounds, buffer_km=20):
    """
    Extract a subset of REMA around the track bounds.
    Returns the elevation array and its extent for plotting.
    """
    buffer_m = buffer_km * 1000
    minx, miny, maxx, maxy = bounds
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m
    
    with rasterio.open(dem_path) as src:
        # Get window from bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        # Read the subset
        data = src.read(1, window=window)
        
        # Replace nodata
        data = np.where(data == src.nodata, np.nan, data)
        
        # Get the actual bounds of what we read
        window_transform = src.window_transform(window)
        height, width = data.shape
        
        # Calculate extent for imshow [left, right, bottom, top]
        left = window_transform.c
        top = window_transform.f
        right = left + width * window_transform.a
        bottom = top + height * window_transform.e  # e is negative
        
        extent = [left, right, bottom, top]
    
    return data, extent


def make_hillshade(elevation, extent):
    """Create a hillshade from elevation data."""
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Calculate pixel size from extent
    dx = (extent[1] - extent[0]) / elevation.shape[1]
    
    hillshade = ls.hillshade(elevation, vert_exag=2, dx=dx, dy=dx)
    return hillshade


def calculate_incidence_angle(flight_x, flight_y, flow_x, flow_y):
    """
    Calculates angle between flight path tangent and ice flow vector.
    Returns angle in degrees [0, 90].
    """
    # Calculate Flight Direction (Tangent)
    dt_x = np.gradient(flight_x)
    dt_y = np.gradient(flight_y)
    
    # Normalize flight vectors
    mag_t = np.sqrt(dt_x**2 + dt_y**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_x = dt_x / mag_t
        t_y = dt_y / mag_t
    
    # Calculate Dot Product with Flow Vector
    dot_prod = t_x * flow_x + t_y * flow_y
    
    # Arccos to get angle
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    angle_rad = np.arccos(dot_prod)
    angle_deg = np.degrees(angle_rad)
    
    # Fold to [0, 90] (we don't care about upstream vs downstream)
    angle_deg = np.minimum(angle_deg, 180 - angle_deg)
    
    return angle_deg


def calculate_along_track_distance(x, y):
    """Calculate cumulative along-track distance in km."""
    dx = np.diff(x)
    dy = np.diff(y)
    segment_distances = np.sqrt(dx**2 + dy**2)
    dist_m = np.concatenate([[0], np.cumsum(segment_distances)])
    return dist_m / 1000  # Convert to km


def load_datasets():
    """Returns a list of dictionaries: {'name': label, 'data': df}"""
    base_path = BASE_PATH
    all_dfs = []
    
    target_files = [
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'TEST_Aurora_SB',
        #     'subset': lambda df: df.iloc[8508112:8508112+17528].copy(),
        # },
        
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv', 
        #     'label': 'ROSS_ICECAP',
        #     'subset': lambda df: df[df['trajectory_id'].astype(str).str.contains('IR1HI2_2009033_DMC_JKB1a_WLKX10b', na=False)].copy()
        # },

        # {
        #     'file': 'PRIC_2016_CHA2_AIR_BM3.csv', 
        #     'label': 'PEL_CHA2',
        #     # skip the exact number of rows in 'Segment 1'
        #     'subset': lambda df: df.iloc[410823 : 410823 + 54566].copy(),
        #     'force_id': 'PRIC_2016_CHA2',
        # },

        # {
        #     'file': 'BAS_2010_IMAFI_AIR_BM3.csv', 
        #     'label': 'Moller_Stream'
        # },    # Institute-Möller Ice Stream
        
        # {
        #     'file': 'BAS_2018_Thwaites_AIR_BM3.csv',
        #     'label':'Thwaites_BAS'
        # },    # Thwaites Glacier
        
        # {
        #     'file': 'CRESIS_2009_Thwaites_AIR_BM3.csv',
        #     'label': 'Thwaites_CR'
        # },   # Thwaites Swath
        
        # {
        #   'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
        #   'label': 'DML_AniRES'
        #  },   # Dronning Maud Land

        ##############################################################################
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig4_Aurora_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -71.0,
        #         'lon_min': 105.0, 'lon_max': 125.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2F_Resolution_SH',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -73.0,
        #         'lon_min': 135.0, 'lon_max': 150.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2G_Highland_A',
        #     'subset': lambda df, _r={
        #         'lat_min': -76.0, 'lat_max': -73.0,
        #         'lon_min': 118.0, 'lon_max': 132.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
        #     'label': 'ASB_ICECAP_2010_Fig2H_Golicyna_SH',
        #     'subset': lambda df, _r={
        #         'lat_min': -75.0, 'lat_max': -72.0,
        #         'lon_min': 103.0, 'lon_max': 117.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        ##############################################################################
        # {
        #     'file': 'BAS_2012_ICEGRAV_AIR_BM3.csv',
        #     'label': 'Rec_Catch_Fig2D_Recovery_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -83.5, 'lat_max': -80.5,
        #         'lon_min': -35.0, 'lon_max': -15.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

        # {
        #     'file': 'NASA_2018_ICEBRIDGE_AIR_BM3.csv',
        #     'label': '2018_Rec_SB_Fig2D_Recovery_SB',
        #     'subset': lambda df, _r={
        #         'lat_min': -83.5, 'lat_max': -80.5,
        #         'lon_min': -35.0, 'lon_max': -15.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },
        ##############################################################################
        {
            'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
            'label': 'POLARGAP_2015_Fig1_Pensacola_Pole',
            'subset': lambda df, _r={
                'lat_min': -88.0, 'lat_max': -82.0,
                'lon_min': -60.0, 'lon_max': -20.0,
            }: df[
                (df['latitude (degree_north)'] >= _r['lat_min']) &
                (df['latitude (degree_north)'] <= _r['lat_max']) &
                (df['longitude (degree_east)']  >= _r['lon_min']) &
                (df['longitude (degree_east)']  <= _r['lon_max'])
            ].copy(),
        },

        # {
        #     'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        #     'label': 'POLARGAP_2015_Fig2C_Hercules_Dome',
        #     'subset': lambda df, _r={
        #         'lat_min': -87.5, 'lat_max': -85.0,
        #         'lon_min': -120.0, 'lon_max': -100.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },
    ]

    file_cache = {}

    for item in target_files:
        filename = item['file']
        label = item['label']
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f" Warning: {filename} not found. Skipping.")
            continue

        try:
            if filepath not in file_cache:
                print(f"  Reading {filename}...")
                file_cache[filepath] = pd.read_csv(filepath, comment='#', low_memory=False)
            df = file_cache[filepath].copy()
            
            # Apply subset if specified (for manual row slicing)
            if 'subset' in item:
                df = item['subset'](df)
            
            if 'force_id' in item:
                df['trajectory_id'] = item['force_id']
            
            # Cleaning Bedmap3 specific nulls (-9999) 
            has_valid_bed = df['bedrock_altitude (m)'] != -9999
            has_valid_traj = (df['trajectory_id'] != -9999) | ('force_id' in item)
            df = df[has_valid_bed & has_valid_traj].copy()
            
            df['trajectory_id'] = df['trajectory_id'].astype(str)
            
            # Filter to specific trajectories if specified
            if 'trajectories' in item:
                traj_list = [str(t) for t in item['trajectories']]
                df_filtered = df[df['trajectory_id'].isin(traj_list)]
                
                # Create separate dataset for each trajectory
                for traj_id in traj_list:
                    traj_df = df_filtered[df_filtered['trajectory_id'] == traj_id].copy()
                    if len(traj_df) > 0:
                        print(f"✓ {label}/{traj_id} loaded: {len(traj_df)} rows")
                        all_dfs.append({'name': f"{label}_{traj_id}", 'data': traj_df})
                    else:
                        print(f"✗ {label}/{traj_id}: No data found")
            else:
                # No trajectory filter - use entire file (or subset)
                if len(df) > 0:
                    print(f"✓ {label} loaded: {len(df)} rows")
                    all_dfs.append({'name': label, 'data': df})
                else:
                    print(f"✗ {label} resulted in 0 rows.")
                
        except Exception as e:
            print(f"✗ Error loading {label}: {e}")

    del file_cache
    return all_dfs


def detect_segments(df, x, y, gap_threshold=2000, min_segment_length=50, min_segment_km=10):
    """
    Detect segments based on gaps in the flight track.
    Matches the logic in bed_analysis.py.
    Returns list of tuples: (segment_df, start_idx, end_idx)
    """
    # Calculate distances between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Find gaps
    gap_indices = np.where(distances > gap_threshold)[0]
    
    # Build segment boundaries
    # Gap indices mark the END of a segment (last point before gap)
    # and the START of the next segment is gap_index + 1
    split_points = [0]
    for gap_idx in gap_indices:
        split_points.append(gap_idx + 1)  # End of current segment
        split_points.append(gap_idx + 1)  # Start of next segment
    split_points.append(len(x))
    
    # Pair up start/end points
    segments = []
    for i in range(0, len(split_points) - 1, 2):
        start = split_points[i]
        end = split_points[i + 1]
        length_km = (np.sqrt((x[end-1] - x[start])**2 + (y[end-1] - y[start])**2)) / 1000
        if end - start >= min_segment_length and length_km >= min_segment_km:
            segments.append((df.iloc[start:end].copy(), start, end))
    
    return segments


def filter_segments_by_thickness(segments, x, y, dem_path, cache, thickness_threshold=0.20):
    """
    Filter out segments with insufficient ice thickness data.
    Returns valid segments along with pre-computed surface elevations and thickness.
    """
    valid_segments = []

    for seg_idx, (segment_df, start, end) in enumerate(segments):
        seg_x = x[start:end]
        seg_y = y[start:end]

        # Get surface elevation and calculate thickness
        surface_elevs = extract_rema_elevation(seg_x, seg_y, dem_path, cache)
        bedrock_elevs = segment_df['bedrock_altitude (m)'].values
        ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_elevs)

        # Check validity
        thickness_validity = np.sum(~np.isnan(ice_thickness)) / len(ice_thickness)

        if thickness_validity < thickness_threshold:
            print(f"   Skipping Segment {seg_idx+1}: Insufficient thickness data ({thickness_validity*100:.1f}% valid)")
            continue

        # Return pre-computed values to avoid duplicate work
        valid_segments.append({
            'df': segment_df,
            'start': start,
            'end': end,
            'seg_num': seg_idx + 1,
            'surface_elevs': surface_elevs,
            'ice_thickness': ice_thickness,
        })
        print(f"   Segment {seg_idx+1}: Valid ({thickness_validity*100:.1f}% thickness data)")

    return valid_segments


def get_orientation_color(angle):
    """Return color based on orientation class."""
    if angle < 30:
        return '#2E7D32'  # Dark green
    elif angle < 60:
        return '#F9A825'  # Amber
    else:
        return '#C62828'  # Dark red


def main(dataset_dict):
    region_label = dataset_dict['name']
    df = dataset_dict['data']

    print(f"Visualizing Flow Orientation for: {region_label}")

    # Get the REMA cache (loads DEM once, reused for all operations)
    cache = get_rema_cache()
    cache.load(DEM_PATH)

    # Project Coordinates (using module-level transformer)
    x, y = _TRANSFORMER.transform(df['longitude (degree_east)'].values,
                                  df['latitude (degree_north)'].values)

    # Detect segments PER TRAJECTORY (matching bed_analysis.py logic)
    # This ensures trajectory boundaries are respected, not just spatial gaps
    print("   Detecting segments per trajectory...")
    raw_segments = []

    for traj_id in df['trajectory_id'].unique():
        # Get indices for this trajectory in the original dataframe
        traj_mask = df['trajectory_id'] == traj_id
        traj_indices = np.where(traj_mask)[0]

        if len(traj_indices) < 20:
            continue

        # Extract trajectory data
        traj_df = df[traj_mask].copy()
        traj_x = x[traj_mask]
        traj_y = y[traj_mask]

        # Detect segments within this trajectory
        traj_segments = detect_segments(traj_df, traj_x, traj_y)

        # Convert local indices back to global indices
        for seg_df, local_start, local_end in traj_segments:
            global_start = traj_indices[local_start]
            global_end = traj_indices[local_end - 1] + 1  # end is exclusive
            raw_segments.append((seg_df, global_start, global_end))

    print(f"   Found {len(raw_segments)} raw segments across {df['trajectory_id'].nunique()} trajectories")

    print("   Filtering by ice thickness...")
    segments = filter_segments_by_thickness(raw_segments, x, y, DEM_PATH, cache)

    if not segments:
        print(f"   No valid segments for {region_label}")
        return

    print(f"   {len(segments)} valid segments after filtering")

    # --- Extract REMA subset for background ---
    print("   Extracting REMA hillshade...")
    bounds = (x.min(), y.min(), x.max(), y.max())
    elevation, extent = extract_rema_subset(DEM_PATH, bounds, buffer_km=10)
    hillshade = make_hillshade(elevation, extent)

    # --- Compute flow vectors using pre-computed thickness (no duplicate work) ---
    print("   Computing flow vectors...")
    segment_flow = {}
    for seg in segments:
        seg_x, seg_y = x[seg['start']:seg['end']], y[seg['start']:seg['end']]
        fx, fy = extract_rema_flow_vector(seg_x, seg_y, DEM_PATH, seg['ice_thickness'], cache)
        inc = calculate_incidence_angle(seg_x, seg_y, fx, fy)
        segment_flow[seg['seg_num']] = {'flow_x': fx, 'flow_y': fy, 'incidence': inc}
    
    # Convert to km for plotting
    x_km, y_km = x / 1000, y / 1000
    extent_km = [e / 1000 for e in extent]
    
    # --- Plotting ---
    # Create figure with main plot and side panel for segment list
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    
    # Background: hillshade
    ax.imshow(hillshade, extent=extent_km, cmap='gray', alpha=0.7, origin='upper')
    
    # Overlay: surface elevation contours (subtle)
    elev_x = np.linspace(extent_km[0], extent_km[1], elevation.shape[1])
    elev_y = np.linspace(extent_km[3], extent_km[2], elevation.shape[0])  # Note: top to bottom
    ax.contour(elev_x, elev_y, elevation, levels=15, colors='white', linewidths=0.3, alpha=0.4)
    
    # Plot track colored by segment orientation
    for seg in segments:
        mean_angle = np.nanmean(segment_flow[seg['seg_num']]['incidence'])
        color = get_orientation_color(mean_angle)

        ax.plot(x_km[seg['start']:seg['end']], y_km[seg['start']:seg['end']], color=color, linewidth=3,
                solid_capstyle='round', zorder=3)
    
    # Flow vectors (subsampled, only for valid segments)
    for seg in segments:
        seg_len = seg['end'] - seg['start']
        step = max(1, seg_len // 10)  # ~10 arrows per segment

        seg_x_km = x_km[seg['start']:seg['end']]
        seg_y_km = y_km[seg['start']:seg['end']]
        fx = segment_flow[seg['seg_num']]['flow_x']
        fy = segment_flow[seg['seg_num']]['flow_y']

        ax.quiver(seg_x_km[::step], seg_y_km[::step], fx[::step], fy[::step],
                  color='royalblue', alpha=0.7, scale=30, width=0.004,
                  headwidth=4, headlength=5, zorder=2)

    # Segment number labels on track
    for seg in segments:
        mid_idx = (seg['start'] + seg['end']) // 2
        ax.text(x_km[mid_idx], y_km[mid_idx], str(seg['seg_num']),
                fontsize=6, fontweight='bold', ha='center', va='center',
                color='white', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', alpha=0.7, linewidth=0))

    # Build segment data for side panel
    segment_info = []
    for seg in segments:
        mean_angle = np.nanmean(segment_flow[seg['seg_num']]['incidence'])
        color = get_orientation_color(mean_angle)
        length_km = calculate_along_track_distance(x[seg['start']:seg['end']], y[seg['start']:seg['end']])[-1]
        segment_info.append({
            'num': seg['seg_num'],
            'angle': mean_angle,
            'color': color,
            'length': length_km,
        })
    
    # Styling
    ax.set_xlabel('Easting (km, EPSG:3031)', fontsize=11)
    ax.set_ylabel('Northing (km, EPSG:3031)', fontsize=11)
    ax.set_title(f'Ice Flow Orientation: {region_label}\n{traj_id}', fontsize=12, pad=10)
    ax.set_aspect('equal')
    
    # --- Find emptiest corner for legend ---
    x_mid = (x_km.min() + x_km.max()) / 2
    y_mid = (y_km.min() + y_km.max()) / 2
    
    # Count points in each quadrant
    corners = {
        'upper left':  np.sum((x_km < x_mid) & (y_km > y_mid)),
        'upper right': np.sum((x_km > x_mid) & (y_km > y_mid)),
        'lower left':  np.sum((x_km < x_mid) & (y_km < y_mid)),
        'lower right': np.sum((x_km > x_mid) & (y_km < y_mid)),
    }
    
    # Sort corners by emptiness (least points first)
    sorted_corners = sorted(corners.items(), key=lambda x: x[1])
    legend_loc = sorted_corners[0][0]
    
    # Legend
    legend_elements = [
        Patch(facecolor='#2E7D32', label='Parallel (<30°)'),
        Patch(facecolor='#F9A825', label='Oblique (30–60°)'),
        Patch(facecolor='#C62828', label='Perpendicular (>60°)'),
        Line2D([0], [0], color='royalblue', marker='>', linestyle='', markersize=8, label='Ice flow'),
    ]
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=9, framealpha=0.95)

    # --- Side panel: Segment list ---
    ax_list = fig.add_subplot(gs[1])
    ax_list.set_xlim(0, 1)
    ax_list.set_ylim(0, 1)
    ax_list.axis('off')

    # Sort segments by number for display
    segment_info_sorted = sorted(segment_info, key=lambda s: s['num'])

    # Calculate layout with multiple columns
    n_segs = len(segment_info_sorted)
    line_height = 0.017
    max_lines_per_col = int(0.88 / line_height)
    n_cols = max(1, (n_segs + max_lines_per_col - 1) // max_lines_per_col)  # Ceiling division
    n_cols = min(n_cols, 3)  # Cap at 3 columns max

    col_width = 1.0 / n_cols

    # Header
    ax_list.text(0.5, 0.98, 'Segment List', fontsize=11, fontweight='bold',
                 ha='center', va='top', transform=ax_list.transAxes)

    # Column headers
    for col in range(n_cols):
        col_x = col * col_width
        ax_list.text(col_x + 0.06 * col_width + 0.02, 0.94, '#', fontsize=7, fontweight='bold', ha='left', va='top')
        ax_list.text(col_x + col_width * 0.5, 0.94, 'Angle', fontsize=7, fontweight='bold', ha='center', va='top')
        ax_list.text(col_x + col_width * 0.88, 0.94, 'km', fontsize=7, fontweight='bold', ha='center', va='top')

    # Draw horizontal separator line
    ax_list.axhline(y=0.92, xmin=0.01, xmax=0.99, color='gray', linewidth=0.5)

    # Draw vertical separator lines between columns
    for col in range(1, n_cols):
        ax_list.axvline(x=col * col_width, ymin=0.02, ymax=0.92, color='gray', linewidth=0.5)

    # Draw entries in columns
    for i, seg in enumerate(segment_info_sorted):
        col = i // max_lines_per_col
        if col >= n_cols:
            break
        row = i % max_lines_per_col

        col_x = col * col_width
        y_pos = 0.90 - row * line_height

        # Colored marker
        ax_list.plot(col_x + 0.02, y_pos - 0.004, 's', color=seg['color'], markersize=5)
        # Segment number
        ax_list.text(col_x + 0.06, y_pos, f"{seg['num']}", fontsize=6, ha='left', va='top')
        # Angle
        ax_list.text(col_x + col_width * 0.5, y_pos, f"{seg['angle']:.0f}°", fontsize=6, ha='center', va='top')
        # Length
        ax_list.text(col_x + col_width * 0.88, y_pos, f"{seg['length']:.1f}", fontsize=6, ha='center', va='top')

    plt.tight_layout()
    output_file = f'flow_orientation_{region_label}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Done! Saved to {output_file}")
    
    # Print summary
    print(f"\n   Segment summary:")
    for seg in segments:
        mean_angle = np.nanmean(segment_flow[seg['seg_num']]['incidence'])
        length_km = calculate_along_track_distance(x[seg['start']:seg['end']], y[seg['start']:seg['end']])[-1]
        orientation = 'Parallel' if mean_angle < 30 else ('Oblique' if mean_angle < 60 else 'Perpendicular')
        print(f"     Seg {seg['seg_num']}: {orientation:13s} ({mean_angle:.1f}°) | {length_km:.1f} km")


if __name__=="__main__":
    datasets = load_datasets()
    for ds in datasets:
        main(ds)
