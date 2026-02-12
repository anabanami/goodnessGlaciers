import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import the local tools
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness

BASE_PATH = 'shortcut_to_culled-data' 
DEM_PATH = os.path.join(BASE_PATH, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')


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
        #     'label': 'ASB_ICECAP',
        #     'subset': lambda df: df.iloc[8508112 : 8508112 + 17528].copy()
        # },
        # {
        #     'file': 'UTIG_2010_ICECAP_AIR_BM3.csv', 
        #     'label': 'ROSS_ICECAP',
        #     'subset': lambda df: df[df['trajectory_id'].astype(str).str.contains('IR1HI2_2009033_DMC_JKB1a_WLKX10b', na=False)].copy()
        # },
        {
            'file': 'PRIC_2016_CHA2_AIR_BM3.csv', 
            'label': 'PEL_CHA2',
            # We shift the start index forward to remove the first segment
            # skip the exact number of rows in 'Segment 1'
            'subset': lambda df: df.iloc[410823 : 410823 + 54566].copy(),
            'force_id': 'PRIC_2016_CHA2',
        },
        # {
        #     'file': 'BAS_2010_IMAFI_AIR_BM3.csv', 
        #     'label': 'Moller_Stream'
        # },    # Institute-Möller Ice Stream <<< NOT GREAT       
        # {
        #     'file': 'BAS_2018_Thwaites_AIR_BM3.csv',
        #     'label':'Thwaites_BAS'
        # },    # Thwaites Glacier  <<< NOT GREAT
        # {
        #     'file': 'CRESIS_2009_Thwaites_AIR_BM3.csv',
        #     'label': 'Thwaites_CR'
        # },   # Thwaites Swath <<< NOT GREAT
        # {
        #   'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
        #   'label': 'DML_AniRES'
        #  },   # Dronning Maud Land
    ]

    for item in target_files:
        filename = item['file']
        label = item['label']
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f" Warning: {filename} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(filepath, comment='#')
            
            if 'subset' in item:
                df = item['subset'](df)
            
            if 'force_id' in item:
                df['trajectory_id'] = item['force_id']
            
            # Cleaning Bedmap3 specific nulls (-9999) 
            initial_len = len(df)
            df = df[(df['bedrock_altitude (m)'] != -9999) & 
                    (df['trajectory_id'] != -9999)].copy()
            
            df['trajectory_id'] = df['trajectory_id'].astype(str)
            
            if len(df) > 0:
                print(f"✓ {label} loaded: {len(df)} rows (Filtered {initial_len - len(df)} nulls)")
                all_dfs.append({'name': label, 'data': df})
            else:
                print(f"❌ {label} resulted in 0 rows.")
        except Exception as e:
            print(f"❌ Error loading {label}: {e}")

    return all_dfs


def detect_segments(df, x, y, gap_threshold=2000, min_segment_length=50):
    """
    Detect segments based on gaps in the flight track.
    Matches the logic in bed_analysis_15.py.
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
        if end - start >= min_segment_length:
            segments.append((df.iloc[start:end].copy(), start, end))
    
    return segments


def filter_segments_by_thickness(segments, x, y, dem_path, thickness_threshold=0.20):
    """
    Filter out segments with insufficient ice thickness data.
    Matches the validation in bed_analysis_15.py.
    """
    valid_segments = []
    
    for seg_idx, (segment_df, start, end) in enumerate(segments):
        seg_x = x[start:end]
        seg_y = y[start:end]
        
        # Get surface elevation and calculate thickness
        surface_elevs = extract_rema_elevation(seg_x, seg_y, DEM_PATH)
        bedrock_elevs = segment_df['bedrock_altitude (m)'].values
        ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_elevs)
        
        # Check validity
        thickness_validity = np.sum(~np.isnan(ice_thickness)) / len(ice_thickness)
        
        if thickness_validity < thickness_threshold:
            print(f"   Skipping Segment {seg_idx+1}: Insufficient thickness data ({thickness_validity*100:.1f}% valid)")
            continue
        
        valid_segments.append((segment_df, start, end, seg_idx + 1))  # Keep original segment number
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
    traj_id = df['trajectory_id'].iloc[0] if 'trajectory_id' in df.columns else region_label
    
    print(f"Visualizing Flow Orientation for: {region_label}")
    
    # Project Coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = transformer.transform(df['longitude (degree_east)'].values, 
                                 df['latitude (degree_north)'].values)
    
    # Detect and filter segments (matching bed_analysis_15.py logic)
    print("   Detecting segments...")
    raw_segments = detect_segments(df, x, y)
    print(f"   Found {len(raw_segments)} raw segments")
    
    print("   Filtering by ice thickness...")
    segments = filter_segments_by_thickness(raw_segments, x, y, DEM_PATH)
    
    if not segments:
        print(f"   No valid segments for {region_label}")
        return
    
    print(f"   {len(segments)} valid segments after filtering")
    
    # --- Extract REMA subset for background ---
    print("   Extracting REMA hillshade...")
    bounds = (x.min(), y.min(), x.max(), y.max())
    elevation, extent = extract_rema_subset(DEM_PATH, bounds, buffer_km=10)
    hillshade = make_hillshade(elevation, extent)
    
    # --- Compute flow vectors PER VALID SEGMENT (not full track) ---
    print("   Computing flow vectors...")
    segment_flow = {}
    for segment_df, start, end, orig_seg_num in segments:
        seg_x, seg_y = x[start:end], y[start:end]
        surf = extract_rema_elevation(seg_x, seg_y, DEM_PATH)
        bed = segment_df['bedrock_altitude (m)'].values
        thick = calculate_ice_thickness(surf, bed)
        fx, fy = extract_rema_flow_vector(seg_x, seg_y, DEM_PATH, thick)
        inc = calculate_incidence_angle(seg_x, seg_y, fx, fy)
        segment_flow[orig_seg_num] = {'flow_x': fx, 'flow_y': fy, 'incidence': inc}
    
    # Convert to km for plotting
    x_km, y_km = x / 1000, y / 1000
    extent_km = [e / 1000 for e in extent]
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Background: hillshade
    ax.imshow(hillshade, extent=extent_km, cmap='gray', alpha=0.7, origin='upper')
    
    # Overlay: surface elevation contours (subtle)
    elev_x = np.linspace(extent_km[0], extent_km[1], elevation.shape[1])
    elev_y = np.linspace(extent_km[3], extent_km[2], elevation.shape[0])  # Note: top to bottom
    ax.contour(elev_x, elev_y, elevation, levels=15, colors='white', linewidths=0.3, alpha=0.4)
    
    # Plot track colored by segment orientation
    for segment_df, start, end, orig_seg_num in segments:
        mean_angle = np.nanmean(segment_flow[orig_seg_num]['incidence'])
        color = get_orientation_color(mean_angle)
        
        ax.plot(x_km[start:end], y_km[start:end], color=color, linewidth=3, 
                solid_capstyle='round', zorder=3)
    
    # Flow vectors (subsampled, only for valid segments)
    for segment_df, start, end, orig_seg_num in segments:
        seg_len = end - start
        step = max(1, seg_len // 10)  # ~10 arrows per segment
        
        seg_x_km = x_km[start:end]
        seg_y_km = y_km[start:end]
        fx = segment_flow[orig_seg_num]['flow_x']
        fy = segment_flow[orig_seg_num]['flow_y']
        
        ax.quiver(seg_x_km[::step], seg_y_km[::step], fx[::step], fy[::step],
                  color='royalblue', alpha=0.7, scale=30, width=0.004, 
                  headwidth=4, headlength=5, zorder=2)
    
    # Segment labels (offset perpendicular to track)
    for segment_df, start, end, orig_seg_num in segments:
        mid_idx = (start + end) // 2
        mean_angle = np.nanmean(segment_flow[orig_seg_num]['incidence'])
        
        # Calculate perpendicular offset direction
        # Get local track direction at midpoint
        dx = x_km[min(mid_idx + 5, end - 1)] - x_km[max(mid_idx - 5, start)]
        dy = y_km[min(mid_idx + 5, end - 1)] - y_km[max(mid_idx - 5, start)]
        
        # Perpendicular vector (rotate 90 degrees)
        perp_x, perp_y = -dy, dx
        perp_mag = np.sqrt(perp_x**2 + perp_y**2)
        if perp_mag > 0:
            perp_x, perp_y = perp_x / perp_mag, perp_y / perp_mag
        
        # Offset distance (in km) - scale with plot size
        offset_km = (extent_km[1] - extent_km[0]) * 0.03
        
        # Position label offset from track
        label_x = x_km[mid_idx] + perp_x * offset_km
        label_y = y_km[mid_idx] + perp_y * offset_km
        
        ax.annotate(f'Seg {orig_seg_num}\n({mean_angle:.0f}°)', 
                    xy=(x_km[mid_idx], y_km[mid_idx]),
                    xytext=(label_x, label_y),
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                    zorder=5)
    
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
    
    plt.tight_layout()
    output_file = f'flow_orientation_{region_label}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"   Done! Saved to {output_file}")
    
    # Print summary
    print(f"\n   Segment summary:")
    for segment_df, start, end, orig_seg_num in segments:
        mean_angle = np.nanmean(segment_flow[orig_seg_num]['incidence'])
        length_km = calculate_along_track_distance(x[start:end], y[start:end])[-1]
        orientation = 'Parallel' if mean_angle < 30 else ('Oblique' if mean_angle < 60 else 'Perpendicular')
        print(f"     Seg {orig_seg_num}: {orientation:13s} ({mean_angle:.1f}°) | {length_km:.1f} km")


if __name__=="__main__":
    datasets = load_datasets()
    for ds in datasets:
        main(ds)
