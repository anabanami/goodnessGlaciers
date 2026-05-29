import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.gridspec import GridSpec
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Import the local tools
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, get_rema_cache, MEaSUREs_comparison
from config import Tee
from loading import load_datasets

BASE_PATH = 'all_data/'
DEM_PATH = os.path.join(BASE_PATH, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

# Module-level transformer (created once, reused)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


# Output configuration - creates folders in same directory as this script
OUTPUT_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'flow_plots/',
    )

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
    output_file = os.path.join(OUTPUT_BASE_PATH, f'flow_orientation_{region_label}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"   Done! Saved to {output_file}")
    
    # Print summary
    print(f"\n   Segment summary:")
    for seg in segments:
        mean_angle = np.nanmean(segment_flow[seg['seg_num']]['incidence'])
        length_km = calculate_along_track_distance(x[seg['start']:seg['end']], y[seg['start']:seg['end']])[-1]
        orientation = 'Parallel' if mean_angle < 30 else ('Oblique' if mean_angle < 60 else 'Perpendicular')
        print(f"     Seg {seg['seg_num']}: {orientation:13s} ({mean_angle:.1f}°) | {length_km:.1f} km")

    # Angular coverage diagnostic
    all_angles = np.concatenate([v['incidence'] for v in segment_flow.values()])
    all_angles = all_angles[~np.isnan(all_angles)]
    if len(all_angles) > 0:
        bins = np.arange(0, 91, 10)
        counts, _ = np.histogram(all_angles, bins=bins)
        total = len(all_angles)
        occupied = np.sum(counts > 0)
        print(f"\n   Angular coverage: {occupied}/{len(counts)} bins occupied | "
              f"range {np.min(all_angles):.0f}°–{np.max(all_angles):.0f}°, std={np.std(all_angles):.1f}°")
        for i in range(len(counts)):
            bar = '#' * int(40 * counts[i] / max(counts.max(), 1))
            print(f"     {bins[i]:2.0f}°–{bins[i+1]:2.0f}°: {counts[i]:5d} ({100*counts[i]/total:5.1f}%) {bar}")


def plot_flow_confidence(dataset_dict):
    """
    Plots flight tracks colored by angular difference between
    REMA-derived and MEaSUREs flow vectors (0-90 degrees).
    0 = agreement, 90 = perpendicular disagreement.
    """
    region_label = dataset_dict['name']
    df = dataset_dict['data']

    print(f"Flow confidence map for: {region_label}")

    cache = get_rema_cache()
    cache.load(DEM_PATH)

    # Project coordinates
    x, y = _TRANSFORMER.transform(df['longitude (degree_east)'].values,
                                  df['latitude (degree_north)'].values)

    # Detect and filter segments (reuse existing logic)
    print("   Detecting segments per trajectory...")
    raw_segments = []
    for traj_id in df['trajectory_id'].unique():
        traj_mask = df['trajectory_id'] == traj_id
        traj_indices = np.where(traj_mask)[0]
        if len(traj_indices) < 20:
            continue
        traj_df = df[traj_mask].copy()
        traj_x = x[traj_mask]
        traj_y = y[traj_mask]
        traj_segments = detect_segments(traj_df, traj_x, traj_y)
        for seg_df, local_start, local_end in traj_segments:
            global_start = traj_indices[local_start]
            global_end = traj_indices[local_end - 1] + 1
            raw_segments.append((seg_df, global_start, global_end))

    print(f"   Found {len(raw_segments)} raw segments")

    print("   Filtering by ice thickness...")
    segments = filter_segments_by_thickness(raw_segments, x, y, DEM_PATH, cache)
    if not segments:
        print(f"   No valid segments for {region_label}")
        return
    print(f"   {len(segments)} valid segments after filtering")

    # Extract REMA hillshade
    print("   Extracting REMA hillshade...")
    bounds = (x.min(), y.min(), x.max(), y.max())
    elevation, extent = extract_rema_subset(DEM_PATH, bounds, buffer_km=10)
    hillshade = make_hillshade(elevation, extent)

    # Compute REMA flow vectors and MEaSUREs angular difference per segment
    print("   Computing flow vectors and MEaSUREs comparison...")
    segment_data = []
    for seg in segments:
        seg_x = x[seg['start']:seg['end']]
        seg_y = y[seg['start']:seg['end']]
        ice_thick = seg['ice_thickness']

        fx, fy = extract_rema_flow_vector(seg_x, seg_y, DEM_PATH, ice_thick, cache)

        # Mask where ice thickness is unknown
        invalid_mask = np.isnan(ice_thick)
        fx[invalid_mask] = np.nan
        fy[invalid_mask] = np.nan

        angular_diff, meas_mag = MEaSUREs_comparison(seg_x, seg_y, fx, fy)
        segment_data.append({
            'x': seg_x, 'y': seg_y,
            'angular_diff': angular_diff,
            'seg_num': seg['seg_num'],
        })
        if np.all(np.isnan(angular_diff)):
            print(f"     Seg {seg['seg_num']}: no valid flow data")
        else:
            print(f"     Seg {seg['seg_num']}: mean diff = {np.nanmean(angular_diff):.1f}°, "
                  f"median = {np.nanmedian(angular_diff):.1f}°")

    # Convert to km
    extent_km = [e / 1000 for e in extent]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 10))

    # Hillshade background
    ax.imshow(hillshade, extent=extent_km, cmap='gray', alpha=0.7, origin='upper')

    # Elevation contours
    elev_x = np.linspace(extent_km[0], extent_km[1], elevation.shape[1])
    elev_y = np.linspace(extent_km[3], extent_km[2], elevation.shape[0])
    ax.contour(elev_x, elev_y, elevation, levels=15, colors='white', linewidths=0.3, alpha=0.4)

    # Color tracks by angular difference using LineCollection
    norm = Normalize(vmin=0, vmax=90)
    cmap = plt.get_cmap('berlin')#.reversed()

    for sd in segment_data:
        sx_km = sd['x'] / 1000
        sy_km = sd['y'] / 1000
        diff = np.asarray(sd['angular_diff']).ravel()

        # Build line segments: each segment connects consecutive points
        points = np.column_stack([sx_km, sy_km]).reshape(-1, 1, 2)
        line_segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment by the mean of its two endpoint values
        colors = (diff[:-1] + diff[1:]) / 2

        lc = LineCollection(line_segments, cmap=cmap, norm=norm, linewidths=2.5, zorder=3)
        lc.set_array(colors)
        ax.add_collection(lc)

    # Segment number labels
    for sd in segment_data:
        mid = len(sd['x']) // 2
        ax.text(sd['x'][mid] / 1000, sd['y'][mid] / 1000, str(sd['seg_num']),
                fontsize=6, fontweight='bold', ha='center', va='center',
                color='white', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.15', facecolor='black', alpha=0.7, linewidth=0))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('REMA–MEaSUREs flow direction difference (°)', fontsize=10)

    ax.set_xlabel('Easting (km, EPSG:3031)', fontsize=11)
    ax.set_ylabel('Northing (km, EPSG:3031)', fontsize=11)
    ax.set_title(f'Flow Direction Confidence: {region_label}\nREMA vs MEaSUREs', fontsize=12, pad=10)
    ax.set_aspect('equal')

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_BASE_PATH, f'flow_confidence_{region_label}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"   Saved to {output_file}")


if __name__=="__main__":
    import sys
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    log_path = os.path.join(OUTPUT_BASE_PATH, 'flow_plots_log.txt')
    sys.stdout = Tee(log_path)
    datasets = load_datasets()
    for ds in datasets:
        main(ds)
        plot_flow_confidence(ds)
