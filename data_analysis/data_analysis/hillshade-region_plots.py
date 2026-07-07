import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import os

# Import the local tools
from REMA_extractor import extract_rema_elevation, calculate_ice_thickness, get_rema_cache
from config import Tee, processing_flag_of
from loading import load_datasets
from plotting import flag_title, label_trajectories

BASE_PATH = 'all_data/'
DEM_PATH = os.path.join(BASE_PATH, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

# Module-level transformer (created once, reused)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)


# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_BASE_PATH = os.path.join(_REGION_BASE, 'hillshade-region_plots/')

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

    for seg_idx, (segment_df, start, end, traj_id) in enumerate(segments):
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
            'traj_id': traj_id,
            'surface_elevs': surface_elevs,
            'ice_thickness': ice_thickness,
        })
        print(f"   Segment {seg_idx+1}: Valid ({thickness_validity*100:.1f}% thickness data)")

    return valid_segments


def main(dataset_dict):
    region_label = dataset_dict['name']
    df = dataset_dict['data']
    pflag = processing_flag_of(df)

    print(f"Rendering hillshade for: {region_label}")

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
            raw_segments.append((seg_df, global_start, global_end, traj_id))

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

    # Convert to km for plotting
    x_km, y_km = x / 1000, y / 1000
    extent_km = [e / 1000 for e in extent]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: hillshade
    ax.imshow(hillshade, extent=extent_km, cmap='gray', alpha=0.7, origin='upper')

    # Overlay: surface elevation contours (subtle)
    elev_x = np.linspace(extent_km[0], extent_km[1], elevation.shape[1])
    elev_y = np.linspace(extent_km[3], extent_km[2], elevation.shape[0])  # Note: top to bottom
    ax.contour(elev_x, elev_y, elevation, levels=15, colors='white', linewidths=0.3, alpha=0.4)

    # Plot flight tracks (single neutral color)
    for seg in segments:
        ax.plot(x_km[seg['start']:seg['end']], y_km[seg['start']:seg['end']],
                color='C0', linewidth=2, solid_capstyle='round', zorder=3)

    # Trajectory ID labels: one per trajectory (shared placement with flow_plots.py)
    label_trajectories(ax, df, _TRANSFORMER, drawn_traj_ids={s['traj_id'] for s in segments})

    # Styling
    ax.set_xlabel('Easting (km, EPSG:3031)', fontsize=11)
    ax.set_ylabel('Northing (km, EPSG:3031)', fontsize=11)
    flag_title(ax, f'REMA Hillshade: {region_label}', pflag, fontsize=12)
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
    sorted_corners = sorted(corners.items(), key=lambda c: c[1])
    legend_loc = sorted_corners[0][0]

    # Legend
    legend_elements = [
        Line2D([0], [0], color='C0', lw=2, label='Flight track'),
        Line2D([0], [0], color='0.6', lw=0.6, label='Surface elevation contour'),
    ]
    ax.legend(handles=legend_elements, loc=legend_loc, fontsize=9, framealpha=0.95)

    plt.tight_layout()
    output_file = os.path.join(OUTPUT_BASE_PATH, f'hillshade_{region_label}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Done! Saved to {output_file}")

    # Print summary (per trajectory)
    segs_by_traj = {}
    for seg in segments:
        segs_by_traj.setdefault(seg['traj_id'], []).append(seg)
    print(f"\n   Trajectory summary ({len(segs_by_traj)} trajectories, {len(segments)} gap-segments):")
    for traj_id, traj_segs in segs_by_traj.items():
        total_km = sum(calculate_along_track_distance(x[s['start']:s['end']], y[s['start']:s['end']])[-1]
                       for s in traj_segs)
        print(f"     {traj_id}: {len(traj_segs)} segment(s), {total_km:.1f} km total")


if __name__=="__main__":
    import sys
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    log_path = os.path.join(OUTPUT_BASE_PATH, 'hillshade-region_plots_log.txt')
    sys.stdout = Tee(log_path)
    datasets = load_datasets()
    for ds in datasets:
        main(ds)
