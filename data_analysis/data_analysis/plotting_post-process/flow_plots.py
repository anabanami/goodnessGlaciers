import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# Import the local tools
import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, get_rema_cache, MEaSUREs_comparison
from config import Tee, processing_flag_of
from loading import load_datasets
from plotting import flag_title, label_trajectories
# Shared segmentation — SAME two-step split (gaps + landscape) the pipeline uses,
# so segment numbers here match the PSD plots and the segment-level CSV.
from segmentation import split_into_segments, split_by_landscape

BASE_PATH = 'all_data/'
DEM_PATH = os.path.join(BASE_PATH, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

# Module-level transformer (created once, reused)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# Quiver density knob: one flow arrow roughly every ARROW_SPACING_KM along a track.
# Larger value -> fewer arrows. This is THE place to change it.
ARROW_SPACING_KM = 50.0


# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_BASE_PATH = os.path.join(_REGION_BASE, 'flow_plots/')

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
    Mirrors calculate_flow_incidence in bed_analysis.py.
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


def build_segments(df, dem_path, cache, thickness_threshold=0.20):
    """
    Segment the region EXACTLY like bed_analysis.py so segment identity matches
    the rest of the pipeline:
      per trajectory -> split_into_segments (gaps) -> split_by_landscape (bed gradient)
    Segments are numbered per trajectory by their index in the two-step list
    (seg_idx + 1), and that number is NOT reassigned when a segment is dropped for
    insufficient ice thickness — so seg_num lines up with the PSD filenames and the
    segment-level CSV (which are keyed on (trajectory, segment)).

    Returns a list of dicts with projected coords, ice thickness and metadata.
    """
    out = []
    for traj_id in df['trajectory_id'].unique():
        line = df[df['trajectory_id'] == traj_id].copy()
        if len(line) < 20:
            continue

        lx, ly = _TRANSFORMER.transform(line['longitude (degree_east)'].values,
                                        line['latitude (degree_north)'].values)
        lx, ly = np.asarray(lx), np.asarray(ly)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(lx)**2 + np.diff(ly)**2))])

        gap_segments = split_into_segments(line, dist)
        if not gap_segments:
            continue

        segments = []
        for seg_data, seg_dist in gap_segments:
            segments.extend(split_by_landscape(seg_data, seg_dist))

        for seg_idx, (segment_data, segment_distance, is_transition) in enumerate(segments):
            seg_x, seg_y = _TRANSFORMER.transform(segment_data['longitude (degree_east)'].values,
                                                  segment_data['latitude (degree_north)'].values)
            seg_x, seg_y = np.asarray(seg_x), np.asarray(seg_y)

            bedrock_elevs = segment_data['bedrock_altitude (m)'].values
            surface_elevs = extract_rema_elevation(seg_x, seg_y, dem_path, cache)
            ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_elevs)

            thickness_validity = np.sum(~np.isnan(ice_thickness)) / len(ice_thickness)
            if thickness_validity < thickness_threshold:
                print(f"   Skipping {traj_id} Segment {seg_idx+1}: "
                      f"insufficient thickness data ({thickness_validity*100:.1f}% valid)")
                continue

            out.append({
                'traj_id': traj_id,
                'seg_num': seg_idx + 1,          # per-trajectory, matches pipeline
                'seg_x': seg_x, 'seg_y': seg_y,
                'segment_data': segment_data,
                'segment_distance': segment_distance,
                'ice_thickness': ice_thickness,
                'is_transition': is_transition,
            })
            print(f"   {traj_id} Segment {seg_idx+1}: valid ({thickness_validity*100:.1f}% thickness data)")

    return out


def _seg_label(seg):
    """Unambiguous (trajectory, segment) tag — seg numbers repeat across trajectories."""
    return f"{seg['traj_id']}·{seg['seg_num']}"


def get_orientation_color(angle):
    """Return color based on orientation class."""
    if angle < 30:
        return '#2E7D32'  # Dark green
    elif angle < 60:
        return '#F9A825'  # Amber
    else:
        return '#C62828'  # Dark red


def arrow_indices(seg_x, seg_y, spacing_km=ARROW_SPACING_KM):
    """Indices spaced ~spacing_km apart along a segment (for uniform quiver density)."""
    d = calculate_along_track_distance(seg_x, seg_y)  # cumulative km
    if d[-1] <= 0:
        return np.array([0])
    targets = np.arange(0, d[-1], spacing_km)
    idx = np.unique(np.searchsorted(d, targets))
    return idx[idx < len(seg_x)]


def main(dataset_dict):
    region_label = dataset_dict['name']
    df = dataset_dict['data']
    pflag = processing_flag_of(df)

    print(f"Visualizing Flow Orientation for: {region_label}")

    # Get the REMA cache (loads DEM once, reused for all operations)
    cache = get_rema_cache()
    cache.load(DEM_PATH)

    # Build segments with the same two-step split the pipeline uses
    print("   Building pipeline-consistent segments...")
    segments = build_segments(df, DEM_PATH, cache)
    if not segments:
        print(f"   No valid segments for {region_label}")
        return
    print(f"   {len(segments)} valid segments across "
          f"{len({s['traj_id'] for s in segments})} trajectories")
    print(f"   \n===== TOTAL SEGMENT COUNT ({region_label}): {len(segments)} =====\n")

    # --- Region bounds for the hillshade background (all valid coords) ---
    all_x = np.concatenate([s['seg_x'] for s in segments])
    all_y = np.concatenate([s['seg_y'] for s in segments])

    # --- Extract REMA subset for background ---
    print("   Extracting REMA hillshade...")
    bounds = (all_x.min(), all_y.min(), all_x.max(), all_y.max())
    elevation, extent = extract_rema_subset(DEM_PATH, bounds, buffer_km=10)
    hillshade = make_hillshade(elevation, extent)

    # --- Compute flow vectors and incidence per segment ---
    print("   Computing flow vectors...")
    for seg in segments:
        fx, fy = extract_rema_flow_vector(seg['seg_x'], seg['seg_y'], DEM_PATH, seg['ice_thickness'], cache)
        seg['flow_x'], seg['flow_y'] = fx, fy
        seg['incidence'] = calculate_incidence_angle(seg['seg_x'], seg['seg_y'], fx, fy)

    extent_km = [e / 1000 for e in extent]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(14, 10))

    # Background: hillshade
    ax.imshow(hillshade, extent=extent_km, cmap='gray', alpha=0.7, origin='upper')

    # Overlay: surface elevation contours (subtle)
    elev_x = np.linspace(extent_km[0], extent_km[1], elevation.shape[1])
    elev_y = np.linspace(extent_km[3], extent_km[2], elevation.shape[0])  # Note: top to bottom
    ax.contour(elev_x, elev_y, elevation, levels=15, colors='white', linewidths=0.3, alpha=0.4)

    # Plot track colored by segment orientation
    for seg in segments:
        mean_angle = np.nanmean(seg['incidence'])
        color = get_orientation_color(mean_angle)
        ax.plot(seg['seg_x'] / 1000, seg['seg_y'] / 1000, color=color, linewidth=3,
                solid_capstyle='round', zorder=3)

    # Flow vectors (uniform ~10 km spacing so density doesn't scale with segment count)
    for seg in segments:
        idx = arrow_indices(seg['seg_x'], seg['seg_y'], spacing_km=ARROW_SPACING_KM)
        ax.quiver(seg['seg_x'][idx] / 1000, seg['seg_y'][idx] / 1000,
                  seg['flow_x'][idx], seg['flow_y'][idx],
                  color='royalblue', alpha=0.7, scale=30, width=0.004,
                  headwidth=4, headlength=5, zorder=2)

    # Trajectory labels (segments are colored but not individually labelled).
    # Placement computed from raw geometry so it matches hillshade-region_plots.py.
    label_trajectories(ax, df, _TRANSFORMER, drawn_traj_ids={s['traj_id'] for s in segments})

    # Styling
    ax.set_xlabel('Easting (km, EPSG:3031)', fontsize=11)
    ax.set_ylabel('Northing (km, EPSG:3031)', fontsize=11)
    flag_title(ax, f'Ice Flow Orientation: {region_label}', pflag, fontsize=12)
    ax.set_aspect('equal')

    # --- Find emptiest corner for legend ---
    x_km_all, y_km_all = all_x / 1000, all_y / 1000
    x_mid = (x_km_all.min() + x_km_all.max()) / 2
    y_mid = (y_km_all.min() + y_km_all.max()) / 2

    corners = {
        'upper left':  np.sum((x_km_all < x_mid) & (y_km_all > y_mid)),
        'upper right': np.sum((x_km_all > x_mid) & (y_km_all > y_mid)),
        'lower left':  np.sum((x_km_all < x_mid) & (y_km_all < y_mid)),
        'lower right': np.sum((x_km_all > x_mid) & (y_km_all < y_mid)),
    }
    sorted_corners = sorted(corners.items(), key=lambda c: c[1])
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
    output_file = os.path.join(OUTPUT_BASE_PATH, f'flow_orientation_{region_label}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()
    print(f"   Done! Saved to {output_file}")

    # Print summary
    print(f"\n   Segment summary:")
    for seg in sorted(segments, key=lambda s: (s['traj_id'], s['seg_num'])):
        mean_angle = np.nanmean(seg['incidence'])
        length_km = calculate_along_track_distance(seg['seg_x'], seg['seg_y'])[-1]
        orientation = 'Parallel' if mean_angle < 30 else ('Oblique' if mean_angle < 60 else 'Perpendicular')
        print(f"     {_seg_label(seg):>12s}: {orientation:13s} ({mean_angle:.1f}°) | {length_km:.1f} km")

    # Angular coverage diagnostic
    all_angles = np.concatenate([seg['incidence'] for seg in segments])
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
    pflag = processing_flag_of(df)

    print(f"Flow confidence map for: {region_label}")

    cache = get_rema_cache()
    cache.load(DEM_PATH)

    # Build segments with the same two-step split the pipeline uses
    print("   Building pipeline-consistent segments...")
    segments = build_segments(df, DEM_PATH, cache)
    if not segments:
        print(f"   No valid segments for {region_label}")
        return
    print(f"   {len(segments)} valid segments across "
          f"{len({s['traj_id'] for s in segments})} trajectories")

    # Region bounds for the hillshade background
    all_x = np.concatenate([s['seg_x'] for s in segments])
    all_y = np.concatenate([s['seg_y'] for s in segments])

    print("   Extracting REMA hillshade...")
    bounds = (all_x.min(), all_y.min(), all_x.max(), all_y.max())
    elevation, extent = extract_rema_subset(DEM_PATH, bounds, buffer_km=10)
    hillshade = make_hillshade(elevation, extent)

    # Compute REMA flow vectors and MEaSUREs angular difference per segment
    print("   Computing flow vectors and MEaSUREs comparison...")
    for seg in segments:
        seg_x, seg_y = seg['seg_x'], seg['seg_y']
        ice_thick = seg['ice_thickness']

        fx, fy = extract_rema_flow_vector(seg_x, seg_y, DEM_PATH, ice_thick, cache)

        # Mask where ice thickness is unknown
        invalid_mask = np.isnan(ice_thick)
        fx[invalid_mask] = np.nan
        fy[invalid_mask] = np.nan

        angular_diff, meas_mag = MEaSUREs_comparison(seg_x, seg_y, fx, fy)
        seg['angular_diff'] = angular_diff
        if np.all(np.isnan(angular_diff)):
            print(f"     {_seg_label(seg)}: no valid flow data")
        else:
            print(f"     {_seg_label(seg)}: mean diff = {np.nanmean(angular_diff):.1f}°, "
                  f"median = {np.nanmedian(angular_diff):.1f}°")

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
    # Sequential, CVD-safe, monotonic lightness: agreement stays dark, high
    # disagreement reads bright yellow against the gray hillshade.
    cmap = plt.get_cmap('viridis')

    for seg in segments:
        sx_km = seg['seg_x'] / 1000
        sy_km = seg['seg_y'] / 1000
        diff = np.asarray(seg['angular_diff']).ravel()

        # Build line segments: each segment connects consecutive points
        points = np.column_stack([sx_km, sy_km]).reshape(-1, 1, 2)
        line_segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Color each segment by the mean of its two endpoint values
        colors = (diff[:-1] + diff[1:]) / 2

        lc = LineCollection(line_segments, cmap=cmap, norm=norm, linewidths=2.5, zorder=3)
        lc.set_array(colors)
        ax.add_collection(lc)

    # Trajectory labels (segments are colored but not individually labelled).
    # Placement computed from raw geometry so it matches hillshade-region_plots.py.
    label_trajectories(ax, df, _TRANSFORMER, drawn_traj_ids={s['traj_id'] for s in segments})

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('REMA–MEaSUREs flow direction difference (°)', fontsize=10)

    ax.set_xlabel('Easting (km, EPSG:3031)', fontsize=11)
    ax.set_ylabel('Northing (km, EPSG:3031)', fontsize=11)
    flag_title(ax, f'Flow Direction Confidence: {region_label}\nREMA vs MEaSUREs', pflag, fontsize=12)
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
