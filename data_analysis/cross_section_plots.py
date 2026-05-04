"""
Cross-section plots: bedrock + REMA surface elevation along radar flight lines.
Shows the ice column between bed and surface for each trajectory.

Usage:
    python cross_section_plots.py

Reads the same datasets as bed_analysis_20.py, applies the same segmentation,
then plots bedrock (from radar) and surface (from REMA) vs distance along track.
"""

import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.ndimage import uniform_filter1d
import os
import sys

from bed_analysis_21 import Tee, load_datasets, detect_data_gaps, split_into_segments, split_by_landscape
from REMA_extractor import extract_rema_elevation

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cross_sections')
DEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'shortcut_to_culled-data',
                        'rema_mosaic_100m_v2.0_filled_cop30',
                        'rema_mosaic_100m_v2.0_filled_cop30_dem.tif')
# MAX_PLOTS = 50  # safety cap per dataset


def plot_cross_section(dist, elev, segments, surface_segments, traj_id, output_path):
    """
    Plot bedrock + REMA surface along a flight line.

    dist, elev: full trajectory arrays (for the gray background line)
    segments: list of (seg_data, seg_dist) from segmentation
    surface_segments: list of surface_elev arrays, one per segment
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Gray background: raw bedrock profile (break at gaps)
    plot_elev = elev.copy().astype(float)
    steps = np.diff(dist)
    for idx in np.where(steps > 2000)[0]:
        plot_elev[idx + 1] = np.nan
    ax.plot(dist / 1000, plot_elev, color='0.6', lw=0.8, alpha=0.4, label='Raw bedrock')

    # Per-segment: bedrock, surface, and fill
    colors = plt.cm.tab10.colors
    for i, ((seg_data, seg_dist), surf_elev) in enumerate(zip(segments, surface_segments)):
        c = colors[i % len(colors)]
        d_km = seg_dist / 1000
        bed = seg_data['bedrock_altitude (m)'].values

        ax.plot(d_km, bed, color=c, lw=1.2, label=f'Seg {i+1} bed')
        ax.plot(d_km, surf_elev, color=c, lw=1.2, ls='--', label=f'Seg {i+1} surface')
        ax.fill_between(d_km, bed, surf_elev, color=c, alpha=0.12)

    ax.set_xlabel('Distance along track (km)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title(f'Cross-section: {traj_id}')
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'cross_section_{traj_id}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, 'cross_sections_log.txt')
    sys.stdout = Tee(log_path)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    datasets = load_datasets()
    if not datasets:
        print("No datasets loaded.")
        return

    for bundle in datasets:
        name = bundle['name']
        df = bundle['data']
        print(f"\n=== {name} ===")

        out_path = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_path, exist_ok=True)

        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        count = 0

        for traj_id in valid['trajectory_id'].unique():
            # if count >= MAX_PLOTS:
            #     break

            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20:
                continue

            lons = line['longitude (degree_east)'].values
            lats = line['latitude (degree_north)'].values
            x, y = transformer.transform(lons, lats)
            dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
            elev = line['bedrock_altitude (m)'].values

            # Segmentation (same as bed_analysis_21)
            gap_segments = split_into_segments(line, dist)
            if not gap_segments:
                continue
            segments = []
            for seg_data, seg_dist in gap_segments:
                segments.extend(split_by_landscape(seg_data, seg_dist))

            # Extract REMA surface per segment
            surface_segments = []
            for seg_data, seg_dist in segments:
                sx, sy = transformer.transform(
                    seg_data['longitude (degree_east)'].values,
                    seg_data['latitude (degree_north)'].values,
                )
                surface_segments.append(extract_rema_elevation(sx, sy, DEM_PATH))

            plot_cross_section(dist, elev, segments, surface_segments, traj_id, out_path)
            count += 1
            print(f"  [{count}] {traj_id}")

        print(f"  Saved {count} cross-section plots to {out_path}")


if __name__ == '__main__':
    main()
