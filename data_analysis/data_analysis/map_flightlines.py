"""
Plot radar flight track locations on an Antarctic map.

Extracts longitude/latitude from Bedmap(2,3) CSV files and plots them
using Antarctic Polar Stereographic projection (EPSG:3031).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import os
import re

import sys
from config import Tee, PROCESSING_FLAG_COLORS, PROCESSING_FLAG_NOTE, processing_flag_of
from loading import load_datasets


# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_BASE_PATH = os.path.join(_REGION_BASE, 'map_flightlines/')

def extract_coordinates(datasets):
    """
    Extract lon/lat coordinates from loaded datasets.
    
    Returns a dict of {dataset_name: {'lon': array, 'lat': array, 'trajectories': dict}}
    where trajectories contains per-trajectory coordinates for finer plotting.
    """
    coords = {}
    
    for bundle in datasets:
        name = bundle['name']
        df = bundle['data']
        
        lons = df['longitude (degree_east)'].values
        lats = df['latitude (degree_north)'].values
        
        # Also extract per-trajectory for coloring/grouping
        trajectories = {}
        for traj_id in df['trajectory_id'].unique():
            traj_df = df[df['trajectory_id'] == traj_id]
            trajectories[traj_id] = {
                'lon': traj_df['longitude (degree_east)'].values,
                'lat': traj_df['latitude (degree_north)'].values
            }
        
        coords[name] = {
            'lon': lons,
            'lat': lats,
            'trajectories': trajectories,
            'flag': processing_flag_of(df),
        }
        
        print(f"{name}: {len(lons)} points, lon range [{lons.min():.2f}, {lons.max():.2f}], "
              f"lat range [{lats.min():.2f}, {lats.max():.2f}]")
    
    return coords


def plot_antarctica_overview(coords, output_path='antarctica_tracks_overview.png'):
    """
    Plot all tracks on a full Antarctic map.
    Uses South Polar Stereographic projection (EPSG:3031).
    """
    # Define the Antarctic Polar Stereographic projection
    # This is equivalent to EPSG:3031
    antarctic_stereo = ccrs.SouthPolarStereo()
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=antarctic_stereo)
    
    # Set extent to show Antarctica (in plate carrée coordinates)
    ax.set_extent([-180, 180, -90, -60], crs=ccrs.PlateCarree())
    
    # Add features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(resolution='50m', linewidth=0.5)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, 
                      linestyle='--', color='gray')
    
    # Plot each dataset with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(coords)))
    
    for (name, data), color in zip(coords.items(), colors):
        ax.scatter(data['lon'], data['lat'], 
                   c=[color], s=1, alpha=0.5, 
                   transform=ccrs.PlateCarree(),
                   label=name)
    
    ax.legend(loc='upper left', markerscale=5)
    ax.set_title('Radar Flight Tracks - Antarctica Overview', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview map to {output_path}")
    plt.close()


def plot_regional_detail(coords, output_path='antarctica_tracks_regional.png'):
    """
    Plot tracks zoomed into the region of interest.
    Automatically determines extent from data bounds.
    """
    antarctic_stereo = ccrs.SouthPolarStereo()
    
    # Collect all coordinates to determine bounds
    all_lons = np.concatenate([data['lon'] for data in coords.values()])
    all_lats = np.concatenate([data['lat'] for data in coords.values()])
    
    # Add padding (in degrees)
    padding = 2.0
    lon_min, lon_max = all_lons.min() - padding, all_lons.max() + padding
    lat_min, lat_max = all_lats.min() - padding, all_lats.max() + padding
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=antarctic_stereo)
    
    # Set extent to the data region
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add features
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff', alpha=0.5)
    ax.coastlines(resolution='10m', linewidth=0.8)
    
    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5,
                      linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot each trajectory separately with different colors
    cmap = plt.cm.viridis
    
    for name, data in coords.items():
        trajectories = data['trajectories']
        n_traj = len(trajectories)
        colors = cmap(np.linspace(0.2, 0.8, n_traj))
        
        for (traj_id, traj_data), color in zip(trajectories.items(), colors):
            # Plot as connected line
            ax.plot(traj_data['lon'], traj_data['lat'],
                    c=color, linewidth=1.5, alpha=0.8,
                    transform=ccrs.PlateCarree(),
                    label=f"{name}: {traj_id}" if n_traj <= 5 else None)
            
            # Mark start and end points
            ax.scatter(traj_data['lon'][0], traj_data['lat'][0],
                       c='green', s=30, marker='o', zorder=5,
                       transform=ccrs.PlateCarree())
            ax.scatter(traj_data['lon'][-1], traj_data['lat'][-1],
                       c='red', s=30, marker='s', zorder=5,
                       transform=ccrs.PlateCarree())
    
    # Add legend markers for start/end
    ax.scatter([], [], c='green', s=30, marker='o', label='Track Start')
    ax.scatter([], [], c='red', s=30, marker='s', label='Track End')
    
    if sum(len(data['trajectories']) for data in coords.values()) <= 10:
        ax.legend(loc='best', fontsize=8)
    else:
        ax.legend(handles=ax.get_legend_handles_labels()[0][-2:], loc='best')
    
    ax.set_title(f'Radar Flight Tracks - Regional Detail\n'
                 f'Lon: [{lon_min:.1f}°, {lon_max:.1f}°], Lat: [{lat_min:.1f}°, {lat_max:.1f}°]',
                 fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved regional map to {output_path}")
    plt.close()


def plot_tracks_by_migration(coords, output_path='antarctica_tracks_migration.png'):
    """Regional track map coloured by radar migration status (data provenance)."""
    from matplotlib.lines import Line2D

    antarctic_stereo = ccrs.SouthPolarStereo()
    all_lons = np.concatenate([data['lon'] for data in coords.values()])
    all_lats = np.concatenate([data['lat'] for data in coords.values()])
    padding = 2.0
    lon_min, lon_max = all_lons.min() - padding, all_lons.max() + padding
    lat_min, lat_max = all_lats.min() - padding, all_lats.max() + padding

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=antarctic_stereo)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff', alpha=0.5)
    ax.coastlines(resolution='10m', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False

    present = []
    for name, data in coords.items():
        flag = data['flag'] or 'unmigrated_or_unknown'
        color = PROCESSING_FLAG_COLORS.get(flag, '0.3')
        if flag not in present:
            present.append(flag)
        ax.scatter(data['lon'], data['lat'], c=color, s=2, alpha=0.7,
                   transform=ccrs.PlateCarree())

    handles = [Line2D([], [], marker='o', ls='', color=PROCESSING_FLAG_COLORS.get(f, '0.3'),
                      label=PROCESSING_FLAG_NOTE.get(f, f)) for f in present]
    ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.85, markerscale=2)
    ax.set_title('Radar Flight Tracks — Migration Status', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved migration-status map to {output_path}")
    plt.close()


TIER_ORDER = ['A', 'B', 'C']   # good -> poor; ramp sampled in this order


def load_region_tiers(directory=None):
    """region-stem -> (tier, tier_driver) from coverage_summary.csv (written by coverage_tags.py).

    `tier_driver` is the signature of what failed (sparsity/directionality/sample,
    or '-' for a clean A). Region names carry a window suffix (`_w50km`) that the
    dataset labels don't, so we strip `_w<N>km` to recover the stem `coords` uses.
    """
    directory = directory or _REGION_BASE
    csv = os.path.join(directory, 'coverage_csvs', 'coverage_summary.csv')
    if not os.path.exists(csv):
        print(f"⚠️ {csv} not found — run coverage_tags.py first. Skipping tier map.")
        return None
    df = pd.read_csv(csv)
    stem = df['region'].str.replace(r'_w\d+km$', '', regex=True)
    driver = df['tier_driver'].fillna('-') if 'tier_driver' in df else ['-'] * len(df)
    return {s: (t, d) for s, t, d in zip(stem, df['tier'], driver)}


def plot_tracks_by_tier(coords, output_path='antarctica_tracks_tier.png', directory=None):
    """Regional track map coloured by data-intrinsic coverage tier (A/B/C) from coverage_tags.py."""
    from matplotlib.lines import Line2D

    tiers = load_region_tiers(directory)
    if tiers is None:
        return

    # sequential ramp sampled at the ordered tiers; NA -> neutral grey (off-ramp)
    ramp = plt.cm.viridis(np.linspace(0.9, 0.15, len(TIER_ORDER)))
    tier_color = {t: ramp[i] for i, t in enumerate(TIER_ORDER)}
    NA_COLOR = '0.6'

    antarctic_stereo = ccrs.SouthPolarStereo()
    all_lons = np.concatenate([data['lon'] for data in coords.values()])
    all_lats = np.concatenate([data['lat'] for data in coords.values()])
    padding = 2.0
    lon_min, lon_max = all_lons.min() - padding, all_lons.max() + padding
    lat_min, lat_max = all_lats.min() - padding, all_lats.max() + padding

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=antarctic_stereo)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff', alpha=0.5)
    ax.coastlines(resolution='10m', linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False

    # one legend entry per dataset: name, tier, and what failed (tier_driver)
    entries = []   # (tier, name, driver)
    for name, data in coords.items():
        tier, driver = tiers.get(name, ('NA', '-'))
        if name not in tiers:
            print(f"  ⚠️ no coverage tier for '{name}' — drawn as NA")
        entries.append((tier, name, driver))
        ax.scatter(data['lon'], data['lat'], color=[tier_color.get(tier, NA_COLOR)],
                   s=2, alpha=0.8, transform=ccrs.PlateCarree())

    # sort by tier (A->C, NA last) then name; label shows the driver signature
    rank = {t: i for i, t in enumerate(TIER_ORDER)}
    entries.sort(key=lambda e: (rank.get(e[0], len(TIER_ORDER)), e[1]))
    handles = []
    for tier, name, driver in entries:
        sig = '' if driver in ('-', '', None) else f' ({driver})'
        handles.append(Line2D([], [], marker='o', ls='', color=tier_color.get(tier, NA_COLOR),
                              label=f'{name} — {tier}{sig}'))
    ax.legend(handles=handles, loc='lower left', fontsize=7, framealpha=0.85,
              markerscale=2, title='Coverage tier (driver)')
    ax.set_title('Radar Flight Tracks — Coverage Tier', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved coverage-tier map to {output_path}")
    plt.close()


def plot_tracks_with_elevation(coords, datasets, output_path='antarctica_tracks_elevation.png'):
    """
    Plot tracks colored by bedrock elevation.
    """
    antarctic_stereo = ccrs.SouthPolarStereo()
    
    # Collect bounds
    all_lons = np.concatenate([data['lon'] for data in coords.values()])
    all_lats = np.concatenate([data['lat'] for data in coords.values()])
    
    padding = 2.0
    lon_min, lon_max = all_lons.min() - padding, all_lons.max() + padding
    lat_min, lat_max = all_lats.min() - padding, all_lats.max() + padding
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=antarctic_stereo)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.coastlines(resolution='10m', linewidth=0.8)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5,
                      linestyle='--', color='gray')
    gl.top_labels = False
    gl.right_labels = False
    
    # Collect all elevations for colorbar normalization
    all_elevs = []
    for bundle in datasets:
        df = bundle['data']
        elevs = df['bedrock_altitude (m)'].values
        all_elevs.extend(elevs[elevs != -9999])
    
    vmin, vmax = np.percentile(all_elevs, [2, 98])  # Clip outliers
    
    # Plot with elevation coloring
    for bundle in datasets:
        df = bundle['data']
        lons = df['longitude (degree_east)'].values
        lats = df['latitude (degree_north)'].values
        elevs = df['bedrock_altitude (m)'].values
        
        sc = ax.scatter(lons, lats, c=elevs, s=2, alpha=0.7,
                        cmap='terrain', vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Bedrock Elevation (m)', fontsize=10)
    
    ax.set_title('Radar Flight Tracks - Bedrock Elevation', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved elevation map to {output_path}")
    plt.close()


def print_coordinate_summary(coords):
    """Print a summary of coordinates for verification."""
    print("\n" + "="*60)
    print("COORDINATE SUMMARY")
    print("="*60)
    
    for name, data in coords.items():
        print(f"\n{name}:")
        print(f"  Total points: {len(data['lon'])}")
        print(f"  Longitude: [{data['lon'].min():.4f}°E, {data['lon'].max():.4f}°E]")
        print(f"  Latitude:  [{data['lat'].min():.4f}°N, {data['lat'].max():.4f}°N]")
        print(f"  Trajectories: {len(data['trajectories'])}")
        
        for traj_id, traj_data in data['trajectories'].items():
            print(f"    - {traj_id}: {len(traj_data['lon'])} points")


OCKENDEN_CELL_M = 50_000   # Ockenden metric grid spacing (PS71 metres)


def plot_tracks_on_ockenden(coords, output_path='tracks_on_ockenden.png',
                            metrics_dir='all_data/Ockenden/Data_Science_Zenodo/Data_Science_Zenodo/Metrics/',
                            zoom=False, track_ms=None, casing_ms=None, track_alpha=None):
    """
    Overlay flight tracks on Ockenden et al. Fig 4 landscape classification,
    recreated natively from the published metrics data (no image needed).

    zoom=True frames on the track bundle (padded) instead of the whole continent,
    drops classification cells outside that frame, and grows the cell markers to
    the 50 km grid so they stay contiguous at the zoomed scale.

    track_ms/casing_ms/track_alpha override the track styling; they default to the
    continental values when zoom=False and to a heavier, solid, white-cased line
    when zoom=True (the thin 30%-alpha black reads fine over the whole continent
    but is lost against the dark classes once zoomed).
    """
    from netCDF4 import Dataset
    from matplotlib.lines import Line2D
    import geopandas as gpd

    def load_metric(name):
        ds = Dataset(os.path.join(metrics_dir, name + '.nc'))
        d = ds.variables['data'][:].data
        ds.close()
        return d

    x_ifpa = load_metric('X_ifpa')
    y_ifpa = load_metric('Y_ifpa')

    # Reproduce classification masks from Ockenden source (Antarctic_FIGURES.ipynb cell 45)
    i_rms_slope_h = load_metric('i_rms_slope_h')
    ifpa_count_250 = load_metric('ifpa_count_max_250')
    ifpa_mean = load_metric('ifpa_mean')
    i_std_l = load_metric('i_std_l')
    ifpa_b1_thick = load_metric('ifpa_b1_thickness')
    ifpa_rms_slope = load_metric('ifpa_rms_slope')
    ifpa_rms_curv = load_metric('ifpa_rms_curvature')
    ifpa_count_20 = load_metric('ifpa_count_max_20')
    ifpa_count_100 = load_metric('ifpa_count_max_100')
    ifpa_wav_max = load_metric('ifpa_wav_max_power')

    mountain_mask = (i_rms_slope_h > 2) | (ifpa_count_250 > 10)
    SGM_mask = (~mountain_mask) & ((ifpa_mean > 1000) | (i_std_l > 19))
    SGM_mask2 = (~mountain_mask) & (~SGM_mask) & \
                (ifpa_b1_thick > -5.0) & (ifpa_rms_slope < 1.1) & (ifpa_mean > 500)
    poordetail_mask = (~mountain_mask) & (~SGM_mask) & (~SGM_mask2) & \
                      ((ifpa_rms_curv < 0.025) | (ifpa_count_20 < 15) | (i_rms_slope_h < 0.07))
    dunes_mask = (~poordetail_mask) & (~mountain_mask) & (~SGM_mask) & (~SGM_mask2) & \
                 ((ifpa_rms_slope / ifpa_rms_curv) < 14.75) & \
                 (ifpa_rms_slope < 0.9) & (ifpa_wav_max < 5000) & (ifpa_count_100 == 0)
    icestreams_mask = (~mountain_mask) & (~poordetail_mask) & (~dunes_mask) & (~SGM_mask) & (~SGM_mask2) & \
                      (ifpa_b1_thick < -5.5) & (ifpa_rms_slope > 1.0)
    icestreams_mask2 = (~mountain_mask) & (~poordetail_mask) & (~dunes_mask) & \
                       (~SGM_mask) & (~SGM_mask2) & (~icestreams_mask)

    # Tracks in PS71; the zoom frame is their padded bounding box, clipped to the continent
    to_ps = Transformer.from_crs('EPSG:4326', 'EPSG:3031', always_xy=True)
    tracks = {name: to_ps.transform(data['lon'], data['lat']) for name, data in coords.items()}
    xlim, ylim = (-2.55e6, 2.7e6), (-2.2e6, 2.2e6)
    if zoom:
        tx = np.concatenate([t[0] for t in tracks.values()])
        ty = np.concatenate([t[1] for t in tracks.values()])
        pad = max(0.15 * max(np.ptp(tx), np.ptp(ty)), 3 * OCKENDEN_CELL_M)
        xlim = (max(xlim[0], tx.min() - pad), min(xlim[1], tx.max() + pad))
        ylim = (max(ylim[0], ty.min() - pad), min(ylim[1], ty.max() + pad))

    # Drop classification cells outside the frame (+1 cell) so only the local landscape is drawn
    inview = ((x_ifpa > xlim[0] - OCKENDEN_CELL_M) & (x_ifpa < xlim[1] + OCKENDEN_CELL_M) &
              (y_ifpa > ylim[0] - OCKENDEN_CELL_M) & (y_ifpa < ylim[1] + OCKENDEN_CELL_M))

    # Plot classification (same colors and draw order as Ockenden source)
    fig, ax = plt.subplots(figsize=(12, 10))
    s, marker = 30, 'o'
    classes = [(poordetail_mask, '#f3e738', 'Low relief landscape'),
               (SGM_mask | SGM_mask2, '#ff9248', 'Alpine landscape (subglacial)'),
               (mountain_mask, '#e75921', 'Alpine landscape (subaerial)'),
               (icestreams_mask, '#4399bf', 'Selective erosion (ice streams)'),
               (icestreams_mask2, '#2f64b4', 'Selective erosion (relict)'),
               (dunes_mask, 'white', 'Invalid data (dunes)')]
    cells, handles = [], []
    for mask, color, label in classes:
        mask = mask & inview
        if not mask.any():
            continue
        cells.append(ax.scatter(x_ifpa[mask], y_ifpa[mask], c=color, s=s, marker=marker))
        handles.append(Line2D([], [], marker=marker, ls='', ms=10, color=color,
                              mec='0.4', mew=0.3, label=label))

    # Grounding line
    gl_path = os.path.join(os.path.dirname(metrics_dir), 'GroundingLine_Antarctica_v2.shp')
    if os.path.exists(gl_path):
        gpd.read_file(gl_path).plot(ax=ax, facecolor='None', edgecolor='k', linewidth=0.5)

    # Overlay tracks (continental defaults unchanged; zoomed gets a visible casing + solid black)
    casing_ms = casing_ms if casing_ms is not None else (5.0 if zoom else 2)
    track_ms = track_ms if track_ms is not None else (2.0 if zoom else 1)
    track_alpha = track_alpha if track_alpha is not None else (1.0 if zoom else 0.3)
    for name, (x, y) in tracks.items():
        ax.plot(x, y, '.', color='white', ms=casing_ms, zorder=3)
        ax.plot(x, y, '.', color='black', ms=track_ms, alpha=track_alpha, zorder=4)
        handles.append(Line2D([], [], marker='.', ls='', ms=10, color='black', label=name))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.8)
    ax.set_title('Flight tracks on Ockenden et al. landscape classification'
                 + (' (regional)' if zoom else ''), fontsize=12)
    plt.tight_layout()

    # Grow the cell markers to the 50 km grid once the axes box is final
    if zoom and cells:
        fig.canvas.draw()
        pts_per_m = ax.get_window_extent().width * 72 / fig.dpi / (xlim[1] - xlim[0])
        for pc in cells:
            pc.set_sizes([(OCKENDEN_CELL_M * pts_per_m) ** 2])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overlay map to {output_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    log_path = os.path.join(OUTPUT_BASE_PATH, 'map_flightlines_log.txt')
    sys.stdout = Tee(log_path)
    print("Loading datasets...")
    datasets = load_datasets()
    
    if not datasets:
        print("No datasets loaded. Check file paths.")
        exit(1)
    
    print("\nExtracting coordinates...")
    coords = extract_coordinates(datasets)
    
    print_coordinate_summary(coords)
    
    print("\nGenerating maps...")
    plot_antarctica_overview(coords, os.path.join(OUTPUT_BASE_PATH, 'antarctica_tracks_overview.png'))
    plot_regional_detail(coords, os.path.join(OUTPUT_BASE_PATH, 'antarctica_tracks_regional.png'))
    plot_tracks_by_migration(coords, os.path.join(OUTPUT_BASE_PATH, 'antarctica_tracks_migration.png'))
    plot_tracks_by_tier(coords, os.path.join(OUTPUT_BASE_PATH, 'antarctica_tracks_tier.png'))
    plot_tracks_with_elevation(coords, datasets, os.path.join(OUTPUT_BASE_PATH, 'antarctica_tracks_elevation.png'))
    plot_tracks_on_ockenden(coords, os.path.join(OUTPUT_BASE_PATH, 'tracks_on_ockenden.png'))
    plot_tracks_on_ockenden(coords, os.path.join(OUTPUT_BASE_PATH, 'tracks_on_ockenden_regional.png'), zoom=True)

    print("\nDone! Generated maps:")
    print("  - antarctica_tracks_overview.png (full continent)")
    print("  - antarctica_tracks_regional.png (zoomed to data)")
    print("  - antarctica_tracks_migration.png (colored by migration status)")
    print("  - antarctica_tracks_tier.png (colored by coverage tier)")
    print("  - antarctica_tracks_elevation.png (colored by bed elevation)")
    print("  - tracks_on_ockenden.png (tracks on Ockenden Fig 4)")
    print("  - tracks_on_ockenden_regional.png (same, zoomed to the track bundle)")