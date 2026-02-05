"""
Plot radar flight track locations on an Antarctic map.

Extracts longitude/latitude from Bedmap3 CSV files and plots them
using Antarctic Polar Stereographic projection (EPSG:3031).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer
import os


def load_datasets():
    """Load the Bedmap3 CSV files with the same logic as bed_analysis_15.py"""
    base_path = 'shortcut_to_culled-data'
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
        # {
        #     'file': 'PRIC_2016_CHA2_AIR_BM3.csv', 
        #     'label': 'PEL_CHA2',
        #     # We shift the start index forward to remove the first segment
        #     # skip the exact number of rows in 'Segment 1'
        #     # 'subset': lambda df: df.iloc[410823 : 410823 + 54566].copy(),
        #     'subset': lambda df: df.iloc[410823 + 15451 : 410823 + 54566].copy(),
        #     'force_id': 'PRIC_2016_CHA2'
        # },
        {'file': 'BAS_2010_IMAFI_AIR_BM3.csv', 'label': 'Moller_Stream'},    # Institute-Möller Ice Stream
        {'file': 'BAS_2018_Thwaites_AIR_BM3.csv',  'label':'Thwaites_BAS'},   # Thwaites Glacier
        {'file': 'CRESIS_2009_Thwaites_AIR_BM3.csv', 'label': 'Thwaites_CR'}, # Thwaites Swath
        {'file': 'AWI_2018_ANIRES_AIR_BM3.csv', 'label': 'DML_AniRES'},      # Dronning Maud Land
    ]


    for item in target_files:
        filename = item['file']
        label = item['label']
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: {filename} not found. Skipping.")
            continue

        try:
            df = pd.read_csv(filepath, comment='#')
            
            if 'subset' in item:
                df = item['subset'](df)
            
            if 'force_id' in item:
                df['trajectory_id'] = item['force_id']
            
            # Clean Bedmap3 nulls (-9999)
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
            'trajectories': trajectories
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


if __name__ == "__main__":
    print("Loading datasets...")
    datasets = load_datasets()
    
    if not datasets:
        print("No datasets loaded. Check file paths.")
        exit(1)
    
    print("\nExtracting coordinates...")
    coords = extract_coordinates(datasets)
    
    print_coordinate_summary(coords)
    
    print("\nGenerating maps...")
    plot_antarctica_overview(coords)
    # plot_regional_detail(coords)
    # plot_tracks_with_elevation(coords, datasets)
    
    print("\nDone! Generated maps:")
    print("  - antarctica_tracks_overview.png (full continent)")
    print("  - antarctica_tracks_regional.png (zoomed to data)")
    print("  - antarctica_tracks_elevation.png (colored by bed elevation)")