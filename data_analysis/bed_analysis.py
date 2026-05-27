import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d
from pyproj import Transformer
import os
import re
import sys
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, MEaSUREs_comparison


class Tee:
    """Write to both stdout and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Window parameters for sensitivity testing
WINDOW_SIZE = 50000  # metres
STEP_SIZE = WINDOW_SIZE // 2 # 50% overlap
WINDOW_TYPE = 'rectangular'

# peak masking parameters for sensitivity testing
peak_masking_height_threshold = 2.0
bin_buffer = 5

# Landscape splitting parameters
SMOOTHING_LENGTH = WINDOW_SIZE  # metres — full window for elevation smoothing
GRADIENT_THRESHOLD = 15  # m/km (split where smoothed elevation gradient exceeds this)


# Output configuration - creates folders in same directory as this script
OUTPUT_BASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    # 'Ockenden-regions/',
    'SMUG-regions/',
    
    # 'sensitivity-window-size',
    # f'{WINDOW_SIZE/1000}km'
    # 'sensitivity-peak-masking',
    # f'threshold_{peak_masking_height_threshold}'
    # 'sensitivity-gradient-threshold',
    # f'{GRADIENT_THRESHOLD}m_per_km'
)


def get_region_folder(dataset_name):
    """
    Extract a clean region folder name from the dataset name.
    Includes year prefix for 2008 data to avoid conflicts with 2010.
    E.g., 'ASB_ICECAP_2010_Fig4_Aurora_SB' -> 'Fig4_Aurora_SB'
          'ASB_ICECAP_2008_Fig2G_Highland_A' -> '2008_Fig2G_Highland_A'
    """
    # Check if this is 2008 data (needs separate folder to avoid conflicts)
    is_2008 = '_2008_' in dataset_name

    # Match Fig followed by number/letter and region name
    match = re.search(r'(Fig\w+_\w+)$', dataset_name)
    if match:
        region = match.group(1)
        return f'2008_{region}' if is_2008 else region
    # Fallback: use last part after final underscore pattern
    region = dataset_name.split('_')[-1]
    return f'2008_{region}' if is_2008 else dataset_name


def ensure_output_dirs(base_path, region_folder):
    """
    Create output directory structure for a region.
    Returns dict with paths for trajectories and psd folders.
    """
    region_path = os.path.join(base_path, region_folder)
    trajectories_path = os.path.join(region_path, 'trajectories')
    psd_path = os.path.join(region_path, 'psd')

    os.makedirs(trajectories_path, exist_ok=True)
    os.makedirs(psd_path, exist_ok=True)

    return {
        'region': region_path,
        'trajectories': trajectories_path,
        'psd': psd_path
    }


def load_datasets():
    base_path = 'all_data/bedmap3_data/bedmap*/Results/'
    all_dfs = []
    

    target_files = [
        {
            'file': 'UTIG_2010_ICECAP_AIR_BM3.csv',
            'label': 'TEST_Aurora_SB',
            'subset': lambda df: df.iloc[8508112:8508112+17528].copy(),
        },
        
        {
            'file': 'PRIC_2016_CHA2_AIR_BM3.csv', 
            'label': 'PEL_CHA2',
            # skip the exact number of rows in 'Segment 1'
            'subset': lambda df: df.iloc[410823 : 410823 + 54566].copy(),
            'force_id': 'PRIC_2016_CHA2',
        },

        {
            'file': 'BAS_2010_IMAFI_AIR_BM3.csv', 
            'label': 'Moller_Stream'
        },    # Institute-Möller Ice Stream
        
        {
            'file': 'BAS_2018_Thwaites_AIR_BM3.csv',
            'label':'Thwaites_BAS'
        },    # Thwaites Glacier
        
        {
          'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
          'label': 'DML_AniRES'
         },   # Dronning Maud Land
    ###########################################################################
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

        # {
        #     'file': 'BAS_2015_POLARGAP_AIR_BM3.csv',
        #     'label': 'POLARGAP_2015_Fig1_Pensacola_Pole',
        #     'subset': lambda df, _r={
        #         'lat_min': -88.0, 'lat_max': -82.0,
        #         'lon_min': -60.0, 'lon_max': -20.0,
        #     }: df[
        #         (df['latitude (degree_north)'] >= _r['lat_min']) &
        #         (df['latitude (degree_north)'] <= _r['lat_max']) &
        #         (df['longitude (degree_east)']  >= _r['lon_min']) &
        #         (df['longitude (degree_east)']  <= _r['lon_max'])
        #     ].copy(),
        # },

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
        matches = glob.glob(os.path.join(base_path, filename))
        if not matches:
            print(f"⚠️ Warning: {filename} not found. Skipping.")
            continue

        filepath = matches[0]

        try:
            if filepath not in file_cache:
                print(f"  Reading {filename}...")
                file_cache[filepath] = pd.read_csv(filepath, comment='#', low_memory=False)
            
            df = file_cache[filepath].copy()
            
            if 'subset' in item:
                df = item['subset'](df)
            
            if 'force_id' in item:
                df['trajectory_id'] = item['force_id']
            
            # Clean Bedmap nulls (-9999)
            initial_len = len(df)
            has_valid_bed = df['bedrock_altitude (m)'] != -9999
            has_valid_traj = (df['trajectory_id'] != -9999) | ('force_id' in item)
            df = df[has_valid_bed & has_valid_traj].copy()
            
            df['trajectory_id'] = df['trajectory_id'].astype(str)
            
            if len(df) > 0:
                print(f"✓ {label} loaded: {len(df)} rows (Filtered {initial_len - len(df)} nulls)")
                all_dfs.append({'name': label, 'data': df})
            else:
                print(f"---{label} resulted in 0 rows.---")
                
        except Exception as e:
            print(f"---Error loading {label}: {e}---")

    # Free cache
    del file_cache

    return all_dfs


def plot_raw_data_with_segmentation_check(dist, elev, segments, traj_id, gap_mask=None, output_path=None):
    """
    Visualizes the raw flight line and how it was split.
    Gray = Raw Data
    Red x = Detected gaps
    Colors = Valid Segments (accepted for analysis)
    """
    plt.figure(figsize=(18, 6))
    
    # Create a copy for plotting that breaks at gaps
    plot_elev = elev.copy().astype(float)
    if gap_mask is not None:
        # Find the 'steps' again to know exactly where to break the line
        steps = np.diff(dist)
        # Set the end-point of every large jump to NaN for the line plot
        gap_breaks = np.where(steps > 2000)[0]
        for idx in gap_breaks:
            # Setting the segment immediately following a jump to NaN 
            # breaks the line without losing the red dots at the boundaries
            plot_elev[idx+1] = np.nan

    # 1. Plot the raw profile
    plt.plot(dist/1000, plot_elev, color='0.4', linewidth=0.8, label='Raw Data (with breaks)', alpha=0.5)
    
    # 2. Highlight detected gaps (Now marks both start and end)
    if gap_mask is not None and np.any(gap_mask):
        plt.scatter(dist[gap_mask]/1000, elev[gap_mask], 
                   color='red', marker='x', s=25, zorder=5, label=f'Gap Boundaries')

    # 3. Plot each accepted segment
    for i, (seg_data, seg_dist) in enumerate(segments):
        seg_elev = seg_data['bedrock_altitude (m)'].values
        plt.scatter(seg_dist/1000, seg_elev, s=15, label=f'Segment {i+1}')

    plt.xlabel('Distance along track (km)')
    plt.ylabel('Bed Elevation (m)')
    plt.title(f'Segmentation Check: {traj_id} ({len(segments)} valid segments)')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        save_path = os.path.join(output_path, f'{traj_id}.png')
    else:
        save_path = f'{traj_id}.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()  # Close figure to prevent memory issues


def plot_spectra(dist, detrended, wavelengths, psd, fitted_psd, beta, residual_psd, traj_id, dataset_name, segment_number=None, output_path=None):
    """
   Plots the spatial profile, power spectrum, and whitened residuals.
    The bottom plot (residuals) spans the full width.
    """
    fig = plt.figure(figsize=(20, 15))
    # Grid to place the 3 plots
    gs = fig.add_gridspec(2, 2)
    
    # Spatial Domain
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dist / 1000, detrended, 'k-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Distance along track (km)')
    ax1.set_ylabel('Detrended Bed Elevation (m)')
    segment_label = f' - Segment {segment_number}' if segment_number is not None else ''
    ax1.set_title(f'Spatial Profile: {traj_id}{segment_label}')
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Frequency Domain
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(wavelengths, psd, color='k', alpha=0.8, label='Power spectrum density')
    ax2.plot(wavelengths, fitted_psd, color='C1', label=fR'Power-law fit: $\beta$={beta:.1f}')
    ax2.set_xlabel('Wavelength (m)')
    ax2.set_ylabel('Power Spectral Density ($m^3$)')
    ax2.set_title('Power Spectrum')
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()
    
    # Whitened residuals
    ax3 = fig.add_subplot(gs[1, :])
    ax3.semilogx(wavelengths, residual_psd, color='k', alpha=0.5)

    # Highlight peaks
    peaks, _ = signal.find_peaks(residual_psd, height=peak_masking_height_threshold)
    if len(peaks) > 0:
        # Find min and max peaks
        peak_waves = wavelengths[peaks]
        peak_powers = residual_psd[peaks]
        idx_min = np.argmin(peak_waves)
        idx_max = np.argmax(peak_waves)

        # Longest wavelengths -> red
        ax3.scatter(peak_waves[idx_max], peak_powers[idx_max], color='C3', s=40, alpha=1, label=f'Max λ: {peak_waves[idx_max]:.0f}m')
        # Shortest wavelengths -> blue
        ax3.scatter(peak_waves[idx_min], peak_powers[idx_min], color='C0', s=40, alpha=1, label=f'Min λ: {peak_waves[idx_min]:.0f}m')
        ax3.legend()

    ax3.set_xlabel('Wavelength (m)')
    ax3.set_ylabel('Whitened PSD - ratio to trend')
    ax3.set_title('Whitened Residuals (Normalised)')
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    segment_suffix = f'_seg{segment_number}' if segment_number is not None else ''
    filename = f'psd_analysis_{dataset_name}_{traj_id}{segment_suffix}.png'
    if output_path:
        save_path = os.path.join(output_path, filename)
    else:
        save_path = filename
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()  # Close figure to prevent memory issues


def detect_data_gaps(distance, gap_threshold=2000):
    """
    Detect gaps in the data by looking at distance jumps.
    Returns a mask where True = points on either side of a gap.
    """
    steps = np.diff(distance)
    # find where the jump is too large
    gap_indices = np.where(steps > gap_threshold)[0]
    
    gap_mask = np.zeros(len(distance), dtype=bool)
    # Mark the start of the gap (last valid point before jump)
    gap_mask[gap_indices] = True
    # Mark the end of the gap (first valid point after jump)
    gap_mask[gap_indices + 1] = True
    
    return gap_mask


def split_into_segments(datafile, distance, gap_threshold=2000, min_segment_length=50, min_segment_km=10):
    """
    Separates and stores each data segment based on detected data gaps.
    Segments shorter than min_segment_length points or min segment_km are discarded to ensure sufficient spectral width for powerlaw fitting.
    """
    # Find gaps directly from distance steps
    steps = np.diff(distance)
    gap_indices = np.where(steps > gap_threshold)[0]

    # Build segment boundaries
    split_points = [0]
    for gap_idx in gap_indices:
        split_points.append(gap_idx + 1)  # End of current segment (exclusive)
        split_points.append(gap_idx + 1)  # Start of next segment
    split_points.append(len(distance))

    segments = []
    for i in range(0, len(split_points) - 1, 2):
        start = split_points[i]
        end = split_points[i + 1]
        length_km = (distance[end-1] - distance[start]) / 1000
        if end - start >= min_segment_length and length_km >= min_segment_km:
            print(f"    > Segment {len(segments)+1}: Rows {start} to {end} ({end-start} points), Length: {length_km:.2f} km")
            segments.append((datafile.iloc[start:end].copy(), distance[start:end]))

    return segments


def split_by_landscape(segment_data, segment_distance, smoothing_length=SMOOTHING_LENGTH,
                       gradient_threshold=GRADIENT_THRESHOLD,
                       min_segment_km=10, min_segment_pts=50):
    """
    Further splits a gap-free segment at landscape transitions detected by
    the gradient of the smoothed bedrock elevation profile.

    Transition zones (steep gradients) become their own segments rather than
    being discarded — e.g. basin | rise | highland | descent | basin.

    Returns list of (sub_segment_data, sub_segment_distance) tuples.
    """
    elev = segment_data['bedrock_altitude (m)'].values
    dist = segment_distance.copy()

    # Ensure strictly increasing distances (duplicates/reversals cause np.gradient div-by-zero)
    for i in range(1, len(dist)):
        if dist[i] <= dist[i - 1]:
            dist[i] = dist[i - 1] + 1e-3

    if len(dist) < 2:
        return [(segment_data, segment_distance)]

    # Smoothing kernel width in number of points
    dx_median = np.median(np.diff(dist))
    if dx_median <= 0:
        dx_median = 15.0
    kernel_pts = int(smoothing_length / dx_median)
    kernel_pts = max(3, kernel_pts if kernel_pts % 2 == 1 else kernel_pts + 1)

    smoothed = uniform_filter1d(elev, size=kernel_pts, mode='nearest')

    # Gradient in m/km
    grad = np.gradient(smoothed, dist / 1000)

    # Identify transition zones where |gradient| exceeds threshold
    in_transition = np.abs(grad) > gradient_threshold

    if not np.any(in_transition):
        return [(segment_data, segment_distance)]

    # Find edges of transition zones
    changes = np.diff(in_transition.astype(int))
    t_starts = np.where(changes == 1)[0] + 1   # first point in transition
    t_ends = np.where(changes == -1)[0] + 1     # first point after transition

    if in_transition[0]:
        t_starts = np.concatenate([[0], t_starts])
    if in_transition[-1]:
        t_ends = np.concatenate([t_ends, [len(in_transition)]])

    # Merge nearby transition zones — if the gap between two zones is < merge_gap_km,
    # treat them as one continuous transition
    merge_gap_km = 5.0
    merged_starts, merged_ends = [t_starts[0]], [t_ends[0]]
    for s, e in zip(t_starts[1:], t_ends[1:]):
        gap_km = (dist[s] - dist[merged_ends[-1]]) / 1000
        if gap_km < merge_gap_km:
            merged_ends[-1] = e  # extend the current zone
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    # Build boundaries: split AT the edges of transition zones so transitions
    # become their own segments: [0..t_start1] [t_start1..t_end1] [t_end1..t_start2] ...
    boundaries = set([0, len(dist)])
    for s, e in zip(merged_starts, merged_ends):
        boundaries.add(s)
        boundaries.add(e)
        peak_grad_idx = s + np.argmax(np.abs(grad[s:e]))
        print(f"      transition zone km {dist[s]/1000:.1f}-{dist[min(e,len(dist)-1)]/1000:.1f}, "
              f"peak gradient = {grad[peak_grad_idx]:.1f} m/km")
    boundaries = sorted(boundaries)

    # Build sub-segments (transitions and flat zones alike)
    sub_segments = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            continue
        length_km = (dist[e - 1] - dist[s]) / 1000
        if e - s >= min_segment_pts and length_km >= min_segment_km:
            sub_segments.append((segment_data.iloc[s:e].copy(), dist[s:e]))

    if not sub_segments:
        return [(segment_data, segment_distance)]

    return sub_segments


def flag_wavelength_confidence(wavelengths, profile_length, min_cycles=2.0):
    """
    Categorizes detected peaks by statistical reliability based on the 
    profile length (L). Wavelengths > L/2 are geologically valid but 
    statistically 'unconfirmed' as periodicities.
    """
    threshold = profile_length / min_cycles
    
    # Handle empty arrays from find_peaks
    if len(wavelengths) == 0:
        return {'confirmed': [], 'candidate': [], 'threshold': threshold}
        
    confirmed = wavelengths[wavelengths <= threshold]
    candidate = wavelengths[wavelengths > threshold]
    
    return {
        'confirmed': confirmed.tolist(),
        'candidate': candidate.tolist(),
        'threshold': threshold
    }


def calculate_flow_incidence(x, y, flow_x, flow_y):
    """
    Calculates angle between Flight Line and Flow Vector.
    Returns: Angle in degrees [0=Parallel, 90=Perpendicular]
    """
    # Flight Tangent (Direction of the plane)
    flight_dx = np.gradient(x)
    flight_dy = np.gradient(y)
    flight_mag = np.sqrt(flight_dx**2 + flight_dy**2)
    
    # Dot Product
    dot = flight_dx * flow_x + flight_dy * flow_y
    
    # Clamp for arccos if floating point error
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = dot / (flight_mag * 1.0) # flow is already normalized
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cos_theta))
    # Fold to [0, 90] (Roughness is the same upstream or downstream)
    return np.minimum(angle, 180 - angle)


def analyse_sliding_windows(dist, elev, incidence_array, window_size, step_size, flow_angular_diff=None, flow_speed=None):
    """
    Slides a window across the segment to capture local morphometrics AND 
    build a robust average spectrum.
    """
    # 1. Define Frequency Grid (Global)
    # We fix this so we can average the PSDs later
    dx_median = np.median(np.diff(dist)) if len(dist) > 1 else 100
    if dx_median == 0: dx_median = 15.0 # Safety fallback
    
    # Nyquist limit
    max_freq = 1 / (2 * max(dx_median, 15.0)) 
    # Min freq is based on the WINDOW size, not the full segment
    min_freq = 1 / window_size 
    
    # Generate geometric frequencies
    freqs = np.geomspace(min_freq, max_freq, num=500)
    angular_freqs = freqs * 2 * np.pi
    
    # Wavelength for per-window beta fitting
    wavelengths_calc = 1 / freqs

    # Create mask for geologically relevant wavelength range (250km to 50km)
    mask = (wavelengths_calc >= 250) & (wavelengths_calc <= 50000)

    # Fit power law: P(f) = A * f^(-β)
    log_freqs = np.log10(freqs)

    psd_accumulator = []
    large_features = []
    
    # 2. Slide the Window
    start_dist = dist.min()
    max_dist = dist.max()
    
    current_start = start_dist
    window_idx = 0
    
    while current_start + window_size <= max_dist + 1e-6:  # small epsilon
        current_end = current_start + window_size
        
        # Mask data for this window
        fit_mask = (dist >= current_start) & (dist <= current_end)
        w_dist = dist[fit_mask]
        w_elev = elev[fit_mask]
        
        # Basic checks
        if len(w_dist) > 50: # Ensure enough points
            
            # A. DETREND LOCALLY
            # This removes the "slope" of the valley wall specific to this window
            w_detrended = signal.detrend(w_elev)

            # B. APPLY TAPER (reduces spectral leakage at window edges)
            if WINDOW_TYPE == 'hann':
                taper = signal.windows.hann(len(w_detrended))
                w_detrended = w_detrended * taper
            elif WINDOW_TYPE == 'tukey':
                taper = signal.windows.tukey(len(w_detrended), alpha=0.5)
                w_detrended = w_detrended * taper
            # else: rectangular (no taper)

            # C. SPECTRAL ANALYSIS (For Texture/Beta)
            pgram = signal.lombscargle(w_dist, w_detrended, angular_freqs, normalize=False)

            psd_accumulator.append(pgram)
            
            # C. Calculate per-window beta (power law exponent)
            window_beta = np.nan
            window_beta_uncertainty = np.nan
            if np.sum(mask) >= 2 and np.all(pgram > 0):
                log_psd = np.log10(pgram)
                try:
                    n_fit = np.sum(mask)
                    if n_fit > 2:
                        coeffs, cov = np.polyfit(log_freqs[mask], log_psd[mask], 1, cov=True)
                        window_beta_uncertainty = np.sqrt(cov[0, 0])
                    else:
                        coeffs = np.polyfit(log_freqs[mask], log_psd[mask], 1)
                    window_beta = -coeffs[0]
                except:
                    pass

            window_hurst = (window_beta - 1) / 2
            window_hurst_uncertainty = window_beta_uncertainty / 2

            # D. MORPHOMETRICS (For "Big Mountains")
            local_relief = np.max(w_elev) - np.min(w_elev)

            feature_stats = {
                'window_id': window_idx,
                'start_km': current_start / 1000,
                'end_km': current_end / 1000,
                'local_relief_m': local_relief,
                'bed_elev_mean': np.mean(w_elev),
                'roughness_rms': np.sqrt(np.mean(w_detrended**2)),
                'window_beta': window_beta,
                'window_beta_uncertainty': window_beta_uncertainty,
                'window_hurst': window_hurst,
                'window_hurst_uncertainty': window_hurst_uncertainty
            }

            # Extract the point-by-point incidence for just this window
            window_incidence = incidence_array[fit_mask]

            # Calculate window specific stats
            mean_window_incidence = np.nanmean(window_incidence)

            # add to feature_stats dictionary
            feature_stats['mean_window_incidence'] = mean_window_incidence

            # MEaSUREs flow validation stats for this window
            if flow_angular_diff is not None:
                w_flow_diff = flow_angular_diff[fit_mask]
                feature_stats['flow_error_mean'] = np.nanmean(w_flow_diff)
                feature_stats['flow_error_median'] = np.nanmedian(w_flow_diff)
            else:
                feature_stats['flow_error_mean'] = np.nan
                feature_stats['flow_error_median'] = np.nan

            if flow_speed is not None:
                w_speed = flow_speed[fit_mask]
                feature_stats['measures_speed_mean'] = np.nanmean(w_speed)
            else:
                feature_stats['measures_speed_mean'] = np.nan

            large_features.append(feature_stats)
            
        current_start += step_size
        window_idx += 1
        
    # 3. Average the PSDs
    if psd_accumulator:
        psd_stack = np.array(psd_accumulator)
        avg_psd = np.mean(psd_stack, axis=0)
        log_psd_std = np.std(np.log10(psd_stack), axis=0)
        log_psd_std[log_psd_std == 0] = np.inf  # zero std -> zero weight
        psd_weights = 1.0 / log_psd_std
    else:
        avg_psd = None
        psd_weights = None

    return avg_psd, freqs, large_features, dx_median, psd_weights                                 
                                                                                        

def analyse_bedrock():
    """
    Statistical spectral profiling of radar flight datasets of Antarctic bedrock elevation
    """
    # Setup projection transformer: WGS84 (Lat/Lon) -> Antarctic Stereo (Meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    datasets_bundle = load_datasets()

    # paths
    base_path = 'all_data/'
    dem_path = os.path.join(base_path, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

    all_results = {}

    for bundle in datasets_bundle:
        dataset_name = bundle['name']
        df = bundle['data']
        print(f"\nStarting analysis of {dataset_name}...")

        # Create output directory structure for this dataset
        region_folder = get_region_folder(dataset_name)
        window_suffix = f"_w{WINDOW_SIZE // 1000}km" if WINDOW_TYPE == 'rectangular' else f"_w{WINDOW_SIZE // 1000}km_{WINDOW_TYPE}"
        region_folder = f"{region_folder}{window_suffix}"
        output_paths = ensure_output_dirs(OUTPUT_BASE_PATH, region_folder)
        print(f"  Output folder: {output_paths['region']}")

        # Filter invalid data
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        print(f"  Valid data points: {len(valid)}")
        print(f"  Unique trajectories: {len(valid['trajectory_id'].unique())}")

        results = {}
        plot_count = 0  # Counter to limit number of plots popping up

        for j, traj_id in enumerate(valid['trajectory_id'].unique()):
            if j > 0 and j % 10 == 0: # print every 10 trajectories
                print(f"  Processed {j} trajectories")

            line = valid[valid['trajectory_id'] == traj_id].copy()
            
            if len(line) < 20: continue 

            # COORDINATE TRANSFORMATION
            longs = line['longitude (degree_east)'].values
            lats = line['latitude (degree_north)'].values

            # Convert to Meters (EPSG:3031)
            # Note: pyproj expects (Long, Lat) order for x, y output
            x, y = transformer.transform(longs, lats)
            
            # Vectorized Distance Calculation (Euclidean in projected space)
            dx_seg = np.diff(x)
            dy_seg = np.diff(y)
            segment_distances = np.sqrt(dx_seg**2 + dy_seg**2)
            
            # Cumulative distance starting at 0
            dist = np.concatenate([[0], np.cumsum(segment_distances)])

            # SIGNAL PROCESSING with segmentation
            elev = line['bedrock_altitude (m)'].values
            
            # Detect gaps in the data
            gap_mask = detect_data_gaps(dist)
            n_gaps = np.sum(gap_mask) // 2
            print(f"  {traj_id}: Found {n_gaps} gaps (>=2000m) in data")
            
            # Split into valid segments (gap-based)
            gap_segments = split_into_segments(line, dist)

            if not gap_segments:
                print(f" Skipping trajectory {traj_id}: no valid segments found")
                continue

            # Further split at landscape transitions (elevation gradient)
            segments = []
            for seg_data, seg_dist in gap_segments:
                sub_segs = split_by_landscape(seg_data, seg_dist)
                segments.extend(sub_segs)

            n_landscape_splits = len(segments) - len(gap_segments)
            print(f"{len(gap_segments)} gap segments -> {len(segments)} after landscape splitting"
                  + (f" (+{n_landscape_splits} landscape splits)" if n_landscape_splits > 0 else ""))

            # Process each segment separately 
            valid_segments = []
            segment_results = []

            for seg_idx, (segment_data, segment_distance) in enumerate(segments):
                bedrock_segment_elev = segment_data['bedrock_altitude (m)'].values

                detrended = signal.detrend(bedrock_segment_elev)

                seg_lons = segment_data['longitude (degree_east)'].values
                seg_lats = segment_data['latitude (degree_north)'].values
                seg_x, seg_y = transformer.transform(seg_lons, seg_lats)

                # Thickness calculation and validity check
                surface_elevs = extract_rema_elevation(seg_x, seg_y, dem_path)
                valid_ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_segment_elev)
                thickness_validity = np.sum(~np.isnan(valid_ice_thickness)) / len(valid_ice_thickness)
                if thickness_validity < 0.20: # If less than 20% thickness data is valid
                    print(f" Skipping Segment {seg_idx+1}: Insufficient thickness data (only {thickness_validity*100:.1f}% is valid)")
                    continue

                print(f" > Segment {seg_idx+1}: Valid ice thickness for count: {np.sum(~np.isnan(valid_ice_thickness))} / {len(valid_ice_thickness)}")

                global_relief = bedrock_segment_elev.max() - bedrock_segment_elev.min()

                # append valid segment to list 
                valid_segments.append((segment_data, segment_distance))

                # Window parameters defined at top of script

                # 1. Get Flow Direction from REMA (Smoothed)
                vx, vy = extract_rema_flow_vector(seg_x, seg_y, dem_path, valid_ice_thickness)
                # If we don't know the thickness, we don't know the smoothing scale.
                # Force these to NaN so they don't count as 90 degree (perpendicular) flow.
                invalid_mask = np.isnan(valid_ice_thickness)
                vx[invalid_mask] = np.nan
                vy[invalid_mask] = np.nan

                # MEaSUREs validation call:
                angular_diff, measures_speed = MEaSUREs_comparison(seg_x, seg_y, vx, vy)
                print(f"Flow validation: mean diff = {np.nanmean(angular_diff):.1f}°, median ={np.nanmedian(angular_diff):.1f}°, mean speed = {np.nanmean(measures_speed):.1f} m/yr")

                # 2. Calculate Incidence_array
                incidence_array = calculate_flow_incidence(seg_x, seg_y, vx, vy) # array
                mean_incidence = np.nanmean(incidence_array)

                # Check if segment is long enough for at least one window  
                segment_len_m = segment_distance.max() - segment_distance.min()  

                if segment_len_m < WINDOW_SIZE:
                    # fallback if segment is valid and short treat the whole segment as one window
                    avg_psd, freqs, window_stats, dx_median, psd_weights = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev, incidence_array,
                        window_size=segment_len_m, step_size=segment_len_m,
                        flow_angular_diff=angular_diff, flow_speed=measures_speed
                    )

                else:
                    # Standard processing
                    avg_psd, freqs, window_stats, dx_median, psd_weights = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev, incidence_array,
                        window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                        flow_angular_diff=angular_diff, flow_speed=measures_speed
                    )

                # Identifying largest features found in windows
                if window_stats:
                    # Find the window with the highest vertical relief
                    max_relief_window = max(window_stats, key= lambda x: x['local_relief_m'])
                    max_local_relief = max_relief_window['local_relief_m']
                    loc_of_max_relief = max_relief_window['start_km']

                    # average (RMS) roughness accross the whole segment
                    avg_rms_roughness = np.mean([w['roughness_rms'] for w in window_stats])

                else:
                    max_local_relief = 0
                    loc_of_max_relief = 0
                    avg_rms_roughness = 0

                print(f" >>>>>>>>>: {dataset_name} | {traj_id} | Segment {seg_idx+1}: mean incidence {mean_incidence:.1f}°")

                stats_dict = {
                    'elevation_range': global_relief,
                    'elevation_min': float(bedrock_segment_elev.min()),
                    'elevation_max': float(bedrock_segment_elev.max()),
                    'max_local_relief': max_local_relief,
                    'loc_of_max_relief': loc_of_max_relief,
                    'rms_roughness': avg_rms_roughness,
                    'skewness': stats.skew(detrended, bias=False),
                    'kurtosis': stats.kurtosis(detrended, bias=False),
                    'ice_thickness_mean': np.nanmean(valid_ice_thickness),
                    'ice_thickness_range': np.nanmax(valid_ice_thickness) - np.nanmin(valid_ice_thickness),
                    'flow_incidence_deg': mean_incidence,
                    'flow_error_mean': np.nanmean(angular_diff),
                    'flow_error_median': np.nanmedian(angular_diff),
                    'measures_speed_mean': np.nanmean(measures_speed),
                    'window_stats': [{**w, 'segment': seg_idx + 1} for w in window_stats]
                }
                
                # SPECTRAL ANALYSIS
                # Guard against zero PSD (no valid spectral windows were processed)
                if avg_psd is None or np.all(avg_psd == 0) or np.any(avg_psd < 0):
                    print(f"  Skipping segment {seg_idx+1} spectral fit: Invalid PSD values")
                    segment_results.append(stats_dict)
                    continue

                # Wavelengths
                wavelengths_calc = 1 / freqs

                # Create mask for geologically relevant wavelength range (250m to 50km)
                fit_mask = (wavelengths_calc >= 250) & (wavelengths_calc <= 50000)
                
                # If the mask is empty (or has too few points), skip the fit
                if np.sum(fit_mask) >= 2:

                    # Fit power law: P(f) = A * f^(-β)
                    log_freqs = np.log10(freqs)
                    log_psd = np.log10(avg_psd)

                    # PASS 1: FInd dominant waves
                    slope_init, intercept_init = np.polyfit(log_freqs[fit_mask], log_psd[fit_mask], 1)
                    fitted_psd_init = 10 ** (intercept_init + slope_init * np.log10(freqs))

                    # Calculate residuals to find peaks
                    residual_psd = avg_psd / fitted_psd_init
                    peaks, _ = signal.find_peaks(residual_psd, height=peak_masking_height_threshold)

                    # PASS 2: Mask large peaks for "texture only" fit
                    clean_mask = fit_mask.copy()
                    if len(peaks) > 0:
                        for p_idx in peaks:
                            # mask out small buffer around the peak to remove edge effects
                            start = max(0, p_idx - bin_buffer)
                            end = min(len(clean_mask), p_idx + bin_buffer + 1)
                            clean_mask[start:end] = False

                    # REFIT
                    if np.sum(clean_mask) >= 2:
                        # Fit only on masked data
                        # Use the spread across windows as weights:
                        psd_weights[~np.isfinite(psd_weights)] = 0
                        w = psd_weights[clean_mask]
                        # If weights are all zero (e.g. single window → zero std), fall back to unweighted fit
                        if np.all(w == 0):
                            w = None
                        (slope, intercept), cov = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1, w=w, cov=True)

                        beta = -slope # Power law exponent
                        beta_uncertainty = np.sqrt(cov[0, 0]) # beta std error

                        # Apply fit to the full range
                        fitted_psd = 10**(intercept + slope * log_freqs)
                        residual_psd = avg_psd / fitted_psd

                    else: # fallback
                        beta = -slope_init
                        beta_uncertainty = np.nan
                        fitted_psd = fitted_psd_init

                    dominant_wavelengths = wavelengths_calc[peaks] if len(peaks) > 0 else []

                    profile_length = segment_distance.max() - segment_distance.min()
                    confidence_flags = flag_wavelength_confidence(dominant_wavelengths, profile_length)

                    # Calculate Hurst exponent from spectral exponent
                    # For 1D profiles: β = 2H + 1, so H = (β - 1) / 2
                    hurst_exponent = (beta - 1) / 2
                    # uncertainty
                    hurst_uncertainty = beta_uncertainty / 2
                    
                    stats_dict.update({
                        'median_spacing': dx_median,
                        'profile_length': profile_length,
                        'dominant_wavelengths': dominant_wavelengths,
                        'confirmed_wavelengths': confidence_flags['confirmed'],
                        'candidate_wavelengths': confidence_flags['candidate'],
                        'confidence_threshold': confidence_flags['threshold'],
                        'power_law_exponent': beta,
                        'beta_uncertainty': beta_uncertainty, 
                        'hurst_exponent': hurst_exponent,
                        'hurst_uncertainty': hurst_uncertainty
                    })

                    # Plot the first n lines
                    if plot_count < 10:
                        plot_spectra(segment_distance, detrended, wavelengths_calc, avg_psd, fitted_psd, beta, residual_psd, traj_id, dataset_name, segment_number=seg_idx+1, output_path=output_paths['psd'])
                        plot_count += 1

                else:
                    print(f"Skipping Line {traj_id}: Not enough data points in 250m–50km range.")

                segment_results.append(stats_dict)

            if valid_segments:
                plot_raw_data_with_segmentation_check(dist, elev, valid_segments, traj_id, gap_mask, output_path=output_paths['trajectories'])

            if segment_results:
                # Aggregate statistics
                combined_stats = {}
                # Keys that are ALREADY lists inside the segment dict
                list_keys = ['dominant_wavelengths', 'confirmed_wavelengths', 'candidate_wavelengths', 'window_stats']
                
                # Keys that are SINGLE VALUES in the segment dict, but we want to KEEP as a list 
                list_keys_collect = ['power_law_exponent', 'hurst_exponent', 'beta_uncertainty', 'hurst_uncertainty', 'flow_incidence_deg', 'flow_error_mean', 'flow_error_median', 'measures_speed_mean', 'elevation_min', 'elevation_max']

                for key in segment_results[0].keys():
                    # 1. Extract values for the CURRENT key immediately
                    values = [seg[key] for seg in segment_results if key in seg]

                    if key == 'max_local_relief':
                        # Take the MAX of the maxes, not the mean
                        combined_stats[key] = np.max(values)
                        # Find which segment had that max to get the correct location
                        idx_of_max = np.argmax(values)
                        combined_stats['loc_of_max_relief'] = segment_results[idx_of_max]['loc_of_max_relief']
                        continue # Skip the standard averaging below
                    
                    if key == 'loc_of_max_relief':
                        continue # Skip this, handled above

                    if key in list_keys:
                        # FLATTEN lists (e.g. [[10, 20], [30]] -> [10, 20, 30])
                        combined_stats[key] = [w for seg in segment_results for w in seg.get(key, [])]
                    
                    elif key in list_keys_collect:
                        # COLLECT values (e.g. [1.7, 2.2, 1.8])
                        combined_stats[key] = values 
                    
                    elif key == 'profile_length':
                        combined_stats[key] = np.mean(values)

                    # Flow Orientation
                    elif isinstance(values[0], str):
                        # Calculate Mode (Most common string)
                        if values:
                            combined_stats[key] = max(set(values), key=values.count)

                    else:
                        # Average other stats
                        if values: # Check if not empty
                            combined_stats[key] = np.mean(values)

                results[traj_id] = combined_stats
                print(f"  Trajectory {traj_id}: {len(segments)} segments, combined median spacing = {combined_stats.get('median_spacing', 0):.1f}m, Nyquist = {2*combined_stats.get('median_spacing', 0):.1f}m")
            
        all_results[dataset_name] = results

        print(f"{dataset_name} is finished processing")

    return all_results


def results_summary(results):
    if not results: return "no valid data found :("

    def format_stat(values, unit=""):
        """Helper to formatting stats without confusing ranges for single values"""
        if not values: return "N/A"
        
        # Safety check for non-numeric data passed to format_stat
        if isinstance(values[0], str):
            return "N/A (String Data)"

        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        if min_val == max_val:
            return f"{mean_val:.1f}{unit} (Single Value)"
        else:
            return f"Mean: {mean_val:.1f}{unit} | Range: [{min_val:.1f}, {max_val:.1f}]{unit}"

    print("-" * 60)
    print(f"  RESULTS SUMMARY ({len(results)} trajectories aggregated)")
    print("-" * 60)

    # 1. Vertical Relief
    reliefs = [r['elevation_range'] for r in results.values() if 'elevation_range' in r]
    print(f"VERTICAL RELIEF (Max-Min):\n  -> {format_stat(reliefs, 'm')}")

    # 1b. Absolute Elevation (elevation_min/max are now per-segment lists)
    elev_mins = [v for r in results.values() for v in r.get('elevation_min', [])]
    elev_maxs = [v for r in results.values() for v in r.get('elevation_max', [])]
    if elev_mins and elev_maxs:
        print(f"ABSOLUTE BED ELEVATION:\n  -> Min: {np.nanmin(elev_mins):.1f}m | Max: {np.nanmax(elev_maxs):.1f}m  (across all trajectories)")

    # 2. Segment Lengths
    lengths = [r['profile_length'] for r in results.values() if 'profile_length' in r]
    print(f"AVG SEGMENT LENGTH:\n  -> {format_stat(lengths, 'm')}")

    # 3b. Skewness and Kurtosis
    skews = [r['skewness'] for r in results.values() if 'skewness' in r and np.isfinite(r['skewness'])]
    kurts = [r['kurtosis'] for r in results.values() if 'kurtosis' in r and np.isfinite(r['kurtosis'])]
    if skews:
        print(f"SKEWNESS:\n  -> {format_stat(skews)}")
    if kurts:
        print(f"KURTOSIS (excess):\n  -> {format_stat(kurts)}")

    # 4. Ice Thickness
    thickness = [r['ice_thickness_mean'] for r in results.values() if 'ice_thickness_mean' in r and not np.isnan(r['ice_thickness_mean'])]
    if thickness:
        print(f"MEAN ICE THICKNESS:\n  -> {format_stat(thickness, 'm')}")

    print("." * 60)

    # 6. Wavelength
    conf = [w for r in results.values() for w in r.get('confirmed_wavelengths', [])]
    cand = [w for r in results.values() for w in r.get('candidate_wavelengths', [])]

    if conf:
        print(f"CONFIRMED WAVELENGTHS (Physically valid < L/2):")
        print(f"  -> Count: {len(conf)}")
        print(f"  -> {format_stat(conf, 'm')}")
    else:
        print("CONFIRMED WAVELENGTHS: None found.")

    if cand:
        print(f"CANDIDATE WAVELENGTHS (Statistically present > L/2):")
        print(f"  -> Count: {len(cand)}")
        print(f"  -> Range: [{min(cand):.0f}m, {max(cand):.0f}m]")
    
    if not conf and not cand:
        print("  -> Topography appears Scale Invariant (Fractal/No dominant peaks)")

    # 7. Largest Detected Structures (The "Big Mountains")
    max_reliefs = [r['max_local_relief'] for r in results.values() if 'max_local_relief' in r]
    locs = [r['loc_of_max_relief'] for r in results.values() if 'loc_of_max_relief' in r]
    
    if max_reliefs:
        print("." * 60)
        print(f"LARGEST LOCAL STRUCTURES ({WINDOW_SIZE/1000:.0f}km Window):")
        # Zip them to print pairs
        for relief, loc in zip(max_reliefs, locs):
            print(f"  -> Relief: {relief:.1f}m at km {loc:.1f}")
        print("." * 60)
    print("=" * 60)
    return {}


if __name__=="__main__":

    log_path = os.path.join(OUTPUT_BASE_PATH, 'bed_analysis_log.txt')
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(log_path)

    results = analyse_bedrock()
    print(f"\n---")
    print(f"Analysed {len(results)} regions")

    for region_name, region_results in results.items():                                       
        print(f"\n=== {region_name} SUMMARY ===")                                             
        results_summary(region_results)                                                       

        # --- Window-level CSV ---
        all_window_data = []
        for traj_id, traj_data in region_results.items():
            for w in traj_data.get('window_stats', []):
                all_window_data.append({
                    'trajectory': traj_id,
                    'segment': w.get('segment'),
                    'window_id': w.get('window_id'),
                    'incidence_deg': w.get('mean_window_incidence'),
                    'beta': w.get('window_beta'),
                    'beta_uncertainty': w.get('window_beta_uncertainty'),
                    'hurst': w.get('window_hurst'),
                    'hurst_uncertainty': w.get('window_hurst_uncertainty'),
                    'relief_m': w.get('local_relief_m'),
                    'bed_elev_mean': w.get('bed_elev_mean'),
                    'rms_roughness': w.get('roughness_rms'),
                    'flow_error_mean': w.get('flow_error_mean'),
                    'flow_error_median': w.get('flow_error_median'),
                    'measures_speed_mean': w.get('measures_speed_mean')
                })
        csv_suffix = f"_w{WINDOW_SIZE // 1000}km" if WINDOW_TYPE == 'rectangular' else f"_w{WINDOW_SIZE // 1000}km_{WINDOW_TYPE}"
        region_output = os.path.join(OUTPUT_BASE_PATH, f'{get_region_folder(region_name)}{csv_suffix}')
        window_csv_dir = os.path.join(OUTPUT_BASE_PATH, 'window_csvs')
        os.makedirs(window_csv_dir, exist_ok=True)
        pd.DataFrame(all_window_data).to_csv(os.path.join(window_csv_dir, f'{region_name}{csv_suffix}_window_stats.csv'), index=False)

        # --- Segment-level CSV (for cos²θ regression) ---
        # Build window DF first so we can derive per-segment distribution stats
        window_df = pd.DataFrame(all_window_data)

        all_segment_data = []
        for traj_id, traj_data in region_results.items():
            betas = traj_data.get('power_law_exponent', [])
            beta_uncerts = traj_data.get('beta_uncertainty', [])
            incidences = traj_data.get('flow_incidence_deg', [])
            hursts = traj_data.get('hurst_exponent', [])
            hurst_uncerts = traj_data.get('hurst_uncertainty', [])
            flow_err_means = traj_data.get('flow_error_mean', [])
            flow_err_medians = traj_data.get('flow_error_median', [])
            speed_means = traj_data.get('measures_speed_mean', [])
            elev_mins = traj_data.get('elevation_min', [])
            elev_maxs = traj_data.get('elevation_max', [])

            n_segs = min(len(betas), len(incidences))
            for i in range(n_segs):
                seg_num = i + 1
                row = {
                    'trajectory': traj_id,
                    'segment': seg_num,
                    'incidence_deg': incidences[i],
                    'beta': betas[i],
                    'beta_uncertainty': beta_uncerts[i] if i < len(beta_uncerts) else np.nan,
                    'hurst': hursts[i] if i < len(hursts) else np.nan,
                    'hurst_uncertainty': hurst_uncerts[i] if i < len(hurst_uncerts) else np.nan,
                    'flow_error_mean': flow_err_means[i] if i < len(flow_err_means) else np.nan,
                    'flow_error_median': flow_err_medians[i] if i < len(flow_err_medians) else np.nan,
                    'measures_speed_mean': speed_means[i] if i < len(speed_means) else np.nan,
                    'elevation_min': elev_mins[i] if i < len(elev_mins) else np.nan,
                    'elevation_max': elev_maxs[i] if i < len(elev_maxs) else np.nan,
                }

                # Window-beta distribution stats for this segment
                if len(window_df) > 0:
                    wm = window_df[(window_df['trajectory'] == traj_id) & (window_df['segment'] == seg_num)]
                    wb = wm['beta'].dropna()
                    row['n_windows'] = len(wb)
                    row['beta_median'] = wb.median() if len(wb) > 0 else np.nan
                    if len(wb) > 1:
                        row['beta_iqr'] = wb.quantile(0.75) - wb.quantile(0.25)
                    else:
                        row['beta_iqr'] = np.nan
                    row['relief_median'] = wm['relief_m'].median() if len(wm) > 0 else np.nan
                else:
                    row.update({'n_windows': 0, 'beta_median': np.nan, 'beta_iqr': np.nan, 'relief_median': np.nan})

                all_segment_data.append(row)

        segment_csv_dir = os.path.join(OUTPUT_BASE_PATH, 'segment_csvs')
        os.makedirs(segment_csv_dir, exist_ok=True)
        pd.DataFrame(all_segment_data).to_csv(os.path.join(segment_csv_dir, f'{region_name}{csv_suffix}_segment_stats.csv'), index=False)

        # --- Wavelength detections CSV (for threshold sensitivity analysis) ---
        all_wavelength_data = []
        for traj_id, traj_data in region_results.items():
            for wl in traj_data.get('confirmed_wavelengths', []):
                all_wavelength_data.append({
                    'trajectory': traj_id,
                    'wavelength_m': wl,
                    'type': 'confirmed'
                })
            for wl in traj_data.get('candidate_wavelengths', []):
                all_wavelength_data.append({
                    'trajectory': traj_id,
                    'wavelength_m': wl,
                    'type': 'candidate'
                })
        pd.DataFrame(all_wavelength_data).to_csv(os.path.join(region_output, f'{region_name}{csv_suffix}_wavelength_detections.csv'), index=False)

        print(f"Exported {len(all_window_data)} window rows, {len(all_segment_data)} segment rows, {len(all_wavelength_data)} wavelength detections")
