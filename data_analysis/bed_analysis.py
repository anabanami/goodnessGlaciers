import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from pyproj import Transformer
import os
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness


def load_datasets():
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
        {
          'file': 'AWI_2018_ANIRES_AIR_BM3.csv',
          'label': 'DML_AniRES'
         },   # Dronning Maud Land
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


def plot_raw_data_with_segmentation_check(dist, elev, segments, traj_id, gap_mask=None):
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
    plt.savefig(f'{traj_id}.png', dpi=500, bbox_inches='tight')
    # plt.show()


def plot_spectra(dist, detrended, wavelengths, psd, fitted_psd, beta, residual_psd, traj_id, dataset_name, segment_number=None):
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
    peaks, _ = signal.find_peaks(residual_psd, height=2.0)
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
    plt.savefig(f'psd_analysis_{dataset_name}_{traj_id}{segment_suffix}.png', dpi=500, bbox_inches='tight')
    # plt.show()


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


def split_into_segments(datafile, distance, gap_threshold=2000, min_segment_length=50):
    """
    Split data into valid segments based on distance gaps.
    """
    # Use the existing function to find gap locations
    gap_mask = detect_data_gaps(distance, gap_threshold)
    gap_indices = np.where(gap_mask)[0]
    
    # Define start and end indices for slicing
    split_points = np.concatenate([[0], gap_indices, [len(distance)]])
    
    segments = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        if end - start >= min_segment_length:
            print(f"    > Segment {len(segments)+1}: Rows {start} to {end} ({end-start} points), Length: {(distance[end-1]-distance[start])/1000:.2f} km")
            segments.append((datafile.iloc[start:end].copy(), distance[start:end]))
    
    return segments


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


def analyse_sliding_windows(dist, elev, window_size=50000, step_size=25000):
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
        mask = (dist >= current_start) & (dist <= current_end)
        w_dist = dist[mask]
        w_elev = elev[mask]
        
        # Basic checks
        if len(w_dist) > 50: # Ensure enough points
            
            # A. DETREND LOCALLY
            # This removes the "slope" of the valley wall specific to this window
            w_detrended = signal.detrend(w_elev)
            
            # B. SPECTRAL ANALYSIS (For Texture/Beta)
            pgram = signal.lombscargle(w_dist, w_detrended, angular_freqs, normalize=False)
            psd_accumulator.append(pgram)
            
            # C. MORPHOMETRICS (For "Big Mountains")
            local_relief = np.max(w_elev) - np.min(w_elev)
            
            feature_stats = {
                'window_id': window_idx,
                'start_km': current_start / 1000,
                'end_km': current_end / 1000,
                'local_relief_m': local_relief,
                'roughness_rms': np.sqrt(np.mean(w_detrended**2))
            }
            large_features.append(feature_stats)
            
        current_start += step_size
        window_idx += 1
        
    # 3. Average the PSDs
    if psd_accumulator:
        avg_psd = np.mean(psd_accumulator, axis=0)
    else:
        avg_psd = None
        
    return avg_psd, freqs, large_features, dx_median


def analyse_bedrock():
    """
    Statistical spectral profiling of radar flight datasets of Antarctic bedrock elevation
    """
    # Setup projection transformer: WGS84 (Lat/Lon) -> Antarctic Stereo (Meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    datasets_bundle = load_datasets()

    # paths
    base_path = 'shortcut_to_culled-data/'
    dem_path = os.path.join(base_path, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')


    all_results = {}

    for bundle in datasets_bundle:
        dataset_name = bundle['name']
        df = bundle['data']
        print(f"\nStarting analysis of {dataset_name}...")

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
            
            # Split into valid segments
            segments = split_into_segments(line, dist)

            if len(segments) != 0:
                print(f"{len(segments)} data segments found")
            else:
                print(f" Skipping trajectory {traj_id}: no valid segments found")
                continue

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

                # Define window parameters
                WINDOW_SIZE = 50000
                STEP_SIZE = 10000

                # Check if segment is long enough for at least one window
                segment_len_m = segment_distance.max() - segment_distance.min()

                if segment_len_m < WINDOW_SIZE:
                    # fallback if segment is valid and short treat the whole segment as one window
                    avg_psd, freqs, window_stats, dx_median = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev,
                        window_size=segment_len_m, step_size=segment_len_m
                    )

                else:
                    # Standard processing
                    avg_psd, freqs, window_stats, dx_median = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev,
                        window_size=WINDOW_SIZE, step_size=STEP_SIZE
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


                # 1. Get Flow Direction from REMA (Smoothed)
                vx, vy = extract_rema_flow_vector(seg_x, seg_y, dem_path, valid_ice_thickness)

                # If we don't know the thickness, we don't know the smoothing scale.
                # Force these to NaN so they don't count as 90 degree (perpendicular) flow.
                invalid_mask = np.isnan(valid_ice_thickness)
                vx[invalid_mask] = np.nan
                vy[invalid_mask] = np.nan

                # 2. Calculate Incidence
                incidence = calculate_flow_incidence(seg_x, seg_y, vx, vy)
                mean_incidence = np.nanmean(incidence)
                # 3. Categorize
                flow_orientation = "Oblique"
                if mean_incidence < 30: flow_orientation = "Parallel"
                elif mean_incidence > 60: flow_orientation = "Perpendicular"
                print(f" >>>>>>>>>: {dataset_name} | {traj_id} | Segment {seg_idx+1}: {flow_orientation} (incidence: {mean_incidence:.1f})")

                stats_dict = {
                    'elevation_range': global_relief,
                    'max_local_relief': max_local_relief,
                    'loc_of_max_relief': loc_of_max_relief,
                    'rms_roughness': avg_rms_roughness,
                    'skewness': stats.skew(detrended, bias=False),
                    'kurtosis': stats.kurtosis(detrended, bias=False),
                    'ice_thickness_mean': np.nanmean(valid_ice_thickness),
                    'ice_thickness_range': np.nanmax(valid_ice_thickness) - np.nanmin(valid_ice_thickness),
                    'flow_incidence_deg': mean_incidence,
                    'flow_orientation': flow_orientation
                }
                
                # SPECTRAL ANALYSIS
                # PSD units
                pos_freqs = freqs
                pos_psd = avg_psd

                # Guard against zero PSD (no valid spectral windows were processed)
                if pos_psd is None or np.all(pos_psd == 0) or np.any(pos_psd < 0):
                    print(f"  Skipping segment {seg_idx+1} spectral fit: Invalid PSD values")
                    segment_results.append(stats_dict)
                    continue

                # Wavelengths
                wavelengths_calc = 1 / pos_freqs

                # Create mask for geologically relevant wavelength range (250km to 50km)
                mask = (wavelengths_calc >= 250) & (wavelengths_calc <= 50000)
                
                # If the mask is empty (or has too few points), skip the fit
                if np.sum(mask) >= 2:

                    # Fit power law: P(f) = A * f^(-β)
                    log_freqs = np.log10(pos_freqs)
                    log_psd = np.log10(pos_psd)

                    # PASS 1: FInd dominant waves
                    slope_init, intercept_init = np.polyfit(log_freqs[mask], log_psd[mask], 1)
                    fitted_psd_init = 10 ** (intercept_init + slope_init * np.log10(pos_freqs))

                    # Calculate residuals to find peaks
                    residual_psd = pos_psd / fitted_psd_init
                    peaks, _ = signal.find_peaks(residual_psd, height=2.0)

                    # PASS 2: Mask large peaks for "texture only" fit
                    clean_mask = mask.copy()
                    if len(peaks) > 0:
                        for p_idx in peaks:
                            # mask out small buffer around the peak to remove edge effects
                            buffer = 5
                            start = max(0, p_idx - buffer)
                            end = min(len(clean_mask), p_idx + buffer + 1)
                            clean_mask[start:end] = False

                    # REFIT
                    if np.sum(clean_mask) >= 2:
                        # Fit only on masked data
                        slope, intercept = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1)
                        beta = -slope # Power law exponent

                        # Apply fit to the full range
                        fitted_psd = 10**(intercept + slope * log_freqs)
                        residual_psd = pos_psd / fitted_psd

                    else: # fallback
                        beta = -slope_init
                        fitted_psd = fitted_psd_init

                    dominant_wavelengths = wavelengths_calc[peaks] if len(peaks) > 0 else []

                    profile_length = segment_distance.max() - segment_distance.min()
                    confidence_flags = flag_wavelength_confidence(dominant_wavelengths, profile_length)

                    # Calculate Hurst exponent from spectral exponent
                    # For 1D profiles: β = 2H + 1, so H = (β - 1) / 2
                    hurst_exponent = (beta - 1) / 2

                    stats_dict.update({
                        'median_spacing': dx_median,
                        'profile_length': profile_length,
                        'dominant_wavelengths': dominant_wavelengths,
                        'confirmed_wavelengths': confidence_flags['confirmed'],
                        'candidate_wavelengths': confidence_flags['candidate'],
                        'confidence_threshold': confidence_flags['threshold'],
                        'power_law_exponent': beta,
                        'hurst_exponent': hurst_exponent
                    })


                    # Plot the first n lines
                    if plot_count < 5:
                        plot_spectra(segment_distance, detrended, wavelengths_calc, pos_psd, fitted_psd, beta, residual_psd, traj_id, dataset_name, segment_number=seg_idx+1)
                        plot_count += 1

                else:
                    print(f"Skipping Line {traj_id}: Not enough data points in 100m-10km range.")

                segment_results.append(stats_dict)

            if valid_segments:
                plot_raw_data_with_segmentation_check(dist, elev, valid_segments, traj_id, gap_mask)

            if segment_results:
                # Aggregate statistics
                combined_stats = {}
                # Keys that are ALREADY lists inside the segment dict
                list_keys = ['dominant_wavelengths', 'confirmed_wavelengths', 'candidate_wavelengths']
                
                # Keys that are SINGLE VALUES in the segment dict, but we want to KEEP as a list 
                list_keys_collect = ['power_law_exponent', 'flow_orientation', 'hurst_exponent']

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

    # 2. Segment Lengths
    lengths = [r['profile_length'] for r in results.values() if 'profile_length' in r]
    print(f"AVG SEGMENT LENGTH:\n  -> {format_stat(lengths, 'm')}")

    # 3. Power Law and Hurst Exponents
    betas = [b for r in results.values() for b in r.get('power_law_exponent', [])]
    print(f"POWER LAW EXPONENT (Beta):\n  -> {format_stat(betas)}")
    hurst_exponents = [H_e for r in results.values() for H_e in r.get('hurst_exponent', [])]
    print(f"HURST EXPONENT:\n  -> {format_stat(hurst_exponents)}")
    
    # 4. Ice Thickness
    thickness = [r['ice_thickness_mean'] for r in results.values() if 'ice_thickness_mean' in r and not np.isnan(r['ice_thickness_mean'])]
    if thickness:
        print(f"MEAN ICE THICKNESS:\n  -> {format_stat(thickness, 'm')}")

    # 5. Flow Orientation (All Segments)
    all_orientations = [o for r in results.values() for o in r.get('flow_orientation', [])]
    if all_orientations:
        print("." * 60)
        print("FLOW ORIENTATION (All Segments):")
        counts = {o: all_orientations.count(o) for o in set(all_orientations)}
        total = len(all_orientations)
        for o, count in counts.items():
            print(f"  -> {o}: {count} segments ({100*count/total:.1f}%)")
        
        # Beta breakdown by orientation
        print("\n  STRATIFIED BETA (Roughness) BY ORIENTATION:")
        for target_orient in ['Parallel', 'Oblique', 'Perpendicular']:
            betas_for_orient = []
            for r in results.values():
                traj_orientations = r.get('flow_orientation', [])
                traj_betas = r.get('power_law_exponent', [])

                # Check if they match in length (they should)
                if len(traj_orientations) == len(traj_betas):
                    # Zip them together to pair (Orientation, Beta)
                    for orient, beta in zip(traj_orientations, traj_betas):
                        if orient == target_orient:
                            betas_for_orient.append(beta)
            
            if betas_for_orient:
                avg_b = np.mean(betas_for_orient)
                print(f"  -> {target_orient:<15}: Mean Beta = {avg_b:.2f}  (n={len(betas_for_orient)})")
            else:
                print(f"  -> {target_orient:<15}: No segments found")


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
        print("LARGEST LOCAL STRUCTURES (50km Window):")
        # Zip them to print pairs
        for relief, loc in zip(max_reliefs, locs):
            print(f"  -> Relief: {relief:.1f}m at km {loc:.1f}")
        print("." * 60)
    print("=" * 60)
    return {}


if __name__=="__main__":

    results = analyse_bedrock()
    print(f"\n---")
    print(f"Analysed {len(results)} regions")

    for region_name, region_results in results.items():
              print(f"\n=== {region_name} SUMMARY ===")
              results_summary(region_results)
