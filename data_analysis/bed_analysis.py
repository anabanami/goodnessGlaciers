import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from pyproj import Transformer
import pprint
import os
from REMA_extractor import extract_rema_elevation, calculate_ice_thickness


def load_datasets():
    # for convenience 
    # Using os.path.join() for each filename:
    base_path = 'shortcut_to_culled-data'
    # DATA LOADING (Existing hardcoded ranges as per JR's analysis)

    filename_test = os.path.join(base_path, 'test.csv')
    test = pd.read_csv(filename_test, comment='#')
    test['trajectory_id'] = 'TEST_LINE' # Ensure it has an ID
    print(f"✓ TEST loaded: {len(test)} rows")

    filename_stanford = os.path.join(base_path, 'STANFORD_1971_SPRI-NSF-TUD_AIR_BM3.csv')
    header_df = pd.read_csv(filename_stanford, comment='#', nrows=0)
    df_stanford = pd.read_csv(filename_stanford, comment='#', skiprows=range(1, 18271), nrows=1501, names=header_df.columns)
    # Assign trajectory_id
    df_stanford['trajectory_id'] = 'STANFORD_1971'
    
    filename_UTIG_2010 = os.path.join(base_path, 'UTIG_2010_ICECAP_AIR_BM3.csv')
    header_df = pd.read_csv(filename_UTIG_2010, comment='#', nrows=0)
    df_UTIG_2010 = pd.read_csv(filename_UTIG_2010, comment='#', skiprows=range(1, 8508113), nrows=17528, names=header_df.columns)
    ASB = pd.concat([df_stanford, df_UTIG_2010], ignore_index=True)
    print(f"✓ ASB loaded: {len(ASB)} rows")

    filename_ROSS = os.path.join(base_path, 'UTIG_2010_ICECAP_AIR_BM3.csv')
    df_UTIG_full = pd.read_csv(filename_ROSS, comment='#')
    # Filter by string match
    ROSS = df_UTIG_full[df_UTIG_full['trajectory_id'].astype(str).str.contains('IR1HI2_2009033_DMC_JKB1a_WLKX10b', na=False)]
    print(f"✓ ROSS loaded: {len(ROSS)} rows")

    filename_PEL = os.path.join(base_path, 'PRIC_2016_CHA2_AIR_BM3.csv')
    header_df = pd.read_csv(filename_PEL, comment='#', nrows=0)
    PEL = pd.read_csv(filename_PEL, comment='#', skiprows=range(1, 410824), nrows=54566, names=header_df.columns)
    # Assign trajectory_id
    PEL['trajectory_id'] = 'PRIC_2016_001'
    print(f"✓ PEL loaded: {len(PEL)} rows")

    # filename_DMA = os.path.join(base_path, 'NASA_2010_ICEBRIDGE_AIR_BM2.csv')
    # header_df = pd.read_csv(filename_DMA, comment='#', nrows=0)
    # DMA = pd.read_csv(filename_DMA, comment='#', header=0, names=header_df.columns)
    # # Assign trajectory_id
    # DMA['trajectory_id'] = 'NASA_2010_ICEBRIDGE_001'
    # print(f"✓ DMA loaded: {len(DMA)} rows")

    ## "Legacy" dataset. It contains airborne radio-echo sounding (RES) data collected between 1966 and 2000.
    # filename_QML = os.path.join(base_path, 'BEDMAP1_1966-2000_AIR_BM1.csv')
    # df_QML_full = pd.read_csv(filename_QML, comment='#')
    # mask1 = (df_QML_full['trajectory_id'] >= 960000) & (df_QML_full['trajectory_id'] <= 980000)
    # mask2 = (df_QML_full['trajectory_id'] >= 1770000) & (df_QML_full['trajectory_id'] <= 1780000)
    # QML = df_QML_full[mask1 | mask2]
    # # # Assign trajectory_id
    # QML['trajectory_id'] = 'BEDMAP1_1966-2000_001'
    # print(f"✓ QML loaded: {len(QML)} rows")

    return [test, ASB, ROSS, PEL]
    # return [test]


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
    Uses detect_data_gaps to ensure logical consistency.
    """
    # Use the existing function to find gap locations
    gap_mask = detect_data_gaps(distance, gap_threshold)
    gap_indices = np.where(gap_mask)[0]
    
    # Define start and end indices for slicing
    split_points = np.concatenate([[0], gap_indices, [len(distance)]])
    
    segments = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        if end - start >= min_segment_length:
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


def analyse_bedrock():
    """
    Statistical spectral profiling of radar flight datasets of Antarctic bedrock elevation
    """
    # Setup projection transformer: WGS84 (Lat/Lon) -> Antarctic Stereo (Meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    datasets = load_datasets()

    # DEM path
    base_path = 'shortcut_to_culled-data/'
    dem_path = os.path.join(base_path, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

    # Loop through each bedrock dataset with a name/label
    # Accumulate results separately for each region
    # dataset_names = ['test']
    dataset_names = ['test', 'ASB', 'ROSS', 'PEL']

    all_results = {}

    for i, df in enumerate(datasets):
        dataset_name = dataset_names[i]
        print(f"\nStarting analysis of {dataset_name}...")

        # Filter invalid data
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        print(f"  Valid data points: {len(valid)}")
        print(f"  Unique trajectories: {len(valid['trajectory_id'].unique())}")
        print(f"  Sample bedrock values: {valid['bedrock_altitude (m)'].head(10).values}")

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
            print(f"  {traj_id}: Found {np.sum(gap_mask)} >=2000m gaps in data")
            
            # Split into valid segments
            segments = split_into_segments(line, dist)

            # Check raw data
            plot_raw_data_with_segmentation_check(dist, elev, segments, traj_id, gap_mask)

            if len(segments) != 0:
                print(f"{len(segments)} valid data segments found")
            else:
                print(f" Skipping trajectory {traj_id}: no valid segments found")
                continue


            # Process each segment separately 
            segment_results = []
            for seg_idx, (segment_data, segment_distance) in enumerate(segments):
                bedrock_segment_elev = segment_data['bedrock_altitude (m)'].values
                detrended = signal.detrend(bedrock_segment_elev)

                seg_lons = segment_data['longitude (degree_east)'].values
                seg_lats = segment_data['latitude (degree_north)'].values

                seg_x, seg_y = transformer.transform(seg_lons, seg_lats)

                # 1. Skip REMA for 'test' dataset; use column values instead
                if dataset_name == 'test':
                    if 'ice_thickness (m)' in segment_data.columns:
                        valid_ice_thickness = segment_data['ice_thickness (m)'].values
                    else:
                        # If you only care about bedrock spectra, NaNs are fine here
                        valid_ice_thickness = np.full(len(bedrock_segment_elev), np.nan)

                else:
                    surface_elevs = extract_rema_elevation(seg_x, seg_y, dem_path)
                    # print(f"  >>>>>>>>>: Surface elevs range: {np.nanmin(surface_elevs):.1f} to {np.nanmax(surface_elevs):.1f}")
                    # print(f"  >>>>>>>>>: Bedrock elevs range: {np.min(bedrock_segment_elev):.1f} to {np.max(bedrock_segment_elev):.1f}")
                    valid_ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_segment_elev)
                    print(f"  >>>>>>>>>: Valid ice thickness count: {np.sum(~np.isnan(valid_ice_thickness))} / {len(valid_ice_thickness)}")

                stats_dict = {
                    'elevation_range': bedrock_segment_elev.max() - bedrock_segment_elev.min(),
                    'skewness': stats.skew(detrended, bias=False),
                    'kurtosis': stats.kurtosis(detrended, bias=False),
                    'rms_roughness': np.sqrt(np.mean(detrended**2)),
                    'ice_thickness_mean': np.nanmean(valid_ice_thickness),
                    'ice_thickness_range': np.nanmax(valid_ice_thickness) - np.nanmin(valid_ice_thickness)
                }
                
                # SPECTRAL ANALYSIS
                # dx_median sampling interval (Used for stats only)
                # Filter out zero-steps to get a valid median
                valid_steps = np.diff(segment_distance)
                valid_steps = valid_steps[valid_steps > 1e-6] # Filter tiny/zero steps

                if len(valid_steps) > 0:
                    dx_median = np.median(valid_steps)
                else:
                    dx_median = 0
                
                if dx_median > 0 and len(detrended) > 10:
                    # Defining frequencies to query
                    min_freq = 1.0 / (segment_distance.max() - segment_distance.min())
                    
                    limit_spacing = max(dx_median, 15.0) # 10-20m is the physical limit for most ice-penetrating radars.
                    max_freq = 1 / (2 * limit_spacing)#  <-- Nyquist limit of the tightest sampling
                    freqs = np.geomspace(min_freq, max_freq, num=len(detrended) * 10)

                    # Calculating power using Lomb-Scargle Periodogram
                    periodogram = signal.lombscargle(segment_distance, detrended, freqs * 2 * np.pi, normalize=False)

                    # PSD units
                    pos_freqs = freqs
                    pos_psd = periodogram

                    # Wavelengths
                    wavelengths_calc = 1 / pos_freqs

                    # Create mask for geologically relevant wavelength range (1km to 50km)
                    mask = (wavelengths_calc >= 1000) & (wavelengths_calc <= 50000)
                    
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

            if segment_results:
                # Aggregate statistics
                combined_stats = {}
                list_keys = ['dominant_wavelengths', 'confirmed_wavelengths', 'candidate_wavelengths']
                
                for key in segment_results[0].keys():
                    
                    # 1. Extract values for the CURRENT key immediately
                    values = [seg[key] for seg in segment_results if key in seg]

                    if key in list_keys:
                        # Special handling for lists (flattening)
                        combined_stats[key] = [w for seg in segment_results for w in seg.get(key, [])]
                    
                    elif key == 'profile_length':
                        # 2. Now 'values' actually contains the profile lengths
                        combined_stats[key] = np.max(values)
                    
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
        
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # If we only have 1 trajectory, or all trajectories are identical
        if min_val == max_val:
            return f"{mean_val:.1f}{unit} (Single Value)"
        else:
            return f"Mean: {mean_val:.1f}{unit} | Range: [{min_val:.1f}, {max_val:.1f}]{unit}"

    print("-" * 60)
    print(f"  RESULTS SUMMARY ({len(results)} trajectories aggregated)")
    print("-" * 60)

    # 1. Vertical Relief (formerly 'elevation_range')
    # Use list comprehension to gather values from all trajectories
    reliefs = [r['elevation_range'] for r in results.values() if 'elevation_range' in r]
    print(f"VERTICAL RELIEF (Max-Min):\n  -> {format_stat(reliefs, 'm')}")

    # 2. Segment Lengths (formerly 'profile_length')
    lengths = [r['profile_length'] for r in results.values() if 'profile_length' in r]
    print(f"AVG SEGMENT LENGTH:\n  -> {format_stat(lengths, 'm')}")

    # 3. Power Law Exponent (Roughness)
    betas = [r['power_law_exponent'] for r in results.values() if 'power_law_exponent' in r]
    print(f"POWER LAW EXPONENT (Beta):\n  -> {format_stat(betas)}")
    
    # 4. Ice Thickness
    thickness = [r['ice_thickness_mean'] for r in results.values() if 'ice_thickness_mean' in r and not np.isnan(r['ice_thickness_mean'])]
    if thickness:
        print(f"MEAN ICE THICKNESS:\n  -> {format_stat(thickness, 'm')}")

    print("." * 60)

    # 5. Wavelength Reporting
    # We aggregate ALL confirmed wavelengths from ALL lines into one list
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

    print("=" * 60)
    return {}


if __name__=="__main__":

    results = analyse_bedrock()
    print(f"\n---")
    print(f"Analysed {len(results)} regions")

    for region_name, region_results in results.items():
              print(f"\n=== {region_name} SUMMARY ===")
              results_summary(region_results)