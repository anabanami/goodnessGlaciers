import numpy as np
import pandas as pd
from scipy import signal, stats
from pyproj import Transformer
import os
import sys

from config import (WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, STANDARD_WINDOW, DATA_ROOT,
                    peak_masking_height_threshold, bin_buffer,
                    ENABLE_LANDSCAPE_SPLITTING,
                    Tee, get_region_folder, ensure_output_dirs)
from loading import  OUTPUT_BASE_PATH, load_datasets
from segmentation import detect_data_gaps, split_into_segments, split_by_landscape
from plotting import plot_raw_data_with_segmentation_check, plot_spectra
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, MEaSUREs_comparison


def flag_wavelength_confidence(wavelengths, profile_length, min_cycles=2.0):
    threshold = profile_length / min_cycles
    if len(wavelengths) == 0:
        return {'confirmed': [], 'candidate': [], 'threshold': threshold}
    confirmed = wavelengths[wavelengths <= threshold]
    candidate = wavelengths[wavelengths > threshold]
    return {'confirmed': confirmed.tolist(), 'candidate': candidate.tolist(), 'threshold': threshold}


def calculate_flow_incidence(x, y, flow_x, flow_y):
    flight_dx = np.gradient(x)
    flight_dy = np.gradient(y)
    flight_mag = np.sqrt(flight_dx**2 + flight_dy**2)
    dot = flight_dx * flow_x + flight_dy * flow_y
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_theta = dot / (flight_mag * 1.0)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return np.minimum(angle, 180 - angle)


def analyse_sliding_windows(dist, elev, incidence_array, window_size, step_size, flow_angular_diff=None, flow_speed=None):
    dx_median = np.median(np.diff(dist)) if len(dist) > 1 else 100
    if dx_median == 0: dx_median = 15.0

    max_freq = 1 / (2 * max(dx_median, 15.0))
    min_freq = 1 / window_size
    freqs = np.geomspace(min_freq, max_freq, num=500)
    angular_freqs = freqs * 2 * np.pi
    wavelengths_calc = 1 / freqs
    mask = (wavelengths_calc >= 250) & (wavelengths_calc <= 50000)
    log_freqs = np.log10(freqs)

    psd_accumulator = []
    large_features = []

    start_dist = dist.min()
    max_dist = dist.max()
    current_start = start_dist
    window_idx = 0

    while current_start + window_size <= max_dist + 1e-6:
        current_end = current_start + window_size
        fit_mask = (dist >= current_start) & (dist <= current_end)
        w_dist = dist[fit_mask]
        w_elev = elev[fit_mask]

        if len(w_dist) > 50:
            w_detrended = signal.detrend(w_elev)

            if WINDOW_TYPE == 'hann':
                w_detrended = w_detrended * signal.windows.hann(len(w_detrended))
            elif WINDOW_TYPE == 'tukey':
                w_detrended = w_detrended * signal.windows.tukey(len(w_detrended), alpha=0.5)

            pgram = signal.lombscargle(w_dist, w_detrended, angular_freqs, normalize=False)
            psd_accumulator.append(pgram)

            window_beta = np.nan
            window_beta_uncertainty = np.nan
            window_psd_intercept = np.nan
            window_psd_intercept_uncertainty = np.nan
            if np.sum(mask) >= 2 and np.all(pgram > 0):
                log_psd = np.log10(pgram)
                try:
                    n_fit = np.sum(mask)
                    if n_fit > 2:
                        coeffs, cov = np.polyfit(log_freqs[mask], log_psd[mask], 1, cov=True)
                        window_beta_uncertainty = np.sqrt(cov[0, 0])
                        window_psd_intercept_uncertainty = np.sqrt(cov[1, 1])
                    else:
                        coeffs = np.polyfit(log_freqs[mask], log_psd[mask], 1)
                    window_beta = -coeffs[0]
                    window_psd_intercept = coeffs[1]
                except:
                    pass

            window_hurst = (window_beta - 1) / 2
            window_hurst_uncertainty = window_beta_uncertainty / 2

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
                'window_psd_intercept': window_psd_intercept,
                'window_psd_intercept_uncertainty': window_psd_intercept_uncertainty,
                'window_hurst': window_hurst,
                'window_hurst_uncertainty': window_hurst_uncertainty,
            }

            window_incidence = incidence_array[fit_mask]
            feature_stats['mean_window_incidence'] = np.nanmean(window_incidence)

            if flow_angular_diff is not None:
                w_flow_diff = flow_angular_diff[fit_mask]
                feature_stats['flow_error_mean'] = np.nanmean(w_flow_diff)
                feature_stats['flow_error_median'] = np.nanmedian(w_flow_diff)
            else:
                feature_stats['flow_error_mean'] = np.nan
                feature_stats['flow_error_median'] = np.nan

            if flow_speed is not None:
                feature_stats['measures_speed_mean'] = np.nanmean(flow_speed[fit_mask])
            else:
                feature_stats['measures_speed_mean'] = np.nan

            large_features.append(feature_stats)

        current_start += step_size
        window_idx += 1

    if psd_accumulator:
        psd_stack = np.array(psd_accumulator)
        avg_psd = np.mean(psd_stack, axis=0)
        log_psd_std = np.std(np.log10(psd_stack), axis=0)
        log_psd_std[log_psd_std == 0] = np.inf
        psd_weights = 1.0 / log_psd_std
    else:
        avg_psd = None
        psd_weights = None

    return avg_psd, freqs, large_features, dx_median, psd_weights


def analyse_bedrock():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    datasets_bundle = load_datasets()

    dem_path = os.path.join(DATA_ROOT, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

    all_results = {}

    for bundle in datasets_bundle:
        dataset_name = bundle['name']
        df = bundle['data']
        print(f"\nStarting analysis of {dataset_name}...")

        region_folder = get_region_folder(dataset_name)
        window_suffix = f"_w{WINDOW_SIZE // 1000}km" if WINDOW_TYPE == STANDARD_WINDOW else f"_w{WINDOW_SIZE // 1000}km_{WINDOW_TYPE}"
        region_folder = f"{region_folder}{window_suffix}"
        output_paths = ensure_output_dirs(OUTPUT_BASE_PATH, region_folder)
        print(f"  Output folder: {output_paths['region']}")

        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        print(f"  Valid data points: {len(valid)}")
        print(f"  Unique trajectories: {len(valid['trajectory_id'].unique())}")

        results = {}
        # plot_count = 0

        for j, traj_id in enumerate(valid['trajectory_id'].unique()):
            if j > 0 and j % 10 == 0:
                print(f"  Processed {j} trajectories")

            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20: continue

            longs = line['longitude (degree_east)'].values
            lats = line['latitude (degree_north)'].values
            x, y = transformer.transform(longs, lats)

            dx_seg = np.diff(x)
            dy_seg = np.diff(y)
            segment_distances = np.sqrt(dx_seg**2 + dy_seg**2)
            dist = np.concatenate([[0], np.cumsum(segment_distances)])

            elev = line['bedrock_altitude (m)'].values

            gap_mask = detect_data_gaps(dist)
            n_gaps = np.sum(gap_mask) // 2
            print(f"  {traj_id}: Found {n_gaps} gaps (>=2000m) in data")

            gap_segments = split_into_segments(line, dist)
            if not gap_segments:
                print(f" Skipping trajectory {traj_id}: no valid segments found")
                continue

            if ENABLE_LANDSCAPE_SPLITTING:
                segments = []
                for seg_data, seg_dist in gap_segments:
                    segments.extend(split_by_landscape(seg_data, seg_dist))

                n_landscape_splits = len(segments) - len(gap_segments)
                print(f"{len(gap_segments)} gap segments -> {len(segments)} after landscape splitting"
                      + (f" (+{n_landscape_splits} landscape splits)" if n_landscape_splits > 0 else ""))
            else:
                segments = list(gap_segments)
                print(f"{len(gap_segments)} gap segments (landscape splitting disabled)")

            valid_segments = []
            segment_results = []

            for seg_idx, (segment_data, segment_distance) in enumerate(segments):
                bedrock_segment_elev = segment_data['bedrock_altitude (m)'].values
                detrended = signal.detrend(bedrock_segment_elev)

                seg_lons = segment_data['longitude (degree_east)'].values
                seg_lats = segment_data['latitude (degree_north)'].values
                seg_x, seg_y = transformer.transform(seg_lons, seg_lats)

                surface_elevs = extract_rema_elevation(seg_x, seg_y, dem_path)
                valid_ice_thickness = calculate_ice_thickness(surface_elevs, bedrock_segment_elev)
                thickness_validity = np.sum(~np.isnan(valid_ice_thickness)) / len(valid_ice_thickness)
                if thickness_validity < 0.20:
                    print(f" Skipping Segment {seg_idx+1}: Insufficient thickness data (only {thickness_validity*100:.1f}% is valid)")
                    continue

                print(f" > Segment {seg_idx+1}: Valid ice thickness for count: {np.sum(~np.isnan(valid_ice_thickness))} / {len(valid_ice_thickness)}")

                global_relief = bedrock_segment_elev.max() - bedrock_segment_elev.min()
                valid_segments.append((segment_data, segment_distance))

                vx, vy = extract_rema_flow_vector(seg_x, seg_y, dem_path, valid_ice_thickness)
                invalid_mask = np.isnan(valid_ice_thickness)
                vx[invalid_mask] = np.nan
                vy[invalid_mask] = np.nan

                angular_diff, measures_speed = MEaSUREs_comparison(seg_x, seg_y, vx, vy)
                print(f"Flow validation: mean diff = {np.nanmean(angular_diff):.1f}°, median ={np.nanmedian(angular_diff):.1f}°, mean speed = {np.nanmean(measures_speed):.1f} m/yr")

                incidence_array = calculate_flow_incidence(seg_x, seg_y, vx, vy)
                mean_incidence = np.nanmean(incidence_array)

                segment_len_m = segment_distance.max() - segment_distance.min()

                if segment_len_m < WINDOW_SIZE:
                    avg_psd, freqs, window_stats, dx_median, psd_weights = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev, incidence_array,
                        window_size=segment_len_m, step_size=segment_len_m,
                        flow_angular_diff=angular_diff, flow_speed=measures_speed)
                else:
                    avg_psd, freqs, window_stats, dx_median, psd_weights = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev, incidence_array,
                        window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                        flow_angular_diff=angular_diff, flow_speed=measures_speed)

                if window_stats:
                    max_relief_window = max(window_stats, key=lambda x: x['local_relief_m'])
                    max_local_relief = max_relief_window['local_relief_m']
                    loc_of_max_relief = max_relief_window['start_km']
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

                if avg_psd is None or np.all(avg_psd == 0) or np.any(avg_psd < 0):
                    print(f"  Skipping segment {seg_idx+1} spectral fit: Invalid PSD values")
                    segment_results.append(stats_dict)
                    continue

                wavelengths_calc = 1 / freqs
                fit_mask = (wavelengths_calc >= 250) & (wavelengths_calc <= 50000)

                if np.sum(fit_mask) >= 2:
                    log_freqs = np.log10(freqs)
                    log_psd = np.log10(avg_psd)

                    # PASS 1: find dominant waves
                    slope_init, intercept_init = np.polyfit(log_freqs[fit_mask], log_psd[fit_mask], 1)
                    fitted_psd_init = 10 ** (intercept_init + slope_init * np.log10(freqs))
                    residual_psd = avg_psd / fitted_psd_init
                    peaks, _ = signal.find_peaks(residual_psd, height=peak_masking_height_threshold)

                    # PASS 2: mask peaks, refit
                    clean_mask = fit_mask.copy()
                    if len(peaks) > 0:
                        for p_idx in peaks:
                            start = max(0, p_idx - bin_buffer)
                            end = min(len(clean_mask), p_idx + bin_buffer + 1)
                            clean_mask[start:end] = False

                    if np.sum(clean_mask) >= 2:
                        psd_weights[~np.isfinite(psd_weights)] = 0
                        w = psd_weights[clean_mask]
                        if np.all(w == 0):
                            w = None
                        (slope, intercept), cov = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1, w=w, cov=True)
                        beta = -slope
                        beta_uncertainty = np.sqrt(cov[0, 0])
                        C = intercept
                        C_uncertainty = np.sqrt(cov[1, 1])
                        fitted_psd = 10**(intercept + slope * log_freqs)
                        residual_psd = avg_psd / fitted_psd
                    else:
                        beta = -slope_init
                        beta_uncertainty = np.nan
                        C = intercept_init
                        C_uncertainty = np.nan
                        fitted_psd = fitted_psd_init

                    dominant_wavelengths = wavelengths_calc[peaks] if len(peaks) > 0 else []
                    profile_length = segment_distance.max() - segment_distance.min()
                    confidence_flags = flag_wavelength_confidence(dominant_wavelengths, profile_length)

                    hurst_exponent = (beta - 1) / 2
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
                        'power_law_intercept': C,
                        'C_uncertainty': C_uncertainty,
                        'hurst_exponent': hurst_exponent,
                        'hurst_uncertainty': hurst_uncertainty
                    })

                    # if plot_count < 10:
                    #     plot_spectra(segment_distance, detrended, wavelengths_calc, avg_psd, fitted_psd, beta, residual_psd, traj_id, dataset_name, segment_number=seg_idx+1, output_path=output_paths['psd'])
                    #     plot_count += 1
                else:
                    print(f"Skipping Line {traj_id}: Not enough data points in 250m–50km range.")

                segment_results.append(stats_dict)

            if valid_segments:
                plot_raw_data_with_segmentation_check(dist, elev, valid_segments, traj_id, gap_mask, output_path=output_paths['trajectories'])

            if segment_results:
                combined_stats = {}
                list_keys = ['dominant_wavelengths', 'confirmed_wavelengths', 'candidate_wavelengths', 'window_stats']
                list_keys_collect = ['power_law_exponent', 'hurst_exponent', 'beta_uncertainty', 'hurst_uncertainty', 'power_law_intercept', 'C_uncertainty', 'flow_incidence_deg', 'flow_error_mean', 'flow_error_median', 'measures_speed_mean', 'elevation_min', 'elevation_max']

                for key in segment_results[0].keys():
                    values = [seg[key] for seg in segment_results if key in seg]

                    if key == 'max_local_relief':
                        combined_stats[key] = np.max(values)
                        idx_of_max = np.argmax(values)
                        combined_stats['loc_of_max_relief'] = segment_results[idx_of_max]['loc_of_max_relief']
                        continue
                    if key == 'loc_of_max_relief':
                        continue

                    if key in list_keys:
                        combined_stats[key] = [w for seg in segment_results for w in seg.get(key, [])]
                    elif key in list_keys_collect:
                        combined_stats[key] = values
                    elif key == 'profile_length':
                        combined_stats[key] = np.mean(values)
                    elif isinstance(values[0], str):
                        if values:
                            combined_stats[key] = max(set(values), key=values.count)
                    else:
                        if values:
                            combined_stats[key] = np.mean(values)

                results[traj_id] = combined_stats
                print(f"  Trajectory {traj_id}: {len(segments)} segments, combined median spacing = {combined_stats.get('median_spacing', 0):.1f}m, Nyquist = {2*combined_stats.get('median_spacing', 0):.1f}m")

        all_results[dataset_name] = results
        print(f"{dataset_name} is finished processing")

    return all_results


def results_summary(results):
    if not results: return "no valid data found :("

    def format_stat(values, unit=""):
        if not values: return "N/A"
        if isinstance(values[0], str): return "N/A (String Data)"
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

    reliefs = [r['elevation_range'] for r in results.values() if 'elevation_range' in r]
    print(f"VERTICAL RELIEF (Max-Min):\n  -> {format_stat(reliefs, 'm')}")

    elev_mins = [v for r in results.values() for v in r.get('elevation_min', [])]
    elev_maxs = [v for r in results.values() for v in r.get('elevation_max', [])]
    if elev_mins and elev_maxs:
        print(f"ABSOLUTE BED ELEVATION:\n  -> Min: {np.nanmin(elev_mins):.1f}m | Max: {np.nanmax(elev_maxs):.1f}m  (across all trajectories)")

    lengths = [r['profile_length'] for r in results.values() if 'profile_length' in r]
    print(f"AVG SEGMENT LENGTH:\n  -> {format_stat(lengths, 'm')}")

    skews = [r['skewness'] for r in results.values() if 'skewness' in r and np.isfinite(r['skewness'])]
    kurts = [r['kurtosis'] for r in results.values() if 'kurtosis' in r and np.isfinite(r['kurtosis'])]
    if skews:
        print(f"SKEWNESS:\n  -> {format_stat(skews)}")
    if kurts:
        print(f"KURTOSIS (excess):\n  -> {format_stat(kurts)}")

    thickness = [r['ice_thickness_mean'] for r in results.values() if 'ice_thickness_mean' in r and not np.isnan(r['ice_thickness_mean'])]
    if thickness:
        print(f"MEAN ICE THICKNESS:\n  -> {format_stat(thickness, 'm')}")

    print("." * 60)

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

    max_reliefs = [r['max_local_relief'] for r in results.values() if 'max_local_relief' in r]
    locs = [r['loc_of_max_relief'] for r in results.values() if 'loc_of_max_relief' in r]

    if max_reliefs:
        print("." * 60)
        print(f"LARGEST LOCAL STRUCTURES ({WINDOW_SIZE/1000:.0f}km Window):")
        for relief, loc in zip(max_reliefs, locs):
            print(f"  -> Relief: {relief:.1f}m at km {loc:.1f}")
        print("." * 60)
    print("=" * 60)
    return {}


if __name__ == "__main__":

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
                    'psd_intercept': w.get('window_psd_intercept'),
                    'psd_intercept_uncertainty': w.get('window_psd_intercept_uncertainty'),
                    'psd_amplitude_1km': w.get('window_psd_intercept', np.nan) + 3 * w.get('window_beta', np.nan),
                    'hurst': w.get('window_hurst'),
                    'hurst_uncertainty': w.get('window_hurst_uncertainty'),
                    'relief_m': w.get('local_relief_m'),
                    'bed_elev_mean': w.get('bed_elev_mean'),
                    'rms_roughness': w.get('roughness_rms'),
                    'flow_error_mean': w.get('flow_error_mean'),
                    'flow_error_median': w.get('flow_error_median'),
                    'measures_speed_mean': w.get('measures_speed_mean')
                })
        csv_suffix = f"_w{WINDOW_SIZE // 1000}km" if WINDOW_TYPE == STANDARD_WINDOW else f"_w{WINDOW_SIZE // 1000}km_{WINDOW_TYPE}"
        region_output = os.path.join(OUTPUT_BASE_PATH, f'{get_region_folder(region_name)}{csv_suffix}')
        window_csv_dir = os.path.join(OUTPUT_BASE_PATH, 'window_csvs')
        os.makedirs(window_csv_dir, exist_ok=True)
        pd.DataFrame(all_window_data).to_csv(os.path.join(window_csv_dir, f'{region_name}{csv_suffix}_window_stats.csv'), index=False)

        # --- Segment-level CSV ---
        window_df = pd.DataFrame(all_window_data)

        all_segment_data = []
        for traj_id, traj_data in region_results.items():
            betas = traj_data.get('power_law_exponent', [])
            beta_uncerts = traj_data.get('beta_uncertainty', [])
            psd_intercepts = traj_data.get('psd_intercept', [])
            psd_intercept_uncerts = traj_data.get('psd_intercept_uncertainty', [])
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
                    'psd_intercept': psd_intercepts[i] if i < len(psd_intercepts) else np.nan,
                    'psd_intercept_uncertainty': psd_intercept_uncerts[i] if i < len(psd_intercept_uncerts) else np.nan,
                    'psd_amplitude_1km': (psd_intercepts[i] if i < len(psd_intercepts) else np.nan) + 3 * betas[i],
                    'hurst': hursts[i] if i < len(hursts) else np.nan,
                    'hurst_uncertainty': hurst_uncerts[i] if i < len(hurst_uncerts) else np.nan,
                    'flow_error_mean': flow_err_means[i] if i < len(flow_err_means) else np.nan,
                    'flow_error_median': flow_err_medians[i] if i < len(flow_err_medians) else np.nan,
                    'measures_speed_mean': speed_means[i] if i < len(speed_means) else np.nan,
                    'elevation_min': elev_mins[i] if i < len(elev_mins) else np.nan,
                    'elevation_max': elev_maxs[i] if i < len(elev_maxs) else np.nan,
                }

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

        # --- Wavelength detections CSV ---
        all_wavelength_data = []
        for traj_id, traj_data in region_results.items():
            for wl in traj_data.get('confirmed_wavelengths', []):
                all_wavelength_data.append({'trajectory': traj_id, 'wavelength_m': wl, 'type': 'confirmed'})
            for wl in traj_data.get('candidate_wavelengths', []):
                all_wavelength_data.append({'trajectory': traj_id, 'wavelength_m': wl, 'type': 'candidate'})
        pd.DataFrame(all_wavelength_data).to_csv(os.path.join(region_output, f'{region_name}{csv_suffix}_wavelength_detections.csv'), index=False)

        print(f"Exported {len(all_window_data)} window rows, {len(all_segment_data)} segment rows, {len(all_wavelength_data)} wavelength detections")
