import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from pyproj import Transformer
import os
import sys

from config import (WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, STANDARD_WINDOW,
                    peak_masking_height_threshold, bin_buffer, WINDOW_MASK,
                    HILL_BOX_M, HILL_RELIEF_THRESHOLDS, HILL_THRESHOLD_M, BEDFORM_BAND_M, FIT_BAND_M,
                    Tee, get_region_folder, ensure_output_dirs)
from loading import  OUTPUT_BASE_PATH, load_datasets
from segmentation import detect_data_gaps, split_into_segments, split_by_landscape
from plotting import plot_raw_data_with_segmentation_check, plot_spectra, psd_spectrum_plot, psd_residuals_plot
from REMA_extractor import extract_rema_elevation, extract_rema_flow_vector, calculate_ice_thickness, MEaSUREs_comparison


def flag_wavelength_confidence(wavelengths, profile_length, min_cycles=2.0):
    threshold = profile_length / min_cycles
    if len(wavelengths) == 0:
        return {'confirmed': [], 'candidate': [], 'threshold': threshold}
    confirmed = wavelengths[wavelengths <= threshold]
    candidate = wavelengths[wavelengths > threshold]
    return {'confirmed': confirmed.tolist(), 'candidate': candidate.tolist(), 'threshold': threshold}


def _amp_uncertainty(cov):
    """1-sigma on psd_amplitude_1km = intercept + 3*beta, beta = -slope.
    polyfit coeffs are [slope, intercept], so var(amp) = var(b) + 9 var(m)
    - 6 cov(m, b). The cross term is subtractive because 1 km sits near the
    centroid of the log-f fit band: most of the lever-arm variance cancels and
    sigma_amp lands below either marginal uncertainty. Cannot be rebuilt from
    the CSVs afterwards, since cov[0,1] is not exported.
    """
    var = cov[1, 1] + 9 * cov[0, 0] - 6 * cov[0, 1]
    return np.sqrt(var) if var > 0 else np.nan


def _window_azimuth(x, y):
    """Track heading within a window via PCA principal axis, orientation-only (0-180 deg)."""
    if len(x) < 2:
        return np.nan
    cov = np.cov(x - x.mean(), y - y.mean())
    if not np.all(np.isfinite(cov)):
        return np.nan
    vx, vy = np.linalg.eigh(cov)[1][:, -1]
    return float(np.degrees(np.arctan2(vy, vx)) % 180.0)


def _downsample_idx(x, y, step_m=1000.0):
    """Indices of points spaced ~step_m apart along-track (for the coverage point cloud)."""
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    keep, last = [0], 0.0
    for i in range(1, len(d)):
        if d[i] - last >= step_m:
            keep.append(i)
            last = d[i]
    return np.asarray(keep)


def _hill_counts(detrended, dx, thresholds=HILL_RELIEF_THRESHOLDS, box_m=HILL_BOX_M):
    """Ockenden bumpiness read along a transect: count points that are the maximum of
    a box_m box whose relief clears the threshold, dropping maxima on the window edge.

    The areal original plane-removes the tile first and detrending the window is the 1-D
    equivalent, not cosmetic: on a sloping bed the box maximum sits at the box edge nearly
    everywhere, so an undetrended profile returns almost no hills. The count is per window
    and cannot exceed about WINDOW_SIZE / box_m.
    """
    n = max(3, int(round(box_m / dx)))
    if n % 2 == 0:
        n += 1
    if len(detrended) < n:
        return {t: np.nan for t in thresholds}

    box_max = maximum_filter1d(detrended, n, mode='nearest')
    box_relief = box_max - minimum_filter1d(detrended, n, mode='nearest')

    counts = {}
    for t in thresholds:
        idx = np.flatnonzero((detrended == box_max) & (box_relief > t))
        if idx.size == 0:
            counts[t] = 0
            continue
        # Plateaux flag as runs of adjacent points; each run is one hill.
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([idx[:1], idx[breaks + 1]])
        ends = np.concatenate([idx[breaks], idx[-1:]])
        counts[t] = int(np.sum((starts > 0) & (ends < len(detrended) - 1)))
    return counts


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


def analyse_sliding_windows(dist, elev, incidence_array, window_size, step_size, flow_angular_diff=None, flow_speed=None, seg_x=None, seg_y=None, flow_undefined=None):
    dx_median = np.median(np.diff(dist)) if len(dist) > 1 else 100
    if dx_median == 0: dx_median = 15.0

    max_freq = 1 / (2 * max(dx_median, 15.0))
    min_freq = 1 / window_size
    freqs = np.geomspace(min_freq, max_freq, num=500)
    angular_freqs = freqs * 2 * np.pi
    wavelengths_calc = 1 / freqs
    mask = (wavelengths_calc >= FIT_BAND_M[0]) & (wavelengths_calc <= FIT_BAND_M[1])
    band = (wavelengths_calc >= BEDFORM_BAND_M[0]) & (wavelengths_calc <= BEDFORM_BAND_M[1])
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

            # Taper is for the spectral estimate only. Keep it in a separate
            # array so roughness_rms below stays on the untapered residuals.
            w_tapered = w_detrended
            if WINDOW_TYPE == 'hann':
                w_tapered = w_detrended * signal.windows.hann(len(w_detrended))
            elif WINDOW_TYPE == 'tukey':
                w_tapered = w_detrended * signal.windows.tukey(len(w_detrended), alpha=0.5)

            pgram = signal.lombscargle(w_dist, w_tapered, angular_freqs, normalize=False)
            psd_accumulator.append(pgram)

            local_relief = np.max(w_elev) - np.min(w_elev)

            # Window beta is fit after the loop, once the averaged PSD and its
            # peak mask are known (see the masked window fit below), so that
            # window-level beta responds to peak masking like segment beta does.
            feature_stats = {
                'window_id': window_idx,
                'start_km': current_start / 1000,
                'end_km': current_end / 1000,
                'local_relief_m': local_relief,
                'bed_elev_mean': np.mean(w_elev),
                'roughness_rms': np.sqrt(np.mean(w_detrended**2)),
                # Phase asymmetry, on the untapered residuals like roughness_rms: a Hann
                # taper suppresses the window ends and would bias both moments.
                'window_skewness': float(stats.skew(w_detrended, bias=False)),
                'window_kurtosis': float(stats.kurtosis(w_detrended, bias=False)),
            }

            for _t, _c in _hill_counts(w_detrended, dx_median).items():
                feature_stats[f'hill_count_{_t}'] = _c

            window_incidence = incidence_array[fit_mask]
            feature_stats['mean_window_incidence'] = np.nanmean(window_incidence)

            if seg_x is not None and seg_y is not None:
                wx, wy = seg_x[fit_mask], seg_y[fit_mask]
                feature_stats['center_x'] = float(np.mean(wx))
                feature_stats['center_y'] = float(np.mean(wy))
                feature_stats['azimuth_deg'] = _window_azimuth(wx, wy)
            else:
                feature_stats['center_x'] = np.nan
                feature_stats['center_y'] = np.nan
                feature_stats['azimuth_deg'] = np.nan

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

            # Fraction of the window where the surface gradient fell below the DEM
            # noise floor, so the flow bearing (and hence the incidence angle) is
            # undefined. These points drop out of the nanmeans above, so a high
            # fraction means flow_error_mean rests on few points: treat theta with
            # caution (flat surface: subglacial lake, divide or shelf).
            if flow_undefined is not None:
                feature_stats['flow_undefined_frac'] = float(np.mean(flow_undefined[fit_mask]))
            else:
                feature_stats['flow_undefined_frac'] = np.nan

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

    # Peak mask for the window-level fits, derived from the averaged PSD using
    # the same first-pass recipe the segment two-pass uses in analyse_bedrock
    # (fit the band, take residual = PSD / fit, flag peaks above the threshold,
    # widen by bin_buffer). Applying it to the per-window fits below is preferred
    # over per-window peak detection, which is noisy on a single periodogram.
    # config.WINDOW_MASK = False bypasses this window-level masking (segment
    # two-pass unaffected); used by the window-type taper-isolation ladder.
    clean_mask = mask.copy()
    if WINDOW_MASK \
            and avg_psd is not None and np.sum(mask) >= 2 and np.all(avg_psd[mask] > 0):
        slope0, int0 = np.polyfit(log_freqs[mask], np.log10(avg_psd[mask]), 1)
        fitted0 = 10 ** (int0 + slope0 * log_freqs)
        resid0 = avg_psd / fitted0
        peaks0, _ = signal.find_peaks(resid0, height=peak_masking_height_threshold)
        for p_idx in peaks0:
            start = max(0, p_idx - bin_buffer)
            end = min(len(clean_mask), p_idx + bin_buffer + 1)
            clean_mask[start:end] = False

    # Fit each window's beta over the masked band. clean_mask is identical to the
    # mask the segment two-pass builds (same avg_psd, same fit band, same peaks),
    # so the window and segment scales are masked consistently.
    n_fit = int(np.sum(clean_mask))
    for feat, pgram in zip(large_features, psd_accumulator):
        window_beta = np.nan
        window_beta_uncertainty = np.nan
        window_psd_intercept = np.nan
        window_psd_intercept_uncertainty = np.nan
        window_psd_amplitude_uncertainty = np.nan
        if n_fit >= 2 and np.all(pgram > 0):
            log_psd = np.log10(pgram)
            try:
                if n_fit > 2:
                    coeffs, cov = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1, cov=True)
                    window_beta_uncertainty = np.sqrt(cov[0, 0])
                    window_psd_intercept_uncertainty = np.sqrt(cov[1, 1])
                    window_psd_amplitude_uncertainty = _amp_uncertainty(cov)
                else:
                    coeffs = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1)
                window_beta = -coeffs[0]
                window_psd_intercept = coeffs[1]
            except Exception:
                pass
        feat['window_beta'] = window_beta
        feat['window_beta_uncertainty'] = window_beta_uncertainty
        feat['window_psd_intercept'] = window_psd_intercept
        feat['window_psd_intercept_uncertainty'] = window_psd_intercept_uncertainty
        feat['window_psd_amplitude_uncertainty'] = window_psd_amplitude_uncertainty
        feat['window_hurst'] = (window_beta - 1) / 2
        feat['window_hurst_uncertainty'] = window_beta_uncertainty / 2
        # H in [0,1] requires beta in [1,3]; outside that the window is not
        # self-affine and window_hurst is not a valid exponent (see Jordan 2017).
        feat['window_self_affine_valid'] = bool(np.isfinite(window_beta) and 1.0 <= window_beta <= 3.0)

        # Detections on this window's own periodogram, same recipe as the segment
        # first pass. Most segments hold one window, so detecting per window makes
        # every detection one measurement instead of a mix of averaged and not.
        dets = []
        if np.sum(mask) >= 2 and np.all(pgram > 0):
            s0, i0 = np.polyfit(log_freqs[mask], np.log10(pgram[mask]), 1)
            resid = pgram / 10 ** (i0 + s0 * log_freqs)
            pk, props = signal.find_peaks(resid, height=peak_masking_height_threshold)
            hit = mask[pk]
            wls = wavelengths_calc[pk[hit]]
            # Each window is window_size long, so that is the resolvability limit.
            thr = flag_wavelength_confidence(wls, window_size)['threshold']
            dets = [{'window_id': feat['window_id'],
                     'wavelength_m': float(wl),
                     'type': 'confirmed' if wl <= thr else 'candidate',
                     'residual_height': float(h),
                     'incidence_deg': feat['mean_window_incidence']}
                    for wl, h in zip(wls, props['peak_heights'][hit])]
        feat['_detections'] = dets

        # Li_2010 two-parameter index over the bedform band. xi is the band integral of
        # the elevation PSD; eta is xi over the same integral of the slope spectrum,
        # (2 pi k)^2 S. 2 pi sqrt(eta) is a wavelength and the 2 pi factors cancel, so it
        # is written here as sqrt(int S dk / int k^2 S dk). The amplitude divides out,
        # which is the axis rms_slope could not supply.
        xi = eta_wl = np.nan
        if np.sum(band) >= 2 and np.all(pgram[band] > 0):
            xi = float(np.trapz(pgram[band], freqs[band]))
            xi_k2 = float(np.trapz(freqs[band]**2 * pgram[band], freqs[band]))
            if xi_k2 > 0:
                eta_wl = float(np.sqrt(xi / xi_k2))
        feat['window_xi_band'] = xi
        feat['window_eta_wavelength'] = eta_wl

    return avg_psd, freqs, large_features, dx_median, psd_weights


def analyse_bedrock():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    datasets_bundle = load_datasets()

    base_path = 'all_data/'
    dem_path = os.path.join(base_path, 'rema_mosaic_100m_v2.0_filled_cop30/rema_mosaic_100m_v2.0_filled_cop30_dem.tif')

    all_results = {}
    region_point_clouds = {}

    for bundle in datasets_bundle:
        dataset_name = bundle['name']
        df = bundle['data']
        pflag = df['processing_flag'].iloc[0]
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
        region_pc = []
        for j, traj_id in enumerate(valid['trajectory_id'].unique()):
            if j > 0 and j % 10 == 0:
                print(f"  Processed {j} trajectories")

            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20: continue

            longs = line['longitude (degree_east)'].values
            lats = line['latitude (degree_north)'].values
            x, y = transformer.transform(longs, lats)

            keep = _downsample_idx(np.asarray(x), np.asarray(y))
            region_pc.append(pd.DataFrame({'x': np.asarray(x)[keep],
                                           'y': np.asarray(y)[keep],
                                           'trajectory_id': traj_id}))

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

            segments = []
            for seg_data, seg_dist in gap_segments:
                segments.extend(split_by_landscape(seg_data, seg_dist))

            n_trans = sum(1 for *_, t in segments if t)
            print(f"{len(segments)} segments ({n_trans} transition zones)")

            valid_segments = []
            segment_results = []

            for seg_idx, (segment_data, segment_distance, is_transition) in enumerate(segments):
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

                vx, vy, flow_undefined = extract_rema_flow_vector(
                    seg_x, seg_y, dem_path, valid_ice_thickness, return_undefined=True)
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
                        flow_angular_diff=angular_diff, flow_speed=measures_speed,
                        seg_x=seg_x, seg_y=seg_y, flow_undefined=flow_undefined)
                else:
                    avg_psd, freqs, window_stats, dx_median, psd_weights = analyse_sliding_windows(
                        segment_distance, bedrock_segment_elev, incidence_array,
                        window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                        flow_angular_diff=angular_diff, flow_speed=measures_speed,
                        seg_x=seg_x, seg_y=seg_y, flow_undefined=flow_undefined)

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
                    'flow_undefined_frac': float(np.mean(flow_undefined)),
                    'is_transition': is_transition,
                    'processing_flag': pflag,
                    'window_stats': [{**w, 'segment': seg_idx + 1, 'is_transition': is_transition, 'processing_flag': pflag} for w in window_stats],
                    # Built here rather than beside the segment fit below so the
                    # detections survive a segment whose spectral fit is skipped.
                    'wavelength_detections': [
                        {'segment': seg_idx + 1, **d,
                         'is_transition': is_transition, 'processing_flag': pflag}
                        for w in window_stats for d in w.get('_detections', [])]
                }

                if avg_psd is None or np.all(avg_psd == 0) or np.any(avg_psd < 0):
                    print(f"  Skipping segment {seg_idx+1} spectral fit: Invalid PSD values")
                    segment_results.append(stats_dict)
                    continue

                wavelengths_calc = 1 / freqs
                fit_mask = (wavelengths_calc >= FIT_BAND_M[0]) & (wavelengths_calc <= FIT_BAND_M[1])

                if np.sum(fit_mask) >= 2:
                    log_freqs = np.log10(freqs)
                    log_psd = np.log10(avg_psd)

                    # PASS 1: find dominant waves
                    slope_init, intercept_init = np.polyfit(log_freqs[fit_mask], log_psd[fit_mask], 1)
                    fitted_psd_init = 10 ** (intercept_init + slope_init * np.log10(freqs))
                    residual_psd = avg_psd / fitted_psd_init
                    # Segment-level peaks drive the pass-2 mask and the confirmed and
                    # candidate lists. Detections are found per window instead, so the
                    # heights are taken there and not here.
                    peaks, _ = signal.find_peaks(residual_psd, height=peak_masking_height_threshold)

                    # PASS 2: mask peaks, refit
                    # Single-window segments: skip two-pass (weights degenerate
                    # to unweighted OLS, +0.20 artifact). Use window first-pass
                    # OLS directly as segment beta.
                    if len(window_stats) == 1:
                        beta = window_stats[0]['window_beta']
                        beta_uncertainty = window_stats[0]['window_beta_uncertainty']
                        psd_intercept = window_stats[0]['window_psd_intercept']
                        psd_intercept_uncertainty = window_stats[0]['window_psd_intercept_uncertainty']
                        psd_amplitude_uncertainty = window_stats[0]['window_psd_amplitude_uncertainty']
                        fitted_psd = 10**(psd_intercept + (-beta) * log_freqs)
                        residual_psd = avg_psd / fitted_psd
                    else:
                        clean_mask = fit_mask.copy()
                        if len(peaks) > 0:
                            for p_idx in peaks:
                                start = max(0, p_idx - bin_buffer)
                                end = min(len(clean_mask), p_idx + bin_buffer + 1)
                                clean_mask[start:end] = False

                        if np.sum(clean_mask) >= 2:
                            psd_weights[~np.isfinite(psd_weights)] = 0
                            w = psd_weights[clean_mask]
                            # psd_weights = 1/std(log10 PSD) across windows, so at
                            # two windows the spread is a 2-sample estimate: bins
                            # where the pair happens to agree get weights ~1e4x the
                            # median and pivot the slope (RSL G17a seg5: beta 2.3 ->
                            # 3.5). The zero-std guard upstream catches only exact
                            # ties. Below three windows, fit unweighted.
                            if np.all(w == 0) or len(window_stats) < 3:
                                w = None
                            (slope, intercept), cov = np.polyfit(log_freqs[clean_mask], log_psd[clean_mask], 1, w=w, cov=True)
                            beta = -slope
                            beta_uncertainty = np.sqrt(cov[0, 0])
                            psd_intercept = intercept
                            psd_intercept_uncertainty = np.sqrt(cov[1, 1])
                            psd_amplitude_uncertainty = _amp_uncertainty(cov)
                            fitted_psd = 10**(intercept + slope * log_freqs)
                            residual_psd = avg_psd / fitted_psd
                        else:
                            beta = -slope_init
                            beta_uncertainty = np.nan
                            psd_intercept = intercept_init
                            psd_intercept_uncertainty = np.nan
                            psd_amplitude_uncertainty = np.nan
                            fitted_psd = fitted_psd_init

                    # The PASS 2 mask above keeps all peaks (near-edge sub-band
                    # peaks legitimately guard the band edge), but detections are
                    # filtered to the fit band: below 250 m the residual compares
                    # the spectrum against an extrapolated fit at the radar
                    # resolution limit, so those peaks are not bed periodicities.
                    in_band = fit_mask[peaks] if len(peaks) > 0 else np.zeros(0, dtype=bool)
                    in_band_peaks = peaks[in_band] if len(peaks) > 0 else []
                    dominant_wavelengths = wavelengths_calc[in_band_peaks] if len(in_band_peaks) > 0 else []
                    profile_length = segment_distance.max() - segment_distance.min()
                    # The 2-cycle resolvability limit is set by the PSD grid, whose
                    # longest wavelength is min(segment length, WINDOW_SIZE), not the
                    # full segment length. Using profile_length alone never binds and
                    # lets unresolvable long wavelengths through as "confirmed".
                    grid_length = min(profile_length, WINDOW_SIZE)
                    confidence_flags = flag_wavelength_confidence(dominant_wavelengths, grid_length)

                    # log10 PSD at λ=1 km: intercept + (-beta)*log10(1e-3)
                    psd_amplitude_1km = psd_intercept + 3 * beta

                    hurst_exponent = (beta - 1) / 2
                    hurst_uncertainty = beta_uncertainty / 2
                    # hurst = (beta-1)/2 (Turcotte 1992, along-track), so H in [0,1]
                    # requires beta in [1,3]. Outside that the surface is not
                    # self-affine and the exported hurst is not a valid exponent.
                    self_affine_valid = bool(np.isfinite(beta) and 1.0 <= beta <= 3.0)

                    stats_dict.update({
                        'median_spacing': dx_median,
                        'profile_length': profile_length,
                        'dominant_wavelengths': dominant_wavelengths,
                        'confirmed_wavelengths': confidence_flags['confirmed'],
                        'candidate_wavelengths': confidence_flags['candidate'],
                        'confidence_threshold': confidence_flags['threshold'],
                        'power_law_exponent': beta,
                        'beta_uncertainty': beta_uncertainty,
                        'psd_intercept': psd_intercept,
                        'psd_intercept_uncertainty': psd_intercept_uncertainty,
                        'psd_amplitude_uncertainty': psd_amplitude_uncertainty,
                        'hurst_exponent': hurst_exponent,
                        'hurst_uncertainty': hurst_uncertainty,
                        'self_affine_valid': self_affine_valid
                    })

                    plot_spectra(segment_distance, detrended, wavelengths_calc, avg_psd, fitted_psd, beta, psd_intercept, psd_amplitude_1km, residual_psd, traj_id, dataset_name, segment_number=seg_idx+1, output_path=output_paths['psd'], processing_flag=pflag)
                    psd_spectrum_plot(wavelengths_calc, avg_psd, fitted_psd, beta, psd_intercept, psd_amplitude_1km, traj_id, dataset_name, segment_number=seg_idx+1, output_path=output_paths['psd'], processing_flag=pflag)
                    psd_residuals_plot(wavelengths_calc, residual_psd, traj_id, dataset_name, segment_number=seg_idx+1, output_path=output_paths['psd'], processing_flag=pflag)
                else:
                    print(f"Skipping Line {traj_id}: Not enough data points in 250m–50km range.")

                segment_results.append(stats_dict)

            if valid_segments:
                plot_raw_data_with_segmentation_check(dist, elev, valid_segments, traj_id, gap_mask, output_path=output_paths['trajectories'], processing_flag=pflag)

            if segment_results:
                combined_stats = {}
                list_keys = ['dominant_wavelengths', 'confirmed_wavelengths', 'candidate_wavelengths', 'window_stats', 'wavelength_detections']
                list_keys_collect = ['power_law_exponent', 'hurst_exponent', 'beta_uncertainty', 'hurst_uncertainty', 'psd_intercept', 'psd_intercept_uncertainty', 'psd_amplitude_uncertainty', 'flow_incidence_deg', 'flow_error_mean', 'flow_error_median', 'measures_speed_mean', 'flow_undefined_frac', 'elevation_min', 'elevation_max', 'is_transition', 'self_affine_valid', 'skewness', 'kurtosis']

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
        region_point_clouds[dataset_name] = (pd.concat(region_pc, ignore_index=True)
                                             if region_pc else pd.DataFrame(columns=['x', 'y', 'trajectory_id']))
        print(f"{dataset_name} is finished processing")

    return all_results, region_point_clouds


def results_summary(results):
    if not results: return "no valid data found :("

    def format_stat(values, unit="", dp=1):
        if not values: return "N/A"
        if isinstance(values[0], str): return "N/A (String Data)"
        # nan-safe: one bad segment used to turn the whole line into nan
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if v.size == 0: return "N/A"
        mean_val, min_val, max_val = v.mean(), v.min(), v.max()
        if min_val == max_val:
            return f"{mean_val:.{dp}f}{unit} (Single Value)"
        else:
            return f"Mean: {mean_val:.{dp}f}{unit} | Range: [{min_val:.{dp}f}, {max_val:.{dp}f}]{unit}"

    def flatten(key):
        return [v for r in results.values() for v in r.get(key, [])]

    print("-" * 60)
    print(f"  RESULTS SUMMARY ({len(results)} trajectories aggregated)")
    print("-" * 60)
    # The block below aggregates values that were already averaged per
    # trajectory at the combine step, so its ranges are across trajectory
    # means, not across segments. The spectral block further down is
    # per-segment.
    print("MORPHOMETRY (per-trajectory means of segment values):")

    reliefs = [r['elevation_range'] for r in results.values() if 'elevation_range' in r]
    print(f"VERTICAL RELIEF (Max-Min):\n  -> {format_stat(reliefs, 'm')}")

    elev_mins = [v for r in results.values() for v in r.get('elevation_min', [])]
    elev_maxs = [v for r in results.values() for v in r.get('elevation_max', [])]
    if elev_mins and elev_maxs:
        print(f"ABSOLUTE BED ELEVATION:\n  -> Min: {np.nanmin(elev_mins):.1f}m | Max: {np.nanmax(elev_maxs):.1f}m  (across all trajectories)")

    lengths = [r['profile_length'] for r in results.values() if 'profile_length' in r]
    print(f"AVG SEGMENT LENGTH:\n  -> {format_stat(lengths, 'm')}")

    # Now collected per segment (see list_keys_collect), so flatten across trajectories
    # rather than reading one trajectory-mean each. The printed value is a mean over
    # segments, not a mean of trajectory means.
    skews = [v for r in results.values() for v in np.atleast_1d(r.get('skewness', [])) if np.isfinite(v)]
    kurts = [v for r in results.values() for v in np.atleast_1d(r.get('kurtosis', [])) if np.isfinite(v)]
    if skews:
        print(f"SKEWNESS:\n  -> {format_stat(skews)}")
    if kurts:
        print(f"KURTOSIS (excess):\n  -> {format_stat(kurts)}")

    thickness = [r['ice_thickness_mean'] for r in results.values() if 'ice_thickness_mean' in r and not np.isnan(r['ice_thickness_mean'])]
    if thickness:
        print(f"MEAN ICE THICKNESS:\n  -> {format_stat(thickness, 'm')}")

    print("." * 60)

    # Per-segment spectral fit: the quantities the anisotropy and bed-class
    # work is built on. psd_amplitude_1km is not collected as its own list, so
    # rebuild it here the same way the CSV export does (intercept + 3*beta).
    betas = flatten('power_law_exponent')
    if betas:
        amps = [i + 3 * b for r in results.values()
                for i, b in zip(r.get('psd_intercept', []), r.get('power_law_exponent', []))]
        sav = flatten('self_affine_valid')
        trans = flatten('is_transition')
        pflags = [r['processing_flag'] for r in results.values() if 'processing_flag' in r]

        def med(key):
            v = np.asarray(flatten(key), float)
            v = v[np.isfinite(v)]
            return f"median sigma {np.median(v):.3f}" if v.size else "sigma N/A"

        print(f"POWER-LAW FIT ({len(betas)} segments):")
        print(f"  -> beta:             {format_stat(betas, dp=2)}   {med('beta_uncertainty')}")
        print(f"  -> hurst:            {format_stat(flatten('hurst_exponent'), dp=2)}   {med('hurst_uncertainty')}")
        print(f"  -> log10 PSD @ 1 km: {format_stat(amps, dp=2)}   {med('psd_amplitude_uncertainty')}")
        # hurst = (beta-1)/2 is only a valid exponent for beta in [1,3];
        # transitional segments are excluded from the anisotropy fits.
        print(f"  -> {int(np.sum(sav))}/{len(sav)} self-affine valid | "
              f"{int(np.sum(trans))} transitional | "
              f"processing: {max(set(pflags), key=pflags.count) if pflags else 'unknown'}")
        print("." * 60)

    conf = [w for r in results.values() for w in r.get('confirmed_wavelengths', [])]
    cand = [w for r in results.values() for w in r.get('candidate_wavelengths', [])]

    if conf:
        c = np.asarray(conf, float)
        print(f"CONFIRMED WAVELENGTHS (resolvable, <= grid 2-cycle limit):")
        print(f"  -> Count: {len(conf)}")
        print(f"  -> Median: {np.median(c):.0f}m  (P10 {np.percentile(c, 10):.0f}m, P90 {np.percentile(c, 90):.0f}m)")
    else:
        print("CONFIRMED WAVELENGTHS: None found.")

    if cand:
        # Beyond the grid 2-cycle limit, so not resolvable as periodicities.
        # Report count and median only: the maximum is drawn from the unresolvable
        # low-frequency tail and is not a bed measurement.
        cd = np.asarray(cand, float)
        print(f"CANDIDATE WAVELENGTHS (beyond 2-cycle limit, not resolvable):")
        print(f"  -> Count: {len(cand)}")
        print(f"  -> Median: {np.median(cd):.0f}m")

    if not conf and not cand:
        print("  -> Topography appears Scale Invariant (Fractal/No dominant peaks)")

    # Keyed by trajectory: an unlabelled list is unreadable once a run covers
    # more than one track.
    largest = [(tid, r['max_local_relief'], r.get('loc_of_max_relief', np.nan))
               for tid, r in results.items() if 'max_local_relief' in r]

    if largest:
        print("." * 60)
        print(f"LARGEST LOCAL STRUCTURES ({WINDOW_SIZE/1000:.0f}km Window):")
        for tid, relief, loc in sorted(largest, key=lambda t: -t[1]):
            print(f"  -> {tid}: Relief: {relief:.1f}m at km {loc:.1f}")
        print("." * 60)
    print("=" * 60)
    return {}


if __name__ == "__main__":

    log_path = os.path.join(OUTPUT_BASE_PATH, 'bed_analysis_log.txt')
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(log_path)

    results, region_point_clouds = analyse_bedrock()
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
                    'psd_amplitude_uncertainty': w.get('window_psd_amplitude_uncertainty', np.nan),
                    'hurst': w.get('window_hurst'),
                    'hurst_uncertainty': w.get('window_hurst_uncertainty'),
                    'self_affine_valid': w.get('window_self_affine_valid'),
                    'relief_m': w.get('local_relief_m'),
                    'bed_elev_mean': w.get('bed_elev_mean'),
                    'rms_roughness': w.get('roughness_rms'),
                    'skewness': w.get('window_skewness'),
                    'kurtosis': w.get('window_kurtosis'),
                    # One column at the adopted gate. The snapshot pipeline that ran the
                    # threshold sweep emits hill_count_<threshold> for each of the four.
                    'hill_count': w.get(f'hill_count_{HILL_THRESHOLD_M}'),
                    'eta_wavelength_m': w.get('window_eta_wavelength'),
                    'xi_band': w.get('window_xi_band'),
                    'flow_error_mean': w.get('flow_error_mean'),
                    'flow_error_median': w.get('flow_error_median'),
                    'measures_speed_mean': w.get('measures_speed_mean'),
                    'flow_undefined_frac': w.get('flow_undefined_frac'),
                    'center_x': w.get('center_x'),
                    'center_y': w.get('center_y'),
                    'azimuth_deg': w.get('azimuth_deg'),
                    'is_transition': w.get('is_transition', False),
                    'processing_flag': w.get('processing_flag')
                })
        csv_suffix = f"_w{WINDOW_SIZE // 1000}km" if WINDOW_TYPE == STANDARD_WINDOW else f"_w{WINDOW_SIZE // 1000}km_{WINDOW_TYPE}"
        region_output = os.path.join(OUTPUT_BASE_PATH, f'{get_region_folder(region_name)}{csv_suffix}')
        window_csv_dir = os.path.join(OUTPUT_BASE_PATH, 'window_csvs')
        os.makedirs(window_csv_dir, exist_ok=True)
        pd.DataFrame(all_window_data).to_csv(os.path.join(window_csv_dir, f'{region_name}{csv_suffix}_window_stats.csv'), index=False)

        # --- Coverage point cloud (~1 km), for data-intrinsic coverage tags ---
        pc = region_point_clouds.get(region_name)
        if pc is not None and len(pc):
            coverage_csv_dir = os.path.join(OUTPUT_BASE_PATH, 'coverage_csvs')
            os.makedirs(coverage_csv_dir, exist_ok=True)
            pc.to_csv(os.path.join(coverage_csv_dir, f'{region_name}{csv_suffix}_track_points.csv'), index=False)

        # --- Segment-level CSV ---
        window_df = pd.DataFrame(all_window_data)

        all_segment_data = []
        for traj_id, traj_data in region_results.items():
            betas = traj_data.get('power_law_exponent', [])
            beta_uncerts = traj_data.get('beta_uncertainty', [])
            psd_intercepts = traj_data.get('psd_intercept', [])
            psd_intercept_uncerts = traj_data.get('psd_intercept_uncertainty', [])
            psd_amp_uncerts = traj_data.get('psd_amplitude_uncertainty', [])
            incidences = traj_data.get('flow_incidence_deg', [])
            hursts = traj_data.get('hurst_exponent', [])
            hurst_uncerts = traj_data.get('hurst_uncertainty', [])
            flow_err_means = traj_data.get('flow_error_mean', [])
            flow_err_medians = traj_data.get('flow_error_median', [])
            speed_means = traj_data.get('measures_speed_mean', [])
            flow_undef_fracs = traj_data.get('flow_undefined_frac', [])
            elev_mins = traj_data.get('elevation_min', [])
            elev_maxs = traj_data.get('elevation_max', [])
            seg_skews = traj_data.get('skewness', [])
            seg_kurts = traj_data.get('kurtosis', [])
            is_trans = traj_data.get('is_transition', [])
            self_affine_valids = traj_data.get('self_affine_valid', [])

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
                    'psd_amplitude_uncertainty': psd_amp_uncerts[i] if i < len(psd_amp_uncerts) else np.nan,
                    'hurst': hursts[i] if i < len(hursts) else np.nan,
                    'hurst_uncertainty': hurst_uncerts[i] if i < len(hurst_uncerts) else np.nan,
                    'self_affine_valid': self_affine_valids[i] if i < len(self_affine_valids) else False,
                    'flow_error_mean': flow_err_means[i] if i < len(flow_err_means) else np.nan,
                    'flow_error_median': flow_err_medians[i] if i < len(flow_err_medians) else np.nan,
                    'measures_speed_mean': speed_means[i] if i < len(speed_means) else np.nan,
                    'flow_undefined_frac': flow_undef_fracs[i] if i < len(flow_undef_fracs) else np.nan,
                    'elevation_min': elev_mins[i] if i < len(elev_mins) else np.nan,
                    'elevation_max': elev_maxs[i] if i < len(elev_maxs) else np.nan,
                    'skewness': seg_skews[i] if i < len(seg_skews) else np.nan,
                    'kurtosis': seg_kurts[i] if i < len(seg_kurts) else np.nan,
                    'is_transition': is_trans[i] if i < len(is_trans) else False,
                    'processing_flag': traj_data.get('processing_flag'),
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
            for d in traj_data.get('wavelength_detections', []):
                all_wavelength_data.append({'trajectory': traj_id, **d})
        pd.DataFrame(all_wavelength_data).to_csv(os.path.join(region_output, f'{region_name}{csv_suffix}_wavelength_detections.csv'), index=False)

        print(f"Exported {len(all_window_data)} window rows, {len(all_segment_data)} segment rows, {len(all_wavelength_data)} wavelength detections")

    # --- Data-intrinsic coverage tags across all regions ---
    print("\n=== COVERAGE TAGS ===")
    try:
        import coverage_tags
        coverage_tags.run_all(directory=OUTPUT_BASE_PATH)
    except Exception as e:
        print(f"  coverage tagging skipped: {e}")
