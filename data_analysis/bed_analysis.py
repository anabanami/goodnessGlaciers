import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from pyproj import Transformer
import pprint


def plot_spectra(dist, detrended, wavelengths, psd, fitted_psd, residual_psd, traj_id):
    """
   Plots the spatial profile, power spectrum, and whitened residuals.
    The bottom plot (residuals) spans the full width.
    """
    fig = plt.figure(figsize=(20, 15))
    # grid to place the 3 plots
    gs = fig.add_gridspec(2, 2)
    
    # Spatial Domain
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dist / 1000, detrended, 'k-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Distance along track (km)')
    ax1.set_ylabel('Detrended Bed Elevation (m)')
    ax1.set_title(f'Spatial Profile: {traj_id}')
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Frequency Domain
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(wavelengths, psd, color='k', alpha=0.8, label='Power spectrum density')
    ax2.plot(wavelengths, fitted_psd, color='C1', label='Power-law fit')

    ax2.set_xlabel('Wavelength (m)')
    ax2.set_ylabel('Power Spectral Density ($m^3$)')
    ax2.set_title('Power Spectrum')
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()
    
    # whitened residuals
    ax3 = fig.add_subplot(gs[1, :])
    ax3.semilogx(wavelengths, residual_psd, color='k', alpha=0.5)

    # Highlight peaks
    peaks, _ = signal.find_peaks(residual_psd, height=2.0)
    if len(peaks) > 0:
        # find min and max peaks
        peak_waves = wavelengths[peaks]
        peak_powers = residual_psd[peaks]
        idx_min = np.argmin(peak_waves)
        idx_max = np.argmax(peak_waves)

        # longest wavelengths -> red
        ax3.scatter(peak_waves[idx_max], peak_powers[idx_max], color='C3', s=40, label=f'Max λ: {peak_waves[idx_max]:.0f}m')
        # shortest wavelengths -> blue
        ax3.scatter(peak_waves[idx_min], peak_powers[idx_min], color='C0', s=40, label=f'Min λ: {peak_waves[idx_min]:.0f}m')
        ax3.legend()

    ax3.set_xlabel('Wavelength (m)')
    ax3.set_ylabel('Whitened PSD - ratio to trend')
    ax3.set_title('Whitened Residuals (Normalised)')
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('psd_analysis.png', dpi=500, bbox_inches='tight')
    plt.show()


def analyse_bedrock():
    # Setup projection transformer: WGS84 (Lat/Lon) -> Antarctic Stereo (Meters)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    # DATA LOADING (Existing hardcoded ranges as per JR's analysis)
    filename1 = 'STANFORD_1971_SPRI-NSF-TUD_AIR_BM3.csv' 
    header_df = pd.read_csv(filename1, comment='#', nrows=0)
    df1 = pd.read_csv(filename1, comment='#', skiprows=range(1, 18271), nrows=1501, names=header_df.columns)
    
    print("Stanford Data Preview:")
    print(df1[['bedrock_altitude (m)', 'trajectory_id']].head(),"\n")

    filename2 = 'UTIG_2010_ICECAP_AIR_BM3.csv'
    header_df = pd.read_csv(filename2, comment='#', nrows=0)
    df2 = pd.read_csv(filename2, comment='#', skiprows=range(1, 8508114), nrows=17528, names=header_df.columns)
    
    df = pd.concat([df1, df2], ignore_index=True)

    # Filter invalid data
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]

    results = {}
    plot_count = 0  # Counter to limit number of plots popping up

    for traj_id in valid['trajectory_id'].unique():
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

        # SIGNAL PROCESSING
        elev = line['bedrock_altitude (m)'].values
        detrended = signal.detrend(elev)

        ice_thickness = line['land_ice_thickness (m)'].values
        valid_ice_thickness = ice_thickness[ice_thickness != -9999]

        stats_dict = {
            'elevation_range': elev.max() - elev.min(),
            'skewness': stats.skew(detrended),
            'kurtosis': stats.kurtosis(detrended),
            'rms_roughness': np.sqrt(np.mean(detrended**2))
        }
        
        if len(valid_ice_thickness) > 0:
            stats_dict['ice_thickness_avg'] = np.mean(valid_ice_thickness)

        # SPECTRAL ANALYSIS
        # Median sampling interval
        dx_median = np.median(np.diff(dist))
        
        if dx_median > 0 and len(detrended) > 10:
            # Hanning window
            windowed_data = detrended * np.hanning(len(detrended))
            
            # Pad to power of 2
            N = 2**int(np.ceil(np.log2(len(windowed_data))))
            padded_data = np.zeros(N)
            padded_data[:len(windowed_data)] = windowed_data
            
            freqs = np.fft.fftfreq(N, dx_median)
            ft_vals = np.abs(np.fft.fft(padded_data))**2

            # Filter positive frequencies
            pos_mask = freqs > 0
            pos_freqs = freqs[pos_mask]
            pos_psd = ft_vals[pos_mask]
            wavelengths_calc = 1 / pos_freqs

            #####################################
            # Create mask for 100m - 10km range
            mask = (wavelengths_calc >= 100) & (wavelengths_calc <= 10000)
            # Fit power law: P(f) = A * f^(-β)
            log_freqs = np.log10(pos_freqs)
            log_psd = np.log10(pos_psd)
            # Fit only on masked data
            slope, intercept = np.polyfit(log_freqs[mask], log_psd[mask], 1)
            beta = -slope # Power law exponent
            # Apply fit to the full range
            fitted_psd = 10**(intercept + slope * log_freqs)

            # Calculate residuals by ratio(whitened spectrum)
            residual_psd = pos_psd / fitted_psd
            #####################################

            # Find peaks
            peaks, _ = signal.find_peaks(residual_psd, height=2.0)
            dominant_wavelengths = wavelengths_calc[peaks] if len(peaks) > 0 else []

            stats_dict.update({
                'median_spacing': dx_median,
                'profile_length': dist.max() - dist.min(),
                'dominant_wavelengths': dominant_wavelengths,
                'power_law_exponent': beta
            })

            # Plot the first n lines
            if plot_count < 5:
                plot_spectra(dist, detrended, wavelengths_calc, pos_psd, fitted_psd, residual_psd, traj_id)
                plot_count += 1

        results[traj_id] = stats_dict

    return results


def results_summary(results):
    if not results:
        return "no valid data found :("

    def make_stats(values):
        return {'mean': np.mean(values), 'range': [np.min(values), np.max(values)]}

    params = {param: [r[param] for r in results.values() if param in r]
            # for param in ['elevation_range', 'skewness', 'kurtosis', 'rms_roughness', 'ice_thickness_avg', 
            #              'median_spacing', 'profile_length', 'power_law_exponent']}
            for param in ['elevation_range', 'profile_length', 'power_law_exponent']}
    summary = {param: make_stats(vals) for param, vals in params.items() if vals}

    wavelengths = [w for r in results.values() if 'dominant_wavelengths' in r for w in r['dominant_wavelengths']]

    # Explicitly report if no periodicities were found
    if wavelengths:
        summary['wavelengths'] = make_stats(wavelengths)
    else:
        summary['wavelengths'] = 'No significant periodicities detected (Scale Invariant)'

    pprint.pprint(summary)
    return summary

if __name__=="__main__":
    results = analyse_bedrock()
    print(f"Analysed {len(results)} flight lines")
    results_summary(results)