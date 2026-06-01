import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from config import peak_masking_height_threshold


def plot_raw_data_with_segmentation_check(dist, elev, segments, traj_id, gap_mask=None, output_path=None):
    plt.figure(figsize=(18, 6))

    plot_elev = elev.copy().astype(float)
    if gap_mask is not None:
        steps = np.diff(dist)
        gap_breaks = np.where(steps > 2000)[0]
        for idx in gap_breaks:
            plot_elev[idx+1] = np.nan

    plt.plot(dist/1000, plot_elev, color='0.4', linewidth=0.8, label='Raw Data (with breaks)', alpha=0.5)

    if gap_mask is not None and np.any(gap_mask):
        plt.scatter(dist[gap_mask]/1000, elev[gap_mask],
                   color='red', marker='x', s=25, zorder=5, label='Gap Boundaries')

    for i, (seg_data, seg_dist) in enumerate(segments):
        seg_elev = seg_data['bedrock_altitude (m)'].values
        plt.scatter(seg_dist/1000, seg_elev, s=15, label=f'Segment {i+1}')

    plt.xlabel('Distance along track (km)')
    plt.ylabel('Bed Elevation (m)')
    plt.title(f'Segmentation Check: {traj_id} ({len(segments)} valid segments)')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_path, f'{traj_id}.png') if output_path else f'{traj_id}.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def plot_spectra(dist, detrended, wavelengths, psd, fitted_psd, beta, C,  residual_psd,
                 traj_id, dataset_name, segment_number=None, output_path=None):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dist / 1000, detrended, 'k-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Distance along track (km)')
    ax1.set_ylabel('Detrended Bed Elevation (m)')
    segment_label = f' - Segment {segment_number}' if segment_number is not None else ''
    ax1.set_title(f'Spatial Profile: {traj_id}{segment_label}')
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(wavelengths, psd, color='k', alpha=0.8, label='Power spectrum density')
    ax2.plot(wavelengths, fitted_psd, color='C1', label=fR'Power-law fit: $\beta$={beta:.1f}, $C_{{\log}}$={C:.1f}')
    ax2.set_xlabel('Wavelength (m)')
    ax2.set_ylabel('Power Spectral Density ($m^3$)')
    ax2.set_title('Power Spectrum')
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, :])
    ax3.semilogx(wavelengths, residual_psd, color='k', alpha=0.5)

    peaks, _ = signal.find_peaks(residual_psd, height=peak_masking_height_threshold)
    if len(peaks) > 0:
        peak_waves = wavelengths[peaks]
        peak_powers = residual_psd[peaks]
        idx_min = np.argmin(peak_waves)
        idx_max = np.argmax(peak_waves)
        ax3.scatter(peak_waves[idx_max], peak_powers[idx_max], color='C3', s=40, alpha=1, label=f'Max λ: {peak_waves[idx_max]:.0f}m')
        ax3.scatter(peak_waves[idx_min], peak_powers[idx_min], color='C0', s=40, alpha=1, label=f'Min λ: {peak_waves[idx_min]:.0f}m')
        ax3.legend()

    ax3.set_xlabel('Wavelength (m)')
    ax3.set_ylabel('Whitened PSD - ratio to trend')
    ax3.set_title('Whitened Residuals (Normalised)')
    ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    segment_suffix = f'_seg{segment_number}' if segment_number is not None else ''
    filename = f'psd_analysis_{dataset_name}_{traj_id}{segment_suffix}.png'
    save_path = os.path.join(output_path, filename) if output_path else filename
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
