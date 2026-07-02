import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
from scipy import signal
import os
from config import peak_masking_height_threshold, PROCESSING_FLAG_COLORS as _FLAG_COLOR


def _two_colour_title(base_title, processing_flag, fontsize):
    """HPacker of a black base title and a colour-coded [flag] tag."""
    base = TextArea(base_title, textprops=dict(color='black', fontsize=fontsize))
    tag = TextArea(f'[{processing_flag}]',
                   textprops=dict(color=_FLAG_COLOR.get(processing_flag, '0.3'), fontsize=fontsize))
    return HPacker(children=[base, tag], sep=6, pad=0, align='center')


def label_trajectories(ax, df, transformer, drawn_traj_ids=None, scale=1000.0,
                       gap_threshold=2000, min_points=20):
    """Draw one label per trajectory, anchored at the midpoint of its longest
    gap-free run. Placement is computed from the RAW trajectory coordinates (not
    from any downstream segmentation), so it is identical across plots that share
    the same df and transformer. Coordinates are divided by `scale` (metres->km).

    drawn_traj_ids: if given, only label these trajectories (e.g. the ones that
    actually survived filtering and got drawn), so no label floats over a blank track.
    """
    keep = None if drawn_traj_ids is None else set(map(str, drawn_traj_ids))

    for tid in df['trajectory_id'].unique():
        if keep is not None and str(tid) not in keep:
            continue
        line = df[df['trajectory_id'] == tid]
        if len(line) < min_points:
            continue

        lx, ly = transformer.transform(line['longitude (degree_east)'].values,
                                       line['latitude (degree_north)'].values)
        lx, ly = np.asarray(lx), np.asarray(ly)
        if len(lx) < 2:
            continue

        # Split into gap-free runs and pick the longest
        step = np.sqrt(np.diff(lx) ** 2 + np.diff(ly) ** 2)
        bounds = np.concatenate([[0], np.where(step > gap_threshold)[0] + 1, [len(lx)]])
        s, e = max(zip(bounds[:-1], bounds[1:]), key=lambda se: se[1] - se[0])
        mid = (s + e) // 2

        ax.text(lx[mid] / scale, ly[mid] / scale, str(tid),
                fontsize=6, fontweight='bold', ha='center', va='center',
                color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, linewidth=0))


def flag_title(ax, base_title, processing_flag, fontsize=None):
    """Axes title: base name in black, migration-status tag in colour."""
    if processing_flag is None:
        ax.set_title(base_title, fontsize=fontsize)
        return
    ax.set_title('')  # clear default so only the two-colour title shows
    pack = _two_colour_title(base_title, processing_flag, fontsize or plt.rcParams['axes.titlesize'])
    ab = AnnotationBbox(pack, (0.5, 1.0), xycoords='axes fraction',
                        box_alignment=(0.5, 0.0), frameon=False, pad=0,
                        annotation_clip=False)
    ax.add_artist(ab)


def flag_suptitle(fig, base_title, processing_flag, fontsize=14):
    """Figure title: base name in black, migration-status tag in colour."""
    if not processing_flag:
        fig.suptitle(base_title, fontsize=fontsize)
        return
    pack = _two_colour_title(base_title, processing_flag, fontsize)
    # Anchor just above the figure's top edge (growing upward) so it clears subplot
    # titles, mirroring fig.suptitle(y=1.02); bbox_inches='tight' captures it on save.
    ab = AnnotationBbox(pack, (0.5, 1.0), xycoords='figure fraction',
                        box_alignment=(0.5, 0.0), frameon=False, pad=0,
                        annotation_clip=False)
    fig.add_artist(ab)


def plot_raw_data_with_segmentation_check(dist, elev, segments, traj_id, gap_mask=None, output_path=None,
                                          processing_flag=None):
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
    flag_title(plt.gca(), f'Segmentation Check: {traj_id} ({len(segments)} valid segments)', processing_flag)
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_path, f'{traj_id}.png') if output_path else f'{traj_id}.png'
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def plot_spectra(dist, detrended, wavelengths, psd, fitted_psd, beta, psd_intercept,  residual_psd,
                 traj_id, dataset_name, segment_number=None, output_path=None, processing_flag=None):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dist / 1000, detrended, 'k-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Distance along track (km)')
    ax1.set_ylabel('Detrended Bed Elevation (m)')
    segment_label = f' - Segment {segment_number}' if segment_number is not None else ''
    flag_title(ax1, f'Spatial Profile: {traj_id}{segment_label}', processing_flag)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(wavelengths, psd, color='k', alpha=0.8, label='Power spectrum density')
    ax2.plot(wavelengths, fitted_psd, color='C1', label=f'Power-law fit: β={beta:.1f}, psd_intercept={psd_intercept:.1f}')
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
