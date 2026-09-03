"""Directional beta from 1D profiles at 5 degree azimuth intervals.

Lays parallel transects across a DEM at one bearing, cuts each into 50 km
windows and fits beta on every window, then rotates and repeats. The window
fit is the one weighted_anisotropy consumes in production.


python azimuth_rake.py
python azimuth_rake.py null_seeds/seed_001        # rake only one bed

"""
import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from scipy import signal
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

ROOT_DIR = '/home/ana/Desktop/code/Data/ODSA'

import sys
sys.path.insert(0, ROOT_DIR)
from config import (FIT_BAND_M, WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE,
                    peak_masking_height_threshold, bin_buffer)
from bed_analysis import _amp_uncertainty


# ---------------------------------------------------------------- geometry

def chord(bounds, bearing_deg, offset_m):
    """Endpoints of the line at this perpendicular offset, clipped to bounds.

    Bearing is degrees clockwise from grid north. Returns None if the line
    misses the rectangle.
    """
    b = np.radians(bearing_deg)
    d = np.array([np.sin(b), np.cos(b)])
    n = np.array([np.cos(b), -np.sin(b)])
    p0 = offset_m * n
    s_lo, s_hi = -np.inf, np.inf
    for axis, lo, hi in ((0, bounds.left, bounds.right), (1, bounds.bottom, bounds.top)):
        if abs(d[axis]) < 1e-12:
            if not lo <= p0[axis] <= hi:
                return None
        else:
            a, c = (lo - p0[axis]) / d[axis], (hi - p0[axis]) / d[axis]
            s_lo, s_hi = max(s_lo, min(a, c)), min(s_hi, max(a, c))
    if s_hi <= s_lo:
        return None
    return p0 + s_lo * d, p0 + s_hi * d


def offsets(bounds, bearing_deg, spacing_m):
    """Perpendicular offsets of a rake of parallel lines covering the bounds."""
    b = np.radians(bearing_deg)
    n = np.array([np.cos(b), -np.sin(b)])
    corners = np.array([[bounds.left, bounds.bottom], [bounds.right, bounds.bottom],
                        [bounds.right, bounds.top], [bounds.left, bounds.top]])
    u = corners @ n
    return np.arange(u.min() + spacing_m / 2, u.max(), spacing_m)


def sample(src, arr, nodata_mask, p_start, p_end, step_m):
    """Elevation sampled along a line at step_m, with masked cells dropped.

    Returns distance along the line and elevation, both from the surviving
    samples only, so the spacing is uneven wherever the mask bites.
    """
    length = np.hypot(*(p_end - p_start))
    n = int(length // step_m) + 1
    dist = np.arange(n) * step_m
    pts = p_start + np.outer(dist, (p_end - p_start) / length)
    rows, cols = rasterio.transform.rowcol(src.transform, pts[:, 0], pts[:, 1])
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    elev = map_coordinates(arr, [rows, cols], order=1, mode='nearest')
    good = np.isfinite(elev)
    if nodata_mask is not None:
        # Nearest on the mask, so a masked cell never bleeds into its neighbours
        good &= map_coordinates(nodata_mask, [rows, cols], order=0, mode='nearest') == 0
    return dist[good], elev[good]


# ---------------------------------------------------------------- spectra

def window_psds(dist, elev, freqs, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Copy of the window loop in bed_analysis.analyse_sliding_windows (~line 128).

    One periodogram per window: detrend, taper, Lomb-Scargle on the shared
    frequency grid. Returns the centre distance of each window alongside its
    periodogram, since a window holding too few samples is dropped and the
    position in the list is therefore not the position along the transect.
    """
    angular_freqs = freqs * 2 * np.pi
    centres, out = [], []
    start = dist.min()
    while start + window_size <= dist.max() + 1e-6:
        m = (dist >= start) & (dist <= start + window_size)
        w_dist, w_elev = dist[m], elev[m]
        if len(w_dist) > 50:
            w_detrended = signal.detrend(w_elev)
            w_tapered = w_detrended
            if WINDOW_TYPE == 'hann':
                w_tapered = w_detrended * signal.windows.hann(len(w_detrended))
            elif WINDOW_TYPE == 'tukey':
                w_tapered = w_detrended * signal.windows.tukey(len(w_detrended), alpha=0.5)
            out.append(signal.lombscargle(w_dist, w_tapered, angular_freqs, normalize=False))
            centres.append(start + window_size / 2)
        start += step_size
    return centres, out


def window_betas(pgrams, freqs):
    """Copy of the per-window fit in bed_analysis.analyse_sliding_windows (~line 227).
    The peak mask comes from the transect's averaged PSD, since per-window peak
    detection is noisy on a single periodogram.
    """
    log_freqs = np.log10(freqs)
    wavelengths = 1 / freqs
    band = (wavelengths >= FIT_BAND_M[0]) & (wavelengths <= FIT_BAND_M[1])
    avg = np.mean(np.array(pgrams), axis=0)

    clean = band.copy()
    if np.sum(band) >= 2 and np.all(avg[band] > 0):
        s0, i0 = np.polyfit(log_freqs[band], np.log10(avg[band]), 1)
        residual = avg / 10 ** (i0 + s0 * log_freqs)
        for p_idx in signal.find_peaks(residual, height=peak_masking_height_threshold)[0]:
            clean[max(0, p_idx - bin_buffer):p_idx + bin_buffer + 1] = False

    n_fit = int(np.sum(clean))
    out = []
    for pgram in pgrams:
        row = dict(beta=np.nan, beta_uncertainty=np.nan, psd_intercept=np.nan,
                   psd_intercept_uncertainty=np.nan, A_1km_uncertainty=np.nan)
        if n_fit >= 2 and np.all(pgram > 0):
            log_psd = np.log10(pgram)
            try:
                if n_fit > 2:
                    coeffs, cov = np.polyfit(log_freqs[clean], log_psd[clean], 1, cov=True)
                    row.update(beta_uncertainty=np.sqrt(cov[0, 0]),
                               psd_intercept_uncertainty=np.sqrt(cov[1, 1]),
                               A_1km_uncertainty=_amp_uncertainty(cov))
                else:
                    coeffs = np.polyfit(log_freqs[clean], log_psd[clean], 1)
                row.update(beta=-coeffs[0], psd_intercept=coeffs[1])
            except (np.linalg.LinAlgError, ValueError):
                pass
        out.append(row)
    return out


def freq_grid(step_m, window_size=WINDOW_SIZE):
    # Same grid bed_analysis builds, including its 15 m floor on sample spacing
    return np.geomspace(1 / window_size, 1 / (2 * max(step_m, 15.0)), num=500)


def azimuth_windows(src, arr, nodata_mask, bearing_deg, spacing_m, step_m):
    """Every 50 km window at one bearing, each fitted on its own periodogram."""
    freqs = freq_grid(step_m)
    windows, n_transects = [], 0
    for off in offsets(src.bounds, bearing_deg, spacing_m):
        ends = chord(src.bounds, bearing_deg, off)
        if ends is None:
            continue
        dist, elev = sample(src, arr, nodata_mask, *ends, step_m)
        if len(dist) < 50:
            continue
        centres, pgrams = window_psds(dist, elev, freqs)
        if pgrams:
            # Same mapping sample() uses, so a centre distance gives the map point
            unit = (ends[1] - ends[0]) / np.hypot(*(ends[1] - ends[0]))
            for k, (c, row) in enumerate(zip(centres, window_betas(pgrams, freqs))):
                x, y = ends[0] + c * unit
                windows.append(dict(bearing_deg=bearing_deg, transect=n_transects,
                                    window=k, centre_m=c, x=x, y=y, **row))
            n_transects += 1
    return windows, n_transects


# ---------------------------------------------------------------- driver

def load_fabric(fabric_csv):
    """Fabric nodes as a lookup tree, their bearings, and the lattice spacing."""
    with open(fabric_csv) as f:
        rows = list(csv.DictReader(f))
    xy = np.array([[float(r['x']), float(r['y'])] for r in rows])
    bearing = np.array([float(r['bearing_deg']) for r in rows])
    ux = np.unique(xy[:, 0])
    step = float(np.median(np.diff(ux))) if len(ux) > 1 else WINDOW_SIZE
    return cKDTree(xy), bearing, step


def window_flow(tree, bearing, step, centre, unit, window_size=WINDOW_SIZE):
    """Fabric axis over the ground one window covers, and two flags on it.

    The nearest node is taken at points spaced `step` apart along the window,
    and the bearings are averaged as axes rather than directions. Returns the
    axis, the concentration of the average, and the mean distance to the nodes
    the average was taken from.
    """
    n = max(int(window_size // step) + 1, 2)
    offs = np.linspace(-window_size / 2, window_size / 2, n)
    dist, idx = tree.query(centre + np.outer(offs, unit))
    a = np.radians(bearing[idx]) * 2
    c, s = np.cos(a).mean(), np.sin(a).mean()
    return (np.degrees(np.arctan2(s, c)) / 2 % 180, float(np.hypot(c, s)),
            float(dist.mean()))


def rake(dem_path, water_path=None, fabric_csv=None, bearings=None,
         spacing_m=5000.0, step_m=10.0):
    bearings = np.arange(0, 180, 5) if bearings is None else bearings
    fabric = load_fabric(fabric_csv) if fabric_csv else None
    with rasterio.open(dem_path) as src:
        arr = src.read(1).astype(np.float32)
        bad = ~np.isfinite(arr)
        if src.nodata is not None:
            bad |= arr == src.nodata
        if water_path:
            with rasterio.open(water_path) as w:
                bad |= w.read(1) > 0
        bad = bad.astype(np.uint8) if bad.any() else None
        rows = []
        for b in bearings:
            wins, n_transects = azimuth_windows(src, arr, bad, float(b), spacing_m, step_m)
            unit = np.array([np.sin(np.radians(b)), np.cos(np.radians(b))])
            for w in wins:
                flow, R, fdist = (window_flow(*fabric, np.array([w['x'], w['y']]), unit)
                                  if fabric else (np.nan, np.nan, np.nan))
                # theta is the acute angle between transect and fabric, so 0 is along flow
                w['theta_deg'] = abs((float(b) - flow + 90) % 180 - 90)
                w['flow_bearing_deg'] = flow
                w['fabric_R'] = R
                w['fabric_dist_m'] = fdist
            rows.extend(wins)
            betas = [w['beta'] for w in wins if np.isfinite(w['beta'])]
            print(f"  {b:5.1f} deg  transects {n_transects:3d}  windows {len(wins):4d}  "
                  f"median beta {np.median(betas):.3f}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('seed_dir', help='folder holding <name>_dem.tif')
    p.add_argument('--spacing-m', type=float, default=5000.0)
    p.add_argument('--step-m', type=float, default=10.0)
    p.add_argument('--out', default=None)
    p.add_argument('--remove-dem', action='store_true',
                   help='delete <name>_dem.tif once the CSV holds a usable fit')
    a = p.parse_args()

    d = Path(a.seed_dir)
    name = d.name
    dem = d / f'{name}_dem.tif'
    water = d / f'{name}_water.tif'
    fabric = d / f'{name}_fabric.csv'
    rows = rake(dem, water if water.exists() else None,
                fabric if fabric.exists() else None,
                spacing_m=a.spacing_m, step_m=a.step_m)

    out = Path(a.out) if a.out else d / f'{name}_window_beta.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {out}, {len(rows)} windows')

    if a.remove_dem:
        # Only once the fit is on disk, so a failed rake never costs the bed
        if sum(np.isfinite(r['beta']) for r in rows) >= 2:
            mb = dem.stat().st_size / 1e6
            dem.unlink()
            print(f'removed {dem.name} ({mb:.0f} MB), regenerate with '
                  f"null_seeds.py --start {name.split('_')[-1].lstrip('0')} --n 1")
        else:
            print(f'kept {dem.name}: fewer than 2 bearings produced a beta')


if __name__ == '__main__':
    main()
