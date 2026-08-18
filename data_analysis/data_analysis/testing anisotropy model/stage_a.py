"""Stage A — dense-grid beta(theta), and the first real test of the cos^2 model form.

Cuts parallel 1-D profiles through a prepped DEM at every azimuth and runs each through the
production spectral estimator (imported from config / bed_analysis's recipe), so beta is
the same quantity the pipeline reports and Stage B's sparse-track run is comparable to it.
theta comes from the local mapped fabric, not one site-wide bearing.

    python stage_a.py                       # all ready sites in prep/
    python stage_a.py --hull                # read prep_hull/ instead
    python stage_a.py --site "Site F Nunavut" --az-step 5 --sample-m 20

Water and nodata samples are dropped, not interpolated over — Lomb-Scargle takes the
irregular remainder, which is exactly how the pipeline handles gappy tracks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage, optimize, signal, spatial, stats

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
sys.path.insert(0, str(HERE.parent))
from config import (FIT_BAND_M, WINDOW_SIZE, WINDOW_TYPE,          # noqa: E402
                    bin_buffer, peak_masking_height_threshold)
from bed_analysis import _amp_uncertainty                          # noqa: E402
from weighted_anisotropy import cos2_model, fit_cos2               # noqa: E402

AZ_STEP = 5.0           # degrees
SAMPLE_M = 10.0         # along-profile spacing; matches the 10 m mosaics. --sample-m overrides
SPACING_M = 5000.0      # between parallel profiles
MAX_MASKED = 0.30       # reject a profile that loses more than this to water/nodata
MIN_SAMPLES = 200
N_FREQ = 500


def profile_beta(dist, elev, length, band_lo=FIT_BAND_M[0], band_hi=FIT_BAND_M[1]):
    """One profile through the production recipe: detrend, Hann, Lomb-Scargle, fit the band.
    Returns beta, its sigma, profile rms, and psd_amplitude_1km with its sigma. The amplitude
    is the in-band quantity; rms is whole-profile and carries wavelengths above the band."""
    freqs = np.geomspace(1 / length, 1 / (2 * max(SAMPLE_M, 15.0)), N_FREQ)
    wl = 1 / freqs
    band = (wl >= band_lo) & (wl <= min(band_hi, length))
    log_f = np.log10(freqs)

    d = signal.detrend(elev)
    taper = signal.windows.hann(len(d)) if WINDOW_TYPE == 'hann' else \
        signal.windows.tukey(len(d), alpha=0.5) if WINDOW_TYPE == 'tukey' else np.ones(len(d))
    pgram = signal.lombscargle(dist, d * taper, freqs * 2 * np.pi, normalize=False)
    if not np.all(pgram > 0) or band.sum() < 2:
        return np.nan, np.nan, np.sqrt(np.mean(d ** 2)), np.nan, np.nan

    # First-pass peak mask, same recipe and constants as the per-window path.
    clean = band.copy()
    s0, i0 = np.polyfit(log_f[band], np.log10(pgram[band]), 1)
    resid = pgram / 10 ** (i0 + s0 * log_f)
    for p in signal.find_peaks(resid, height=peak_masking_height_threshold)[0]:
        clean[max(0, p - bin_buffer):p + bin_buffer + 1] = False
    if clean.sum() < 3:
        clean = band

    # psd_amplitude_1km = intercept + 3*beta, since log10(1/1000 m) sits at -3.
    (slope, icept), cov = np.polyfit(log_f[clean], np.log10(pgram[clean]), 1, cov=True)
    return (-slope, float(np.sqrt(cov[0, 0])), float(np.sqrt(np.mean(d ** 2))),
            float(icept - 3 * slope), float(_amp_uncertainty(cov)))


def _chord(base, u, lo, hi):
    """Parameter range where base + t*u stays inside the raster rectangle (slab clip)."""
    tmin, tmax = -np.inf, np.inf
    for o, d, a, b in zip(base, u, lo, hi):
        if abs(d) < 1e-9:
            if o < a or o > b:
                return None
            continue
        t1, t2 = sorted(((a - o) / d, (b - o) / d))
        tmin, tmax = max(tmin, t1), min(tmax, t2)
    return (tmin, tmax) if tmax > tmin else None


def chords(shape, tr, az_deg):
    """Yield (base, u, tmin, tmax) for each parallel line crossing the grid at this azimuth."""
    h, w = shape
    px, x0, y0 = tr.a, tr.c, tr.f
    lo = (x0 + px, y0 - h * px + px)
    hi = (x0 + w * px - px, y0 - px)
    cx, cy = x0 + w * px / 2, y0 - h * px / 2
    a = np.radians(az_deg)
    u, v = np.array([np.sin(a), np.cos(a)]), np.array([np.cos(a), -np.sin(a)])
    diag = np.hypot(w, h) * px
    for off in np.arange(-diag / 2, diag / 2, SPACING_M):
        base = np.array([cx, cy]) + off * v
        seg = _chord(base, u, lo, hi)
        if seg:
            yield base, u, seg[0], seg[1]


def max_chord(shape, tr, az_step):
    """Longest profile that fits at *every* azimuth. Length must not vary with azimuth:
    band truncation reads short profiles steeper, so a length that tracks azimuth would
    forge the very Δβ this stage measures."""
    return min((max((t1 - t0 for *_, t0, t1 in chords(shape, tr, az)), default=0.0)
                for az in np.arange(0, 180, az_step)), default=0.0)


def cut(dem, mask, tr, az_deg, length):
    """Parallel non-overlapping fixed-length profiles at one azimuth, in map coords."""
    px, x0, y0 = tr.a, tr.c, tr.f
    t = np.arange(0, length, SAMPLE_M)
    for base, u, tmin, tmax in chords(dem.shape, tr, az_deg):
        for start in np.arange(tmin, tmax - length, length):
            p = base + np.outer(start + t, u)
            col, row = (p[:, 0] - x0) / px, (y0 - p[:, 1]) / px
            bad = ndimage.map_coordinates(mask, [row, col], order=0, mode='nearest') > 0
            if bad.mean() > MAX_MASKED or (~bad).sum() < MIN_SAMPLES:
                continue
            z = ndimage.map_coordinates(dem, [row, col], order=1, mode='nearest')
            yield t[~bad], z[~bad], p[len(p) // 2], bad.mean()


def aic(resid, k):
    n = len(resid)
    return n * np.log(np.sum(resid ** 2) / n) + 2 * k


def compare_models(th, b):
    """Is cos^2 the right interpolation, or just a curve that happens to fit?"""
    out, preds = {}, {}
    preds['flat'] = np.full_like(b, b.mean())
    try:
        p, _ = optimize.curve_fit(cos2_model, th, b, p0=[b.mean(), b.mean()], maxfev=5000)
        preds['cos2'] = cos2_model(th, *p)
    except (RuntimeError, ValueError):
        pass
    abscos = lambda t, lo, hi: lo + (hi - lo) * np.abs(np.cos(np.radians(t)))  # noqa: E731
    try:
        p, _ = optimize.curve_fit(abscos, th, b, p0=[b.mean(), b.mean()], maxfev=5000)
        preds['abscos'] = abscos(th, *p)
    except (RuntimeError, ValueError):
        pass
    near = th < 45
    if near.any() and (~near).any():
        preds['step45'] = np.where(near, b[near].mean(), b[~near].mean())

    aics = {n: float(aic(b - p, 1 if n == 'flat' else 2)) for n, p in preds.items()}
    out |= {f'aic_{n}': round(v, 2) for n, v in aics.items()}
    best = min(aics, key=aics.get)
    out['best_model'] = best
    out['delta_aic_cos2_vs_best'] = round(aics.get('cos2', np.nan) - aics[best], 2)

    # Cooper's two-bin contrast, reported alongside rather than as a competing fit.
    par, perp = th <= 20, th >= 70
    if par.sum() >= 5 and perp.sum() >= 5:
        out |= {'cooper_par': round(float(b[par].mean()), 3),
                'cooper_perp': round(float(b[perp].mean()), 3),
                'cooper_delta': round(float(b[par].mean() - b[perp].mean()), 3),
                'cooper_n': [int(par.sum()), int(perp.sum())]}
    return out


def grid_bias(th, b, ax):
    """Interpolation along an arbitrary bearing is angle dependent (none at 0/90, worst near
    45). Correlate the cos^2 residual against distance from the nearest grid axis, then refit
    with a linear term in it removed: if beta(theta) tracks the sampler, delta moves."""
    p, _ = optimize.curve_fit(cos2_model, th, b, p0=[b.mean(), b.mean()], maxfev=5000)
    r, pv = stats.pearsonr(b - cos2_model(th, *p), ax)
    coef = np.linalg.lstsq(np.c_[np.ones(len(b)), np.cos(np.radians(th)) ** 2, ax],
                           b, rcond=None)[0][2]
    out = {'grid_axis_r': round(float(r), 3), 'grid_axis_p': float(f'{pv:.1e}'),
           'grid_axis_coef': round(float(coef), 5)}
    fit = fit_cos2(th, b - coef * ax, quiet=True)
    if fit:
        out |= {'delta_beta_grid_corrected': round(fit['delta'], 3),
                'delta_se_grid_corrected': round(fit['delta_se'], 3)}
    return out


def run(site, outdir, az_step, length=None, band_lo=FIT_BAND_M[0], band_hi=FIT_BAND_M[1]):
    name = site.name
    # Outputs are suffixed when the band edge is not production, so runs sit side by side.
    sfx = '' if (band_lo, band_hi) == FIT_BAND_M else f'_b{int(band_lo)}-{int(band_hi)}'
    print(f'\n{name}')
    with rasterio.open(outdir / f'{name}_dem.tif') as src:
        dem, tr, nod = src.read(1).astype('float32'), src.transform, src.nodata
    with rasterio.open(outdir / f'{name}_water.tif') as src:
        mask = src.read(1)
    mask = (mask > 0) | (dem == (nod if nod is not None else -9999))

    fab = pd.read_csv(outdir / f'{name}_fabric.csv')
    tree = spatial.cKDTree(fab[['x', 'y']].values)

    fits = max_chord(dem.shape, tr, az_step)
    L = length or WINDOW_SIZE
    if L > fits:
        print(f'  NOTE  {L/1000:.0f} km does not fit at every azimuth '
              f'(max isotropic length {fits/1000:.1f} km) — using {fits*0.95/1000:.1f} km')
        L = np.floor(fits * 0.95 / 1000) * 1000
    print(f'  profile length {L/1000:.0f} km, band {band_lo:.0f} m - {min(band_hi, L):.0f} m'
          f'  (production {WINDOW_SIZE/1000:.0f} km, {FIT_BAND_M[0]:.0f} m)')

    rows = []
    for az in np.arange(0, 180, az_step):
        for dist, z, mid, mfrac in cut(dem, mask, tr, az, L):
            d, i = tree.query(mid)
            if d > SPACING_M:
                continue
            beta, se, rms, amp, amp_se = profile_beta(dist, z, L, band_lo, band_hi)
            if not np.isfinite(beta):
                continue
            bearing = fab.bearing_deg.values[i]
            th = np.abs(az - bearing) % 180
            rows.append(dict(azimuth_deg=az, bearing_deg=bearing,
                             incidence_deg=min(th, 180 - th), beta=beta, beta_se=se,
                             rms=rms, amp=amp, amp_se=amp_se,
                             grid_axis_deg=min(az % 90, 90 - az % 90),
                             x=mid[0], y=mid[1], masked_frac=mfrac, n=len(dist)))
        print(f'  az {az:5.1f}  cumulative profiles {len(rows)}', end='\r')

    d = pd.DataFrame(rows)
    if d.empty:
        print(f'  no usable profiles')
        return {'site': name, 'status': 'no profiles'}
    d.to_csv(outdir / f'{name}_stage_a_profiles{sfx}.csv', index=False)

    by_az = d.groupby('azimuth_deg').agg(
        n=('beta', 'size'), beta_median=('beta', 'median'),
        beta_p16=('beta', lambda v: v.quantile(.16)),
        beta_p84=('beta', lambda v: v.quantile(.84)),
        theta_median=('incidence_deg', 'median')).reset_index()
    by_az.to_csv(outdir / f'{name}_stage_a_beta_theta{sfx}.csv', index=False)

    th, b = d.incidence_deg.values, d.beta.values
    qc = {'site': name, 'band_m': [band_lo, min(band_hi, L)], 'sample_m': SAMPLE_M,
          'profile_len_km': L / 1000, 'production_window_km': WINDOW_SIZE / 1000,
          'max_isotropic_len_km': round(fits / 1000, 1),
          'n_profiles': len(d), 'n_azimuths': int(d.azimuth_deg.nunique()),
          'theta_range': [round(float(th.min()), 1), round(float(th.max()), 1)],
          'beta_median': round(float(np.median(b)), 3),
          'masked_frac_median': round(float(d.masked_frac.median()), 3)}
    ref = fit_cos2(th, b, quiet=True)
    if ref:
        qc |= {'delta_beta_ref': round(ref['delta'], 3), 'delta_se': round(ref['delta_se'], 3),
               'beta_par': round(ref['beta_par'], 3), 'beta_perp': round(ref['beta_perp'], 3),
               'r2': round(ref['r2'], 4)}
    qc |= compare_models(th, b)

    qc |= grid_bias(th, b, d.grid_axis_deg.values)
    (outdir / f'{name}_stage_a{sfx}.json').write_text(json.dumps(qc, indent=1))
    print(f"\n  {len(d)} profiles, theta {qc['theta_range']}, "
          f"delta_beta_ref {qc.get('delta_beta_ref')} +/- {qc.get('delta_se')}, "
          f"best model {qc['best_model']}")
    return qc


if __name__ == '__main__':
    args = sys.argv[1:]
    sub = 'prep_hull' if '--hull' in args else 'prep'
    want = args[args.index('--site') + 1] if '--site' in args else None
    step = float(args[args.index('--az-step') + 1]) if '--az-step' in args else AZ_STEP
    if '--sample-m' in args:
        SAMPLE_M = float(args[args.index('--sample-m') + 1])
    ln = float(args[args.index('--len-km') + 1]) * 1000 if '--len-km' in args else None
    los = ([float(v) for v in args[args.index('--band-lo') + 1].split(',')]
           if '--band-lo' in args else [FIT_BAND_M[0]])
    hi = float(args[args.index('--band-hi') + 1]) if '--band-hi' in args else FIT_BAND_M[1]
    floor = 5 * SAMPLE_M   # fewer than 5 samples per wavelength and interpolation dominates
    if min(los) < floor:
        sys.exit(f'--band-lo below {floor:.0f} m is unusable at {SAMPLE_M:.0f} m sampling')

    sites = [s for s in sorted(DATA.iterdir())
             if (s / sub / f'{s.name}_dem.tif').exists() and (want is None or s.name == want)]
    if not sites:
        sys.exit(f'no prepped sites in {sub}/' + (f' matching {want!r}' if want else ''))

    rows = [run(s, s / sub, step, ln, lo, hi) for s in sites for lo in los]
    pd.DataFrame(rows).to_csv(HERE / f'stage_a_summary{"_hull" if "--hull" in args else ""}.csv',
                              index=False)
    print(f'\n{len(rows)} sites done')
