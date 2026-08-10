#!/usr/bin/env python3
"""
Per-window MEaSUREs velocity error, written as a sidecar beside the window CSVs.

VELOCITY_ERROR_M_YR = 5.0 is a continent-wide constant standing in for a spatially varying
error field that MEaSUREs already ships. The sweep in v23/velocity_error_sweep.py showed the
ONSET/DIVIDE seam is decided by that constant and by a hard threshold: the seam is possible
only when the envelope is wide enough to jump the `low` band, 2*K_SIGMA*err > 5, so
err > 1.25 m/yr. A constant cannot answer that question, since whether a segment clears the
threshold depends entirely on where it sits in the error field.

ERRX/ERRY are in all_data/measures_velocity/antarctica_ice_velocity_450m_v2.nc, the same file
REMA_extractor.MEaSUREs_comparison already opens for VX/VY. This samples them at each window
and writes trajectory / segment / window_id keyed rows, the same shape as the delta_beta
sidecar, so nothing upstream has to be re-run.

REGISTERED PREDICTION, before the run: continent-wide the RMS component error has a median of
about 2.6 m/yr, which is above the 1.25 threshold, so if the test regions look like the
continent the seam survives in most of them. It should fail unevenly rather than uniformly:
the seam vanishes where MEaSUREs is well stacked and persists where it is not.

Two choices worth knowing about when reading the output.

  - The stencil. window_stats carries only center_x / center_y, not the track points the
    speed was averaged over, so the error is sampled on a 3x3 stencil at +/-15 km about the
    centre and reduced by nanmedian. That is a footprint proxy for a 50 km window, not the
    window itself. `err_iqr` reports how much the field moves across the stencil; where it is
    large the proxy is doing real work and should be distrusted.
  - The propagation. Above the 5 m/yr flow-weight ramp the direction is trustworthy, so the
    speed error is the components projected on it. Below it the direction is noise
    (Rignot_2011), so the isotropic RMS of the two components is used instead. Same cutoff
    weighted_anisotropy already treats as the trust boundary, for the same reason.

Nothing here classifies and nothing is overwritten outside <region>/velocity/.

      python velocity_error_sidecar.py
      python velocity_error_sidecar.py --root Ockenden-regions
      python velocity_error_sidecar.py --dry-run          # summary only, writes nothing
"""
import argparse, glob, os, sys

from config import Tee
import numpy as np
import pandas as pd
import xarray as xr

MEASURES_NC = 'all_data/measures_velocity/antarctica_ice_velocity_450m_v2.nc'
STENCIL_M = 15_000.0          # 3x3 half-offset, a footprint proxy for a 50 km window
DIRECTION_TRUSTED_M_YR = 5.0  # below this MEaSUREs direction is noise, so propagate isotropically
SEAM_THRESHOLD = 1.25         # 5 / (2 * K_SIGMA); above this the ONSET/DIVIDE seam is possible
KEYS = ['trajectory', 'segment', 'window_id']


def nearest_index(coord, want):
    """Index of the nearest grid node, handling either coordinate direction."""
    if coord[0] > coord[-1]:
        i = len(coord) - 1 - np.searchsorted(coord[::-1], want)
    else:
        i = np.searchsorted(coord, want)
    i = np.clip(i, 1, len(coord) - 1)
    return np.where(np.abs(coord[i] - want) < np.abs(coord[i - 1] - want), i, i - 1)


def sample(ds, xs, ys):
    """VX, VY, ERRX, ERRY at the nearest node to each (x, y). Vectorised isel, so only the
    points asked for are read off disk -- the grid is 12445^2 and must not be loaded."""
    gx, gy = ds.x.values, ds.y.values
    ix = xr.DataArray(nearest_index(gx, xs), dims='pt')
    iy = xr.DataArray(nearest_index(gy, ys), dims='pt')
    out = [ds[v].isel(x=ix, y=iy).values.astype(float)
           for v in ('VX', 'VY', 'ERRX', 'ERRY', 'CNT')]
    return [np.where(np.abs(a) > 1e20, np.nan, a) for a in out]


def nanpct(a, ok, q):
    """nanpercentile down the stencil, leaving all-NaN columns as NaN without the warning.
    An all-NaN column is a window outside MEaSUREs coverage entirely."""
    out = np.full(a.shape[1], np.nan)
    if ok.any():
        out[ok] = np.nanpercentile(a[:, ok], q, axis=0)
    return out


def nanmed(a, ok):
    return nanpct(a, ok, 50)


def speed_error(vx, vy, ex, ey):
    """Error on |v|. Above the direction-trust cutoff the component errors project onto the
    flow direction; below it the direction carries no information, so use the isotropic RMS."""
    v = np.hypot(vx, vy)
    with np.errstate(invalid='ignore', divide='ignore'):
        projected = np.sqrt((vx * ex) ** 2 + (vy * ey) ** 2) / v
    isotropic = np.sqrt((ex ** 2 + ey ** 2) / 2.0)
    return v, np.where(np.isfinite(projected) & (v >= DIRECTION_TRUSTED_M_YR),
                       projected, isotropic)


def region(ds, csv_path):
    df = pd.read_csv(csv_path)
    missing = [k for k in KEYS + ['center_x', 'center_y'] if k not in df.columns]
    if missing:
        print(f"    skipped, missing columns: {missing}")
        return None

    offs = [-STENCIL_M, 0.0, STENCIL_M]
    cx, cy = df.center_x.to_numpy(float), df.center_y.to_numpy(float)
    per_point = []
    for dx in offs:
        for dy in offs:
            vx, vy, ex, ey, cnt = sample(ds, cx + dx, cy + dy)
            v, s = speed_error(vx, vy, ex, ey)
            per_point.append((v, s, cnt))
    speeds = np.vstack([p[0] for p in per_point])
    errs = np.vstack([p[1] for p in per_point])
    cnts = np.vstack([p[2] for p in per_point])
    ok = np.isfinite(errs).any(axis=0)

    out = df[KEYS].copy()
    out['measures_err_m_yr'] = nanmed(errs, ok)
    out['measures_err_iqr'] = nanpct(errs, ok, 75) - nanpct(errs, ok, 25)
    out['measures_speed_sampled'] = nanmed(speeds, ok)
    # CNT is the number of stacked observations behind the pixel. It is the column that
    # explains the error field: 2 to 4 inside the InSAR pole hole against 74 to 206 outside.
    out['measures_cnt'] = nanmed(cnts, ok)
    out['n_stencil_ok'] = np.isfinite(errs).sum(axis=0)
    return out


def cross_check(df, out):
    """The sampled speed must track measures_speed_mean already in the window stats. If it
    does not, the coordinates or the grid indexing are wrong and the errors are meaningless.

    The test is median |diff| against the region's own median speed, NOT r. r is the wrong
    statistic here: it collapses on a region with few windows over a narrow speed range even
    when every sample is good (MSB came back r = 0.909 on a median |diff| of 0.18 m/yr), and
    it is also depressed by genuine field heterogeneity where MEaSUREs is thin, which is a
    result rather than a bug. r is still printed, as context for the disagreement."""
    if 'measures_speed_mean' not in df.columns:
        return "  (no measures_speed_mean to cross-check against)"
    a = df.measures_speed_mean.to_numpy(float)
    b = out.measures_speed_sampled.to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return "  (too few finite pairs to cross-check)"
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    med = np.nanmedian(np.abs(a[ok] - b[ok]))
    rel = med / np.nanmedian(a[ok]) if np.nanmedian(a[ok]) > 0 else np.nan
    flag = '   ** check the coordinates **' if rel > 0.25 else ''
    return (f"  speed cross-check: median |diff| = {med:.2f} m/yr = {rel:.0%} of the region "
            f"median speed (r = {r:.3f}){flag}")


def main(root, dry):
    if not os.path.exists(MEASURES_NC):
        sys.exit(f"MEaSUREs grid not found: {MEASURES_NC}")
    files = sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'), recursive=True))
    if not files:
        sys.exit(f"No *_window_stats.csv under {root}")

    ds = xr.open_dataset(MEASURES_NC)
    print(f"MEaSUREs: {MEASURES_NC}")
    print(f"Stencil: 3x3 at +/-{STENCIL_M/1000:.0f} km, reduced by nanmedian")
    print(f"Seam threshold: err > {SEAM_THRESHOLD} m/yr makes ONSET|DIVIDE possible\n")

    summary = []
    for f in files:
        name = os.path.basename(f).replace('_window_stats.csv', '')
        print(f"  {name}")
        out = region(ds, f)
        if out is None:
            continue
        df = pd.read_csv(f)
        print(cross_check(df, out))

        e = out.measures_err_m_yr
        over = float((e > SEAM_THRESHOLD).mean())
        # No MEaSUREs coverage at all. The axis must widen on these, never narrow, so they
        # have to reach landscape_vector as NaN and not as a filled number.
        n_missing = int(e.isna().sum())
        print(f"  windows {len(out):4d}   err median {e.median():6.2f}   "
              f"p25 {e.quantile(.25):5.2f}  p75 {e.quantile(.75):5.2f}   "
              f"over threshold {over:.0%}")
        print(f"  CNT median {out.measures_cnt.median():6.0f}   "
              f"partial stencils {int((out.n_stencil_ok.between(1, 8)).sum()):3d}   "
              f"no coverage {n_missing:3d}"
              + ('   ** these must stay NaN downstream **' if n_missing else ''))
        summary.append({'region': name, 'n_windows': len(out),
                        'err_median': e.median(), 'err_p25': e.quantile(.25),
                        'err_p75': e.quantile(.75), 'frac_over_threshold': over,
                        'stencil_iqr_median': out.measures_err_iqr.median(),
                        'cnt_median': out.measures_cnt.median(), 'n_no_coverage': n_missing})

        if not dry:
            d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(f))), 'velocity')
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f'{name}_velocity_error.csv')
            out.to_csv(p, index=False)
            print(f"  -> {p}")
        print()

    s = pd.DataFrame(summary)
    print("=" * 96)
    print("  PER REGION")
    print("=" * 96)
    s = s.sort_values('cnt_median')
    print(f"  {'region':40s}{'n':>5s}{'CNT':>7s}{'median':>9s}{'p25':>8s}{'p75':>8s}"
          f"{'>1.25':>8s}{'stencil IQR':>13s}{'no cov':>8s}")
    for _, r in s.iterrows():
        print(f"  {r.region[:40]:40s}{int(r.n_windows):>5d}{r.cnt_median:>7.0f}"
              f"{r.err_median:>9.2f}{r.err_p25:>8.2f}{r.err_p75:>8.2f}"
              f"{r.frac_over_threshold:>7.0%}{r.stencil_iqr_median:>13.2f}"
              f"{int(r.n_no_coverage):>8d}")
    print(f"\n  Registered: if the regions look like the continent (median ~2.6 m/yr) the seam")
    print("  survives in most of them, and fails unevenly rather than uniformly.")
    print("  A region well under 1.25 loses the seam outright; one well over keeps it whole.")
    print("\n  Sorted on CNT because that is what the error field tracks, monotonically and")
    print("  with no exceptions. The test set is bimodal on CNT itself -- the two POLARGAP")
    print("  regions sit in the InSAR pole hole at 2 and 3, everything else stacks 74 to 206")
    print("  -- but the error range IS sampled around 1.25 (2.61, 1.32, 0.82), which is the")
    print("  part that matters. The threshold is a cliff per window, since it is arithmetic")
    print("  and not empirical, and a ramp per region: the >1.25 column is the fraction of")
    print("  each region on the wrong side of it.")
    if not dry:
        os.makedirs('v23/velocity_error', exist_ok=True)
        p = 'v23/velocity_error/measures_error_summary.csv'
        s.to_csv(p, index=False)
        print(f"\n  Saved: {p}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='individual_region_TEST')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.dry_run:
        sys.stdout = Tee(os.path.join(a.root, 'velocity_error_sidecar_log.txt'))
    main(a.root, a.dry_run)
