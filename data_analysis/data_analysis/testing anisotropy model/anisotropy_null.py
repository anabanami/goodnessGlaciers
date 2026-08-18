"""Tier 1 rotation surrogate: the Δβ a cos²θ fit returns when the flow bearing is arbitrary.

Keeps every β, weight and segment as measured and replaces the assumed flow bearing with a
uniform ψ swept through 180°. The spread of the resulting Δβ is the noise floor. Runs from
this folder; imports the production fit from the ODSA root so the estimator is identical.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from weighted_anisotropy import _do_curve_fit, flow_weight  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOTS = ['individual_region_TEST']   # --all appends 'new'
N_PSI = 360
BANDS = [15, 30, 45, 90]


def fold(a):
    """Unfolded heading difference (0-180) to incidence angle (0-90)."""
    a = np.abs(a) % 180
    return np.minimum(a, 180 - a)


def axial_mean(deg):
    """Circular mean of axial (mod 180) angles."""
    r = np.radians(deg) * 2
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) / 2 % 180


def delta(theta, beta, w):
    """Point Δβ only — replicates fit_cos2's p0 and skips its 2000-draw bootstrap."""
    low, high = theta < 30, theta > 60
    p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
    p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)
    popt, _ = _do_curve_fit(theta, beta, w, p0=[p0_perp, p0_par])
    return popt[1] - popt[0]


def sweep(az, beta, w, n_psi=N_PSI):
    """Δβ at every uniform bearing ψ. Weights are held fixed throughout."""
    out = []
    for psi in np.linspace(0, 180, n_psi, endpoint=False):
        try:
            out.append((psi, delta(fold(az - psi), beta, w)))
        except (RuntimeError, ValueError):
            continue
    psi, d = np.array(out).T
    ok = np.isfinite(d)
    return psi[ok], d[ok]


def band_floors(az, beta, w, bands=BANDS, step=5.0, min_n=8):
    """Null width inside a capped azimuth band: narrower bands cap the θ spread the fit can
    use, which is the coverage curve. Reports the most populated window per band."""
    out = {}
    for width in bands:
        best = None
        for lo in np.arange(0, 180, step):
            m = fold(az - (lo + width / 2)) <= width / 2
            if m.sum() < min_n or (best is not None and m.sum() <= best[1]):
                continue
            try:
                _, nb = sweep(az[m], beta[m], None if w is None else w[m], n_psi=90)
            except (RuntimeError, ValueError, TypeError):
                continue
            best = (float(np.percentile(np.abs(nb), 95)), int(m.sum()))
        out[f'floor95_{width}deg'] = best[0] if best else np.nan
        out[f'n_{width}deg'] = best[1] if best else 0
    return out


def find(region, sub):
    for r in ROOTS:
        hits = sorted((ROOT / r / region / f'{sub}_csvs').glob(f'*_{sub}_stats.csv'))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'no {sub} CSV for {region}')


def load(region, level):
    """Window or segment table with an azimuth column, transitions dropped."""
    d = pd.read_csv(find(region, level))

    if level == 'segment':
        # Segment CSVs carry no azimuth; take the axial mean over each segment's windows.
        az = (pd.read_csv(find(region, 'window')).groupby(['trajectory', 'segment'])
              .azimuth_deg.apply(axial_mean).rename('azimuth_deg'))
        d = d.merge(az, on=['trajectory', 'segment'], how='left')

    d = d[~d.is_transition.astype(bool)]
    return d.dropna(subset=['beta', 'azimuth_deg', 'incidence_deg'])


def analyse(region, level):
    d = load(region, level)
    if len(d) < 8:
        return None

    az, beta, inc = d.azimuth_deg.values, d.beta.values, d.incidence_deg.values
    w = flow_weight(d.flow_error_mean.values, d.measures_speed_mean.values)
    degenerate = w.sum() == 0
    if degenerate:
        w = None

    obs = delta(inc, beta, w)
    psi, null = sweep(az, beta, w)

    # Anchor: the uniform bearing that best reproduces the real spatially varying θ.
    err = [np.mean(np.abs(fold(az - p) - inc)) for p in psi]
    i = int(np.argmin(err))

    row = dict(region=region, level=level, n=len(d), weights_degenerate=degenerate,
               delta_obs=obs, psi_anchor=psi[i], delta_anchor=null[i],
               floor95=np.percentile(np.abs(null), 95),
               p_null=float((np.abs(null) >= abs(obs)).mean()))
    row['clears'] = abs(obs) > row['floor95']

    return row | band_floors(az, beta, w)


if __name__ == '__main__':
    if '--all' in sys.argv[1:]:
        ROOTS.append('new')
    regions = sorted({q.parent.parent.name for r in ROOTS
                      for q in (ROOT / r).glob('*/window_csvs/*_window_stats.csv')})

    rows = []
    for region in regions:
        for level in ('window', 'segment'):
            try:
                row = analyse(region, level)
            except (StopIteration, KeyError, ValueError) as e:
                print(f'{region:10s} {level:8s} skipped: {e}')
                continue
            if row:
                rows.append(row)
                flag = 'CLEARS' if row['clears'] else 'fails'
                print(f"{region:10s} {level:8s} n={row['n']:4d}  obs={row['delta_obs']:+.3f}  "
                      f"floor95={row['floor95']:.3f}  p={row['p_null']:.3f}  {flag}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'anisotropy_null_floor.csv', index=False)
    print(f"\n{len(df)} fits -> {OUT / 'anisotropy_null_floor.csv'}")
