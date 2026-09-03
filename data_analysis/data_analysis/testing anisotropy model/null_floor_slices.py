"""Noise floor from the null seed beds, sliced the same way as the real sites.

Every seed bed is isotropic by construction, so its true delta_beta is zero. Each bed is
sliced along x and y by DEM_slicer and fitted with weighted_anisotropy's cos^2 model, and
the spread of delta_beta across beds is the floor a real delta_beta has to clear.

    python null_floor_slices.py
    python null_floor_slices.py --spacing-km 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import DEM_slicer
import run_wa

HERE = Path(__file__).resolve().parent
SEEDS = HERE / 'null_seeds'
OUT = HERE / 'null_floor_slices.csv'


def row_for(folder, fit_cos2, spacing_m):
    name, d = DEM_slicer.slice_bed(folder, spacing_m)
    fit = fit_cos2(d.incidence_deg.to_numpy(float), d.beta.to_numpy(float), quiet=True)
    r = dict(name=name, n=len(d), beta_median=float(d.beta.median()),
             beta_sd=float(d.beta.std(ddof=1)),
             theta_lo=float(d.incidence_deg.min()), theta_hi=float(d.incidence_deg.max()))
    if fit:
        r |= dict(beta_par=fit['beta_par'], beta_perp=fit['beta_perp'],
                  delta=fit['delta'], delta_se=fit['delta_se'], r2=fit['r2'])
    print(f"  {name}  n={r['n']:3d}  beta {r['beta_median']:.3f}  "
          f"delta {r.get('delta', np.nan):+.4f} +/- {r.get('delta_se', np.nan):.4f}")
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spacing-km', type=float, default=DEM_slicer.SPACING_M / 1000)
    a = p.parse_args()

    folders = sorted(f for f in SEEDS.iterdir() if f.is_dir() and any(f.glob('*_dem.tif')))
    if not folders:
        raise SystemExit(f'no seed beds under {SEEDS}')

    fit_cos2 = run_wa.load()['fit_cos2']
    print(f'{len(folders)} seed beds, slice spacing {a.spacing_km:.0f} km')
    rows = [row_for(f, fit_cos2, a.spacing_km * 1000) for f in folders]

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    d = df.delta.dropna().values
    print(f'\nFLOOR over {len(d)} beds, true delta_beta = 0')
    print(f'  delta_beta   mean {d.mean():+.4f}   sd {d.std(ddof=1):.4f}   '
          f'range {d.min():+.4f} to {d.max():+.4f}')
    print(f'  |delta_beta| p50 {np.percentile(abs(d), 50):.4f}   '
          f'p95 {np.percentile(abs(d), 95):.4f}   max {abs(d).max():.4f}')
    print(f'  bootstrap SE, median over beds {df.delta_se.median():.4f}')
    print(f'\n-> {OUT}')
