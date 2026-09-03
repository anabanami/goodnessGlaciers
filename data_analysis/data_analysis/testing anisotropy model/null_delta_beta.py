"""Delta_beta and its bootstrap standard error for every raked null seed.

Reads the per window (theta, beta) pairs each seed's azimuth rake wrote, fits
the production cos2 model unweighted, and writes one row per seed.


python null_delta_beta.py
python null_delta_beta.py --seeds null_seeds/seed_001
python null_delta_beta.py --fabric-r-min 0.7        # drop windows over incoherent fabric

"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_DIR = '/home/ana/Desktop/code/Data/ODSA'

import sys
sys.path.insert(0, ROOT_DIR)   # config, plotting and loading, which the fitted module imports
sys.path.insert(0, str(HERE))  # the module under test, ahead of the ODSA module of the same name
from config import Tee
from weighted_anisotropy import fit_cos2


def fit_seed(csv_path, n_boot=2000, r_min=None):
    """One seed's Delta_beta. Unweighted, so the fit measures the estimator alone."""
    df = pd.read_csv(csv_path)
    ok = np.isfinite(df['beta']) & np.isfinite(df['theta_deg'])
    if r_min is not None:
        ok &= df['fabric_R'] >= r_min
    theta = df.loc[ok, 'theta_deg'].to_numpy(dtype=float)
    beta = df.loc[ok, 'beta'].to_numpy(dtype=float)
    if len(theta) < 4:
        return None

    fit = fit_cos2(theta, beta, n_boot=n_boot, quiet=True)
    if fit is None:
        return None

    return dict(seed=Path(csv_path).parent.name,
                n_points=len(theta),
                beta_par=fit['beta_par'],
                beta_perp=fit['beta_perp'],
                delta=fit['delta'],
                delta_se=fit['delta_se'],
                beta_perp_se=fit['perr'][0],
                beta_par_se=fit['perr'][1],
                r2=fit['r2'])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', default='null_seeds',
                   help='folder of seed folders, or one seed folder')
    p.add_argument('--n-boot', type=int, default=2000)
    p.add_argument('--fabric-r-min', type=float,
                   help='keep only windows whose fabric_R reaches this')
    p.add_argument('--out', default='null_delta_beta.csv')
    a = p.parse_args()

    stem = '*_window_beta.csv'
    out = Path(a.out)
    sys.stdout = Tee(out.with_name(out.stem + '_log.txt'))

    root = Path(a.seeds)
    paths = sorted(root.glob(stem)) or sorted(root.glob(f'seed_*/{stem}'))
    if not paths:
        sys.exit(f'no {stem} under {root}')

    rows = []
    for path in paths:
        row = fit_seed(path, n_boot=a.n_boot, r_min=a.fabric_r_min)
        if row is None:
            print(f'  {path.parent.name}  fit failed')
            continue
        rows.append(row)
        print(f"  {row['seed']}  delta {row['delta']:+.4f}  se {row['delta_se']:.4f}  r2 {row['r2']:.3f}")

    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cut = f', fabric_R at least {a.fabric_r_min}' if a.fabric_r_min else ''
    print(f'wrote {out}, {len(rows)} of {len(paths)} seeds fitted{cut}')


if __name__ == '__main__':
    main()
