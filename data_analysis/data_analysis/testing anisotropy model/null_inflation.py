"""Bootstrap standard error compared with the true spread of Delta_beta across null seeds.

The true Delta_beta is zero on every null bed, so the across seed spread of the
fitted Delta_beta is the estimator's real uncertainty, and the ratio of that
spread to the reported delta_se is DELTA_BETA_BOOTSTRAP_INFLATION.


python null_inflation.py
python null_inflation.py --fits null_delta_beta.csv

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = '/home/ana/Desktop/code/Data/ODSA'

import sys
sys.path.insert(0, ROOT_DIR)
from config import Tee


def robust_half(x):
    """The bootstrap's own definition of a standard error, applied across seeds."""
    return (np.percentile(x, 84) - np.percentile(x, 16)) / 2


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fits', default='null_delta_beta.csv')
    p.add_argument('--log', default=None)
    a = p.parse_args()

    fits = Path(a.fits)
    sys.stdout = Tee(a.log or fits.with_name(fits.stem.replace('null_delta_beta',
                                                               'null_inflation') + '_log.txt'))
    df = pd.read_csv(fits)
    delta, se, r2 = df['delta'].to_numpy(), df['delta_se'].to_numpy(), df['r2'].to_numpy()
    n = len(delta)

    print(f'\n{n} seeds from {fits}\n')

    # True Delta_beta is zero, so the mean is bias and the RMS is total error
    sem = delta.std(ddof=1) / np.sqrt(n)
    print('Delta_beta across seeds')
    print(f'  mean                  {delta.mean():+.5f}  +/- {sem:.5f}  (bias, true value is 0)')
    print(f'  standard deviation     {delta.std(ddof=1):.5f}')
    print(f'  robust half interval   {robust_half(delta):.5f}')
    print(f'  RMS about zero         {np.sqrt((delta**2).mean()):.5f}')
    print(f'  range                 {delta.min():+.5f} to {delta.max():+.5f}')

    print('\nReported delta_se')
    print(f'  mean                   {se.mean():.5f}')
    print(f'  median                 {np.median(se):.5f}')
    print(f'  range                  {se.min():.5f} to {se.max():.5f}')

    z = delta / se
    print('\nInflation, spread divided by reported standard error')
    print(f'  std / median se        {delta.std(ddof=1) / np.median(se):.2f}')
    print(f'  robust half / median se {robust_half(delta) / np.median(se):.2f}')
    print(f'  RMS / median se        {np.sqrt((delta**2).mean()) / np.median(se):.2f}')
    print(f'  robust half of z       {robust_half(z):.2f}   (1.00 if the bootstrap is honest)')

    print('\nDelta_beta in units of its own reported standard error')
    print(f'  mean |z|               {np.abs(z).mean():.2f}   (0.80 if honest)')
    for k, expected in ((1, 0.317), (2, 0.046), (3, 0.003)):
        count = int((np.abs(z) > k).sum())
        print(f'  |z| > {k}                {count / n:.3f}   ({expected:.3f} if honest, {count} seeds)')

    print('\nFit quality')
    print(f'  r2 median              {np.median(r2):.3f}')
    print(f'  r2 range               {r2.min():.3f} to {r2.max():.3f}')
    print(f'  corr(r2, |Delta_beta|) {np.corrcoef(r2, np.abs(delta))[0, 1]:+.2f}')
    print(f'  corr(r2, delta_se)     {np.corrcoef(r2, se)[0, 1]:+.2f}')
    print(f'  beta_par mean          {df["beta_par"].mean():.4f}')
    print(f'  beta_perp mean         {df["beta_perp"].mean():.4f}')


if __name__ == '__main__':
    main()
