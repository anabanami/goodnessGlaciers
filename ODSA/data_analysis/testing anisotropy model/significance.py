"""Stage 5. A site's Delta_beta against the null distribution of its own masked batch.

A masked batch holds 100 isotropic beds carrying the site's mask and its fabric, so its
Delta_beta values are what the estimator returns on that ground when the true value is
zero. They carry the mask's bias as well as its spread, which is why a site is read as a
percentile against them rather than as a multiple of a reported standard error.

    python significance.py                  # all three sites
    python significance.py --site site_f    # one
    python significance.py --tag r07        # fits written under a fabric_R cut

"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_DIR = '/home/ana/Desktop/code/Data/ODSA'

sys.path.insert(0, ROOT_DIR)
from config import Tee

SITES = {'site_e': 'Site E Prince of Wales',
         'site_f': 'Site F Nunavut',
         'dubawnt': 'Dubawnt'}


def robust_half(v):
    """The spread `bootstrap_cos2_uncertainty` reports, (p84 - p16)/2."""
    lo, hi = np.percentile(v, [16, 84])
    return (hi - lo) / 2


def compare(slug, tag):
    """One site against its masked null. Counts are of null seeds, p is (count + 1)/(n + 1)."""
    site = pd.read_csv(HERE / f'{slug}{tag}_delta_beta.csv').iloc[0]
    null = pd.read_csv(HERE / f'{slug}_masked{tag}_delta_beta.csv')['delta'].to_numpy()

    obs, n, centre = float(site['delta']), len(null), float(np.median(null))
    ge = int((null >= obs).sum())
    far = int((np.abs(null - centre) >= abs(obs - centre)).sum())
    sd = float(null.std(ddof=1))

    return dict(slug=slug, obs=obs, se=float(site['delta_se']), r2=float(site['r2']),
                n=n, mean=float(null.mean()), sd=sd, half=robust_half(null),
                ge=ge, far=far, z=(obs - null.mean()) / sd,
                p_one=(ge + 1) / (n + 1), p_two=(far + 1) / (n + 1))


def report(c):
    print(f"\n{SITES[c['slug']]}")
    print(f"  Delta_beta              {c['obs']:+.4f}   delta_se {c['se']:.4f}   r2 {c['r2']:.3f}")
    print(f"  masked null, {c['n']} beds   mean {c['mean']:+.4f}   sd {c['sd']:.4f}   "
          f"robust half {c['half']:.4f}")
    print(f"  Delta_beta on that null {c['z']:+.2f}   (over the null mean, in null "
          f"standard deviations)")
    print(f"  null beds at least this large   {c['ge']:3d} of {c['n']}   p {c['p_one']:.3f}")
    print(f"  null beds at least this far     {c['far']:3d} of {c['n']}   p {c['p_two']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', choices=sorted(SITES), action='append',
                   help='repeatable, default all three')
    p.add_argument('--tag', default='',
                   help='a suffix on the fits names, as written by a fabric_R cut')
    a = p.parse_args()

    tag = f'_{a.tag}' if a.tag else ''
    sys.stdout = Tee(HERE / f'significance{tag}_log.txt')

    rows = [compare(slug, tag) for slug in a.site or sorted(SITES)]
    for c in rows:
        report(c)

    print(f'\n{"site":24s} {"Delta_beta":>10s} {"null mean":>10s} {"null sd":>8s} '
          f'{"one sided":>10s} {"two sided":>10s}')
    for c in rows:
        print(f"{SITES[c['slug']]:24s} {c['obs']:>+10.4f} {c['mean']:>+10.4f} "
              f"{c['sd']:>8.4f} {c['p_one']:>10.3f} {c['p_two']:>10.3f}")


if __name__ == '__main__':
    main()
