"""Fabric columns on a rake that is already done, under a different lattice.

An unmasked bed gives the same beta per window whatever fabric it carries, since the
fabric enters a window only through its theta. So the no mask corner of a control is
the isotropic rake with its four fabric columns recomputed, and needs no rake of its
own.

    python retheta.py masked_seeds/site_e_fabric          # from null_seeds
    python retheta.py masked_seeds/site_e_fabric --check  # compare, write nothing

"""
import argparse
import csv
from pathlib import Path

import numpy as np

from azimuth_rake import load_fabric, window_flow

HERE = Path(__file__).resolve().parent
SEEDS = HERE / 'null_seeds'

FABRIC_COLS = ('theta_deg', 'flow_bearing_deg', 'fabric_R', 'fabric_dist_m')


def retheta(rows, fabric):
    """Rewrite the fabric columns of raked windows, as `azimuth_rake.rake` sets them."""
    for r in rows:
        b = float(r['bearing_deg'])
        unit = np.array([np.sin(np.radians(b)), np.cos(np.radians(b))])
        flow, R, fdist = window_flow(*fabric, np.array([float(r['x']), float(r['y'])]), unit)
        r['theta_deg'] = abs((b - flow + 90) % 180 - 90)
        r['flow_bearing_deg'] = flow
        r['fabric_R'] = R
        r['fabric_dist_m'] = fdist


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('batch', help='a staged folder holding seed_NNN subfolders')
    p.add_argument('--source', default=str(SEEDS), help='the rake to read, default null_seeds')
    p.add_argument('--check', action='store_true',
                   help='compare against the CSV already in the batch, write nothing')
    a = p.parse_args()

    batch, source = Path(a.batch), Path(a.source)
    done = differ = 0
    for d in sorted(batch.glob('seed_*')):
        name = d.name
        src = source / name / f'{name}_window_beta.csv'
        if not src.exists():
            print(f'  {name}: no source rake, skipping')
            continue
        with open(src) as f:
            rows = list(csv.DictReader(f))
        retheta(rows, load_fabric(d / f'{name}_fabric.csv'))

        out = d / f'{name}_window_beta.csv'
        if a.check:
            with open(out) as f:
                raked = list(csv.DictReader(f))
            bad = sum(1 for x, y in zip(rows, raked)
                      if any(str(x[c]) != y[c] for c in FABRIC_COLS))
            differ += bad
            if bad:
                print(f'  {name}: {bad} of {len(rows)} windows differ')
        else:
            with open(out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        done += 1

    verb = 'checked' if a.check else 'wrote'
    print(f'{verb} {done} seeds in {batch}' + (f', {differ} windows differ' if a.check else ''))


if __name__ == '__main__':
    main()
