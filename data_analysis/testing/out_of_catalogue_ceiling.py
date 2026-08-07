#!/usr/bin/env python3
"""
Re-derive the preregistered out-of-catalogue ceiling.

The ceiling is what the OUT-OF-CATALOGUE rate would be if every axis resolved exactly: the
cells of observable space that match no catalogue entry, weighted by how often the survey
lands in them. It is a property of the catalogue read through the data, so **it moves
whenever the catalogue moves and must be re-derived and re-registered, not quietly edited.**

History: 44% against 44 uncovered cells. Widening DIVIDE to {very_low, low} on 2026-08-07
covered 4 heavily-occupied cells and took it to 31.9% against 40 uncovered.

Observable space is the four measurable axes only (beta x relief x elevation x velocity =
144 cells). delta_beta and beta_spread are left free: a cell counts as covered if ANY
setting of those two admits an entry, because neither is something the survey resolves
independently of the others.

Reading it: approaching the ceiling means the coverage gap fully expressed, a catalogue
result. Materially below means ambiguity is still rescuing cells, so the rate describes the
error bars and not the bed. Compare like with like -- this is a WINDOW rate, so it is not
directly comparable to a segment-level OUT-OF-CATALOGUE count.

      python out_of_catalogue_ceiling.py
      python out_of_catalogue_ceiling.py --root ../Ockenden-regions
"""
import argparse, collections, glob, itertools, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ODSA = HERE.parent
sys.path.insert(0, str(ODSA))
import landscape_vector as lv                                          # noqa: E402

AXES = [('beta_class', 'beta', lv.BED_CLASSES),
        ('relief_class', 'relief_m', lv.RELIEF_CLASSES),
        ('elevation_class', 'bed_elev_mean', lv.ELEVATION_CLASSES),
        ('velocity_band', 'measures_speed_mean', lv.VELOCITY_CLASSES)]


def classify(v, classes):
    """Exact class for a value, or None if it is missing."""
    if not np.isfinite(v):
        return None
    return next((n for n, lo, hi in classes if lo <= v < hi), None)


def covered_cells():
    """Cells admitting at least one entry for some delta_beta / beta_spread setting."""
    out = set()
    free = list(itertools.product(lv.AXIS_VALUES['delta_beta'], lv.AXIS_VALUES['beta_spread']))
    for combo in itertools.product(*[[n for n, _, _ in cl] for _, _, cl in AXES]):
        p = dict(zip([a for a, _, _ in AXES], combo))
        if any(all(dict(p, delta_beta=d, beta_spread=b)[a] in allowed
                   for a, allowed in c['c'].items())
               for d, b in free for c in lv.CATALOGUE):
            out.add(combo)
    return out


def main(root):
    cov = covered_cells()
    n_cells = int(np.prod([len(cl) for _, _, cl in AXES]))
    print(f"Observable cells {n_cells}: {len(cov)} covered, {n_cells - len(cov)} uncovered "
          f"({len(cov)/n_cells:.0%} covered by construction)")

    frames = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        d = pd.read_csv(f).dropna(subset=['beta'])
        if 'is_transition' in d.columns:
            d = d[~d['is_transition']]
        frames.append(d.assign(region=os.path.basename(os.path.dirname(os.path.dirname(f)))))
    if not frames:
        sys.exit(f"No *_window_stats.csv under {root}")
    w = pd.concat(frames)

    w['cell'] = [tuple(classify(r[col], cl) for _, col, cl in AXES)
                 for _, r in w.iterrows()]
    ok = w[[None not in c for c in w.cell]]
    print(f"Windows {len(w)} after dropping transitions, {len(ok)} with all four axes assignable")

    miss = ok[[c not in cov for c in ok.cell]]
    print(f"\n>>> CEILING: {len(miss)}/{len(ok)} = {len(miss)/len(ok):.1%} of windows would be "
          f"OUT-OF-CATALOGUE if every axis resolved exactly\n")
    print("  per region")
    for r, g in ok.groupby('region'):
        m = sum(c not in cov for c in g.cell)
        print(f"    {r:10s} {m:4d}/{len(g):4d}   {m/len(g):5.1%}")
    print("\n  most-occupied uncovered cells (beta, relief, elevation, velocity)")
    for c, n in collections.Counter(miss.cell).most_common(8):
        print(f"    {n:4d}  {', '.join(c)}")
    print("\n  Re-register this number in WIP/NEXT.md if the catalogue has changed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ODSA / 'individual_region_TEST'))
    main(ap.parse_args().root)
