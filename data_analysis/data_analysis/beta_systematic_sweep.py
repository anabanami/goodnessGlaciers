"""What an honest beta envelope costs, swept.

K_SIGMA = 2 on beta's formal fit error alone (median 0.056) is a lower bound. The taper moves
beta by a mean absolute 0.251 with a region-dependent sign and window size does the same, so
the sign-varying part belongs in the envelope (Roughness_and_Anisotropy.md §2). Band truncation
is one-directional and is a correction, not an envelope, so it is not swept here.

Sweeps BETA_SYSTEMATIC_ERROR with the migration widening OFF — the envelope replaces it, it does
not stack on it. Production is the 0.05 row. The first row is the pre-envelope reference, widening
on with no systematic, and is not a config that is run.

Vectors are built once and re-observed per value, so the cost is one pass, not one per value.

    python beta_systematic_sweep.py [individual_region_TEST]
"""
import glob, os, sys
import numpy as np, pandas as pd
from config import Tee
import landscape_vector as lv
from landscape_vector import (load_region, build_vector, observe, match, units_from,
                              CATALOGUE, ALL_AXES)

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'individual_region_TEST'
VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LEVEL = 'window'


def build_once(root):
    """(region, vector, pflag) per unit. observe() is re-run per sweep value; this is not."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        region = os.path.basename(os.path.dirname(os.path.dirname(f)))
        df, pflag = load_region(f, quiet=True)
        if df is None:
            continue
        for unit, g in units_from(df, LEVEL):
            out.append((region, build_vector(g, f'{LEVEL}:{unit}', pflag), pflag))
    return out


def evaluate(built, systematic, widen):
    lv.BETA_SYSTEMATIC_ERROR, lv.MIGRATION_WIDENS_BETA = systematic, widen
    sizes, nbeta, regions = [], [], []
    for region, v, pflag in built:
        obs = observe(v, pflag)
        sizes.append(len(match(obs)))
        nbeta.append(len(obs['beta_class']['set']))
        regions.append(region)
    s, b = np.array(sizes), np.array(nbeta)
    return dict(beta_systematic=systematic, widening=widen,
                beta_1class_pct=round(100 * float((b == 1).mean()), 1),
                beta_mean_classes=round(float(b.mean()), 2),
                median_set=float(np.median(s)), mean_set=round(float(s.mean()), 2),
                resolved_pct=round(100 * float((s == 1).mean()), 1),
                ooc_pct=round(100 * float((s == 0).mean()), 1),
                degenerate_pct=round(100 * float((s > 1).mean()), 1)), np.array(regions), s


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'beta_systematic_sweep_log.txt'))
    built = build_once(ROOT)
    print(f"{len(built)} {LEVEL} units, {len(CATALOGUE)} catalogue entries\n")

    rows, per_region = [], {}
    ref, _, _ = evaluate(built, 0.0, True)
    ref['beta_systematic'] = 'reference (widening on, no systematic)'
    rows.append(ref)
    for s in VALUES:
        r, regions, sizes = evaluate(built, s, False)
        rows.append(r)
        per_region[s] = pd.Series(sizes).groupby(regions).apply(
            lambda x: round(100 * (x == 1).mean(), 1))

    t = pd.DataFrame(rows)
    print("=== BETA ENVELOPE SWEEP (widening off except the reference row) ===")
    print(t.to_string(index=False), "\n")

    print("=== resolved % by region, by beta_systematic ===")
    print(pd.DataFrame(per_region).to_string(), "\n")

    print("Read: beta_1class_pct is the beta axis resolving to a single class. Where that "
          "falls to near zero,\nbeta has stopped classifying and whatever resolution remains "
          "is carried by the other axes.")
    t.to_csv(os.path.join(ROOT, 'beta_systematic_sweep.csv'), index=False)
    pd.DataFrame(per_region).to_csv(os.path.join(ROOT, 'beta_systematic_sweep_by_region.csv'))
    print(f"\nSaved: {os.path.join(ROOT, 'beta_systematic_sweep.csv')}")
