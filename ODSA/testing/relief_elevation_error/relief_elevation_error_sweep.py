#!/usr/bin/env python3
"""
Relief / elevation error sensitivity sweep for the landscape vector.

RELIEF_ERROR_M and ELEVATION_ERROR_M were None until 2026-08-07, so relief and elevation
resolved against their class breaks carrying no uncertainty at all: 306 of 379 segments
(81%) read `assumed-exact` on both. **They are now set, 30 m and 10 m, from [Pritchard_2025],
and `assumed-exact` no longer fires anywhere.** This sweep is what justified setting them and
it is kept as the sensitivity record.

The two axes go exact on exactly the SAME segments in all 7 regions -- a single-window
segment has neither an across-window spread nor a formal error, so both lose their error
bar together. The diagonal therefore carries most of the signal; the off-diagonal cells
say which of the two is doing the work.

PREDICTION, registered before the first run: widening an envelope admits more archetypes, so
OUT-OF-CATALOGUE should FALL, DEGENERATE should RISE and RESOLVED should FALL. That is
the opposite direction to the delta_beta run, which narrowed envelopes and walked the
out-of-catalogue rate up toward the preregistered ceiling. Confirmed at every level in all
three runs so far.

NOTE: 0 and None are the same setting -- `if nominal:` in observe() treats 0 as falsy -- so
the 0/0 cell means "no nominal error on either axis". **It is no longer production.** It is
now the counterfactual that isolates what setting the constants bought, and the cell that
must reproduce the live reports is PROD_RELIEF / PROD_ELEV, which is what baseline_check
compares and which is added to the grid automatically.

Nothing here writes into the region trees; results go to v23/relief_elevation_error/.

      python relief_elevation_error_sweep.py
      python relief_elevation_error_sweep.py --errors 0 50          # cheap 2x2
      python relief_elevation_error_sweep.py --root ../Ockenden-regions
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA root
OUT_ROOT = HERE / 'relief_elevation_error'
sys.path.insert(0, str(ODSA))
import landscape_vector as lv                                          # noqa: E402

ERRORS_M = [0, 25, 50, 100]
# The live values, so the sweep always contains the production cell to check against.
PROD_RELIEF = int(lv.RELIEF_ERROR_M or 0)
PROD_ELEV = int(lv.ELEVATION_ERROR_M or 0)
VERDICTS = ['RESOLVED', 'RESOLVED-WITH-EXTERNAL', 'DEGENERATE', 'OUT-OF-CATALOGUE']
SHORT = {'RESOLVED': 'resolved', 'RESOLVED-WITH-EXTERNAL': '+external',
         'DEGENERATE': 'degenerate', 'OUT-OF-CATALOGUE': 'out-of-cat'}


def load_region(csv_path):
    """The same frame process_region classifies: transitions dropped, local delta_beta merged."""
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    pflag = lv.region_flag(df)
    if 'is_transition' in df.columns:
        df = df[~df['is_transition']].copy()
    dbl = lv.load_delta_beta(csv_path)
    if dbl is not None:
        keys = [c for c in ('trajectory', 'segment', 'window_id')
                if c in dbl.columns and c in df.columns]
        df = df.merge(dbl[keys + ['delta_beta_local', 'delta_beta_local_se',
                                  'delta_beta_status', 'delta_beta_label']],
                      on=keys, how='left')
    # Without this the sampled velocity error is absent, velocity falls back to the
    # VELOCITY_ERROR_M_YR constant, and the sweep silently runs at the old baseline.
    vel = lv.load_velocity_error(csv_path)
    if vel is not None:
        keys = [c for c in ('trajectory', 'segment', 'window_id')
                if c in vel.columns and c in df.columns]
        df = df.merge(vel[keys + ['measures_err_m_yr']], on=keys, how='left')
    return df, pflag


def classify(vecs, pflag):
    """Verdicts under whatever lv.RELIEF_ERROR_M / ELEVATION_ERROR_M are set to now.
    The vectors themselves are error-independent -- the constants enter at observe()."""
    rows = []
    for v in vecs:
        obs = lv.observe(v, pflag)
        cases = lv.match(obs)
        kind, _, _ = lv.verdict(cases, obs)
        rows.append({'verdict': kind, 'n_admissible': len(cases),
                     'exact_relief': obs['relief_class']['status'] == 'assumed-exact',
                     'exact_elev': obs['elevation_class']['status'] == 'assumed-exact',
                     'amb_relief': obs['relief_class']['status'] == 'ambiguous',
                     'amb_elev': obs['elevation_class']['status'] == 'ambiguous'})
    return pd.DataFrame(rows)


def sweep(root, errors):
    found = {os.path.basename(f).replace('_window_stats.csv', ''): f
             for f in sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'),
                                       recursive=True))}
    if not found:
        sys.exit(f"No *_window_stats.csv under {root}")
    print(f"Regions: {len(found)}")

    # Build each region's segment vectors once; only the observation step is swept.
    frames = {}
    for r, f in found.items():
        df, pflag = load_region(f)
        vecs = [lv.build_vector(g, u, pflag) for u, g in lv.units_from(df, 'segment')]
        frames[r] = (vecs, pflag)
        print(f"  {r:48s} {len(vecs):4d} segments")

    rows = []
    for re_m in errors:
        for ee_m in errors:
            lv.RELIEF_ERROR_M = re_m or None
            lv.ELEVATION_ERROR_M = ee_m or None
            for r, (vecs, pflag) in frames.items():
                s = classify(vecs, pflag)
                matched = s.loc[s.n_admissible > 0, 'n_admissible']
                rec = {'relief_error_m': re_m, 'elevation_error_m': ee_m, 'region': r,
                       'n_segments': len(s),
                       'median_admissible': matched.median() if len(matched) else np.nan,
                       'exact_both': int((s.exact_relief & s.exact_elev).sum()),
                       'ambiguous_relief': int(s.amb_relief.sum()),
                       'ambiguous_elevation': int(s.amb_elev.sum())}
                rec.update({v: int((s.verdict == v).sum()) for v in VERDICTS})
                rows.append(rec)
    # Restore the live values, NOT None -- these are set in production since 2026-08-07.
    lv.RELIEF_ERROR_M, lv.ELEVATION_ERROR_M = PROD_RELIEF or None, PROD_ELEV or None
    return pd.DataFrame(rows)


def baseline_check(root, d):
    """The cell matching the LIVE constants must reproduce the archetype reports; if it does
    not, this script is not classifying the way process_region does and nothing below it is
    trustworthy.

    That cell is not 0/0 any more. Relief and elevation were unset until 2026-08-07 and are now
    30 m and 10 m, so 0/0 is a counterfactual (`no nominal error`) rather than production. The
    live values are added to the swept grid so this check always has a cell to land on."""
    files = glob.glob(os.path.join(root, '*', 'landscape_vector', '*_archetype_report.csv'))
    if not files:
        print("\n  (no archetype reports under root -- baseline not cross-checked)")
        return
    live = pd.concat([pd.read_csv(f) for f in files])
    live = live[live.level == 'segment'].verdict.value_counts()
    mine = d[(d.relief_error_m == PROD_RELIEF) &
             (d.elevation_error_m == PROD_ELEV)][VERDICTS].sum()
    bad = {v: (int(mine[v]), int(live.get(v, 0))) for v in VERDICTS
           if int(mine[v]) != int(live.get(v, 0))}
    if bad:
        print("\n  ** BASELINE MISMATCH vs the live reports (sweep, report): "
              + ', '.join(f"{v} {a} vs {b}" for v, (a, b) in bad.items())
              + " -- do not read the sweep until this is explained **")
    else:
        print(f"\n  Baseline check: {PROD_RELIEF}/{PROD_ELEV} reproduces the live reports exactly "
              f"({', '.join(f'{SHORT[v]} {int(mine[v])}' for v in VERDICTS if mine[v])})")


def grid(tot, col, errors):
    """One verdict as a relief x elevation grid, summed over regions."""
    print(f"\n  {SHORT.get(col, col).upper()}  (rows = relief error m, cols = elevation error m)")
    print("        " + ''.join(f"{e:>8d}" for e in errors))
    for re_m in errors:
        cells = [int(tot[(tot.relief_error_m == re_m) &
                         (tot.elevation_error_m == e)][col].iloc[0]) for e in errors]
        print(f"  {re_m:5d} " + ''.join(f"{c:>8d}" for c in cells))


def report(d, errors, out_root):
    tot = d.groupby(['relief_error_m', 'elevation_error_m'], as_index=False).sum(numeric_only=True)
    n = int(d[(d.relief_error_m == 0) & (d.elevation_error_m == 0)].n_segments.sum())

    base = tot[(tot.relief_error_m == 0) & (tot.elevation_error_m == 0)].iloc[0]
    print(f"\n{'='*100}\n  0/0, NO nominal error on either axis  {n} segments"
          f"   (production is {PROD_RELIEF}/{PROD_ELEV}, checked above)\n{'='*100}")
    print("    " + '  '.join(f"{SHORT[v]} {int(base[v])}" for v in VERDICTS if base[v]))
    print(f"    both axes assumed-exact: {int(base.exact_both)} "
          f"({base.exact_both / n:.0%})")

    for v in ('RESOLVED', 'OUT-OF-CATALOGUE', 'DEGENERATE'):
        grid(tot, v, errors)
    grid(tot, 'exact_both', errors)

    print(f"\n{'='*100}\n  THE DIAGONAL (both errors equal)\n{'='*100}")
    hdr = ['error_m'] + [SHORT[v] for v in VERDICTS] + ['exact_both', 'med|set|']
    print('  ' + ''.join(f"{h:>14s}" for h in hdr))
    for e in errors:
        r = tot[(tot.relief_error_m == e) & (tot.elevation_error_m == e)].iloc[0]
        med = d[(d.relief_error_m == e) & (d.elevation_error_m == e)]['median_admissible'].median()
        vals = [f"{e}"] + [f"{int(r[v])}" for v in VERDICTS] + \
               [f"{int(r.exact_both)}", f"{med:.0f}" if np.isfinite(med) else "-"]
        print('  ' + ''.join(f"{x:>14s}" for x in vals))

    print(f"\n{'='*100}\n  PER REGION, RESOLVED (+external) on the diagonal\n{'='*100}")
    print(f"  {'region':30s}{'n':>6s}" + ''.join(f"{f'{e} m':>12s}" for e in errors))
    for r, g in d.groupby('region'):
        cells = []
        for e in errors:
            x = g[(g.relief_error_m == e) & (g.elevation_error_m == e)].iloc[0]
            cells.append(f"{int(x.RESOLVED + x['RESOLVED-WITH-EXTERNAL'])}")
        print(f"  {r[:30]:30s}{int(g.n_segments.iloc[0]):>6d}" +
              ''.join(f"{c:>12s}" for c in cells))

    print(f"\n{'='*100}\n  AGAINST THE PREDICTION\n{'='*100}")
    print("  Registered: out-of-catalogue FALLS, degenerate RISES, resolved FALLS.")
    for e in errors[1:]:
        r = tot[(tot.relief_error_m == e) & (tot.elevation_error_m == e)].iloc[0]
        dres = int(r.RESOLVED - base.RESOLVED)
        dout = int(r['OUT-OF-CATALOGUE'] - base['OUT-OF-CATALOGUE'])
        ddeg = int(sum(r[v] for v in VERDICTS if v.startswith('DEGEN'))
                   - sum(base[v] for v in VERDICTS if v.startswith('DEGEN')))
        ok = 'as predicted' if (dres <= 0 and dout <= 0 and ddeg >= 0) else '** AGAINST **'
        print(f"  {e:4d} m : resolved {dres:+4d}   out-of-cat {dout:+4d}   "
              f"degenerate {ddeg:+4d}   {ok}")

    os.makedirs(out_root, exist_ok=True)
    p1 = os.path.join(out_root, 'relief_elevation_error_sweep.csv')
    p2 = os.path.join(out_root, 'relief_elevation_error_totals.csv')
    d.to_csv(p1, index=False)
    tot.to_csv(p2, index=False)
    print(f"\n  Saved: {p1}\n         {p2}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ODSA / 'individual_region_TEST'))
    ap.add_argument('--errors', nargs='+', type=int, default=ERRORS_M)
    a = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    sys.stdout = lv.Tee(str(OUT_ROOT / 'relief_elevation_error_sweep_log.txt'))
    errors = sorted(set(a.errors) | {PROD_RELIEF, PROD_ELEV})
    print(f"Relief / elevation error sweep: {errors} m on each axis, {len(errors)**2} cells")
    print(f"Root: {a.root}")

    d = sweep(a.root, errors)
    baseline_check(a.root, d)
    report(d, errors, str(OUT_ROOT))
