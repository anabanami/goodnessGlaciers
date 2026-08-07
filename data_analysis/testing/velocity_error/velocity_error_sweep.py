#!/usr/bin/env python3
"""
Velocity error sensitivity sweep for the landscape vector: the ONSET/DIVIDE seam.

VELOCITY_ERROR_M_YR = 5.0 at K_SIGMA = 2 puts a +/-10 m/yr envelope on every segment.
The within-unit spread adds almost nothing (half-width runs 10.0 min, 10.0 median, 12.4
max over the 379 test segments), so the envelope is effectively a constant -- and it is
20 m/yr wide while every break it has to resolve (very_low|low at 5, low|moderate at 10)
sits inside 0-10 m/yr. Median segment speed is 8.15 m/yr.

The consequence is that 279 of 379 segments (74%) carry the identical envelope
{very_low, low, moderate}, and all 127 segments admitting both ONSET and DIVIDE are
inside that group with no exceptions. On those 127 every other axis has already done what
it can: elevation is resolved on 113 and never reads `elevated` (the one value that drops
DIVIDE), relief on 93, delta_beta on 58 -- and delta_beta resolving to `zero` cannot
help, since both entries allow zero. Velocity is the only axis left in the comparison.

This sweeps the constant and tabulates what the classification does, so that the seam is
diagnosed as an envelope-width problem or ruled out as one before the catalogue is touched.

PREDICTION, registered before the run: narrowing an envelope admits fewer archetypes, so
ONSET|DIVIDE should FALL, DEGENERATE should FALL, RESOLVED should RISE and OUT-OF-CATALOGUE
should RISE toward the preregistered 44% ceiling. Same direction as the delta_beta run and
the opposite of the relief/elevation sweep, which widened.

Two things to hold while reading it:

  - 0 and None are the same setting (`if nominal:` in observe() treats 0 as falsy), so the
    0 cell is NOT the truth. It drops back to the within-unit spread alone, which a
    single-window segment does not have, so velocity goes `assumed-exact` on those and the
    axis resolves against the 5 m/yr break with no uncertainty at all. That is the
    over-confidence the constant was added to prevent. Read `exact_velocity` alongside the
    verdicts: gains bought by that column are not gains.
  - Nothing here changes the catalogue. Filling the transitional+`low` hole is monotone
    (a wider allowed set only ever ADDS archetypes), so it cannot move the 127 either way.
    The hole is unreachable until this envelope narrows; the two are one fix, not two.

The production baseline is the 5.0 cell and the script asserts it reproduces the live
reports. Nothing writes into the region trees; results go to v23/velocity_error/.

      python velocity_error_sweep.py
      python velocity_error_sweep.py --errors 0 5              # cheap baseline pair
      python velocity_error_sweep.py --root ../Ockenden-regions
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA root
OUT_ROOT = HERE / 'velocity_error'
sys.path.insert(0, str(ODSA))
import landscape_vector as lv                                          # noqa: E402

ERRORS_M_YR = [0.0, 1.0, 2.0, 5.0, 10.0]
BASELINE = 5.0                                  # production
SEAM = frozenset({'very_low', 'low', 'moderate'})
SEAM_PAIR = ('ONSET', 'DIVIDE')
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
    return df, pflag


def classify(vecs, pflag):
    """Verdicts under whatever lv.VELOCITY_ERROR_M_YR is set to now. The vectors are
    error-independent -- the constant enters at observe(), folded in by hypot."""
    rows = []
    for v in vecs:
        obs = lv.observe(v, pflag)
        cases = lv.match(obs)
        kind, _, _ = lv.verdict(cases, obs)
        ids = {c['id'] for c, _ in cases}
        o = obs['velocity_band']
        rows.append({'verdict': kind, 'n_admissible': len(cases),
                     'seam_pair': set(SEAM_PAIR) <= ids,
                     'n_bands': len(o['set']),
                     'seam_envelope': frozenset(o['set']) == SEAM,
                     'half_width': lv.K_SIGMA * o['sigma'] if np.isfinite(o['sigma']) else np.nan,
                     'exact_vel': o['status'] == 'assumed-exact',
                     'amb_vel': o['status'] == 'ambiguous',
                     'unavail_vel': o['status'] == 'unavailable'})
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
    for err in errors:
        lv.VELOCITY_ERROR_M_YR = err or None
        for r, (vecs, pflag) in frames.items():
            s = classify(vecs, pflag)
            matched = s.loc[s.n_admissible > 0, 'n_admissible']
            rec = {'velocity_error_m_yr': err, 'region': r, 'n_segments': len(s),
                   'median_admissible': matched.median() if len(matched) else np.nan,
                   'onset_divide': int(s.seam_pair.sum()),
                   'seam_envelope': int(s.seam_envelope.sum()),
                   'mean_n_bands': s.n_bands.mean(),
                   'median_half_width': s.half_width.median(),
                   'exact_velocity': int(s.exact_vel.sum()),
                   'ambiguous_velocity': int(s.amb_vel.sum()),
                   'unavailable_velocity': int(s.unavail_vel.sum())}
            rec.update({v: int((s.verdict == v).sum()) for v in VERDICTS})
            rows.append(rec)
    lv.VELOCITY_ERROR_M_YR = BASELINE                    # leave the module as we found it
    return pd.DataFrame(rows)


def baseline_check(root, d):
    """The 5.0 cell must reproduce the live archetype reports; if it does not, this script
    is not classifying the way process_region does and nothing below it is trustworthy."""
    files = glob.glob(os.path.join(root, '*', 'landscape_vector', '*_archetype_report.csv'))
    if not files:
        print("\n  (no archetype reports under root -- baseline not cross-checked)")
        return
    live = pd.concat([pd.read_csv(f) for f in files])
    live = live[live.level == 'segment']
    seam = int((live.archetypes.fillna('').str.contains(SEAM_PAIR[0]) &
                live.archetypes.fillna('').str.contains(SEAM_PAIR[1])).sum())
    counts = live.verdict.value_counts()
    mine = d[d.velocity_error_m_yr == BASELINE]
    bad = {v: (int(mine[v].sum()), int(counts.get(v, 0))) for v in VERDICTS
           if int(mine[v].sum()) != int(counts.get(v, 0))}
    if int(mine.onset_divide.sum()) != seam:
        bad['ONSET|DIVIDE'] = (int(mine.onset_divide.sum()), seam)
    if bad:
        print("\n  ** BASELINE MISMATCH vs the live reports (sweep, report): "
              + ', '.join(f"{v} {a} vs {b}" for v, (a, b) in bad.items())
              + " -- do not read the sweep until this is explained **")
    else:
        print(f"\n  Baseline check: {BASELINE} m/yr reproduces the live reports exactly "
              f"({', '.join(f'{SHORT[v]} {int(mine[v].sum())}' for v in VERDICTS if mine[v].sum())}"
              f", ONSET|DIVIDE {seam})")


def report(d, errors, out_root):
    tot = d.groupby('velocity_error_m_yr', as_index=False).sum(numeric_only=True)
    base = tot[tot.velocity_error_m_yr == BASELINE]
    base = base.iloc[0] if len(base) else None
    n = int(d[d.velocity_error_m_yr == errors[0]].n_segments.sum())

    if base is not None:
        print(f"\n{'='*104}\n  BASELINE ({BASELINE} m/yr, = production)  {n} segments\n{'='*104}")
        print("    " + '  '.join(f"{SHORT[v]} {int(base[v])}" for v in VERDICTS if base[v]))
        print(f"    ONSET|DIVIDE co-admitted: {int(base.onset_divide)} "
              f"({base.onset_divide / n:.0%})")
        print(f"    carrying the {{very_low, low, moderate}} envelope: {int(base.seam_envelope)} "
              f"({base.seam_envelope / n:.0%})")

    print(f"\n{'='*104}\n  THE SWEEP\n{'='*104}")
    hdr = ['err m/yr'] + [SHORT[v] for v in VERDICTS] + \
          ['ONSET|DIV', 'seam env', 'exact_vel', 'mean|band|', 'med|set|']
    print('  ' + ''.join(f"{h:>12s}" for h in hdr))
    for e in errors:
        r = tot[tot.velocity_error_m_yr == e].iloc[0]
        g = d[d.velocity_error_m_yr == e]
        vals = [f"{e:g}"] + [f"{int(r[v])}" for v in VERDICTS] + \
               [f"{int(r.onset_divide)}", f"{int(r.seam_envelope)}", f"{int(r.exact_velocity)}",
                f"{g.mean_n_bands.mean():.2f}", f"{g.median_admissible.median():.0f}"]
        print('  ' + ''.join(f"{x:>12s}" for x in vals))

    print(f"\n{'='*104}\n  ONSET|DIVIDE PER REGION\n{'='*104}")
    print(f"  {'region':30s}{'n':>6s}" + ''.join(f"{f'{e:g}':>10s}" for e in errors))
    for r, g in d.groupby('region'):
        cells = [f"{int(g[g.velocity_error_m_yr == e].onset_divide.iloc[0])}" for e in errors]
        print(f"  {r[:30]:30s}{int(g.n_segments.iloc[0]):>6d}" +
              ''.join(f"{c:>10s}" for c in cells))

    print(f"\n{'='*104}\n  WHAT THE VELOCITY AXIS IS DOING\n{'='*104}")
    print("  exact_vel is the trap: at 0 the axis loses its error bar on single-window")
    print("  segments and resolves against the 5 m/yr break with no uncertainty. Any")
    print("  resolution bought there is over-confidence, not measurement.")
    print(f"\n  {'err m/yr':>10s}{'med half-width':>16s}{'exact':>8s}{'ambiguous':>11s}"
          f"{'unavailable':>13s}")
    for e in errors:
        r = tot[tot.velocity_error_m_yr == e].iloc[0]
        hw = d[d.velocity_error_m_yr == e].median_half_width.median()
        print(f"  {e:>10g}{hw if np.isfinite(hw) else float('nan'):>16.1f}"
              f"{int(r.exact_velocity):>8d}{int(r.ambiguous_velocity):>11d}"
              f"{int(r.unavailable_velocity):>13d}")

    if base is not None:
        print(f"\n{'='*104}\n  AGAINST THE PREDICTION\n{'='*104}")
        print(f"  Registered, relative to the {BASELINE} m/yr baseline: narrowing the envelope")
        print("  cuts ONSET|DIVIDE and DEGENERATE, raises RESOLVED and OUT-OF-CATALOGUE.")
        for e in [x for x in errors if x < BASELINE]:
            r = tot[tot.velocity_error_m_yr == e].iloc[0]
            dsm = int(r.onset_divide - base.onset_divide)
            dres = int(r.RESOLVED - base.RESOLVED)
            dout = int(r['OUT-OF-CATALOGUE'] - base['OUT-OF-CATALOGUE'])
            ddeg = int(r.DEGENERATE - base.DEGENERATE)
            ok = 'as predicted' if (dsm <= 0 and dres >= 0 and dout >= 0 and ddeg <= 0) \
                 else '** AGAINST **'
            print(f"  {e:4g} m/yr : ONSET|DIVIDE {dsm:+4d}   resolved {dres:+4d}   "
                  f"out-of-cat {dout:+4d}   degenerate {ddeg:+4d}   {ok}")
        print("\n  If ONSET|DIVIDE barely moves, the seam is not envelope width and the")
        print("  catalogue hole is not the other half of it either -- look elsewhere.")

    os.makedirs(out_root, exist_ok=True)
    p1 = os.path.join(out_root, 'velocity_error_sweep.csv')
    p2 = os.path.join(out_root, 'velocity_error_totals.csv')
    d.to_csv(p1, index=False)
    tot.to_csv(p2, index=False)
    print(f"\n  Saved: {p1}\n         {p2}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ODSA / 'individual_region_TEST'))
    ap.add_argument('--errors', nargs='+', type=float, default=ERRORS_M_YR)
    a = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    sys.stdout = lv.Tee(str(OUT_ROOT / 'velocity_error_sweep_log.txt'))
    errors = sorted(set(a.errors))
    print(f"Velocity error sweep: {[f'{e:g}' for e in errors]} m/yr, {len(errors)} cells")
    print(f"K_SIGMA = {lv.K_SIGMA}, so the envelope half-width is K_SIGMA x error")
    print(f"Root: {a.root}")

    d = sweep(a.root, errors)
    baseline_check(a.root, d)
    report(d, errors, str(OUT_ROOT))
