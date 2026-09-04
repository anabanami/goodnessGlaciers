#!/usr/bin/env python3
"""
Velocity error sensitivity sweep for the landscape vector: the ONSET/DIVIDE seam.

Velocity is the only axis that separates ONSET from DIVIDE, so the width of its envelope
decides how many segments admit both. The half-width is K_SIGMA * velocity_sigma, where
velocity_sigma combines the sampled per-window MEaSUREs error with the across-window spread
of segment speed, in quadrature.

This sweeps a multiplier on the sampled error and tabulates what the classification does, so
that the seam is diagnosed as an envelope-width problem or ruled out as one before the
catalogue is touched. The multiplier scales the sampled term alone: the across-window spread
is a measurement of the segment itself and stays fixed, so the envelope grows in quadrature
rather than linearly.

The scale enters velocity_sigma inside build_vector, so the vectors are rebuilt for each
cell. Scale 1.0 is production, and the script asserts that cell reproduces the live
archetype reports.

Reading it: narrowing the envelope admits fewer archetypes, so ONSET|DIVIDE and DEGENERATE
fall while RESOLVED and OUT-OF-CATALOGUE rise. Widening does the reverse. A cell that breaks
that ordering is flagged. If ONSET|DIVIDE barely moves across the grid, the seam is not a
question of envelope width and the answer is elsewhere.

Filling the transitional+`low` catalogue hole cannot move the seam either way, because a
wider allowed set only ever ADDS archetypes. That hole is unreachable until this envelope
narrows, so the two are one fix rather than two.

The sidecar is required. Without it velocity falls back to the VELOCITY_ERROR_M_YR constant,
which is one number for the whole survey and offers the scale nothing to act on.

Nothing here changes the catalogue and nothing writes into the region trees; results go to
v23/velocity_error/.

      python velocity_error_sweep.py
      python velocity_error_sweep.py --scales 0.5 1 2          # cheap triple
      python velocity_error_sweep.py --root ../individual_region_TEST
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

SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]
BASELINE = 1.0                                  # production
SEAM = frozenset({'very_low', 'low', 'moderate'})
SEAM_PAIR = ('ONSET', 'DIVIDE')
VERDICTS = ['RESOLVED', 'RESOLVED-WITH-EXTERNAL', 'DEGENERATE', 'OUT-OF-CATALOGUE']
SHORT = {'RESOLVED': 'resolved', 'RESOLVED-WITH-EXTERNAL': '+external',
         'DEGENERATE': 'degenerate', 'OUT-OF-CATALOGUE': 'out-of-cat'}


def load_region(csv_path):
    """The frame process_region classifies: transitions dropped and the velocity sidecar
    merged, so every cell is measured against production's own inputs."""
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    pflag = lv.region_flag(df)
    if 'is_transition' in df.columns:
        df = df[~df['is_transition']].copy()
    vel = lv.load_velocity_error(csv_path)
    if vel is None:
        sys.exit(f"No *_velocity_error.csv beside {csv_path}. The scale has nothing to act "
                 f"on without it; run velocity_error_sidecar.py first.")
    keys = [c for c in ('trajectory', 'segment', 'window_id')
            if c in vel.columns and c in df.columns]
    df = df.merge(vel[keys + ['measures_err_m_yr', 'measures_cnt']], on=keys, how='left')
    return df, pflag


def vectors_at(df, pflag, scale):
    """Segment vectors with the sampled error scaled. _agg folds it into velocity_sigma in
    quadrature with the standard error of the median, which the scale leaves alone."""
    g = df.copy()
    g['measures_err_m_yr'] = g['measures_err_m_yr'] * scale
    return [lv.build_vector(sub, u, pflag) for u, sub in lv.units_from(g, 'segment')]


def classify(vecs, pflag):
    """Verdicts for one region's segment vectors. The scale is already inside them, since it
    enters velocity_sigma at build time rather than at observe()."""
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


def sweep(root, scales):
    found = {os.path.basename(f).replace('_window_stats.csv', ''): f
             for f in sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'),
                                       recursive=True))}
    if not found:
        sys.exit(f"No *_window_stats.csv under {root}")
    print(f"Regions: {len(found)}")

    # Read each region once; the scale is applied to a copy of the frame per cell.
    frames = {}
    for r, f in found.items():
        df, pflag = load_region(f)
        frames[r] = (df, pflag)
        print(f"  {r:48s} {df.groupby(['trajectory', 'segment']).ngroups:4d} segments")

    rows = []
    for k in scales:
        for r, (df, pflag) in frames.items():
            s = classify(vectors_at(df, pflag, k), pflag)
            matched = s.loc[s.n_admissible > 0, 'n_admissible']
            rec = {'error_scale': k, 'region': r, 'n_segments': len(s),
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
    return pd.DataFrame(rows)


def baseline_check(root, d):
    """Scale 1.0 must reproduce the live archetype reports; if it does not, this script is
    not classifying the way process_region does and nothing below it is trustworthy."""
    files = glob.glob(os.path.join(root, '*', 'landscape_vector', '*_archetype_report.csv'))
    if not files:
        print("\n  (no archetype reports under root -- baseline not cross-checked)")
        return
    live = pd.concat([pd.read_csv(f) for f in files])
    live = live[live.level == 'segment']
    seam = int((live.archetypes.fillna('').str.contains(SEAM_PAIR[0]) &
                live.archetypes.fillna('').str.contains(SEAM_PAIR[1])).sum())
    counts = live.verdict.value_counts()
    mine = d[d.error_scale == BASELINE]
    if not len(mine):
        print(f"\n  (scale {BASELINE:g} not in the grid -- baseline not cross-checked)")
        return
    bad = {v: (int(mine[v].sum()), int(counts.get(v, 0))) for v in VERDICTS
           if int(mine[v].sum()) != int(counts.get(v, 0))}
    if int(mine.onset_divide.sum()) != seam:
        bad['ONSET|DIVIDE'] = (int(mine.onset_divide.sum()), seam)
    if bad:
        print("\n  ** BASELINE MISMATCH vs the live reports (sweep, report): "
              + ', '.join(f"{v} {a} vs {b}" for v, (a, b) in bad.items())
              + " -- do not read the sweep until this is explained **")
    else:
        print(f"\n  Baseline check: scale {BASELINE:g} reproduces the live reports exactly "
              f"({', '.join(f'{SHORT[v]} {int(mine[v].sum())}' for v in VERDICTS if mine[v].sum())}"
              f", ONSET|DIVIDE {seam})")


def report(d, scales, out_root):
    tot = d.groupby('error_scale', as_index=False).sum(numeric_only=True)
    base = tot[tot.error_scale == BASELINE]
    base = base.iloc[0] if len(base) else None
    n = int(d[d.error_scale == scales[0]].n_segments.sum())

    if base is not None:
        print(f"\n{'='*104}\n  BASELINE (scale {BASELINE:g}, = production)  {n} segments\n{'='*104}")
        print("    " + '  '.join(f"{SHORT[v]} {int(base[v])}" for v in VERDICTS if base[v]))
        print(f"    ONSET|DIVIDE co-admitted: {int(base.onset_divide)} "
              f"({base.onset_divide / n:.0%})")
        print(f"    carrying the {{very_low, low, moderate}} envelope: {int(base.seam_envelope)} "
              f"({base.seam_envelope / n:.0%})")

    print(f"\n{'='*104}\n  THE SWEEP\n{'='*104}")
    hdr = ['scale'] + [SHORT[v] for v in VERDICTS] + \
          ['ONSET|DIV', 'seam env', 'exact_vel', 'mean|band|', 'med|set|']
    print('  ' + ''.join(f"{h:>12s}" for h in hdr))
    for k in scales:
        r = tot[tot.error_scale == k].iloc[0]
        g = d[d.error_scale == k]
        vals = [f"{k:g}"] + [f"{int(r[v])}" for v in VERDICTS] + \
               [f"{int(r.onset_divide)}", f"{int(r.seam_envelope)}", f"{int(r.exact_velocity)}",
                f"{g.mean_n_bands.mean():.2f}", f"{g.median_admissible.median():.0f}"]
        print('  ' + ''.join(f"{x:>12s}" for x in vals))

    print(f"\n{'='*104}\n  ONSET|DIVIDE PER REGION\n{'='*104}")
    print(f"  {'region':30s}{'n':>6s}" + ''.join(f"{f'{k:g}':>10s}" for k in scales))
    for r, g in d.groupby('region'):
        cells = [f"{int(g[g.error_scale == k].onset_divide.iloc[0])}" for k in scales]
        print(f"  {r[:30]:30s}{int(g.n_segments.iloc[0]):>6d}" +
              ''.join(f"{c:>10s}" for c in cells))

    print(f"\n{'='*104}\n  WHAT THE VELOCITY AXIS IS DOING\n{'='*104}")
    print("  With the sidecar merged, a unit whose windows carry no MEaSUREs coverage is")
    print("  routed to `unavailable` and widens to every band, so `exact` stays 0 across the")
    print("  grid. A non-zero entry means a unit reached classify_set with no sigma at all.")
    print(f"\n  {'scale':>10s}{'med half-width':>16s}{'exact':>8s}{'ambiguous':>11s}"
          f"{'unavailable':>13s}")
    for k in scales:
        r = tot[tot.error_scale == k].iloc[0]
        hw = d[d.error_scale == k].median_half_width.median()
        print(f"  {k:>10g}{hw if np.isfinite(hw) else float('nan'):>16.1f}"
              f"{int(r.exact_velocity):>8d}{int(r.ambiguous_velocity):>11d}"
              f"{int(r.unavailable_velocity):>13d}")

    if base is not None:
        print(f"\n{'='*104}\n  DIRECTION\n{'='*104}")
        print(f"  Relative to scale {BASELINE:g}: a narrower envelope cuts ONSET|DIVIDE and")
        print("  DEGENERATE and raises RESOLVED and OUT-OF-CATALOGUE. A wider one reverses it.")
        for k in [x for x in scales if x != BASELINE]:
            r = tot[tot.error_scale == k].iloc[0]
            dsm = int(r.onset_divide - base.onset_divide)
            dres = int(r.RESOLVED - base.RESOLVED)
            dout = int(r['OUT-OF-CATALOGUE'] - base['OUT-OF-CATALOGUE'])
            ddeg = int(r.DEGENERATE - base.DEGENERATE)
            s = 1 if k < BASELINE else -1          # expected sign of dres and dout
            ok = 'consistent' if (s * dsm <= 0 and s * dres >= 0 and s * dout >= 0
                                  and s * ddeg <= 0) else '** AGAINST **'
            print(f"  {k:4g} x : ONSET|DIVIDE {dsm:+4d}   resolved {dres:+4d}   "
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
    ap.add_argument('--scales', nargs='+', type=float, default=SCALES)
    a = ap.parse_args()

    scales = sorted(set(a.scales))
    # At scale 0 the sampled error becomes a finite 0.0 rather than absent, so a
    # single-window segment gets sigma 0.0, resolves against the break with no uncertainty
    # and is not flagged `exact`. The cell would read as resolution rather than as the
    # over-confidence it is.
    if any(k <= 0 for k in scales):
        sys.exit("Scales must be positive.")

    os.makedirs(OUT_ROOT, exist_ok=True)
    sys.stdout = lv.Tee(str(OUT_ROOT / 'velocity_error_sweep_log.txt'))
    print(f"Velocity error scale sweep: {[f'{k:g}' for k in scales]} x sampled error, "
          f"{len(scales)} cells")
    print(f"K_SIGMA = {lv.K_SIGMA}, so the envelope half-width is "
          f"K_SIGMA x sqrt(spread^2 + (scale x sampled error)^2)")
    print(f"Root: {a.root}")

    d = sweep(a.root, scales)
    baseline_check(a.root, d)
    report(d, scales, str(OUT_ROOT))
