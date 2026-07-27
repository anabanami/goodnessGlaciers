#!/usr/bin/env python3
"""
Flow-stencil sensitivity sweep for the ODSA anisotropy result.

The incidence angle theta is set by the modelled flow bearing, which
REMA_extractor.extract_rema_flow_vector derives from the surface slope measured
over a stencil of half-width STENCIL_FACTOR * ice_thickness. Production uses
STENCIL_FACTOR = 5; McCormack et al. (2019) recommend ~10. The cos2(theta) fit,
and in particular Pensacola's negative window Delta-beta, depends on theta, so this
brackets STENCIL_FACTOR and reports how Delta-beta moves.

Changing the stencil moves three things at once, all reported so the mechanism is
visible: theta itself, the flat-surface rejection (dz scales with the stencil, so a
wider stencil rejects fewer points), and the MEaSUREs flow-error that sets the
weights.

Re-runs bed_analysis_23.py for Pensacola at each factor into an isolated output
tree (this folder/stencil_sensitivity/runs/x<factor>/), then refits Delta-beta at
window and segment level, unweighted and weighted, exactly as the pipeline does.

Run from v23/:
      python stencil_sensitivity.py                    # full sweep (2.5/5/7.5/10)
      python stencil_sensitivity.py --factors 5,10     # subset
      python stencil_sensitivity.py --skip-runs        # rebuild table from existing runs/
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ODSA = HERE.parent
OUT_ROOT = HERE / 'stencil_sensitivity'
RUNS = OUT_ROOT / 'runs'
sys.path.insert(0, str(ODSA))
from weighted_anisotropy import fit_cos2, flow_weight  # noqa: E402
from config import Tee  # noqa: E402

FACTORS = [2.5, 5.0, 7.5, 10.0]                  # 5.0 = production, 10.0 = McCormack 2019
REGIONS = ['POLARGAP_2015_Pensacola_Pole']       # selective-erosion region; the open sign call
SUFFIX = '_w50km'                                # WINDOW_SIZE unchanged, so filename is fixed


def base_for(factor):
    return RUNS / f'x{factor}'


def run_one(factor):
    env = dict(os.environ,
               ODSA_STENCIL_FACTOR=str(factor),
               ODSA_OUTPUT_BASE=str(base_for(factor)) + os.sep,
               ODSA_REGION_FILTER=','.join(REGIONS))
    print(f"\n{'='*70}\n  STENCIL_FACTOR = {factor} x ice thickness\n{'='*70}")
    subprocess.run([sys.executable, 'bed_analysis_23.py'], cwd=str(ODSA), env=env, check=True)


def _ess(w):
    w = w[w > 0]
    return (w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0


def _aniso(csv, level):
    if not os.path.exists(csv):
        return {}
    df = pd.read_csv(csv).dropna(subset=['incidence_deg', 'beta'])
    if 'is_transition' in df:
        df = df[~df['is_transition']]
    out = {f'{level}_n': len(df)}
    if len(df) < 4:
        return out
    th, b = df['incidence_deg'].values, df['beta'].values
    spd = df['measures_speed_mean'].values if 'measures_speed_mean' in df else None
    w = flow_weight(df['flow_error_mean'].values, speed=spd) if 'flow_error_mean' in df else None
    m = w > 0 if w is not None else np.ones(len(df), bool)
    out[f'{level}_nw'] = int(m.sum())                       # n at weight > 0
    out[f'{level}_ess'] = _ess(w) if w is not None else len(df)
    out[f'{level}_theta_mean'] = float(th[m].mean())        # mean used incidence
    if 'flow_undefined_frac' in df:
        out[f'{level}_undef'] = float(df['flow_undefined_frac'].mean())
    if 'flow_error_mean' in df:
        out[f'{level}_ferr'] = float(np.nanmean(df['flow_error_mean'].values[m]))
    for tag, weights in [('unw', None), ('wt', w)]:
        f = fit_cos2(th[m], b[m], weights=(weights[m] if weights is not None else None)) \
            if (weights is None or np.any(m)) else None
        if f:
            out[f'{level}_{tag}_dbeta'] = f['delta']
            out[f'{level}_{tag}_se'] = f['delta_se']
            out[f'{level}_{tag}_z'] = abs(f['delta'] / f['delta_se']) if f['delta_se'] > 0 else np.inf
            out[f'{level}_{tag}_r2'] = f['r2']
    return out


def build_table():
    rows = []
    for region in REGIONS:
        for fac in FACTORS:
            base = base_for(fac)
            row = {'region': region, 'factor': fac}
            row.update(_aniso(os.path.join(base, 'window_csvs', f'{region}{SUFFIX}_window_stats.csv'), 'win'))
            row.update(_aniso(os.path.join(base, 'segment_csvs', f'{region}{SUFFIX}_segment_stats.csv'), 'seg'))
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_ROOT, 'stencil_sensitivity_comparison.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 220)
    for region in REGIONS:
        sub = df[df.region == region]
        print(f"\n### {region}")
        print("  mechanism (how the stencil moves the inputs):")
        mech = ['factor', 'win_n', 'win_nw', 'win_ess', 'win_theta_mean', 'win_undef', 'win_ferr']
        print(sub[[c for c in mech if c in sub]].to_string(index=False))
        print("\n  anisotropy Delta-beta vs factor (5.0 = production):")
        an = ['factor', 'win_unw_dbeta', 'win_wt_dbeta', 'win_wt_se', 'win_wt_z',
              'seg_wt_dbeta', 'seg_wt_se', 'seg_wt_z']
        print(sub[[c for c in an if c in sub]].to_string(index=False))
    print(f"\nFull table -> {out}")
    return df


if __name__ == '__main__':
    if '--factors' in sys.argv:
        FACTORS = [float(s) for s in sys.argv[sys.argv.index('--factors') + 1].split(',')]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(str(OUT_ROOT / 'stencil_sensitivity_log.txt'))
    if '--skip-runs' not in sys.argv:
        RUNS.mkdir(parents=True, exist_ok=True)
        for fac in FACTORS:
            run_one(fac)
    build_table()
