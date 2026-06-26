#!/usr/bin/env python3
"""
Window-size sensitivity sweep for the ODSA pipeline.

Re-runs bed_analysis_23.py at several WINDOW_SIZE values on a fixed 3-region
subset, each into an isolated output tree (this folder/runs/), then assembles
one comparison table of the window-size-sensitive outputs:
  - sample geometry : n segments, n exported windows
  - beta stability  : mean / median / IQR / range
  - anisotropy      : cos2(theta) dbeta +/- SE, z, R2 (window & segment, unweighted & weighted)

Mechanical trends (confirmed-wavelength count / mean-lambda scaling with window
size) are deterministic consequences of min_freq = 1/window_size and are NOT
re-derived here.

CONFOUND: changing WINDOW_SIZE also moves STEP_SIZE (= W/2) and SMOOTHING_LENGTH
(= W, the landscape-split smoothing kernel, config.py). So this sweeps the window
knob exactly as the pipeline wires it -- segmentation included. To vary the
spectral window alone, decouple SMOOTHING_LENGTH from WINDOW_SIZE in config.py.

Run:  python v23/window_size_sensitivity_test/run_sweep.py                # full sweep (30/50/75/100)
      python v23/window_size_sensitivity_test/run_sweep.py --sizes 50     # cheap single-size spot-check
      python v23/window_size_sensitivity_test/run_sweep.py --skip-runs    # rebuild table from existing runs/
      python v23/window_size_sensitivity_test/run_sweep.py --skip-runs --plots   # (re)draw anisotropy panels
      python v23/window_size_sensitivity_test/run_sweep.py --skip-runs --plots 0  # ...and don't gate on z
"""
import os, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent          # .../v23/window_size_sensitivity_test
ODSA = HERE.parents[1]                           # .../ODSA root (bed_analysis_23.py, config.py, weighted_anisotropy.py)
RUNS = HERE / 'runs'
PLOT_Z_MIN = 1.5         # --plots only draws cells where |z| (unw or wt) reaches this; tune via `--plots <z>`
sys.path.insert(0, str(ODSA))
from weighted_anisotropy import fit_cos2, flow_weight  # noqa: E402

SIZES_KM = [30, 50, 75, 100]
REGIONS = [                                       # all 7 Ockenden regions (loading.py)
    'Rec_Catch_Fig2D_Recovery_SL',                # low-relief,              tier A, n~114, clean coverage
    'POLARGAP_2015_Fig2C_Hercules_Dome',          # alpine/mountainous,      tier B, n~40,  worst fragmentation
    'POLARGAP_2015_Fig1_Pensacola_Pole',          # selective erosion,       tier C, n~283, data-rich, big gaps
    'ASB_ICECAP_2010_Fig4_Aurora_SB_lowrelief',   # low-relief
    'ASB_ICECAP_2010_Fig2A_Maud_SB',              # low-relief / sel. erosion
    'ASB_ICECAP_2010_Fig2G_Highland_A',           # alpine / sel. erosion / low-relief
    'ASB_ICECAP_2010_Fig2H_Golicyna_SM',          # alpine / sel. erosion / low-relief
]


def run_one(size_km):
    env = dict(os.environ,
               ODSA_WINDOW_SIZE=str(size_km * 1000),
               ODSA_WINDOW_TYPE='rectangular',
               ODSA_OUTPUT_BASE=str(RUNS) + os.sep,
               ODSA_REGION_FILTER=','.join(REGIONS))
    print(f"\n{'='*70}\n  WINDOW_SIZE = {size_km} km\n{'='*70}")
    subprocess.run([sys.executable, 'bed_analysis_23.py'], cwd=str(ODSA), env=env, check=True)


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
    out[f'{level}_beta_mean'] = b.mean()
    out[f'{level}_beta_med'] = np.median(b)
    out[f'{level}_beta_iqr'] = np.subtract(*np.percentile(b, [75, 25]))
    out[f'{level}_beta_min'], out[f'{level}_beta_max'] = b.min(), b.max()
    spd = df['measures_speed_mean'].values if 'measures_speed_mean' in df else None
    w = flow_weight(df['flow_error_mean'].values, speed=spd) if 'flow_error_mean' in df else None
    for tag, weights in [('unw', None), ('wt', w)]:
        f = fit_cos2(th, b, weights=weights) if (weights is None or np.any(weights > 0)) else None
        if f:
            out[f'{level}_{tag}_dbeta'] = f['delta']
            out[f'{level}_{tag}_se'] = f['delta_se']
            out[f'{level}_{tag}_z'] = abs(f['delta'] / f['delta_se']) if f['delta_se'] > 0 else np.inf
            out[f'{level}_{tag}_r2'] = f['r2']
    return out


def build_table():
    rows = []
    for region in REGIONS:
        for km in SIZES_KM:
            sfx = f'_w{km}km'
            row = {'region': region, 'window_km': km}
            row.update(_aniso(os.path.join(RUNS, 'window_csvs', f'{region}{sfx}_window_stats.csv'), 'win'))
            row.update(_aniso(os.path.join(RUNS, 'segment_csvs', f'{region}{sfx}_segment_stats.csv'), 'seg'))
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'window_size_comparison.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    for region in REGIONS:
        cols = ['window_km', 'win_n', 'win_beta_mean', 'win_beta_iqr',
                'win_unw_dbeta', 'win_wt_dbeta', 'win_wt_z',
                'seg_n', 'seg_unw_dbeta', 'seg_unw_z']
        sub = df[df.region == region]
        print(f"\n### {region}")
        print(sub[[c for c in cols if c in sub]].to_string(index=False))
    print(f"\nFull table -> {out}")
    return df


def make_plots(df, z_min=PLOT_Z_MIN):
    """Draw the unweighted/weighted cos2 panels (weighted_anisotropy.plot_anisotropy)
    from the already-saved per-size CSVs. Pure post-process: needs no pipeline re-run.
    Gated on |z| so the flat-null cells don't clutter the output (z_min=0 -> draw all)."""
    import weighted_anisotropy as wa  # already imported above; re-bind its output dir
    plotdir = os.path.join(RUNS, 'anisotropy_plots')
    os.makedirs(plotdir, exist_ok=True)
    wa.OUTPUT_BASE_PATH = plotdir + os.sep
    n = 0
    for _, r in df.iterrows():
        region, km = r['region'], int(r['window_km'])
        for level, sub, csvdir in [('window', 'win', 'window_csvs'),
                                   ('segment', 'seg', 'segment_csvs')]:
            zs = [r.get(f'{sub}_{tag}_z') for tag in ('unw', 'wt')]
            zmax = max([z for z in zs if pd.notna(z)], default=0.0)
            if zmax < z_min:
                continue
            csv = os.path.join(RUNS, csvdir, f'{region}_w{km}km_{level}_stats.csv')
            if not os.path.exists(csv):
                continue
            print(f"\n--- {region} {km}km {level} (z={zmax:.2f}) ---")
            wa.plot_anisotropy(csv, level=level)
            n += 1
    print(f"\nDrew {n} anisotropy panel(s) with |z| >= {z_min} -> {plotdir}")


if __name__ == '__main__':
    if '--sizes' in sys.argv:
        SIZES_KM = [int(s) for s in sys.argv[sys.argv.index('--sizes') + 1].split(',')]
    if '--skip-runs' not in sys.argv:
        os.makedirs(RUNS, exist_ok=True)
        for km in SIZES_KM:
            run_one(km)
    df = build_table()
    if '--plots' in sys.argv:
        i = sys.argv.index('--plots') + 1
        z_min = float(sys.argv[i]) if i < len(sys.argv) and not sys.argv[i].startswith('-') else PLOT_Z_MIN
        make_plots(df, z_min=z_min)
