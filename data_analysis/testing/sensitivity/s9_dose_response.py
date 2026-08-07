"""Recover §9 dose-response inputs: n and missing-decade range for the
window-level beta vs missing-band regression (truncated population).

Run from v23/; writes results to v23/TESTING_LANDSCAPE_SPLITTING/."""
import numpy as np, pandas as pd, os
from pyproj import Transformer
from scipy import stats as sst
import sys
HERE = os.path.dirname(os.path.abspath(__file__))            # .../v23
ODSA = os.path.dirname(HERE)                                 # .../ODSA — current codebase + results
OUT = os.path.join(HERE, "TESTING_LANDSCAPE_SPLITTING")      # this script's results folder
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import detect_data_gaps, split_into_segments, split_by_landscape
from config import WINDOW_SIZE, FIT_BAND_M, Tee
RESULTS = os.path.join(ODSA, "Ockenden-regions")  # most recent codebase run
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "s9_dose_response_log.txt"))

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# segment length per (trajectory, seg_idx+1), reproducing bed_analysis_23 ordering
lengths = {}  # (dataset, traj, seg_num) -> length_m
for d in load_datasets():
    name, df = d['name'], d['data']
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
    for traj_id in valid['trajectory_id'].unique():
        line = valid[valid['trajectory_id'] == traj_id].copy()
        if len(line) < 20: continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        line2 = line.copy()
        gap_segments = split_into_segments(line2, dist)
        if not gap_segments: continue
        segs = []
        for sd, sdist in gap_segments:
            segs.extend(split_by_landscape(sd, sdist))
        for i, (sdata, sdist, is_t) in enumerate(segs):
            lengths[(name, str(traj_id), i + 1)] = sdist.max() - sdist.min()

csvs = {
    'Pensacola': 'POLARGAP_2015_Pensacola_Pole_w50km_window_stats.csv',
    'Hercules':  'POLARGAP_2015_Fig2C_Hercules_Dome_w50km_window_stats.csv',
}
datasetname = {'Pensacola': 'POLARGAP_2015_Pensacola_Pole',
               'Hercules':  'POLARGAP_2015_Fig2C_Hercules_Dome'}

# LOG50 is log10 of the fit-band upper wavelength, taken from config so it cannot go
# stale. The name keeps the 50 km value it was written for.
LOG50 = np.log10(float(FIT_BAND_M[1]))
for reg, fn in csvs.items():
    w = pd.read_csv(os.path.join(RESULTS, 'window_csvs', fn))
    w['L'] = [lengths.get((datasetname[reg], str(t), int(s)), np.nan)
              for t, s in zip(w['trajectory'], w['segment'])]
    print(f"\n##### {reg}: {len(w)} windows, {w['L'].isna().sum()} unmatched (no length)")
    d = w[~w['is_transition'].astype(bool)].dropna(subset=['L', 'beta'])  # homogeneous
    trunc = d[d['L'] < WINDOW_SIZE].copy()
    full  = d[d['L'] >= WINDOW_SIZE].copy()
    trunc['md'] = LOG50 - np.log10(trunc['L'])
    print(f"  truncated n={len(trunc)}  full-band n={len(full)}")
    print(f"  missing-decade range: {trunc['md'].min():.3f} – {trunc['md'].max():.3f}  (IQR {trunc['md'].quantile(.25):.3f}–{trunc['md'].quantile(.75):.3f})")

    # within-truncated dose-response slope (Check 2, part 1)
    sl, ic, r, p, se = sst.linregress(trunc['md'], trunc['beta'])
    print(f"  within-truncated slope={sl:.3f}  R2={r**2:.3f}  p={p:.3f}")

    # raw truncated-vs-fullband offset (the '+0.13')
    off = trunc['beta'].mean() - full['beta'].mean()
    print(f"  raw offset  beta_trunc({trunc['beta'].mean():.3f}, relief {trunc['relief_m'].median():.0f}m)"
          f" - beta_full({full['beta'].mean():.3f}, relief {full['relief_m'].median():.0f}m) = {off:+.3f}")

    # relief-matched (§9 style): bin by relief, Δβ = trunc-full within bin
    edges = np.quantile(d['relief_m'], [0, 1/3, 2/3, 1.0]); edges[-1] += 1
    print("  relief-matched:")
    for k, name in enumerate(['low', 'mid', 'high']):
        lo, hi = edges[k], edges[k+1]
        tb = trunc[(trunc['relief_m'] >= lo) & (trunc['relief_m'] < hi)]
        fb = full[(full['relief_m'] >= lo) & (full['relief_m'] < hi)]
        if len(tb) and len(fb):
            print(f"    {name:4s} [{lo:.0f}-{hi:.0f}m]: Δ={tb['beta'].mean()-fb['beta'].mean():+.3f}  "
                  f"(n_t={len(tb)} relief {tb['relief_m'].mean():.0f} | n_f={len(fb)} relief {fb['relief_m'].mean():.0f})")
        else:
            print(f"    {name:4s} [{lo:.0f}-{hi:.0f}m]: n_t={len(tb)} n_f={len(fb)} (one empty)")
