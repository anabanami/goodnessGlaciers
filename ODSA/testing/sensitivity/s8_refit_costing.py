"""§8 refit costing on the homogeneous basis, with length-based band classification.

Replaces two shortcuts in the earlier costing: it classifies full-band by segment
LENGTH (>= WINDOW_SIZE) rather than by window count, and it re-takes the region
median after shifting the affected subset rather than scaling the offset linearly.
Both linear predictions are printed alongside so the gap is visible.

Run from v23/; writes results to v23/TESTING_LANDSCAPE_SPLITTING/."""
import numpy as np, pandas as pd, os, sys, glob
from pyproj import Transformer
HERE = os.path.dirname(os.path.abspath(__file__))            # .../v23
ODSA = os.path.dirname(HERE)                                 # .../ODSA — current codebase + results
OUT = os.path.join(HERE, "TESTING_LANDSCAPE_SPLITTING")      # this script's results folder
sys.path.insert(0, ODSA)
from loading import load_datasets, OUTPUT_BASE_PATH
from segmentation import split_into_segments, split_by_landscape
from config import WINDOW_SIZE, FIT_BAND_M, Tee
from bed_character import BED_EDGES, CLASS_ORDER          # class breaks, imported not mirrored
RESULTS = OUTPUT_BASE_PATH
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "s8_refit_costing_log.txt"))


def csv_map(sub, suffix):
    """Basename -> path for both tree layouts: flat <root>/<sub>/ and per-region
    <root>/<region>/<sub>/."""
    hits = (glob.glob(os.path.join(RESULTS, sub, '*' + suffix)) or
            glob.glob(os.path.join(RESULTS, '*', sub, '*' + suffix)))
    return {os.path.basename(p): p for p in hits}

WINDOWS = csv_map("window_csvs", "_window_stats.csv")
SEGMENTS = csv_map("segment_csvs", "_segment_stats.csv")

# Measured truncated-vs-full-band offset, §9 (Pensacola, relief-matched n-weighted mean).
# Not a mirror of a production constant: a measurement. Re-derive if §9 is re-run.
OFFSET = 0.30
COMMON_BAND = 25000.0   # the least aggressive common band that equalises; §8 discussion

# --- Coupling guard, not a mirror check. FIT_BAND_M's upper edge must equal WINDOW_SIZE
# for the length-based full-band test to mean what it says, but it is a fixed literal in
# config and does not follow an env-overridden WINDOW_SIZE. Warn (non-fatal) on mismatch.
if float(FIT_BAND_M[1]) != float(WINDOW_SIZE):
    print(f"WARNING: fit-band max {FIT_BAND_M[1]:.0f} != WINDOW_SIZE {WINDOW_SIZE} — "
          f"the full-band test is measuring a band longer or shorter than one window.")

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

# segment length per (dataset, trajectory, seg_idx+1), reproducing bed_analysis ordering
lengths, loaded = {}, set()
for d in load_datasets():
    name, df = d['name'], d['data']
    loaded.add(name)
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
    for traj_id in valid['trajectory_id'].unique():
        line = valid[valid['trajectory_id'] == traj_id].copy()
        if len(line) < 20: continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        gap_segments = split_into_segments(line.copy(), dist)
        if not gap_segments: continue
        segs = []
        for sd, sdist in gap_segments:
            segs.extend(split_by_landscape(sd, sdist))
        for i, (sdata, sdist, is_t) in enumerate(segs):
            lengths[(name, str(traj_id), i + 1)] = sdist.max() - sdist.min()

cls = lambda b: CLASS_ORDER[int(np.searchsorted(BED_EDGES[1:-1], b))]
rows = []
for fn in sorted(WINDOWS):
    if not fn.endswith('_window_stats.csv'): continue
    dset = fn.replace('_w50km_window_stats.csv', '')
    if dset not in loaded:
        print(f"WARNING: {fn} has no matching loaded dataset {dset!r} — skipped."); continue
    w = pd.read_csv(WINDOWS[fn])
    w['L'] = [lengths.get((dset, str(t), int(s)), np.nan) for t, s in zip(w.trajectory, w.segment)]
    d = w[~w.is_transition.astype(bool)].dropna(subset=['L', 'beta'])
    n_unmatched = w[~w.is_transition.astype(bool)]['L'].isna().sum()

    b, L = d.beta.values, d.L.values
    trunc, full = L < WINDOW_SIZE, L >= WINDOW_SIZE

    # window-count proxy, for the impurity comparison only
    seg = pd.read_csv(SEGMENTS[fn.replace('_window_stats', '_segment_stats')])
    nwin = seg.set_index(['trajectory', 'segment']).n_windows
    proxy_trunc = np.array([nwin.get(k, np.nan) == 1 for k in zip(d.trajectory, d.segment)])

    med = np.median(b)
    debias = np.median(np.where(trunc, b - OFFSET, b))       # §8: correct the truncated windows
    refit = np.median(np.where(full, b + OFFSET, b))         # common band: degrade the full-band ones
    lin_debias = med - OFFSET * trunc.mean()                 # the linear model, for contrast
    lin_refit = med + OFFSET * full.mean()
    rows.append(dict(region=dset, n=len(d), unmatched=n_unmatched,
                     n_full=int(full.sum()), n_trunc=int(trunc.sum()),
                     exposed_len=100 * trunc.mean(), exposed_proxy=100 * proxy_trunc.mean(),
                     mismatch=int((trunc != proxy_trunc).sum()),
                     median=med, cls_now=cls(med),
                     debias=debias, cls_debias=cls(debias), lin_debias=lin_debias,
                     refit=refit, cls_refit=cls(refit), lin_refit=lin_refit,
                     lost_at_25km=int((L < COMMON_BAND).sum())))

r = pd.DataFrame(rows).sort_values('exposed_len', ascending=False)
pd.set_option('display.width', 200, 'display.max_columns', 50)

print(f"\n### Band classification by segment length (full-band = L >= {WINDOW_SIZE/1000:.0f} km), homogeneous windows")
print(r[['region', 'n', 'unmatched', 'n_full', 'n_trunc', 'exposed_len',
         'exposed_proxy', 'mismatch']].to_string(index=False, float_format='%.0f'))
print("\n  exposed_proxy = the window-count proxy (single-window segment). mismatch = windows")
print("  the proxy classifies differently from length. Non-zero mismatch is the proxy's impurity.")

print(f"\n### §8 de-bias: subtract {OFFSET} from truncated windows, re-take the median")
print(r[['region', 'exposed_len', 'median', 'cls_now', 'debias', 'cls_debias',
         'lin_debias']].to_string(index=False, float_format='%.3f'))

print(f"\n### Common-band refit: add {OFFSET} to full-band windows, re-take the median")
print(r[['region', 'n_full', 'median', 'cls_now', 'refit', 'cls_refit', 'lin_refit',
         'lost_at_25km']].to_string(index=False, float_format='%.3f'))

print("\n### Class changes")
for _, x in r.iterrows():
    # bracket access throughout: 'median' collides with the Series method
    for label, new, c in [('de-bias', x['debias'], x['cls_debias']), ('refit', x['refit'], x['cls_refit'])]:
        if c != x['cls_now']:
            print(f"  {x['region']:42s} {label:8s} {x['median']:.3f} ({x['cls_now']}) -> {new:.3f} ({c})")
    for label, v in [('de-bias', x['debias']), ('refit', x['refit'])]:
        edge = BED_EDGES[1:-1][np.argmin(np.abs(BED_EDGES[1:-1] - v))]
        print(f"  {x['region']:42s} {label:8s} lands {abs(v - edge):.3f} from the {edge:.1f} break")

print("\n### Linear-model error (why the linear estimate was dropped)")
r['err_debias'] = (r.debias - r.lin_debias).abs()
r['err_refit'] = (r.refit - r.lin_refit).abs()
print(r[['region', 'err_debias', 'err_refit']].to_string(index=False, float_format='%.3f'))
print(f"  worst de-bias error {r.err_debias.max():.3f}, worst refit error {r.err_refit.max():.3f}"
      f" — against 0.5-wide class bands.")

r.to_csv(os.path.join(OUT, "s8_refit_costing.csv"), index=False)
print(f"\nwrote {os.path.join(OUT, 's8_refit_costing.csv')}")
sys.stdout.flush()   # Tee buffers; without this a later crash truncates the log
