"""Is the bed self-affine over the fit band, per region?

The deviogram is v(L) = RMS[z(x+L) - z(x)]. A self-affine profile gives v ~ L^H, so log v
against log L is a straight line and beta = 2H + 1 describes the bed. Curvature means no
single exponent describes it, and then neither estimator is measuring the same thing:
production fits the short-wavelength end of the band and the deviogram the long-lag end.

Two departures are measured per window, both zero under self-affinity:

  SAG   the quadratic coefficient of log v against log L, with the lag axis scaled to
        [-1, 1], so it is the vertical departure from the chord at mid-lag in decades.
  dH    H fitted over the upper half of the lag bins minus H over the lower half.

Both are biased by the estimator itself: the structure function saturates as the lag
approaches the window length, which curves the deviogram on a bed that is self-affine by
construction. The null removes that. For each window the real elevations are replaced by
fBm of the window's own fitted H, sampled at the same track positions and detrended the
same way, so the null carries the window's length, point count and sampling gaps exactly
and what survives is the bed.

Control: H fitted here must reproduce H_devio in v23/deviogram/deviogram_windows.csv,
which is the same estimator run by deviogram_validation.py.

Writes v23/self_affinity/. Runs from v23/ or from the ODSA root.

    python self_affinity_test.py                       # individual_region_TEST, else Ockenden-regions
    python self_affinity_test.py Ockenden-regions      # explicit root
"""
import numpy as np, pandas as pd, glob, os, re, sys
from scipy import signal
from pyproj import Transformer
HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(HERE)
OUT = os.path.join(HERE, "self_affinity")
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import split_into_segments, split_by_landscape
from config import WINDOW_SIZE, STEP_SIZE, MIN_SEGMENT_POINTS, Tee
DEFAULT_ROOTS = ('individual_region_TEST', 'Ockenden-regions')

# The estimator, identical to deviogram_validation.py and deviogram_calibration.py. The
# three scripts must run the same estimator, so the definition is read back from the sibling
# and a mismatch is fatal, as it is between the other two.
LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS = 250.0, 20, 30, 8
MIN_SPLIT_BINS = 5       # per half, for the dH split
N_NULL = 50              # fBm draws per window
N_MATCH = 15             # draws per round while matching the null's H to the window's
MATCH_ROUNDS = 3
MATCH_TOL = 0.01
H_SYN_RANGE = (0.0, 1.5)  # the calibration grid's range of self-affine H
SLICE_MULT = 32          # synthesis domain / window length
GRID_N = 16384
SEED = 20260905
TOL = 1e-9


def _estimator_guard():
    src = open(os.path.join(HERE, 'deviogram_calibration.py')).read()
    m = re.search(r"LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS = ([\d.]+), (\d+), (\d+), (\d+)", src)
    if not m:
        raise SystemExit("deviogram_calibration.py no longer states the estimator on one "
                         "line, so the identity check here cannot run. Fix the check.")
    theirs = (float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    if theirs != (LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS):
        raise SystemExit(f"Estimator mismatch: {theirs} in deviogram_calibration.py against "
                         f"{(LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS)} here. The null and the "
                         "deviogram family would not be measuring the same thing.")


_estimator_guard()


def window_stats_files(root):
    return sorted(glob.glob(os.path.join(root, 'window_csvs', '*_window_stats.csv')) +
                  glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv')))


def resolve_root():
    """First positional argument, else the first default tree holding window CSVs."""
    arg = next((a for a in sys.argv[1:] if not a.startswith('-')), None)
    tried = []
    for name in ([arg] if arg else DEFAULT_ROOTS):
        for p in (name, os.path.join(ODSA, name)):
            tried.append(p)
            if window_stats_files(p):
                return p
    raise SystemExit("No <root>/[<region>/]window_csvs/*_window_stats.csv under any of:\n  "
                     + "\n  ".join(tried))


RESULTS = resolve_root()
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "self_affinity_log.txt"))


def deviogram(d, z, edges):
    """RMS deviation per geometric lag bin. Identical to deviogram_validation.deviogram."""
    n = len(d)
    S1 = np.concatenate([[0.0], np.cumsum(z)])
    S2 = np.concatenate([[0.0], np.cumsum(z * z)])
    D1 = np.concatenate([[0.0], np.cumsum(d)])
    L, v = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        jlo = np.maximum(np.searchsorted(d, d + lo, side='left'), np.arange(n) + 1)
        jhi = np.maximum(np.searchsorted(d, d + hi, side='left'), jlo)
        cnt = jhi - jlo
        tot = int(cnt.sum())
        if tot < MIN_PAIRS:
            continue
        sq = np.sum((S2[jhi] - S2[jlo]) - 2 * z * (S1[jhi] - S1[jlo]) + cnt * z * z)
        sep = np.sum((D1[jhi] - D1[jlo]) - cnt * d)
        L.append(sep / tot); v.append(np.sqrt(sq / tot))
    return np.array(L), np.array(v)


def departures(d, z, edges):
    """(H, r2, sag, dH, n_bins). sag and dH are zero for a straight log-log deviogram."""
    L, v = deviogram(d, z, edges)
    ok = (v > 0) & (L > 0)
    L, v = L[ok], v[ok]
    if len(L) < MIN_BINS:
        return np.nan, np.nan, np.nan, np.nan, len(L)
    x, y = np.log10(L), np.log10(v)
    sl, ic = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (ic + sl * x))**2) / np.sum((y - y.mean())**2)
    # Scale the lag axis to [-1, 1] so the quadratic term is the mid-lag sag in decades,
    # comparable across windows of different length.
    xs = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    sag = np.polyfit(xs, y, 2)[0]
    half = len(L) // 2
    dH = (np.polyfit(x[len(L) - half:], y[len(L) - half:], 1)[0]
          - np.polyfit(x[:half], y[:half], 1)[0]) if half >= MIN_SPLIT_BINS else np.nan
    return sl, r2, sag, dH, len(L)


def fbm(n, H, rng, domain):
    """fBm by spectral synthesis: PSD ~ f^-(2H+1), so beta = 2H+1 by construction."""
    f = np.fft.rfftfreq(n, d=domain / n); f[0] = f[1]
    ph = rng.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    z = np.fft.irfft(f ** (-(2 * H + 1) / 2.0) * np.exp(1j * ph), n)
    s = np.std(z)
    return z / s * 100.0 if s > 0 else z


def _draw(d, W, H_syn, edges, rng, n_draw):
    """n_draw self-affine windows at the track positions of d. Returns (H, sag, dH) arrays."""
    domain = W * SLICE_MULT
    xg = np.linspace(0.0, domain, GRID_N)
    out, per_real = [], max(1, n_draw // 5)
    while len(out) < n_draw:
        zg = fbm(GRID_N, H_syn, rng, domain)
        for _ in range(per_real):
            if len(out) >= n_draw:
                break
            s0 = rng.uniform(0.05 * domain, 0.95 * domain - W)
            sel = (xg >= s0) & (xg <= s0 + W)
            if sel.sum() < MIN_SEGMENT_POINTS:
                continue
            z = signal.detrend(np.interp(d, xg[sel] - s0, zg[sel]))
            H, _, s, h, _ = departures(d, z, edges)
            out.append((H, s, h))
    a = np.array(out, float)
    return a[:, 0], a[:, 1], a[:, 2]


def null_departures(d, W, H, edges, rng, n_draw=N_NULL):
    """The same two departures on a self-affine bed at the window's own track positions.

    The deviogram reads H low, so synthesising at the measured H puts the null at the wrong
    roughness and the curvature bias goes with it. The synthesis H is therefore matched so
    that the null's own measured H lands on the window's, which is the observable the two
    share. Returns (sag, dH, H_syn, H_null_mean)."""
    H_syn = float(np.clip(H, *H_SYN_RANGE))
    for _ in range(MATCH_ROUNDS):
        Hn, _, _ = _draw(d, W, H_syn, edges, rng, N_MATCH)
        err = H - np.nanmean(Hn)
        H_syn = float(np.clip(H_syn + err, *H_SYN_RANGE))
        if abs(err) < MATCH_TOL:
            break
    Hn, sag, dH = _draw(d, W, H_syn, edges, rng, n_draw)
    return sag, dH, H_syn, float(np.nanmean(Hn))


def segment_windows(dist, elev):
    """Reproduce analyse_sliding_windows' window loop and detrend. No taper or periodogram:
    the deviogram is a space-domain statistic and takes the detrended residual only."""
    seg_len = dist.max() - dist.min()
    W = seg_len if seg_len < WINDOW_SIZE else WINDOW_SIZE
    step = W if seg_len < WINDOW_SIZE else STEP_SIZE
    wins, cur, widx = [], dist.min(), 0
    while cur + W <= dist.max() + 1e-6:
        m = (dist >= cur) & (dist <= cur + W)
        wd = dist[m]
        if len(wd) > MIN_SEGMENT_POINTS:
            wins.append(dict(window_id=widx, d=wd - wd.min(), z=signal.detrend(elev[m])))
        cur += step; widx += 1
    return wins, W


print(f"run tree: {RESULTS}")
print(f"lag bins: {N_LAGS} geometric from {LAG_MIN:.0f} m to W/3, per window")
print(f"null: {N_NULL} fBm draws per window at the window's own H, same track positions\n")

rng = np.random.default_rng(SEED)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
rows = []
for dset in load_datasets():
    name, df = dset['name'], dset['data']
    n0 = len(rows)
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
    for traj in valid['trajectory_id'].unique():
        line = valid[valid['trajectory_id'] == traj].copy()
        if len(line) < 20: continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        gaps = split_into_segments(line.copy(), dist)
        if not gaps: continue
        segs = []
        for sd, sdist in gaps:
            segs.extend(split_by_landscape(sd, sdist))
        for i, (sdata, sdist, is_t) in enumerate(segs):
            elev = sdata['bedrock_altitude (m)'].to_numpy(float)
            wins, W = segment_windows(sdist, elev)
            lag_max = W / 3.0
            if lag_max <= LAG_MIN * 2:
                continue
            edges = np.geomspace(LAG_MIN, lag_max, N_LAGS + 1)
            for w in wins:
                H, r2, sag, dH, nb = departures(w['d'], w['z'], edges)
                rec = dict(dataset=name, trajectory=str(traj), segment=i + 1,
                           window_id=w['window_id'], is_transition=bool(is_t), W_m=W,
                           n_pts=len(w['d']), H_devio=H, devio_r2=r2, devio_bins=nb,
                           sag=sag, dH=dH)
                if np.isfinite(H):
                    ns, nd, hs, hn = null_departures(w['d'], W, H, edges, rng)
                    rec.update(sag_null=np.nanmean(ns), sag_null_sd=np.nanstd(ns, ddof=1),
                               dH_null=np.nanmean(nd), dH_null_sd=np.nanstd(nd, ddof=1),
                               H_syn=hs, H_null=hn)
                rows.append(rec)
    print(f"  {name}: {len(rows) - n0} windows")

r = pd.DataFrame(rows)
if r.empty:
    raise SystemExit("No windows rebuilt: load_datasets() returned nothing. This script "
                     "resynthesises its own vectors from the raw data, so the production "
                     "region entries in loading.py have to be uncommented, and reading the "
                     f"CSVs under {RESULTS} is not enough.")
r['sag_z'] = (r.sag - r.sag_null) / r.sag_null_sd
r['dH_z'] = (r.dH - r.dH_null) / r.dH_null_sd
print(f"reconstructed {len(r)} windows across {r.dataset.nunique()} regions")

# --- Control. H must reproduce the deviogram_validation run, which is the same estimator.
ref_path = os.path.join(HERE, 'deviogram', 'deviogram_windows.csv')
if os.path.exists(ref_path):
    ref = pd.read_csv(ref_path)[['dataset', 'trajectory', 'segment', 'window_id', 'H_devio']]
    ref['trajectory'] = ref.trajectory.astype(str)
    m = r.merge(ref.rename(columns={'H_devio': 'H_ref'}),
                on=['dataset', 'trajectory', 'segment', 'window_id'], how='inner')
    both = m[m.H_devio.notna() & m.H_ref.notna()]
    dmax = (both.H_devio - both.H_ref).abs().max()
    print(f"\n### Control: {len(m)} of {len(ref)} windows matched to deviogram_windows.csv, "
          f"max |dH| = {dmax:.2e}")
    if dmax > TOL:
        print("WARNING: the estimator here does not reproduce deviogram_validation.py. "
              "Nothing below is comparable to that run.")
else:
    print(f"\nNOTE: {ref_path} absent, so H is not controlled against deviogram_validation.py.")

# --- Homogeneous windows only, the set every reported result is measured over.
h = r[~r.is_transition & r.H_devio.notna()].copy()
h['region'] = h.dataset.str.replace('ASB_ICECAP_2010_|POLARGAP_2015_|Rec_Catch_', '', regex=True)
h['truncated'] = h.W_m < WINDOW_SIZE
h.to_csv(os.path.join(OUT, 'self_affinity_windows.csv'), index=False)

print(f"\n### The estimator's own curvature on a self-affine bed")
print(f"  sag null median {h.sag_null.median():+.4f}, dH null median {h.dH_null.median():+.4f}")
print("  Both are non-zero, so raw curvature cannot be read as non-self-affinity.")
print(f"  Null H matched to the window's: median |H_null - H_devio| = "
      f"{(h.H_null - h.H_devio).abs().median():.4f}")

print("\n### Sign convention")
print("  sag_z < 0: the spectrum steepens toward short wavelengths, so the bed is smoother")
print("             there than a single exponent allows, and H read over short lags exceeds")
print("             H read over long lags.")
print("  sag_z > 0: the reverse. Zero is self-affine.")


def boot_mean(g, col, n_rep=1000, seed=SEED):
    """Mean z with a segment-level bootstrap interval. Windows overlap by 50% and segments
    do not, so segments are the resampling unit."""
    rng_b = np.random.default_rng(seed)
    keys = g[['trajectory', 'segment']].apply(tuple, axis=1)
    groups = [v[col].to_numpy(float) for _, v in g.groupby(keys.values)]
    groups = [a[np.isfinite(a)] for a in groups]
    groups = [a for a in groups if len(a)]
    if len(groups) < 3:
        return np.nan, np.nan, len(groups)
    means = [np.concatenate([groups[i] for i in rng_b.integers(0, len(groups), len(groups))]).mean()
             for _ in range(n_rep)]
    return float(np.percentile(means, 5)), float(np.percentile(means, 95)), len(groups)


print("\n### Per region, homogeneous windows")
rowsr = []
for reg, g in h.groupby('region'):
    lo, hi, nseg = boot_mean(g, 'sag_z')
    rowsr.append(dict(region=reg, n=len(g), n_seg=nseg, H=g.H_devio.median(),
                      r2=g.devio_r2.median(), sag=g.sag.median(),
                      sag_null=g.sag_null.median(), sag_z=g.sag_z.mean(),
                      lo=lo, hi=hi, dH_z=g.dH_z.mean(),
                      frac_z2=float((g.sag_z.abs() > 2).mean()), trunc=g.truncated.mean()))
g = pd.DataFrame(rowsr).set_index('region')
print(g.round(3).to_string())
print("  lo and hi are a 90% segment-bootstrap interval on mean sag_z. An interval that")
print("  excludes zero is a region the single exponent does not describe.")
print(f"  frac_z2 is nominally 0.05 under self-affinity; pooled it reads "
      f"{float((h.sag_z.abs() > 2).mean()):.3f} over {len(h)} windows.")

print("\n### Truncated against full windows")
print(h.groupby('truncated').agg(n=('sag', 'size'), sag=('sag', 'median'),
                                 sag_z=('sag_z', 'median'), dH=('dH', 'median'),
                                 dH_z=('dH_z', 'median'),
                                 bins=('devio_bins', 'median')).round(3).to_string())

# --- Does the departure track the estimator disagreement? Reading 2 predicts that it does.
hom_path = os.path.join(HERE, 'deviogram', 'deviogram_homogeneous.csv')
if os.path.exists(hom_path):
    d2 = pd.read_csv(hom_path)
    d2['trajectory'] = d2.trajectory.astype(str)
    j = h.merge(d2[['dataset', 'trajectory', 'segment', 'window_id', 'beta_csv', 'beta_devio']],
                on=['dataset', 'trajectory', 'segment', 'window_id'], how='inner')
    j['resid'] = j.beta_csv - j.beta_devio
    per = j.groupby('region').agg(n=('resid', 'size'), beta_prod=('beta_csv', 'median'),
                                  beta_devio=('beta_devio', 'median'),
                                  resid=('resid', 'median'), sag_z=('sag_z', 'median'),
                                  dH_z=('dH_z', 'median')).sort_values('beta_prod')
    print(f"\n### Estimator disagreement against departure from self-affinity ({len(j)} windows)")
    print(per.round(3).to_string())
    ok = per.resid.notna() & per.sag_z.notna()
    print(f"\n  region-level rank correlation, resid against sag_z: "
          f"{per.loc[ok, ['resid', 'sag_z']].corr(method='spearman').iloc[0, 1]:+.3f}")
    print(f"  region-level rank correlation, resid against dH_z:  "
          f"{per.loc[ok, ['resid', 'dH_z']].corr(method='spearman').iloc[0, 1]:+.3f}")
    print(f"  window-level, resid against sag_z: "
          f"{j[['resid', 'sag_z']].corr(method='spearman').iloc[0, 1]:+.3f} "
          f"(n = {int(j[['resid', 'sag_z']].dropna().shape[0])})")
    per.to_csv(os.path.join(OUT, 'self_affinity_region.csv'))
else:
    g.to_csv(os.path.join(OUT, 'self_affinity_region.csv'))
    print(f"\nNOTE: {hom_path} absent, so the departure is not joined to the estimator residual.")

print(f"\nWritten to {OUT}")
