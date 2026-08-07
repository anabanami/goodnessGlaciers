"""Open question #4: validate the hann taper against a space-domain estimator.

The deviogram (Shepard 2001; Jordan 2017) is v(L) = RMS[z(x+L) - z(x)] ~ L^H, fitted
as a log-log slope. It is a space-domain statistic: no FFT, no periodogram, no taper,
so it does not share the spectral leakage failure mode that the rect/tukey/hann ladder
cannot rule on. beta = 2H + 1, the same reparameterisation production uses.

Two confounds sit between H_devio and production beta, and both are removed here rather
than argued around:

  BAND.  The deviogram is fitted over lags 250 m to WINDOW_SIZE/3, about 1.8 decades.
         Production fits 250 m to 50 km, about 2.3 decades. Production beta is therefore
         also refitted over the deviogram's band so the two are compared like for like.
  MASK.  Production beta is peak-masked; the deviogram has no analogue. Beta is therefore
         also computed unmasked, at both bands.

Production windows are reconstructed here rather than read, because refitting needs the
periodogram. The reconstruction is checked against the production CSV before any
comparison is reported: if beta_full_masked does not reproduce the CSV, nothing else in
the output is trustworthy and the script says so.

Run from v23/; writes results to v23/deviogram/."""
import numpy as np, pandas as pd, os, re, sys
from scipy import signal
from pyproj import Transformer
HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(HERE)
OUT = os.path.join(HERE, "deviogram")
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import split_into_segments, split_by_landscape
from config import (WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, WINDOW_MASK, FIT_BAND_M,
                    peak_masking_height_threshold, bin_buffer, Tee)
RESULTS = os.path.join(ODSA, "Ockenden-regions")
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "deviogram_validation_log.txt"))

LAG_MIN = 250.0                    # short end, matches the production fit band floor
LAG_MAX = WINDOW_SIZE / 3.0        # long end; beyond ~W/3 the pair count collapses
N_LAGS = 20                        # geometric lag bins
MIN_PAIRS = 30                     # per-bin pair floor (matches beta_sigma_calibration)
MIN_BINS = 8                       # bins needed to fit a slope
TOL = 1e-9                         # reconstruction-vs-CSV tolerance

# --- Drift guard. The reconstruction below reproduces analyse_sliding_windows
# (bed_analysis_23.py). The band comes from config; grid density and dx floor are still
# inline literals there, so read those from source and warn (non-fatal) on drift.
_SRC = open(os.path.join(ODSA, "bed_analysis_23.py")).read()
def _prodval(pat, cast=float):
    m = re.search(pat, _SRC); return cast(m.group(1)) if m else None
N_BINS, DX_FLOOR = 500, 15.0
BAND_MIN, BAND_MAX = FIT_BAND_M
for _label, _mine, _prod in [("grid bins", N_BINS,   _prodval(r"geomspace\([^)]*num=(\d+)", int)),
                             ("dx floor",  DX_FLOOR, _prodval(r"max\(dx_median,\s*([\d.]+)\)"))]:
    if _prod is None: print(f"NOTE: could not cross-check {_label} against bed_analysis_23.py.")
    elif _mine != _prod: print(f"WARNING: {_label} = {_mine} here but {_prod} in production — STALE.")
if WINDOW_TYPE != 'hann':
    print(f"WARNING: WINDOW_TYPE is {WINDOW_TYPE!r}, not 'hann'. This test is about the hann taper.")
if not WINDOW_MASK:
    print("WARNING: WINDOW_MASK is False, so the production CSV is unmasked and the "
          "reconstruction check below compares against an unmasked run.")


def deviogram(d, z, edges):
    """RMS deviation per geometric lag bin. Exact, via prefix sums over the sorted track.

    For lag bin [lo, hi) and each point i, the partners j satisfy d[j]-d[i] in [lo, hi),
    a contiguous index range because d is sorted. Sum (z[j]-z[i])^2 over that range comes
    from the prefix sums of z and z^2, so the cost is O(n log n) per bin rather than O(n^2).
    """
    n = len(d)
    S1 = np.concatenate([[0.0], np.cumsum(z)])
    S2 = np.concatenate([[0.0], np.cumsum(z * z)])
    D1 = np.concatenate([[0.0], np.cumsum(d)])
    L, v, npair = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        jlo = np.searchsorted(d, d + lo, side='left')
        jhi = np.searchsorted(d, d + hi, side='left')
        jlo = np.maximum(jlo, np.arange(n) + 1)          # ordered pairs only, j > i
        jhi = np.maximum(jhi, jlo)
        cnt = jhi - jlo
        tot = int(cnt.sum())
        if tot < MIN_PAIRS:
            continue
        sz = S1[jhi] - S1[jlo]
        sz2 = S2[jhi] - S2[jlo]
        sd = D1[jhi] - D1[jlo]
        sq = np.sum(sz2 - 2 * z * sz + cnt * z * z)      # sum over pairs of (z_j - z_i)^2
        sep = np.sum(sd - cnt * d)                       # sum over pairs of (d_j - d_i)
        L.append(sep / tot); v.append(np.sqrt(sq / tot)); npair.append(tot)
    return np.array(L), np.array(v), np.array(npair)


def fit_H(d, z, edges):
    """Deviogram slope. Returns (H, r2, n_bins, lag range)."""
    L, v, npair = deviogram(d, z, edges)
    ok = (v > 0) & (L > 0)
    L, v = L[ok], v[ok]
    if len(L) < MIN_BINS:
        return np.nan, np.nan, len(L), np.nan, np.nan
    x, y = np.log10(L), np.log10(v)
    sl, ic = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (ic + sl * x))**2) / np.sum((y - y.mean())**2)
    return sl, r2, len(L), L.min(), L.max()


def segment_windows(dist, elev):
    """Reproduce analyse_sliding_windows: same loop, same detrend, same taper, same mask.

    Production falls back to window_size = segment length for sub-50 km segments
    (bed_analysis_23.py:323-326), so a truncated segment yields one window spanning it.
    That fallback is reproduced here, and it is why window length is carried per window:
    the deviogram's lag ceiling and its bias both scale with it.

    Returns (windows, band_full, clean, log_freqs, wavelengths, W).
    """
    seg_len = dist.max() - dist.min()
    W = seg_len if seg_len < WINDOW_SIZE else WINDOW_SIZE
    step = W if seg_len < WINDOW_SIZE else STEP_SIZE

    dx_median = np.median(np.diff(dist)) if len(dist) > 1 else 100
    if dx_median == 0: dx_median = DX_FLOOR
    max_freq = 1 / (2 * max(dx_median, DX_FLOOR))
    min_freq = 1 / W
    freqs = np.geomspace(min_freq, max_freq, num=N_BINS)
    ang = freqs * 2 * np.pi
    wl = 1 / freqs
    band_full = (wl >= BAND_MIN) & (wl <= BAND_MAX)
    log_freqs = np.log10(freqs)

    wins, pgrams = [], []
    cur, widx = dist.min(), 0
    while cur + W <= dist.max() + 1e-6:
        m = (dist >= cur) & (dist <= cur + W)
        wd, we = dist[m], elev[m]
        if len(wd) > 50:
            det = signal.detrend(we)
            tap = det * signal.windows.hann(len(det)) if WINDOW_TYPE == 'hann' else \
                  det * signal.windows.tukey(len(det), alpha=0.5) if WINDOW_TYPE == 'tukey' else det
            pg = signal.lombscargle(wd, tap, ang, normalize=False)
            pgrams.append(pg)
            wins.append(dict(window_id=widx, d=wd - wd.min(), z=det, pgram=pg))
        cur += step; widx += 1

    clean = band_full.copy()
    if pgrams and WINDOW_MASK:
        avg = np.mean(np.array(pgrams), axis=0)
        if np.sum(band_full) >= 2 and np.all(avg[band_full] > 0):
            s0, i0 = np.polyfit(log_freqs[band_full], np.log10(avg[band_full]), 1)
            resid = avg / 10 ** (i0 + s0 * log_freqs)
            peaks, _ = signal.find_peaks(resid, height=peak_masking_height_threshold)
            for p in peaks:
                clean[max(0, p - bin_buffer):min(len(clean), p + bin_buffer + 1)] = False
    return wins, band_full, clean, log_freqs, wl, W


def beta_over(pgram, log_freqs, fit_mask):
    """Production's window fit: OLS of log10(PSD) on log10(f), beta = -slope."""
    n = int(np.sum(fit_mask))
    if n < 2 or not np.all(pgram > 0): return np.nan
    return -np.polyfit(log_freqs[fit_mask], np.log10(pgram[fit_mask]), 1)[0]


# --- Calibration. The deviogram slope is biased low, increasingly so with H, because the
# structure function saturates as the lag approaches the profile length. The bias is
# deterministic, independent of sampling density, and depends on H and window length only,
# so it is measured once by deviogram_calibration.py and forward-mapped here: production H
# is pushed through the estimator's own response, and the prediction is what gets compared
# against the measured H_devio. Without this the test cannot separate "hann over-corrects"
# from "the deviogram under-reads at high H".
CAL = os.path.join(OUT, "deviogram_calibration.csv")
if not os.path.exists(CAL):
    print(f"ERROR: calibration table not found at {CAL}.\n"
          f"       Run deviogram_calibration.py first; the comparison is meaningless without it.")
    sys.stdout.flush(); sys.exit(1)
cal = pd.read_csv(CAL)

# --- Estimator-identity guard. The calibration measures the response of one particular
# deviogram. If these four differ from the ones the table was built with, the forward map
# applies the wrong response and every dH below is wrong with nothing in the output to show
# it. Fatal, not a warning: there is no partially-valid result here.
_want = dict(lag_min=LAG_MIN, n_lags=N_LAGS, min_pairs=MIN_PAIRS, min_bins=MIN_BINS)
_missing = [k for k in _want if k not in cal.columns]
if _missing:
    print(f"ERROR: calibration table predates the estimator-identity guard (no {_missing}).\n"
          f"       Re-run deviogram_calibration.py.")
    sys.stdout.flush(); sys.exit(1)
for _k, _v in _want.items():
    _got = cal[_k].unique()
    if len(_got) != 1 or not np.isclose(float(_got[0]), float(_v)):
        print(f"ERROR: {_k} = {_v} here but {_got} in the calibration table. The table "
              f"describes a different estimator.\n       Re-run deviogram_calibration.py.")
        sys.stdout.flush(); sys.exit(1)

_cw = np.sort(cal.W_km.unique()); _ch = np.sort(cal.H_true.unique())
_cg = cal.pivot_table(index='W_km', columns='H_true', values='H_devio_mean').reindex(
    index=_cw, columns=_ch).to_numpy()

def expected_H(H_true, W_m):
    """E[H_devio] if the bed really had H_true, at this window length. Bilinear on the grid."""
    if not np.isfinite(H_true): return np.nan
    wi = np.clip(np.interp(W_m / 1000.0, _cw, np.arange(len(_cw))), 0, len(_cw) - 1)
    hi = np.clip(np.interp(H_true, _ch, np.arange(len(_ch))), 0, len(_ch) - 1)
    w0, h0 = int(np.floor(wi)), int(np.floor(hi))
    w1, h1 = min(w0 + 1, len(_cw) - 1), min(h0 + 1, len(_ch) - 1)
    fw, fh = wi - w0, hi - h0
    return ((1 - fw) * (1 - fh) * _cg[w0, h0] + fw * (1 - fh) * _cg[w1, h0]
            + (1 - fw) * fh * _cg[w0, h1] + fw * fh * _cg[w1, h1])

print(f"\ncalibration: {len(cal)} grid points, H_true {_ch.min():.2f} to {_ch.max():.2f}, "
      f"W {_cw.min():.1f} to {_cw.max():.1f} km")
print(f"lag bins: {N_LAGS} geometric from {LAG_MIN:.0f} m to W/3, per window")
print(f"production band: {BAND_MIN} m to min(W, {BAND_MAX} m)")
print(f"matched band:    {BAND_MIN} m to W/3, applied to the production recipe\n")

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
rows = []
for dset in load_datasets():
    name, df = dset['name'], dset['data']
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
            wins, band_full, clean, log_freqs, wl, W = segment_windows(sdist, elev)
            lag_max = W / 3.0
            edges = np.geomspace(LAG_MIN, lag_max, N_LAGS + 1) if lag_max > LAG_MIN * 2 else None
            band_match = band_full & (wl <= lag_max)
            clean_match = clean & (wl <= lag_max)
            for w in wins:
                H, r2, nb, lmin, lmax = fit_H(w['d'], w['z'], edges) if edges is not None \
                    else (np.nan, np.nan, 0, np.nan, np.nan)
                bfm = beta_over(w['pgram'], log_freqs, clean)
                rows.append(dict(
                    dataset=name, trajectory=str(traj), segment=i + 1, window_id=w['window_id'],
                    is_transition=bool(is_t), W_m=W,
                    H_devio=H, beta_devio=2 * H + 1, devio_r2=r2, devio_bins=nb,
                    devio_lag_min=lmin, devio_lag_max=lmax,
                    beta_full_masked=bfm,
                    beta_full_unmasked=beta_over(w['pgram'], log_freqs, band_full),
                    beta_match_masked=beta_over(w['pgram'], log_freqs, clean_match),
                    beta_match_unmasked=beta_over(w['pgram'], log_freqs, band_match)))

r = pd.DataFrame(rows)
print(f"reconstructed {len(r)} windows across {r.dataset.nunique()} regions")

# --- Reconstruction check. beta_full_masked must reproduce the production CSV.
prod = []
for fn in sorted(os.listdir(os.path.join(RESULTS, 'window_csvs'))):
    if not fn.endswith('_window_stats.csv'): continue
    w = pd.read_csv(os.path.join(RESULTS, 'window_csvs', fn))
    w['dataset'] = fn.replace('_w50km_window_stats.csv', '')
    prod.append(w[['dataset', 'trajectory', 'segment', 'window_id', 'beta', 'is_transition']])
prod = pd.concat(prod); prod['trajectory'] = prod.trajectory.astype(str)
m = r.merge(prod.rename(columns={'beta': 'beta_csv', 'is_transition': 'tz_csv'}),
            on=['dataset', 'trajectory', 'segment', 'window_id'], how='inner')
d = (m.beta_full_masked - m.beta_csv).abs()
bad = int((d > TOL).sum())
print(f"\n### Reconstruction check: {len(m)} of {len(prod)} production windows matched")
print(f"  max |beta_full_masked - beta_csv| = {np.nanmax(d):.3e}, mismatches over {TOL:g}: {bad}")
if len(m) != len(prod) or bad:
    print("  RECONSTRUCTION FAILED. The comparisons below are not trustworthy. Stopping.")
    r.to_csv(os.path.join(OUT, "deviogram_windows.csv"), index=False); sys.stdout.flush(); sys.exit(1)
print("  reconstruction reproduces production exactly.")

h = m[~m.tz_csv.astype(bool)].dropna(subset=['beta_devio']).copy()
h['region'] = h.dataset.str.replace('ASB_ICECAP_2010_|POLARGAP_2015_|Rec_Catch_', '', regex=True)
print(f"\n  homogeneous windows with a deviogram fit: {len(h)} of {(~m.tz_csv.astype(bool)).sum()}")
print(f"  deviogram fit quality: median R2 {h.devio_r2.median():.3f}, "
      f"median bins used {h.devio_bins.median():.0f} of {N_LAGS}")

print("\n### Per-window agreement, calibrated")
print("  H_pred = what the deviogram would return if production H were the truth, from the")
print("  calibration table at this window's length. dH = H_devio - H_pred, so 0 means the")
print("  space-domain estimator agrees with production once its own bias is accounted for.")
print(f"\n{'comparator':22s} {'med H_prod':>10s} {'med H_pred':>10s} {'med H_devio':>11s} "
      f"{'med dH':>8s} {'r':>7s} {'slope':>7s}")
for col in ['beta_match_masked', 'beta_full_masked', 'beta_match_unmasked', 'beta_full_unmasked']:
    Hp = (h[col] - 1) / 2
    Hx = np.array([expected_H(v, w) for v, w in zip(Hp, h.W_m)])
    ok = np.isfinite(Hx) & np.isfinite(h.H_devio.to_numpy())
    rr = np.corrcoef(Hx[ok], h.H_devio.to_numpy()[ok])[0, 1]
    sl = np.polyfit(Hx[ok], h.H_devio.to_numpy()[ok], 1)[0]
    print(f"{col:22s} {Hp.median():10.3f} {np.nanmedian(Hx):10.3f} {h.H_devio.median():11.3f} "
          f"{np.nanmedian(h.H_devio.to_numpy() - Hx):8.3f} {rr:7.3f} {sl:7.3f}")
    h['dH_' + col] = h.H_devio.to_numpy() - Hx
print("\n  band_match_masked is the like-for-like test: same band, same mask, bias-corrected.")
print("  A median dH near 0 validates the taper. A large negative dH means production beta is")
print("  steeper than the bed supports, which is the over-correction signature.")
print("\n  Two limits on what dH can settle, both from the calibration table itself:")
print("  1. The forward map assumes the bed is self-affine over the lag range, because the")
print("     calibration is measured on fBm. Where the bed is not self-affine, a non-zero dH")
print("     is not attributable to the taper. devio_r2 is the per-window check on that.")
_hi = cal[cal.W_km == _cw[-1]].sort_values('H_true')
_step = np.diff(_hi.H_devio_mean.to_numpy()) / np.diff(_hi.H_true.to_numpy())
print(f"  2. Power falls off at high H. dH_devio/dH_true drops to {_step[-1]:.2f} at the top of")
print(f"     the grid, so a large change in true H moves H_devio little and the test cannot")
print(f"     discriminate well above H ~ {_hi.H_true.to_numpy()[np.argmax(_step < 0.5)]:.1f}"
      f" (beta ~ {2 * _hi.H_true.to_numpy()[np.argmax(_step < 0.5)] + 1:.1f}). This caps its")
print("     reach on the beta > 3 tail, independently of the lag-range limit.")

print("\n### Pooled beta distributions (homogeneous windows)")
print(f"{'estimator':22s} {'median':>7s} {'IQR':>15s} {'frac>3':>7s} {'frac<1':>7s}")
for col in ['beta_devio', 'beta_full_masked', 'beta_match_masked', 'beta_match_unmasked']:
    s = h[col].dropna()
    print(f"{col:22s} {s.median():7.3f} {s.quantile(.25):7.3f}-{s.quantile(.75):<7.3f} "
          f"{(s > 3).mean():7.1%} {(s < 1).mean():7.1%}")

print("\n### Pooled histogram, 0.25-wide bins (fixed bins, not a KDE: the mode is read off")
print("###   the counts, so no bandwidth choice enters. cf. the KDE bug in §C of CONSTANTS_AUDIT)")
hb = np.arange(0.5, 4.5, 0.25)
cols = ['beta_devio', 'beta_full_masked', 'beta_match_masked']
counts = {c: np.histogram(h[c].dropna(), bins=hb)[0] for c in cols}
print(f"{'bin':>12s} " + ' '.join(f"{c:>18s}" for c in cols))
for k in range(len(hb) - 1):
    print(f"{hb[k]:5.2f}-{hb[k+1]:<6.2f} " + ' '.join(f"{counts[c][k]:>18d}" for c in cols))
for c in cols:
    k = int(np.argmax(counts[c]))
    print(f"  {c:20s} modal bin {hb[k]:.2f}-{hb[k+1]:.2f} ({counts[c][k]} windows)")

print("\n### Per region")
piv = h.groupby('region').agg(n=('beta_devio', 'size'), devio=('beta_devio', 'median'),
                              prod=('beta_full_masked', 'median'),
                              match=('beta_match_masked', 'median'),
                              match_unm=('beta_match_unmasked', 'median'),
                              dH=('dH_beta_match_masked', 'median'),
                              r2=('devio_r2', 'median'), W_km=('W_m', lambda s: s.median() / 1000))
print(piv.sort_values('prod').to_string(float_format='%.3f'))
print("  dH is the calibrated residual against band-matched masked production.")

print("\n### Truncation cross-check (open question #2)")
print("  If the +0.30 truncated-vs-full-band offset is method bias, the deviogram should")
print("  not reproduce it: dH should run more negative for truncated windows. If it is real")
print("  bed signal, beta_devio should carry the offset too and dH should be flat.")
h['truncated'] = h.W_m < WINDOW_SIZE - 1
g = h.groupby('truncated').agg(n=('beta_devio', 'size'), devio=('beta_devio', 'median'),
                               match=('beta_match_masked', 'median'),
                               dH=('dH_beta_match_masked', 'median'))
print(g.to_string(float_format='%.3f'))
if len(g) == 2:
    print(f"  beta_devio offset (trunc - full)  = {g.devio.get(True, np.nan) - g.devio.get(False, np.nan):+.3f}")
    print(f"  production offset (trunc - full)  = {g['match'].get(True, np.nan) - g['match'].get(False, np.nan):+.3f}")
    print(f"  dH offset         (trunc - full)  = {g.dH.get(True, np.nan) - g.dH.get(False, np.nan):+.3f}")
print("  Read with care: truncated windows are shorter, so their deviogram spans fewer")
print("  decades and is noisier. The calibration is per window length, so the bias is")
print("  handled, but the variance is not.")

r.to_csv(os.path.join(OUT, "deviogram_windows.csv"), index=False)
h.to_csv(os.path.join(OUT, "deviogram_homogeneous.csv"), index=False)
print(f"\nwrote {OUT}/deviogram_windows.csv and deviogram_homogeneous.csv")
sys.stdout.flush()
