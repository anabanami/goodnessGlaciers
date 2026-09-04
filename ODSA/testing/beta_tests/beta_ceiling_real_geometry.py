"""Does the beta ceiling survive on the real track geometry, or is it my interval model?

beta_level_bias.py found that production's beta estimator saturates under irregular
sampling: on synthetic tracks its mean reading stops rising at 1.92 for ICECAP, 2.93 for
ICEGRAV and 3.04 for POLARGAP no matter how steep the true bed, while regular sampling
tracks truth to beta = 4.5 with no ceiling at all. If that holds it is one-sided censoring
rather than symmetric error, a blanket BETA_SYSTEMATIC_ERROR margin is the wrong shape for
it, and the ordering of the regional beta medians partly tracks survey quality rather than
bed.

That result rests on one assumption I control: intervals drawn i.i.d. by inverse CDF from
each survey's empirical distribution. Real tracks could carry serial structure that a
shuffled draw destroys. Measured lag-1 autocorrelation of real interval sequences is
+0.064 for ICECAP and +0.071 for ICEGRAV, so i.i.d. is close for both, but +0.397 for
POLARGAP. And it does not settle the harder objection, that 21 real ICECAP windows read
beta >= 2.5 when the synthetic says their survey ceilings at 1.92.

This script removes the interval model. It rebuilds the actual along-track positions of
production's own windows from the Bedmap3 CSVs, synthesises fBm at known H onto those
exact coordinates, and asks production's recipe what it reads back. Nothing about the
sampling is modelled: it is the same picks, in the same order, at the same spacings, that
produced the beta in the window CSV.

Two questions, in order of what would change:

  1. Per survey, does the pooled response still ceiling, and at the same level? If the
     ceilings move a long way the i.i.d. model was doing the work and the finding falls.
  2. Per window, and this is the one that matters, what ceiling does the geometry of each
     individual high-beta window permit? If a window that really read 3.05 sits on
     geometry that cannot report above 1.9, the censoring story cannot explain that window
     and something else is going on. If instead the high-beta windows sit on benign
     geometry with high ceilings, the ceiling is window-specific, the survey-level number
     is only its average, and the two results are consistent.

Geometry reconstruction is checked before anything is reported. The segmentation is
reproduced from loading + segmentation, but the REMA ice-thickness filter that drops
segments in production is not, so reconstructed windows are matched to production's own
window CSV by projected centre and only matched windows are used. The match rate and the
centre offsets are printed, and a poor match is fatal rather than a footnote.

Run from v23/; writes to v23/beta_ceiling_real_geometry/."""
import numpy as np, pandas as pd, os, re, sys, glob, contextlib, io
from scipy import signal
from scipy.spatial import cKDTree
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(HERE)
# `production` is the seven classified regions. `domec` is the three campaigns that fly the
# same Dome C box, which is the only ground in the set measured by more than one survey and
# so the only place survey geometry can be separated from processing_flag.
TARGET = (sys.argv[1] if len(sys.argv) > 1 else 'production').lower()
if TARGET not in ('production', 'domec'):
    sys.exit(f"usage: beta_ceiling_real_geometry.py [production|domec], got {TARGET!r}")
OUT = os.path.join(HERE, "beta_ceiling_real_geometry"
                   + ("_domec" if TARGET == 'domec' else ""))
sys.path.insert(0, ODSA)
sys.path.insert(0, os.path.join(ODSA, 'all_data', 'Ockenden'))
from config import (WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, WINDOW_MASK, FIT_BAND_M,
                    peak_masking_height_threshold, bin_buffer, Tee)
from segmentation import split_into_segments, split_by_landscape
from ockenden_coords import _ps71_subset, _ppb_core_subset, _ps71_lowrelief_subset
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "beta_ceiling_real_geometry_log.txt"))
print(f"target: {TARGET}\n")

# True H to test. Spans the class breaks and runs past them far enough to see a ceiling,
# but coarser than beta_level_bias.py's grid because each point here costs a real segment
# rather than a synthetic one. beta = 2H + 1.
H_GRID = np.array([0.25, 0.50, 0.75, 1.05, 1.40, 1.75])
N_REPS = 25             # realisations per (segment, H). SE per window is about 0.05,
                        # which separates a 1.9 ceiling from a 3.0 reading comfortably.
MAX_SEG_PER_SURVEY = 60 # every segment holding a beta >= 2.5 window is kept regardless;
                        # the remainder of the quota is filled at random, seeded
HIGH_BETA = 2.5         # the break this is all about
SLICE_MULT = 32         # synthesis domain / segment length, as in beta_level_bias.py
SAMPLES_PER_BAND_FLOOR = 25.0
GRID_N_CAP = 2 ** 21
MATCH_TOL_M = 500.0     # centre offset allowed when pairing a reconstructed window to
                        # production's; windows are 50 km apart at 25 km step, so this is
                        # far tighter than any ambiguity
MIN_MATCH_FRAC = 0.80   # below this the reconstruction is wrong, not incomplete
SEED = 20260814

BAND_MIN, BAND_MAX = FIT_BAND_M
BEDMAP = os.path.join(ODSA, 'all_data/bedmap3_data/bedmap*/Results/')

# The seven regions, as loading.load_datasets defines them. Duplicated rather than
# imported because loading.py's target_files is currently pointed at a different region
# and this must not depend on what is uncommented there.
REGIONS = [
    dict(reg='PPB', surv='polargap', file='BAS_2015_POLARGAP_AIR_BM3.csv',
         subset=_ppb_core_subset),
    dict(reg='HD', surv='polargap', file='BAS_2015_POLARGAP_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, [-0.6e6, -0.3e6, -0.23e6, 0.07e6])),
    dict(reg='RSL', surv='icegrav', file='BAS_2012_ICEGRAV_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, [0.0e6, 0.30e6, 0.6e6, 0.9e6])),
    dict(reg='ASB-LR', surv='icecap', file='UTIG_2010_ICECAP_AIR_BM3.csv',
         subset=lambda df: _ps71_lowrelief_subset(df, [1.05e6, 2.20e6, -0.80e6, 0.20e6])),
    dict(reg='MSB', surv='icecap', file='UTIG_2010_ICECAP_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, [0.15e6, 0.45e6, 1.025e6, 1.325e6])),
    dict(reg='HA', surv='icecap', file='UTIG_2010_ICECAP_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, [1.90e6, 2.20e6, -0.725e6, -0.425e6])),
    dict(reg='GSM', surv='icecap', file='UTIG_2010_ICECAP_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, [2.15e6, 2.45e6, -0.5e6, -0.2e6])),
]

# The same 300 km Dome C box flown three times, from the same commented block in loading.py.
# True beta is one number here, so any spread between the campaigns is instrument, and the
# ceilings measured on their own coordinates say how much of it geometry alone predicts.
_DOMEC_BOX = [1020000.0, 1320000.0, -1237000.0, -937000.0]
REGIONS_DOMEC = [
    dict(reg='DomeC-WISE', surv='wise', file='BAS_2005_WISE-ISODYN_AIR_BM2.csv',
         subset=lambda df: _ps71_subset(df, _DOMEC_BOX)),
    dict(reg='DomeC-ICECAP', surv='icecap', file='UTIG_2010_ICECAP_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, _DOMEC_BOX)),
    dict(reg='DomeC-IB', surv='icebridge', file='NASA_2013_ICEBRIDGE_AIR_BM3.csv',
         subset=lambda df: _ps71_subset(df, _DOMEC_BOX)),
]

if TARGET == 'domec':
    REGIONS = REGIONS_DOMEC
    PROD_GLOB = os.path.join(ODSA, 'new/*/window_csvs/*.csv')
    # 9 windows at DomeC-IB, so every segment runs and none is sampled away.
    MAX_SEG_PER_SURVEY = 10 ** 6
else:
    PROD_GLOB = os.path.join(ODSA, 'individual_region_TEST/*/window_csvs/*.csv')
SURVEY_ORDER = list(dict.fromkeys(R['surv'] for R in REGIONS))

# Drift guard, same construction as beta_level_bias.py: two of production's constants are
# inline literals in bed_analysis.py and are not importable.
_SRC = open(os.path.join(ODSA, "bed_analysis.py")).read()
def _prodval(pat, cast=float):
    m = re.search(pat, _SRC); return cast(m.group(1)) if m else None
N_BINS, DX_FLOOR = 500, 15.0
for _label, _mine, _prod in [("grid bins", N_BINS,   _prodval(r"geomspace\([^)]*num=(\d+)", int)),
                             ("dx floor",  DX_FLOOR, _prodval(r"max\(dx_median,\s*([\d.]+)\)"))]:
    if _prod is None: print(f"NOTE: could not cross-check {_label} against bed_analysis.py.")
    elif _mine != _prod: print(f"WARNING: {_label} = {_mine} here but {_prod} in production — STALE.")
if WINDOW_TYPE != 'hann':
    print(f"WARNING: WINDOW_TYPE is {WINDOW_TYPE!r}, not 'hann'.")
if not WINDOW_MASK:
    print("WARNING: WINDOW_MASK is False, so production is unmasked and so is this.")


def fbm(n, H, rng, domain):
    """fBm by spectral synthesis: PSD ~ f^-(2H+1), so beta = 2H+1 by construction."""
    f = np.fft.rfftfreq(n, d=domain / n); f[0] = f[1]
    ph = rng.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    z = np.fft.irfft(f ** (-(2 * H + 1) / 2.0) * np.exp(1j * ph), n)
    s = np.std(z)
    return z / s * 100.0 if s > 0 else z


def production_windows(dist, elev, xs=None, ys=None):
    """Reproduce analyse_sliding_windows on a real segment's own sample positions.

    Same loop bounds, detrend, taper, segment-averaged peak mask and per-window OLS as
    beta_level_bias.production_betas, which is itself lifted from
    deviogram_validation.segment_windows. Returns one dict per window.
    """
    seg_len = dist.max() - dist.min()
    W = seg_len if seg_len < WINDOW_SIZE else WINDOW_SIZE
    step = W if seg_len < WINDOW_SIZE else STEP_SIZE

    dx_median = np.median(np.diff(dist)) if len(dist) > 1 else 100
    if dx_median == 0: dx_median = DX_FLOOR
    max_freq = 1 / (2 * max(dx_median, DX_FLOOR))
    freqs = np.geomspace(1 / W, max_freq, num=N_BINS)
    ang = freqs * 2 * np.pi
    wl = 1 / freqs
    band_full = (wl >= BAND_MIN) & (wl <= BAND_MAX)
    log_freqs = np.log10(freqs)

    wins, pgrams = [], []
    cur = dist.min()
    while cur + W <= dist.max() + 1e-6:
        m = (dist >= cur) & (dist <= cur + W)
        wd, we = dist[m], elev[m]
        if len(wd) > 50:
            det = signal.detrend(we)
            tap = det * signal.windows.hann(len(det)) if WINDOW_TYPE == 'hann' else \
                  det * signal.windows.tukey(len(det), alpha=0.5) if WINDOW_TYPE == 'tukey' else det
            pgrams.append(signal.lombscargle(wd, tap, ang, normalize=False))
            wins.append(dict(mask=m, n=int(m.sum()),
                             cx=float(np.mean(xs[m])) if xs is not None else np.nan,
                             cy=float(np.mean(ys[m])) if ys is not None else np.nan))
        cur += step

    clean = band_full.copy()
    if pgrams and WINDOW_MASK:
        avg = np.mean(np.array(pgrams), axis=0)
        if np.sum(band_full) >= 2 and np.all(avg[band_full] > 0):
            s0, i0 = np.polyfit(log_freqs[band_full], np.log10(avg[band_full]), 1)
            resid = avg / 10 ** (i0 + s0 * log_freqs)
            peaks, _ = signal.find_peaks(resid, height=peak_masking_height_threshold)
            for p in peaks:
                clean[max(0, p - bin_buffer):min(len(clean), p + bin_buffer + 1)] = False

    def _fit(pg, mask):
        n = int(np.sum(mask))
        if n < 2 or not np.all(pg[mask] > 0): return np.nan
        return -np.polyfit(log_freqs[mask], np.log10(pg[mask]), 1)[0]

    for w, pg in zip(wins, pgrams):
        w['beta'] = _fit(pg, clean)
        w['beta_unmasked'] = _fit(pg, band_full)
    return wins


# --- Rebuild production's window geometry -----------------------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
prod = []
for f in glob.glob(PROD_GLOB):
    d = pd.read_csv(f); d['reg'] = f.split(os.sep)[-3]
    prod.append(d)
prod = pd.concat(prod, ignore_index=True)
SURV = {r['reg']: r['surv'] for r in REGIONS}
# new/ also holds ASB-SQ and DML, which are not part of either target. Dropping them here
# keeps the match-rate floor below a measure of the reconstruction rather than of the glob.
prod = prod[prod.reg.isin(SURV)]
prod = prod[prod.beta.notna() & prod.center_x.notna()].reset_index(drop=True)
prod['surv'] = prod.reg.map(SURV)
print(f"production windows with a beta and a centre: {len(prod)}  "
      f"({', '.join(f'{k} {v}' for k, v in prod.reg.value_counts().items())})")

segments = []   # one entry per reconstructed segment that matched production
file_cache = {}
for R in REGIONS:
    path = glob.glob(os.path.join(BEDMAP, R['file']))[0]
    if path not in file_cache:
        file_cache[path] = pd.read_csv(path, comment='#', low_memory=False)
    df = R['subset'](file_cache[path].copy())
    df = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)].copy()
    df['trajectory_id'] = df['trajectory_id'].astype(str)

    pr = prod[prod.reg == R['reg']]
    tree = cKDTree(np.c_[pr.center_x.values, pr.center_y.values])
    n_win = n_match = 0
    offs = []
    for traj_id in df['trajectory_id'].unique():
        line = df[df['trajectory_id'] == traj_id]
        if len(line) < 20: continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        x, y = np.asarray(x), np.asarray(y)
        dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        # split_by_landscape prints a transition line per zone; not this script's output
        with contextlib.redirect_stdout(io.StringIO()):
            subs = []
            for sd, sdist in split_into_segments(line, dist):
                subs.extend(split_by_landscape(sd, sdist))
        for sd, sdist, _ in subs:
            # Positional lookup, not searchsorted on distance: repeated positions occur
            # (segmentation nudges them for its gradient) and would resolve to the wrong row.
            pos = line.index.get_indexer(sd.index)
            if np.any(pos < 0) or len(pos) != len(sdist): continue
            sx, sy = x[pos], y[pos]
            elev = sd['bedrock_altitude (m)'].values
            wins = production_windows(np.asarray(sdist, float), elev, sx, sy)
            if not wins: continue
            n_win += len(wins)
            d_, i_ = tree.query(np.c_[[w['cx'] for w in wins], [w['cy'] for w in wins]])
            keep = d_ <= MATCH_TOL_M
            offs.extend(d_[keep])
            if not keep.any(): continue
            n_match += int(keep.sum())
            segments.append(dict(
                reg=R['reg'], surv=R['surv'], traj=traj_id,
                dist=np.asarray(sdist, float) - float(sdist[0]),
                # real beta of every matched window, positionally aligned to the window
                # list production_windows returns for this segment
                keep=keep, prod_beta=pr.beta.values[i_], prod_idx=pr.index.values[i_],
                real_beta_recon=np.array([w['beta'] for w in wins])))
    frac = n_match / max(n_win, 1)
    print(f"  {R['reg']:7s} {R['surv']:9s}  reconstructed {n_win:4d} windows, "
          f"{n_match:4d} matched to production ({frac:.0%}), "
          f"median centre offset {np.median(offs) if offs else np.nan:.1f} m")
del file_cache

matched = sum(int(s['keep'].sum()) for s in segments)
hit = np.concatenate([s['prod_idx'][s['keep']] for s in segments])
if len(hit) != len(set(hit)):
    print(f"WARNING: {len(hit)-len(set(hit))} production windows matched by more than one "
          f"reconstructed window; the pairing is not one to one.")
print(f"\nmatched {matched} of production's {len(prod)} windows "
      f"({matched/len(prod):.0%}), tolerance {MATCH_TOL_M:.0f} m")
if matched / len(prod) < MIN_MATCH_FRAC:
    print(f"ERROR: below the {MIN_MATCH_FRAC:.0%} floor. The geometry reconstruction does "
          f"not reproduce production's windows, so nothing below would mean anything.")
    sys.stdout.flush(); sys.exit(1)

# Sanity check that costs nothing: production_windows re-run on the REAL elevations should
# return production's own beta. Any gap is reconstruction error, and it bounds how much of
# the ceiling below could be an artefact of the rebuild rather than of the sampling.
rb = np.concatenate([s['real_beta_recon'][s['keep']] for s in segments])
pb = np.concatenate([s['prod_beta'][s['keep']] for s in segments])
ok = np.isfinite(rb) & np.isfinite(pb)
print(f"reconstruction vs production beta on the real bed: median |diff| "
      f"{np.median(np.abs(rb[ok]-pb[ok])):.4f}, 95th {np.percentile(np.abs(rb[ok]-pb[ok]),95):.4f}, "
      f"max {np.max(np.abs(rb[ok]-pb[ok])):.4f}  (n={ok.sum()})")

# --- Choose the segments to run ---------------------------------------------------
rng = np.random.default_rng(SEED)
chosen = []
for surv in SURVEY_ORDER:
    pool = [i for i, s in enumerate(segments) if s['surv'] == surv and s['keep'].any()]
    def _maxb(i):
        b = segments[i]['prod_beta'][segments[i]['keep']]
        b = b[np.isfinite(b)]
        return b.max() if len(b) else -np.inf
    hi = [i for i in pool if _maxb(i) >= HIGH_BETA]
    rest = [i for i in pool if i not in set(hi)]
    n_extra = max(0, MAX_SEG_PER_SURVEY - len(hi))
    pick = hi + list(rng.choice(rest, size=min(n_extra, len(rest)), replace=False))
    chosen.extend(pick)
    print(f"  {surv:9s}: {len(pool)} segments available, running {len(pick)} "
          f"({len(hi)} holding a beta >= {HIGH_BETA} window)")

print(f"\n{len(chosen)} segments x {len(H_GRID)} H x {N_REPS} realisations\n")

# --- Synthesise on the real coordinates -------------------------------------------
rows = []
for k, si in enumerate(chosen):
    s = segments[si]
    dist = s['dist']
    seg_len = dist.max() - dist.min()
    domain = SLICE_MULT * seg_len
    grid_n = int(min(GRID_N_CAP,
                     2 ** np.ceil(np.log2(max(4096, domain * SAMPLES_PER_BAND_FLOOR / BAND_MIN)))))
    spw = BAND_MIN / (domain / grid_n)
    xg = np.linspace(0.0, domain, grid_n)
    for H in H_GRID:
        beta_true = 2 * H + 1
        acc = {}      # window index -> list of measured beta
        acc_u = {}
        for _ in range(N_REPS):
            zg = fbm(grid_n, H, rng, domain)
            s0 = rng.uniform(0.05 * domain, 0.95 * domain - seg_len)
            sel = (xg >= s0) & (xg <= s0 + seg_len)
            elev = np.interp(dist, xg[sel] - s0, zg[sel])
            for wi, w in enumerate(production_windows(dist, elev)):
                acc.setdefault(wi, []).append(w['beta'])
                acc_u.setdefault(wi, []).append(w['beta_unmasked'])
        for wi in acc:
            if wi >= len(s['keep']) or not s['keep'][wi]: continue
            b = np.array(acc[wi], float); b = b[np.isfinite(b)]
            u = np.array(acc_u[wi], float); u = u[np.isfinite(u)]
            if not len(b): continue
            rows.append(dict(
                reg=s['reg'], surv=s['surv'], traj=s['traj'], seg_idx=si, win_idx=wi,
                prod_idx=int(s['prod_idx'][wi]), beta_real=float(s['prod_beta'][wi]),
                H_true=H, beta_true=beta_true, n_reps=len(b),
                beta_meas_mean=b.mean(), beta_meas_sd=b.std(),
                beta_meas_se=b.std() / np.sqrt(len(b)),
                beta_unmasked_mean=u.mean() if len(u) else np.nan,
                bias=b.mean() - beta_true, seg_len_km=seg_len / 1000,
                grid_pitch_m=domain / grid_n, samples_per_band_floor=spw))
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(chosen)} segments done")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "beta_ceiling_real_geometry.csv"), index=False)

# Geometry diagnostics per matched window, so a ceiling can be tied to something measured
diag = []
for si in set(df.seg_idx):
    s = segments[si]
    d = s['dist']
    for wi in df[df.seg_idx == si].win_idx.unique():
        W = min(d.max() - d.min(), WINDOW_SIZE)
        step = W if (d.max() - d.min()) < WINDOW_SIZE else STEP_SIZE
        m = (d >= d.min() + wi * step) & (d <= d.min() + wi * step + W)
        dx = np.diff(d[m]); dx = dx[dx > 0]
        diag.append(dict(seg_idx=si, win_idx=wi, n_pts=int(m.sum()),
                         dx_median=float(np.median(dx)), dx_cv=float(dx.std() / dx.mean()),
                         dx_p99=float(np.percentile(dx, 99)),
                         gap_frac=float(dx[dx > 200].sum() / dx.sum())))
df = df.merge(pd.DataFrame(diag), on=['seg_idx', 'win_idx'], how='left')
df.to_csv(os.path.join(OUT, "beta_ceiling_real_geometry.csv"), index=False)

# --- Question 1: does the survey-level ceiling survive? ----------------------------
print("\n### Measured beta against true beta, pooled over each survey's real windows")
piv = df.pivot_table(index='beta_true', columns='surv', values='beta_meas_mean')
print(piv.to_string(float_format='%.3f'))

# beta_level_bias.py, i.i.d. intervals, same estimator, each survey at its own density.
# Only the three production surveys were modelled there, so this arm is production-only.
IID_CEILING = {'icecap': 1.920, 'icegrav': 2.934, 'polargap': 3.041}
print("\n### Ceiling per survey" + (", real geometry against the i.i.d. interval model"
                                    if TARGET == 'production' else ""))
for surv in piv.columns:
    c = piv[surv]
    line = f"  {surv:9s} real geometry ceiling {c.max():.3f} at true beta {c.idxmax():.2f}"
    if TARGET == 'production' and surv in IID_CEILING:
        line += (f"   i.i.d. model {IID_CEILING[surv]:.3f}"
                 f"   shift {c.max()-IID_CEILING[surv]:+.3f}")
    print(line)
    st = np.diff(c.to_numpy())
    print(f"            {'saturates, response turns over' if np.any(st <= 0) else 'still rising, no ceiling in this H range'}")

if TARGET == 'domec':
    # One bed, three campaigns, so true beta is a single number and the spread between the
    # measured medians is instrument. headroom = ceiling - measured: a campaign reading at
    # its own ceiling cannot be distinguished from one reading a steeper bed.
    print("\n### Dome C, one bed measured three times")
    print("###   if geometry drives the spread, the campaign with the lowest headroom")
    print("###   reads the lowest beta; if processing drives it, headroom will not order them")
    dc = df.groupby(['surv', 'reg']).agg(
        n_win=('beta_real', 'nunique'), beta_real=('beta_real', 'median')).reset_index()
    dc['ceiling'] = dc.surv.map(piv.max())
    dc['headroom'] = dc.ceiling - dc.beta_real
    dc['flag'] = dc.reg.map({'DomeC-WISE': 'migrated', 'DomeC-ICECAP': 'partial',
                             'DomeC-IB': 'migrated'})
    print(dc.sort_values('beta_real').to_string(index=False, float_format='%.3f'))
    sp = dc.beta_real.max() - dc.beta_real.min()
    print(f"\n  spread in measured beta across the three campaigns: {sp:.3f}")
    print(f"  spread in geometry ceiling: {dc.ceiling.max()-dc.ceiling.min():.3f}")
    if dc.headroom.min() > 0.5:
        print("  every campaign has headroom over 0.5, so this bed sits below all three")
        print("  ceilings and the site cannot test censoring, whatever the spread says.")
    else:
        print("  at least one campaign is reading near its ceiling, so the site is a live")
        print("  test: compare the beta ordering against the headroom ordering above.")

# --- Question 2: the windows that really read high --------------------------------
print(f"\n### Every window whose real beta is >= {HIGH_BETA}, and what its own geometry allows")
print("###   'ceiling' is the highest mean this window's coordinates returned at any true beta")
hi = df[df.beta_real >= HIGH_BETA]
if not len(hi):
    print("  none")
else:
    g = hi.groupby(['surv', 'reg', 'seg_idx', 'win_idx']).agg(
        beta_real=('beta_real', 'first'), ceiling=('beta_meas_mean', 'max'),
        n_pts=('n_pts', 'first'), dx_median=('dx_median', 'first'),
        dx_cv=('dx_cv', 'first'), gap_frac=('gap_frac', 'first')).reset_index()
    g['excess'] = g.beta_real - g.ceiling
    g = g.sort_values(['surv', 'excess'], ascending=[True, False])
    print(g.to_string(index=False, float_format='%.3f'))
    print("\n###   excess > 0 means the window read higher than its own geometry can "
          "produce on a pure power law")
    for surv, gg in g.groupby('surv'):
        print(f"  {surv:9s} n={len(gg):3d}  median excess {gg.excess.median():+.3f}  "
              f"{(gg.excess > 0).sum()} of {len(gg)} above their own ceiling")

print("\n### Ceiling against window geometry, all windows, quartiles of interval CV")
w = df.groupby(['surv', 'seg_idx', 'win_idx']).agg(
    ceiling=('beta_meas_mean', 'max'), beta_real=('beta_real', 'first'),
    dx_cv=('dx_cv', 'first'), gap_frac=('gap_frac', 'first'),
    n_pts=('n_pts', 'first')).reset_index()
w['cv_q'] = pd.qcut(w.dx_cv, 4, labels=['Q1 lowest', 'Q2', 'Q3', 'Q4 highest'],
                    duplicates='drop')
print(w.groupby('cv_q', observed=True).agg(
    n=('ceiling', 'size'), dx_cv=('dx_cv', 'median'), gap_frac=('gap_frac', 'median'),
    ceiling=('ceiling', 'median'), beta_real=('beta_real', 'median')
).to_string(float_format='%.3f'))
r = w[['dx_cv', 'gap_frac', 'n_pts', 'ceiling', 'beta_real']].corr(method='spearman')
print("\nSpearman, ceiling and real beta against the geometry diagnostics")
print(r.loc[['ceiling', 'beta_real'], ['dx_cv', 'gap_frac', 'n_pts']].to_string(float_format='%+.3f'))

print("\n### How to read this")
print("  If the real-geometry ceilings sit near the i.i.d. ones, the interval model was")
print("  not doing the work and the censoring is a property of the surveys.")
print("  If the high-beta windows mostly sit BELOW their own ceiling (excess < 0) the two")
print("  results are consistent and the ceiling is window-specific, not survey-wide.")
print("  If they sit above it, censoring cannot explain those readings and the real bed")
print("  is doing something a pure power law does not.")
print(f"\nwrote {os.path.join(OUT, 'beta_ceiling_real_geometry.csv')}")
sys.stdout.flush()
