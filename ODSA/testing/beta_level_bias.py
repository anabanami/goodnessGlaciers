"""Is production's absolute beta biased? Both estimators, one known truth, one geometry.

Stage E (testing anisotropy model/) measured production's recipe returning beta = 2.369 on
a synthetic bed built at 2.05, so +0.32. deviogram_calibration.py measured the deviogram
reading LOW, -0.09 in beta at H = 0.5 and -0.29 at H = 0.8. Yet ODSA - open questions.md
reports the calibrated deviogram and production agreeing on pooled level to 0.04 in beta.
Those three statements cannot all be true. Nothing so far has run both estimators on the
same profile with the truth known, which is the only comparison that separates them.

This does. fBm is synthesised at known H, so beta = 2H + 1 by construction, and each
profile is measured twice: once by the production Lomb-Scargle recipe and once by the
deviogram. The difference between the two biases is the quantity ODSA - open questions.md
reports as 0.04.

Why it matters beyond the estimator question: BED_CLASSES breaks at 1.5 / 2.0 / 2.5 are
absolute beta, and they are definitional in H (0.5 is the Brownian persistence boundary,
0.75 its half-step) rather than fitted to any instrument. A definitional threshold carries
no instrument bias to cancel against, so an offset in the estimator moves every window
across the breaks and nothing corrects it.

Geometry is production's, not Stage A's: SEGMENT_KM of synthetic bed is cut into 50 km
windows at 50% overlap, and the peak mask is built from the segment-averaged periodogram
over those windows exactly as bed_analysis does. A single isolated window would mask off
its own periodogram, which production only does for single-window segments.

Both estimators see the SAME sampled profile, so the comparison is paired and the
sampling draw cancels out of the difference.

Run from v23/; writes to v23/beta_level_bias/."""
import numpy as np, pandas as pd, os, re, sys
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(HERE)
OUT = os.path.join(HERE, "beta_level_bias")
sys.path.insert(0, ODSA)
from config import (WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, WINDOW_MASK, FIT_BAND_M,
                    peak_masking_height_threshold, bin_buffer, Tee)
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "beta_level_bias_log.txt"))

# Deviogram estimator definition. Must match deviogram_calibration.py or the two are not
# the same instrument and the comparison against its table is void.
LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS = 250.0, 20, 30, 8

SEGMENT_KM = 100.0      # 3 windows at 50% overlap, so the mask is segment-averaged
N_REPS = 150            # profiles per grid point; the H grid doubled, so this trades a
                        # little precision back for runtime. SEs stay well under 0.02.
SLICE_MULT = 32         # synthesis domain / segment length; a domain equal to the
                        # segment is circulant and suppresses long-lag variance
# Must leave >= SAMPLES_PER_BAND_FLOOR grid points per wavelength at the short end of the
# fit band, or the bed itself has no structure where the log-log slope has its longest
# lever arm and every beta comes back low. Stage E ties its grid to band_lo/25.
GRID_N = 524288
SAMPLES_PER_BAND_FLOOR = 25.0
SEED = 20260814

# H spanning the class breaks (0.25 -> beta 1.5, 0.50 -> 2.0, 0.75 -> 2.5) and far enough
# past them to invert the whole measured range, which runs 0.73 to 4.0 over the 651 windows.
# Measured is biased low, so true beta must extend well above the largest measured value.
H_GRID = np.array([0.00, 0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40,
                   1.60, 1.75])
# Points per 50 km window. Stage E ran at 10 m sampling, so 5000; real radar traces are
# coarser. Swept because "does the bias depend on sampling density" is a live question
# for the PSD estimator, even though deviogram_calibration settled it for the deviogram.
# The three real survey densities, so each survey's own curve needs no interpolation:
# POLARGAP 1631 at 30.7 m, ICECAP 2257 at 22.2 m, ICEGRAV 5365 at 9.3 m.
NPTS_GRID = [1631, 2257, 5365]
# 'regular' is Stage A's grid cut and 'uniform_random' is the opposite worst case, which
# no radar produces. The two named surveys are measured off the Bedmap3 CSVs, valid bed
# picks only. They differ enough to matter: ICECAP puts 37% of its track length inside
# gaps against POLARGAP's 3.7%, and the four ICECAP regions are the four `partial` ones.
# uniform_random dropped after the 2026-08-14 runs: it is a worst case no radar produces
# and it had served its purpose of bracketing the three real surveys from below.
SAMPLING_MODES = ['regular', 'icegrav', 'polargap', 'icecap']
# Empirical interval distributions, sampled by inverse CDF. Measured off the Bedmap3
# Results CSVs, valid bed picks only, as interval / median interval per survey. Parametric
# fits were tried and abandoned: ICECAP is a tight core (79.5% within +/-20% of median)
# with a fat tail (7.1% beyond 3x), which no single jitter distribution reproduces.
SURVEY_Q = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
    86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    99.2, 99.4, 99.6, 99.8, 99.9, 99.95, 99.99, 100])
SURVEY_INTERVALS = {
    # PPB, HD. median 30.7 m, 1631 pts per 50 km window, 3.7% of track length in gaps.
    'polargap': np.array([0.5388, 0.7752, 0.8025, 0.8338, 0.8539, 0.8682, 0.8829, 0.8944,
        0.9026, 0.9108, 0.9156, 0.92, 0.9251, 0.9288, 0.9323, 0.9345, 0.9372, 0.9402,
        0.9431, 0.9448, 0.9473, 0.9501, 0.9513, 0.954, 0.9558, 0.9578, 0.9604, 0.9614,
        0.9641, 0.965, 0.9676, 0.9685, 0.9709, 0.972, 0.9743, 0.9755, 0.9778, 0.9789,
        0.9806, 0.9822, 0.9832, 0.9854, 0.9863, 0.9883, 0.9896, 0.9913, 0.993, 0.9945,
        0.9965, 0.9979, 1, 1.002, 1.004, 1.006, 1.008, 1.01, 1.012, 1.014, 1.017, 1.019,
        1.021, 1.024, 1.026, 1.029, 1.032, 1.034, 1.036, 1.039, 1.042, 1.045, 1.047, 1.05,
        1.053, 1.056, 1.059, 1.063, 1.068, 1.072, 1.077, 1.082, 1.087, 1.092, 1.096, 1.101,
        1.105, 1.109, 1.113, 1.117, 1.121, 1.124, 1.128, 1.131, 1.136, 1.14, 1.145, 1.151,
        1.158, 1.167, 1.181, 1.201, 1.208, 1.218, 1.235, 1.334, 4.295, 13.17, 71.61, 1702]),
    # ASB-LR, MSB, HA, GSM. median 22.2 m, 2257 pts, 37.3% of track length in gaps.
    'icecap': np.array([0.002058, 0.7606, 0.7894, 0.8077, 0.8215, 0.8324, 0.8422, 0.8503,
        0.8582, 0.8662, 0.873, 0.8788, 0.8841, 0.8891, 0.894, 0.8988, 0.9029, 0.9069,
        0.9105, 0.9141, 0.9179, 0.9218, 0.9253, 0.9287, 0.932, 0.9351, 0.9382, 0.9413,
        0.9442, 0.9473, 0.9503, 0.9531, 0.9556, 0.958, 0.9606, 0.9631, 0.9657, 0.9681,
        0.9706, 0.973, 0.9754, 0.9778, 0.98, 0.9824, 0.9847, 0.987, 0.9895, 0.992, 0.9946,
        0.9972, 1, 1.003, 1.005, 1.008, 1.011, 1.013, 1.016, 1.019, 1.022, 1.024, 1.027,
        1.03, 1.033, 1.036, 1.039, 1.042, 1.045, 1.049, 1.052, 1.056, 1.06, 1.065, 1.07,
        1.076, 1.082, 1.089, 1.096, 1.104, 1.114, 1.125, 1.139, 1.157, 1.199, 1.686, 1.814,
        1.887, 1.943, 1.993, 2.044, 2.102, 2.188, 2.475, 2.841, 3.025, 3.247, 3.782, 4.26,
        5.118, 6.491, 9.594, 10.81, 12.56, 15.51, 22.05, 32.61, 51.5, 341.3, 1.032e+04]),
    # RSL. median 9.3 m, 5365 pts, 6.1% of track length in gaps.
    'icegrav': np.array([0.008118, 0.8366, 0.8558, 0.865, 0.8723, 0.8789, 0.8848, 0.89,
        0.8946, 0.8987, 0.9025, 0.9061, 0.9097, 0.9131, 0.9165, 0.9198, 0.923, 0.9261,
        0.9291, 0.9319, 0.9346, 0.9373, 0.9401, 0.9428, 0.9457, 0.9484, 0.9511, 0.9537,
        0.9561, 0.9584, 0.9605, 0.9626, 0.9646, 0.9665, 0.9684, 0.9703, 0.9722, 0.974,
        0.976, 0.9779, 0.9798, 0.9818, 0.9837, 0.9858, 0.9877, 0.9897, 0.9917, 0.9937,
        0.9958, 0.9979, 1, 1.002, 1.004, 1.006, 1.008, 1.011, 1.013, 1.015, 1.017, 1.019,
        1.022, 1.024, 1.026, 1.029, 1.031, 1.033, 1.035, 1.037, 1.04, 1.042, 1.044, 1.046,
        1.048, 1.051, 1.053, 1.055, 1.058, 1.06, 1.063, 1.065, 1.068, 1.07, 1.073, 1.076,
        1.079, 1.082, 1.085, 1.088, 1.091, 1.094, 1.098, 1.102, 1.107, 1.114, 1.121, 1.13,
        1.139, 1.152, 1.168, 1.198, 1.211, 1.227, 1.243, 1.968, 2.287, 3.16, 54.26, 7461]),
}
# segmentation.split_into_segments breaks a trajectory on any step over 2 km, so an
# interval longer than that cannot occur inside a segment and must not be drawn here.
# Expressed per survey in the ratio units the tables use, from each survey's own median.
GAP_THRESHOLD_M = 2000.0
SURVEY_MEDIAN_DX = {'polargap': 30.66, 'icecap': 22.15, 'icegrav': 9.33}

# Drift guard, same construction as deviogram_validation.py: the reconstruction below
# reproduces analyse_sliding_windows, but two of its constants are inline literals there.
_SRC = open(os.path.join(ODSA, "bed_analysis.py")).read()
def _prodval(pat, cast=float):
    m = re.search(pat, _SRC); return cast(m.group(1)) if m else None
N_BINS, DX_FLOOR = 500, 15.0
BAND_MIN, BAND_MAX = FIT_BAND_M
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


def sample_track(mode, npts, rng):
    """Along-track sample positions over SEG at the requested mean density."""
    if mode == 'regular':
        return np.linspace(0.0, SEG, npts)
    if mode == 'uniform_random':
        return np.sort(rng.uniform(0.0, SEG, npts))
    # Survey modes: intervals drawn from the survey's own empirical distribution by
    # inverse CDF, capped at the segmentation gap threshold, then rescaled so the profile
    # spans SEG. Rescaling is a uniform stretch, so it preserves the interval shape.
    tab = SURVEY_INTERVALS[mode]
    cap = GAP_THRESHOLD_M / SURVEY_MEDIAN_DX[mode]
    keep = tab <= cap
    q, v = SURVEY_Q[keep], tab[keep]
    gaps = np.interp(rng.uniform(q[0], q[-1], npts - 1), q, v)
    d = np.concatenate([[0.0], np.cumsum(gaps)])
    return d * (SEG / d[-1])


def deviogram(d, z, edges):
    """RMS deviation per geometric lag bin. Identical to deviogram_calibration.deviogram."""
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
        if tot < MIN_PAIRS: continue
        sq = np.sum((S2[jhi] - S2[jlo]) - 2 * z * (S1[jhi] - S1[jlo]) + cnt * z * z)
        sep = np.sum((D1[jhi] - D1[jlo]) - cnt * d)
        L.append(sep / tot); v.append(np.sqrt(sq / tot))
    return np.array(L), np.array(v)


def devio_H(d, z, edges):
    L, v = deviogram(d, z, edges)
    ok = (v > 0) & (L > 0)
    if ok.sum() < MIN_BINS: return np.nan
    return np.polyfit(np.log10(L[ok]), np.log10(v[ok]), 1)[0]


def production_betas(dist, elev):
    """Reproduce analyse_sliding_windows: same loop, detrend, taper, segment-averaged
    mask, and per-window OLS. Lifted from deviogram_validation.segment_windows, which
    checks its reconstruction against the production CSV before reporting anything.

    Returns one beta per window, plus the detrended window profiles so the deviogram
    can be run on exactly the same samples.
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
            wins.append((wd - wd.min(), det))
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

    # Unmasked is a control, not production. On a pure power law there are no real peaks,
    # so any gap between the two is the mask firing on periodogram noise.
    return [(_fit(pg, clean), _fit(pg, band_full), wd, det)
            for pg, (wd, det) in zip(pgrams, wins)]


rng = np.random.default_rng(SEED)
SEG = SEGMENT_KM * 1000.0
edges = np.geomspace(LAG_MIN, WINDOW_SIZE / 3.0, N_LAGS + 1)
domain = SEG * SLICE_MULT
xg = np.linspace(0.0, domain, GRID_N)

# Fatal, not a warning. An under-resolved bed biases beta low by construction and the
# output would look like a finding rather than an artefact.
_pitch = domain / GRID_N
_spw = BAND_MIN / _pitch
print(f"synthesis grid pitch {_pitch:.2f} m, {_spw:.1f} samples per wavelength at the "
      f"{BAND_MIN:.0f} m band floor")
if _spw < SAMPLES_PER_BAND_FLOOR:
    need = int(2 ** np.ceil(np.log2(domain * SAMPLES_PER_BAND_FLOOR / BAND_MIN)))
    print(f"ERROR: under-resolved. Need >= {SAMPLES_PER_BAND_FLOOR:.0f} samples per "
          f"wavelength at the band floor; set GRID_N = {need}.")
    sys.stdout.flush(); sys.exit(1)

print(f"{len(H_GRID)} H x {len(NPTS_GRID)} sampling densities x {N_REPS} profiles")
print(f"segment {SEGMENT_KM:.0f} km -> 50 km windows at {STEP_SIZE/1000:.0f} km step, "
      f"synthesis domain {SLICE_MULT}x segment, band {BAND_MIN:.0f}-{BAND_MAX:.0f} m\n")

rows = []
for H in H_GRID:
    beta_true = 2 * H + 1
    for mode in SAMPLING_MODES:
        for npts_win in NPTS_GRID:
            npts_seg = int(npts_win * SEGMENT_KM * 1000 / WINDOW_SIZE)
            bp, bu, bd = [], [], []
            for _ in range(N_REPS):
                zg = fbm(GRID_N, H, rng, domain)
                s0 = rng.uniform(0.05 * domain, 0.95 * domain - SEG)
                sel = (xg >= s0) & (xg <= s0 + SEG)
                xs, zs = xg[sel] - s0, zg[sel]
                dist = sample_track(mode, npts_seg, rng)
                elev = np.interp(dist, xs, zs)
                for b, b_unmasked, wd, det in production_betas(dist, elev):
                    h = devio_H(wd, det, edges)      # same samples, same detrend
                    if np.isfinite(b): bp.append(b)
                    if np.isfinite(b_unmasked): bu.append(b_unmasked)
                    if np.isfinite(h): bd.append(2 * h + 1)
            bp, bu, bd = np.array(bp), np.array(bu), np.array(bd)
            rows.append(dict(H_true=H, beta_true=beta_true, sampling=mode,
                             npts_per_window=npts_win, n_windows=len(bp),
                             beta_prod_mean=bp.mean(), beta_prod_sd=bp.std(),
                             beta_prod_se=bp.std() / np.sqrt(len(bp)),
                             beta_prod_unmasked_mean=bu.mean(),
                             beta_devio_mean=bd.mean(), beta_devio_sd=bd.std(),
                             beta_devio_se=bd.std() / np.sqrt(len(bd)),
                             bias_prod=bp.mean() - beta_true,
                             bias_prod_unmasked=bu.mean() - beta_true,
                             bias_devio=bd.mean() - beta_true,
                             prod_minus_devio=bp.mean() - bd.mean(),
                             mask_cost=bp.mean() - bu.mean(),
                             grid_pitch_m=_pitch, samples_per_band_floor=_spw,
                             segment_km=SEGMENT_KM, window_m=WINDOW_SIZE,
                             band_min=BAND_MIN, band_max=BAND_MAX,
                             lag_min=LAG_MIN, n_lags=N_LAGS,
                             min_pairs=MIN_PAIRS, min_bins=MIN_BINS))
            print(f"  H={H:.2f} (beta {beta_true:.2f})  {mode:9s} npts={npts_win:5d}  "
                  f"prod {bp.mean():+.3f} ({bp.mean()-beta_true:+.3f})  "
                  f"unmasked {bu.mean():+.3f}  "
                  f"devio {bd.mean():+.3f} ({bd.mean()-beta_true:+.3f})  "
                  f"gap {bp.mean()-bd.mean():+.3f}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "beta_level_bias.csv"), index=False)

def table(val, title, note=None):
    print(f"\n### {title}")
    if note: print(f"###   {note}")
    print(df.pivot_table(index='H_true', columns=['sampling', 'npts_per_window'],
                         values=val).to_string(float_format='%+.3f'))

table('bias_prod', 'Bias in beta against known truth, production estimator')
table('bias_prod_unmasked', 'Same, peak masking off',
      'a pure power law has no peaks, so any difference is the mask firing on noise')
table('mask_cost', 'What the peak mask costs, masked minus unmasked')
table('bias_devio', 'Bias in beta against known truth, deviogram (uncalibrated)')
table('prod_minus_devio', 'Production minus deviogram, both raw',
      "the quantity 'ODSA - open questions.md' reports as 0.04 pooled on real bed")

print("\n### At the class breaks, production estimator")
for H, br in [(0.25, 1.5), (0.50, 2.0), (0.75, 2.5)]:
    for mode in SAMPLING_MODES:
        g = df[(df.H_true == H) & (df.sampling == mode)]
        if not len(g): continue
        print(f"  beta = {br}, {mode:9s}: reads {g.beta_prod_mean.min():.3f} to "
              f"{g.beta_prod_mean.max():.3f} across sampling densities")

# --- Inversion. Each survey is read at its own density, which is why NPTS_GRID holds the
# three real ones. Monotonicity is checked first: a response that turns over cannot be
# inverted, and silently interpolating through a fold would invent corrections.
OWN = {'polargap': 1631, 'icecap': 2257, 'icegrav': 5365}
print("\n### Measured beta -> true beta, each survey at its own sampling density")
inv = []
for surv, npts in OWN.items():
    g = df[(df.sampling == surv) & (df.npts_per_window == npts)].sort_values('beta_true')
    if not len(g):
        print(f"  {surv}: no rows at npts={npts}, skipped"); continue
    bt, bm = g.beta_true.to_numpy(), g.beta_prod_mean.to_numpy()
    step = np.diff(bm)
    if np.any(step <= 0):
        print(f"  {surv}: NON-MONOTONIC at {int((step <= 0).sum())} steps, NOT invertible "
              f"(min step {step.min():+.4f}). Extend or refine H_GRID before using it.")
        continue
    print(f"  {surv} (npts {npts}): measured {bm.min():.2f} to {bm.max():.2f} covers "
          f"true {bt.min():.2f} to {bt.max():.2f}")
    for m in (1.5, 2.0, 2.5, 3.0):
        if bm.min() <= m <= bm.max():
            print(f"      measured {m:.1f}  ->  true {np.interp(m, bm, bt):.3f}")
        else:
            print(f"      measured {m:.1f}  ->  outside the grid, extend H_GRID")
    inv += [dict(survey=surv, npts=npts, beta_measured=m, beta_true=float(np.interp(m, bm, bt)))
            for m in np.round(np.arange(0.7, 4.01, 0.05), 2) if bm.min() <= m <= bm.max()]
if inv:
    pd.DataFrame(inv).to_csv(os.path.join(OUT, "beta_inversion.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT, 'beta_inversion.csv')}")

print("\n### Read this against three numbers already on record")
print("  Stage E, production recipe on a 2-D bed built at 2.05:        +0.32")
print("  deviogram_calibration.py, W = 50 km, H 0.5 / 0.8:        -0.09 / -0.29 in beta")
print("  ODSA - open questions.md, calibrated devio vs production:  0.04 pooled, real bed")
print("  Stage A sampled a grid regularly, so if the two sampling columns differ in sign")
print("  the +0.32 is a property of cutting profiles out of a grid and does not transfer")
print("  to production, which reads irregularly spaced picks along a track.")
print(f"\nwrote {os.path.join(OUT, 'beta_level_bias.csv')}")
sys.stdout.flush()
