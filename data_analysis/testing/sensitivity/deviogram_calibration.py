"""Calibrate the deviogram slope estimator. Run once, before deviogram_validation.py.

The deviogram slope is biased low, and the bias grows with H, because the structure
function saturates as the lag approaches the profile length. Measured on synthetic
profiles of known H over a 50 km window with lags to W/3:

    H_true 0.25 -> 0.25    H_true 0.75 -> 0.65
    H_true 0.50 -> 0.46    H_true 1.00 -> 0.81

In beta terms a true beta of 3.0 reads about 2.6. That is the same magnitude and sign as
the effect open question #4 exists to look for, so an uncalibrated deviogram cannot tell
"hann over-corrects" from "the estimator under-reads at high H". This script measures the
estimator's response so the comparison can be made against it rather than against H_true.

Two properties make the calibration tractable, both verified before it was written:
the bias does not depend on sampling density (0.65 at 800 through 6000 points per window),
and it does not vanish by shortening the lag range (still -0.14 at H=1 with lags to W/20).
It depends on H and on window length, so those are the two grid axes.

Synthesis is fBm by spectral method on a domain SLICE_MULT times longer than the window,
with windows cut from the interior. A domain equal to the window would be circulant, which
suppresses long-lag variance on its own and inflates the very bias being measured.

Writes v23/deviogram/deviogram_calibration.csv."""
import numpy as np, pandas as pd, os, sys
from scipy import signal
HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(HERE)
OUT = os.path.join(HERE, "deviogram")
sys.path.insert(0, ODSA)
from config import WINDOW_SIZE, Tee
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "deviogram_calibration_log.txt"))

# These four define the estimator being calibrated, and deviogram_validation.py must run
# the identical estimator or the forward map applies the wrong response. They are written
# into the output CSV as columns and checked there on load, so the two cannot diverge
# silently. Change them here; the validation script will refuse the stale table.
LAG_MIN, N_LAGS, MIN_PAIRS, MIN_BINS = 250.0, 20, 30, 8

H_GRID = np.round(np.arange(0.0, 1.55, 0.1), 2)
W_GRID_KM = np.array([10., 12.5, 15., 20., 25., 30., 35., 40., 45., 50.])
N_REPS = 400            # per grid point; sd is about 0.09, so the mean lands near +-0.005
N_PTS = 1500            # sampling density (verified not to matter)
SLICE_MULT = 32         # synthesis domain / window length
GRID_N = 16384          # FFT length for the synthesis domain
SEED = 20260727


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
        if tot < MIN_PAIRS: continue
        sq = np.sum((S2[jhi] - S2[jlo]) - 2 * z * (S1[jhi] - S1[jlo]) + cnt * z * z)
        sep = np.sum((D1[jhi] - D1[jlo]) - cnt * d)
        L.append(sep / tot); v.append(np.sqrt(sq / tot))
    return np.array(L), np.array(v)


def slope(d, z, edges):
    L, v = deviogram(d, z, edges)
    ok = (v > 0) & (L > 0)
    if ok.sum() < MIN_BINS: return np.nan
    return np.polyfit(np.log10(L[ok]), np.log10(v[ok]), 1)[0]


def fbm(n, H, rng, domain):
    """fBm by spectral synthesis: PSD ~ f^-(2H+1), so beta = 2H+1 by construction."""
    f = np.fft.rfftfreq(n, d=domain / n); f[0] = f[1]
    ph = rng.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    z = np.fft.irfft(f ** (-(2 * H + 1) / 2.0) * np.exp(1j * ph), n)
    s = np.std(z)
    return z / s * 100.0 if s > 0 else z


rng = np.random.default_rng(SEED)
print(f"grid: {len(H_GRID)} H x {len(W_GRID_KM)} W, {N_REPS} reps each "
      f"({len(H_GRID) * len(W_GRID_KM) * N_REPS} fits)")
print(f"synthesis domain = {SLICE_MULT}x window, {N_PTS} irregular samples per window, "
      f"linear detrend applied as production does\n")

rows = []
for W_km in W_GRID_KM:
    W = W_km * 1000.0
    lag_max = W / 3.0
    edges = np.geomspace(LAG_MIN, lag_max, N_LAGS + 1)
    for H in H_GRID:
        got = []
        domain = W * SLICE_MULT
        xg = np.linspace(0.0, domain, GRID_N)
        per_real = max(1, N_REPS // 20)          # windows cut from each synthesis
        while len(got) < N_REPS:
            zg = fbm(GRID_N, H, rng, domain)
            for _ in range(per_real):
                if len(got) >= N_REPS: break
                s0 = rng.uniform(0.05 * domain, 0.95 * domain - W)
                sel = (xg >= s0) & (xg <= s0 + W)
                if sel.sum() < 50: continue
                xs, zs = xg[sel] - s0, zg[sel]
                d = np.sort(rng.uniform(0, W, N_PTS))
                z = signal.detrend(np.interp(d, xs, zs))
                got.append(slope(d, z, edges))
        a = np.array(got, float); a = a[np.isfinite(a)]
        rows.append(dict(W_km=W_km, H_true=H, H_devio_mean=a.mean(), H_devio_sd=a.std(),
                         H_devio_se=a.std() / np.sqrt(len(a)), n=len(a),
                         bias=a.mean() - H, decades=np.log10(lag_max / LAG_MIN),
                         lag_min=LAG_MIN, n_lags=N_LAGS,     # estimator definition, checked
                         min_pairs=MIN_PAIRS, min_bins=MIN_BINS))  # by deviogram_validation
    print(f"  W={W_km:5.1f} km ({np.log10(lag_max / LAG_MIN):.2f} lag decades) done")

cal = pd.DataFrame(rows)
cal.to_csv(os.path.join(OUT, "deviogram_calibration.csv"), index=False)

print("\n### Bias (H_devio_mean - H_true) by window length")
print(cal.pivot_table(index='H_true', columns='W_km', values='bias').to_string(float_format='%+.3f'))
print("\n### Standard error on the calibrated mean")
print(f"  worst {cal.H_devio_se.max():.4f}, median {cal.H_devio_se.median():.4f}")
print("\n### Monotonicity check: H_devio_mean must increase with H_true, or the forward map")
print("###   is not invertible and the calibration cannot be trusted.")
for W_km, g in cal.groupby('W_km'):
    g = g.sort_values('H_true')
    d = np.diff(g.H_devio_mean.to_numpy())
    flag = "OK" if np.all(d > 0) else f"NON-MONOTONIC at {int((d <= 0).sum())} steps"
    print(f"  W={W_km:5.1f} km  min step {d.min():+.4f}  {flag}")

print(f"\nwrote {os.path.join(OUT, 'deviogram_calibration.csv')}")
sys.stdout.flush()
