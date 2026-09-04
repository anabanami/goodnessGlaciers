"""
Calibration of the beta uncertainty used for soft bed-class membership.

`beta_uncertainty` in the window CSVs is the formal OLS error on a log-log PSD
slope. It assumes independent periodogram ordinates (they are not: tapering and
Welch averaging correlate adjacent bins), and it knows nothing about fit-band
choice or sub-window bed variability. So it is a LOWER bound on the true sigma,
and soft memberships built on it alone come out overconfident.

Rather than guess the true sigma, this brackets it:

  lower bound   sigma_fit     = median beta_uncertainty (formal only)
  upper bound   sqrt(nugget)  = beta semivariogram extrapolated to zero lag,
                                which absorbs measurement noise, method noise
                                AND real sub-window bed variability (not
                                separable from this data, hence a bound)

The nugget is fitted only on lags where windows share no data (>= 1 window
length apart); windows are 50% overlapping, so anything closer has correlated
errors and would drag the nugget down.

It then sweeps sigma_extra (added in quadrature) across the bracket and reports
how the region class composition moves. If composition is flat across the whole
bracket, the classification does not depend on the uncertainty model. If it
moves, sigma_extra has to be justified and the categorical cannot carry
quantitative weight on its own.

RESULT (650 windows, 7 regions, 50 km windows):

The nugget route does NOT work. It gives 0.229 (x5.3 the formal 0.043), which is
not believable: gamma(h) declines with lag instead of rising to a sill, and at
50 km windows with 50% overlap the shortest data-independent lag IS 50 km, by
which point two windows cover completely different bed. The nugget is counting
real geology as noise. True upper bound, far too loose to use.

The sweep is the useful part. Region composition is robust (1-2% drift over any
plausible sigma), per-window confidence is not (ambiguous windows run 22.9% ->
96.3%). So quote composition, don't lean on per-window confidence.

The formal error is fine, near enough. polyfit(cov=True) scales by the observed
residual variance, so it measures real log-PSD scatter. Only defect is the
geomspace(500) grid oversampling the 50 km window's ~198 independent frequencies
by ~1.8x, so sigma is small by ~1.34x. True sigma ~0.06. Shifts composition ~1%.
SIGMA_EXTRA stays 0.0 and bed_analysis.py stays as it is. See §6.5.

One-off: revisit only if window size, fit band or PSD method changes.

  OUT_ROOT/beta_sigma_calibration.png   semivariogram + nugget | composition sweep
  OUT_ROOT/beta_sigma_calibration_log.txt

Usage:
  python v23/beta_sigma_calibration.py              # all regions pooled
  python v23/beta_sigma_calibration.py Pensacola    # partial match
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA
sys.path.insert(0, str(ODSA))
from config import Tee                                          # noqa: E402
from loading import OUTPUT_BASE_PATH                             # noqa: E402
from bed_character import (BED_COLORS, CLASS_ORDER,             # noqa: E402
                           add_soft_membership, expected_fractions,
                           parse_window_km)

OUT_ROOT = HERE / "beta_sigma_calibration"
MAX_FIT_LAGS = 6        # independent lags used for the linear extrapolation to h=0
MIN_PAIRS = 30          # a lag bin needs this many pairs to be trusted

# The window_csvs/ folders of the run tree, under either layout: flat <root>/window_csvs/
# and per-region <root>/<region>/window_csvs/.
SRC = Path(OUTPUT_BASE_PATH)
csv_dirs = sorted({p.parent for p in
                   (list(SRC.glob("window_csvs/*_window_stats.csv")) or
                    list(SRC.glob("*/window_csvs/*_window_stats.csv")))})


def load(pattern=None):
    paths = sorted(p for d in csv_dirs for p in d.glob("*_window_stats.csv"))
    if pattern:
        paths = [p for p in paths if pattern.lower() in p.name.lower()]
    if not paths:
        sys.exit(f"No window_stats CSVs found"
                 f"{f' matching {pattern!r}' if pattern else ''} in {csv_dirs}")
    frames = []
    for p in paths:
        d = pd.read_csv(p).dropna(subset=['beta'])
        if 'is_transition' in d.columns:
            d = d[~d['is_transition']]
        d['region'] = p.name.replace('_window_stats.csv', '')
        d['step_km'] = parse_window_km(str(p)) / 2      # 50% overlap
        d['window_km'] = parse_window_km(str(p))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def semivariogram(df):
    """gamma(h) = 0.5 * E[(beta(x+h) - beta(x))^2], binned by along-track lag."""
    acc = {}
    for _, g in df.groupby(['region', 'trajectory', 'segment']):
        g = g.sort_values('window_id')
        b = g['beta'].to_numpy()
        w = g['window_id'].to_numpy()
        step = g['step_km'].iloc[0]
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                acc.setdefault((w[j] - w[i]) * step, []).append((b[j] - b[i]) ** 2)
    return pd.DataFrame([{'lag_km': k, 'gamma': 0.5 * np.mean(v), 'n_pairs': len(v)}
                         for k, v in sorted(acc.items())])


def nugget(vg, window_km):
    """Extrapolate gamma to zero lag using only data-independent lags."""
    ind = vg[(vg.lag_km >= window_km) & (vg.n_pairs >= MIN_PAIRS)].head(MAX_FIT_LAGS)
    if len(ind) < 2:
        return np.nan, np.nan, ind
    slope, intercept = np.polyfit(ind.lag_km, ind.gamma, 1)
    return max(intercept, 0.0), slope, ind


def sweep(df, sigmas):
    rows = []
    for se in sigmas:
        d = add_soft_membership(df.copy(), sigma_extra=se)
        frac, _ = expected_fractions(d)
        row = {'sigma_extra': se,
               'ambiguous_pct': (d['class_confidence'] < 0.9).mean() * 100,
               'median_conf': d['class_confidence'].median()}
        row.update(dict(zip(CLASS_ORDER, frac)))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(str(OUT_ROOT / "beta_sigma_calibration_log.txt"))

    df = load(pattern)
    window_km = df['window_km'].median()
    sigma_fit = df['beta_uncertainty'].median()

    print(f"\n{'='*78}\n  BETA SIGMA CALIBRATION  "
          f"({pattern if pattern else 'all regions pooled'})\n{'='*78}")
    print(f"  {len(df)} windows | {df.region.nunique()} regions | window {window_km:.0f} km")
    print(f"  independent lag >= {window_km:.0f} km (50% overlap -> closer lags share data)")

    # ── Bracket ──────────────────────────────────────────────────────
    vg = semivariogram(df)
    nug, slope, fitted = nugget(vg, window_km)
    if np.isnan(nug):
        sys.exit("  Not enough independent lags to estimate a nugget.")
    sigma_upper = np.sqrt(nug)
    sigma_extra_max = np.sqrt(max(nug - sigma_fit ** 2, 0.0))

    print(f"\n  sigma_fit    (formal, LOWER bound) : {sigma_fit:.3f}")
    print(f"  sqrt(nugget) (total,  UPPER bound) : {sigma_upper:.3f}")
    print(f"  implied sigma_extra range          : 0.000 – {sigma_extra_max:.3f}")
    print(f"  inflation factor over formal       : x{sigma_upper / sigma_fit:.1f}")
    print(f"  nugget fitted on {len(fitted)} independent lags "
          f"({fitted.lag_km.min():.0f}–{fitted.lag_km.max():.0f} km), slope {slope:+.2e}")
    print(f"\n  The nugget absorbs real sub-{window_km:.0f} km bed variability as well as")
    print(f"  measurement noise; they are not separable, so it is an upper bound.")

    # ── Sensitivity across the bracket ───────────────────────────────
    sigmas = np.round(np.linspace(0, sigma_extra_max, 6), 3)
    sw = sweep(df, sigmas)
    print(f"\n  Region composition vs sigma_extra (added in quadrature):\n")
    hdr = (f"  {'s_extra':>8s} {'s_eff':>6s} "
           + ' '.join(f"{c[:6]:>7s}" for c in CLASS_ORDER)
           + f" {'ambig%':>7s} {'medConf':>8s}")
    print(hdr + '\n  ' + '-' * (len(hdr) - 2))
    for _, r in sw.iterrows():
        eff = np.sqrt(sigma_fit ** 2 + r['sigma_extra'] ** 2)
        print(f"  {r['sigma_extra']:>8.3f} {eff:>6.3f} "
              + ' '.join(f"{r[c]:>6.1%}" for c in CLASS_ORDER)
              + f" {r['ambiguous_pct']:>6.1f}% {r['median_conf']:>7.0%}")

    drift = max(abs(sw[c].iloc[-1] - sw[c].iloc[0]) for c in CLASS_ORDER)
    print(f"\n  Max class-fraction drift across the full bracket: {drift:.1%}")
    print("  -> composition is robust to the uncertainty model."
          if drift < 0.05 else
          "  -> composition MOVES with the uncertainty model. sigma_extra must be\n"
          "     justified, and the categorical should not carry quantitative weight\n"
          "     on its own.")
    print(f"{'='*78}\n")

    # ── Plots ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(vg.lag_km, vg.gamma, 'o-', color='0.6', ms=4, lw=0.8, label='γ(h)')
    ax1.plot(fitted.lag_km, fitted.gamma, 'o', color='#d62728', ms=7,
             label='independent lags (fitted)')
    hh = np.linspace(0, fitted.lag_km.max(), 50)
    ax1.plot(hh, slope * hh + nug, '--', color='#d62728', lw=1.2)
    ax1.axhline(nug, color='k', ls=':', lw=1)
    ax1.axhline(sigma_fit ** 2, color='#1f77b4', ls=':', lw=1)
    ax1.axvline(window_km, color='0.7', ls='--', lw=1)
    # Both labels sit just above their lines; the y-limit below adds the headroom
    # the nugget label needs, since the nugget rides the top of the data range.
    ax1.annotate(f'nugget = {nug:.4f}  (σ = {sigma_upper:.3f})', (0, nug),
                 xytext=(10, 6), textcoords='offset points', fontsize=9,
                 va='bottom', ha='left')
    ax1.annotate(f'σ_fit² = {sigma_fit**2:.4f}', (0, sigma_fit ** 2),
                 xytext=(10, 6), textcoords='offset points', fontsize=9,
                 va='bottom', ha='left', color='#1f77b4')
    ax1.set_xlabel('Along-track lag h (km)')
    ax1.set_ylabel('Semivariance γ(β)')
    ax1.set_title('β semivariogram: nugget brackets σ from above')
    ax1.set_ylim(0, 1.18 * max(vg.gamma.max(), nug))
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)

    bottom = np.zeros(len(sw))
    for c in CLASS_ORDER:
        ax2.fill_between(sw.sigma_extra, bottom, bottom + sw[c], color=BED_COLORS[c],
                         alpha=0.75, label=c)
        bottom += sw[c].to_numpy()
    ax2.set_xlabel('σ_extra (added in quadrature)')
    ax2.set_ylabel('Expected class fraction')
    ax2.set_title('Composition across the σ bracket')
    ax2.set_xlim(0, max(sigma_extra_max, 1e-6))
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc='center right')

    plt.tight_layout()
    out = OUT_ROOT / ('beta_sigma_calibration.png' if not pattern
                      else f'beta_sigma_calibration_{pattern}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out}\n")


if __name__ == "__main__":
    main()
