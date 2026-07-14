"""
Threshold sensitivity analysis for the hard/transitional β boundary.

Two layers of output, all under transitional_threshold_sensitivity/ :

  * Shared (threshold-independent) — written once to the top-level folder:
    the sweep plot, sweep table, KDE structure, per-region peak table and a
    single sweep log.
  * Per-candidate [2.0, 2.1, 2.2] — written to <value>/ : the artifacts that
    actually move with the boundary (pooled KDE plot, per-region KDE plot,
    classification breakdown), plus that candidate's log.

Data source: the current ODSA window-level CSVs (Ockenden-regions/window_csvs).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelmin

# ── Paths ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA
sys.path.insert(0, str(ODSA))
from config import Tee                          # noqa: E402

OUT_ROOT = HERE / "transitional_threshold"
THRESHOLDS = [2.0, 2.1, 2.2]

# Gather all window-level CSVs from whichever region dirs exist
csv_dirs = [d for d in [
    ODSA / "Ockenden-regions" / "window_csvs",
    ODSA / "SMUG-regions" / "window_csvs",
] if d.exists()]

# ── Step 1: load pooled beta (done once) ─────────────────────────────
frames = []
for d in csv_dirs:
    for f in sorted(d.glob("*_window_stats.csv")):
        df = pd.read_csv(f, usecols=["beta"])
        df["region"] = f.stem.replace("_window_stats", "")
        frames.append(df)

all_beta = pd.concat(frames, ignore_index=True)
beta = all_beta["beta"].values
n = len(beta)

# Pooled KDE — threshold-independent, computed once
x_grid = np.linspace(0.5, 3.0, 500)
kde = gaussian_kde(beta, bw_method=0.08)        # bandwidth 0.08 — tight enough
                                                # to see structure, not so tight
                                                # it's just noise
density = kde(x_grid)

# 1.8–2.5 sub-range used for structure / peak detection
mask = (x_grid >= 1.8) & (x_grid <= 2.5)
x_sub = x_grid[mask]
d_sub = density[mask]


def shared():
    """Threshold-independent analysis — written once to OUT_ROOT."""
    print(f"{'='*70}\nSWEEP — threshold-independent analysis\n{'='*70}")
    print(f"Total windows loaded: {n}")
    print(f"Beta range: [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"Source dirs: {[str(d) for d in csv_dirs]}")
    print(f"Candidate thresholds: {THRESHOLDS}")
    print(f"\nPer-region counts:")
    print(all_beta.groupby("region").size().to_string())

    # ── Threshold sweep ──────────────────────────────────────────────
    # Slide the hard/transitional boundary from 1.8 to 2.6. The chaotic/hard
    # (1.5) and transitional/soft (2.5) boundaries stay fixed; the candidate
    # values are marked for reference.
    sweep = np.arange(1.8, 2.61, 0.01)

    pct_chaotic      = np.array([(beta < 1.5).sum() / n * 100] * len(sweep))
    pct_hard         = np.array([((beta >= 1.5) & (beta < t)).sum() / n * 100 for t in sweep])
    pct_transitional = np.array([((beta >= t) & (beta < 2.5)).sum() / n * 100 for t in sweep])
    pct_soft         = np.array([(beta >= 2.5).sum() / n * 100] * len(sweep))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sweep, pct_hard, label="hard", color="sienna", lw=2)
    ax.plot(sweep, pct_transitional, label="transitional", color="goldenrod", lw=2)
    ax.plot(sweep, pct_chaotic, label="chaotic (fixed)", color="grey", ls=":", lw=1.5)
    ax.plot(sweep, pct_soft, label="soft (fixed)", color="grey", ls="--", lw=1.5)
    cand_label = "candidates: " + ", ".join(f"{t:.1f}" for t in THRESHOLDS)
    for i, t in enumerate(THRESHOLDS):
        ax.axvline(t, color="red", ls="--", lw=1.2, alpha=0.6,
                   label=cand_label if i == 0 else None)
    ax.set_xlabel("hard / transitional threshold")
    ax.set_ylabel("% of windows")
    ax.set_title("Classification sensitivity to threshold position")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "threshold_sweep.png", dpi=150)
    plt.close(fig)
    print("\n✓ Saved threshold_sweep.png")

    # Sweep table
    print("\n threshold   % hard     % trans     Δhard/0.1")
    print("  ──────────┼──────────┼───────────┼───────────")
    for t in [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
        h = ((beta >= 1.5) & (beta < t)).sum() / n * 100
        tr = ((beta >= t) & (beta < 2.5)).sum() / n * 100
        # rate of change: how much does %hard change per 0.1 shift?
        h2 = ((beta >= 1.5) & (beta < t + 0.1)).sum() / n * 100
        marker = "  ← candidate" if any(abs(t - c) < 1e-9 for c in THRESHOLDS) else ""
        print(f"    {t:.1f}     │  {h:5.1f}   │  {tr:5.1f}    │  {h2 - h:+.1f}%{marker}")

    # ── KDE structure between 1.8 and 2.5 ────────────────────────────
    # order=15: point must be lower than 15 neighbours on each side
    minima_idx = argrelmin(d_sub, order=15)[0]
    maxima_idx = argrelmin(-d_sub, order=15)[0]  # local maxima

    print("\n── KDE structure between β=1.8 and β=2.5 ──")
    print("Local maxima (peaks):")
    for i in maxima_idx:
        print(f"  β = {x_sub[i]:.3f}   density = {d_sub[i]:.4f}")
    print("Local minima (saddles):")
    for i in minima_idx:
        print(f"  β = {x_sub[i]:.3f}   density = {d_sub[i]:.4f}")

    # ── Per-region KDE peaks (table only; plots are per-candidate) ────
    print("\n── Per-region KDE peaks (β = 1.8–2.5) ──")
    print(f"  {'region':<55s}   n    peak β   density")
    print("  " + "─" * 85)
    for name, grp in all_beta.groupby("region"):
        b = grp["beta"].values
        if len(b) < 10:
            print(f"  {name:<55s} {len(b):>4d}    (too few)")
            continue
        d = gaussian_kde(b, bw_method=0.08)(x_grid)
        d_sub_r = d[mask]
        max_idx_r = argrelmin(-d_sub_r, order=15)[0]
        if len(max_idx_r) > 0:
            best = max_idx_r[np.argmax(d_sub_r[max_idx_r])]
            print(f"  {name:<55s} {len(b):>4d}    {x_sub[best]:.3f}   {d_sub_r[best]:.4f}")
        else:
            imax = np.argmax(d_sub_r)
            print(f"  {name:<55s} {len(b):>4d}    {x_sub[imax]:.3f}*  {d_sub_r[imax]:.4f}  (*boundary max)")


def run(threshold, outdir):
    """Threshold-dependent artifacts for one boundary value."""
    print(f"{'='*70}\nHARD / TRANSITIONAL THRESHOLD = {threshold:.2f}\n{'='*70}")

    h  = ((beta >= 1.5) & (beta < threshold)).sum() / n * 100
    tr = ((beta >= threshold) & (beta < 2.5)).sum() / n * 100
    print(f"At threshold = {threshold:.2f}:  hard = {h:.1f}%   transitional = {tr:.1f}%   (n = {n})")

    # ── Pooled KDE with the boundary marked ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x_grid, density, alpha=0.3, color="steelblue")
    ax.plot(x_grid, density, color="steelblue", lw=2)
    ax.axvline(threshold, color="red", ls="--", lw=1.5,
               label=f"threshold ({threshold:.1f})")
    ax.axvline(1.5, color="grey", ls=":", lw=1, alpha=0.6, label="chaotic/hard (1.5)")
    ax.axvline(2.5, color="grey", ls=":", lw=1, alpha=0.6, label="transitional/soft (2.5)")
    ax.set_xlabel("β (spectral slope)")
    ax.set_ylabel("density")
    ax.set_title("KDE of pooled window-level β  (n = {})".format(n))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "threshold_kde.png", dpi=150)
    plt.close(fig)
    print("✓ Saved threshold_kde.png")

    # ── Per-region KDEs with the boundary marked ─────────────────────
    regions = all_beta.groupby("region")
    n_regions = regions.ngroups
    cols = 3
    rows = int(np.ceil(n_regions / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows),
                             sharex=True, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, grp) in zip(axes, regions):
        b = grp["beta"].values
        if len(b) < 10:
            ax.set_title(f"{name}\n(n={len(b)}, too few)", fontsize=7)
            continue
        d = gaussian_kde(b, bw_method=0.08)(x_grid)
        ax.fill_between(x_grid, d, alpha=0.3, color="steelblue")
        ax.plot(x_grid, d, color="steelblue", lw=1.5)
        ax.axvline(threshold, color="red", ls="--", lw=1, alpha=0.7)
        ax.axvline(1.5, color="grey", ls=":", lw=0.8, alpha=0.4)
        ax.axvline(2.5, color="grey", ls=":", lw=0.8, alpha=0.4)
        ax.set_title(f"{name}\n(n={len(b)})", fontsize=7)
        ax.tick_params(labelsize=6)

    for i in range(n_regions, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Per-region KDE of β  (red dashed = {threshold:.1f} threshold)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "threshold_kde_per_region.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved threshold_kde_per_region.png")


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _real_stdout = sys.stdout

    # Shared sweep — once, at the top level
    sys.stdout = Tee(str(OUT_ROOT / "transitional_threshold_sweep_log.txt"))
    try:
        shared()
    finally:
        sys.stdout.flush()
        sys.stdout = _real_stdout
    print(f"→ {OUT_ROOT} (shared sweep) done\n")

    # Per-candidate artifacts
    for t in THRESHOLDS:
        outdir = OUT_ROOT / f"{t:.1f}"
        outdir.mkdir(parents=True, exist_ok=True)
        sys.stdout = Tee(str(outdir / "transitional_threshold_sensitivity_log.txt"))
        try:
            run(t, outdir)
        finally:
            sys.stdout.flush()
            sys.stdout = _real_stdout
        print(f"→ {outdir} done\n")
