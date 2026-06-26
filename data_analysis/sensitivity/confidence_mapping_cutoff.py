"""
Cutoff-angle sensitivity of the MEaSUREs-weighted cos²θ anisotropy fit.

Recomputes Δβ and R² for every region at a range of flow-direction weighting
cutoffs, at both window and segment level, straight from the current ODSA
CSVs (Ockenden-regions/{window,segment}_csvs) — no hand-transcribed numbers.
The unweighted fit is the baseline; a tighter cutoff trusts fewer windows
(only those where REMA and MEaSUREs agree closely on flow direction). If the
anisotropy is real and noisy flow directions were diluting it, tightening the
cutoff should strengthen the signal.

  TOP ROW    Weighted Δβ vs cutoff, with the unweighted baseline (dashed) ±1 SE.
  BOTTOM ROW Weighted R² vs cutoff — how well cos²θ explains the data.

Output → confidence_mapping_of_surface_flow/ :
  cutoff_sensitivity.png  +  cutoff_sensitivity_log.txt
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # .../v23
ODSA = HERE.parent                              # .../ODSA
sys.path.insert(0, str(ODSA))
from config import Tee                                       # noqa: E402
from weighted_anisotropy import (discover_regions,           # noqa: E402
                                 flow_weight, fit_cos2)

OUT_ROOT = HERE / "confidence_mapping_of_surface_flow"
CUTOFFS = [45, 50, 60, 70, 75]

# Region CSV roots — discover_regions reads window_csvs/ and segment_csvs/ under each
region_roots = [d for d in [
    ODSA / "Ockenden-regions",
    ODSA / "SMUG-regions",
] if d.exists()]


def short_label(region):
    """Compact, unique label from a long region/dataset name."""
    s = re.sub(r'_w\d+km$', '', region)         # drop window-size suffix
    m = re.search(r'Fig\w+', s)
    return s[m.start():] if m else s


def load_clean(path):
    """Load a window/segment CSV, drop NaN fits and transition windows
    (matches weighted_anisotropy.plot_anisotropy)."""
    df = pd.read_csv(path).dropna(subset=['incidence_deg', 'beta'])
    if 'is_transition' in df.columns:
        df = df[~df['is_transition']]
    return df


def sweep_region(path):
    """Unweighted baseline + weighted fit at every cutoff for one CSV.

    Returns (baseline_fit_or_None, {cutoff: fit_or_None}, n, n_valid_at_60).
    """
    df = load_clean(path)
    if len(df) == 0:
        return None, {}, 0, 0
    theta = df['incidence_deg'].values
    beta = df['beta'].values
    speed = df['measures_speed_mean'].values if 'measures_speed_mean' in df else None
    flow_err = df['flow_error_mean'].values if 'flow_error_mean' in df else None

    baseline = fit_cos2(theta, beta)
    weighted = {}
    n_valid60 = 0
    if flow_err is not None:
        for c in CUTOFFS:
            w = flow_weight(flow_err, speed=speed, angle_cutoff=float(c))
            n_ok = int(np.sum(w > 0))
            if c == 60:
                n_valid60 = n_ok
            # need a handful of weighted points for curve_fit to be meaningful
            weighted[c] = fit_cos2(theta, beta, weights=w) if n_ok >= 5 else None
    return baseline, weighted, len(df), n_valid60


def collect(level, regions):
    """Run the sweep for every region that has this level; print a summary."""
    print(f"\n{'='*70}\n{level.upper()}-LEVEL cutoff sweep\n{'='*70}")
    out = {}
    for region in sorted(regions):
        path = regions[region].get(level)
        if path is None:
            continue
        baseline, weighted, n, n_valid60 = sweep_region(path)
        if baseline is None:
            print(f"  {short_label(region):<45s}  (no valid data — skipped)")
            continue
        out[region] = dict(baseline=baseline, weighted=weighted)
        wb = weighted.get(60)
        msg = (f"  {short_label(region):<45s} n={n:<4d}  "
               f"unw Δβ={baseline['delta']:+.3f} R²={baseline['r2']:.3f}")
        if wb is not None:
            msg += (f"  |  w@60° (n={n_valid60}) Δβ={wb['delta']:+.3f} "
                    f"R²={wb['r2']:.3f}")
        print(msg)
    return out


def main():
    regions = {}
    for root in region_roots:
        for region, files in discover_regions(str(root)).items():
            regions.setdefault(region, {}).update(files)
    if not regions:
        print("No region CSVs found under:", [str(r) for r in region_roots])
        return

    levels = ['window', 'segment']
    data = {lvl: collect(lvl, regions) for lvl in levels}

    # Shared colour per region across both columns
    all_regions = sorted({r for lvl in levels for r in data[lvl]})
    cmap = plt.colormaps['tab10']
    colours = {r: cmap(i % 10) for i, r in enumerate(all_regions)}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for col, level in enumerate(levels):
        ax_db, ax_r2 = axes[0, col], axes[1, col]
        for region, fits in data[level].items():
            colour = colours[region]
            label = short_label(region)
            base = fits['baseline']
            weighted = fits['weighted']

            cuts = [c for c in CUTOFFS if weighted.get(c) is not None]
            if not cuts:
                continue
            db = [weighted[c]['delta'] for c in cuts]
            se = [weighted[c]['delta_se'] for c in cuts]
            r2 = [weighted[c]['r2'] for c in cuts]

            ax_db.errorbar(cuts, db, yerr=se, fmt='o-', color=colour,
                           capsize=4, capthick=1.3, lw=1.5, ms=5, label=label)
            ax_db.axhline(base['delta'], color=colour, ls='--', alpha=0.5, lw=1)
            ax_db.fill_between([CUTOFFS[0] - 3, CUTOFFS[-1] + 3],
                               base['delta'] - base['delta_se'],
                               base['delta'] + base['delta_se'],
                               color=colour, alpha=0.06)

            ax_r2.plot(cuts, r2, 'o-', color=colour, lw=1.5, ms=5, label=label)
            ax_r2.axhline(base['r2'], color=colour, ls='--', alpha=0.5, lw=1)

        ax_db.axhline(0, color='k', lw=0.5, zorder=0)
        ax_r2.axhline(0, color='k', lw=0.5, zorder=0)
        ax_db.set_title(f'{level.capitalize()}-level', fontsize=13)
        ax_db.set_ylabel(r'Weighted $\Delta\beta$ ($\beta_\parallel - \beta_\perp$)')
        ax_r2.set_ylabel(r'Weighted R²')
        ax_r2.set_xlabel('Weighting cutoff angle (°)')
        ax_db.legend(fontsize=8)
        ax_db.grid(True, alpha=0.3)
        ax_r2.grid(True, alpha=0.3)
        ax_r2.set_xticks(CUTOFFS)
        ax_r2.set_xlim(CUTOFFS[0] - 3, CUTOFFS[-1] + 3)

    fig.suptitle('Sensitivity of weighted anisotropy fit to cutoff angle\n'
                 '(dashed lines = unweighted baseline ± 1 SE)',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out_png = OUT_ROOT / 'cutoff_sensitivity.png'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved {out_png}")


if __name__ == "__main__":
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _real_stdout = sys.stdout
    sys.stdout = Tee(str(OUT_ROOT / 'cutoff_sensitivity_log.txt'))
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stdout = _real_stdout
