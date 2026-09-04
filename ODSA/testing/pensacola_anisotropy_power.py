"""#1 Pensacola diagnostic: is the segment-level Δβ null (−0.084±0.174) a
power/coverage problem or a genuine scale difference from the window-level signal
(−0.329±0.153)?

The cos²θ fit's ability to separate β∥ from β⊥ rests entirely on the spread of the
regressor cos²θ (the "lever arm") and on having weighted points in BOTH the parallel
(θ<30) and perpendicular (θ>60) arms. Segment β is an average of its windows, so
segment θ — and the lever — is mechanically compressed. This quantifies that:

  power/coverage problem  -> segment loses one arm, ESS collapses, or the weighted
                             cos²θ spread shrinks toward zero. The null is uninformative.
  genuine scale difference -> segment keeps n, both arms, and lever, yet Δβ ~ 0.
                             The anisotropy really weakens when averaged to segment scale.

Reuses production flow_weight + fit_cos2 so numbers match the pipeline exactly.
Run from v23/.  python pensacola_anisotropy_power.py [region-substring]  (default Pensacola)
"""
import sys, os, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; ODSA = HERE.parent
sys.path.insert(0, str(ODSA))
from weighted_anisotropy import flow_weight, fit_cos2, _do_curve_fit
from config import Tee
from loading import OUTPUT_BASE_PATH

OUT = HERE / "TESTING_LANDSCAPE_SPLITTING"; OUT.mkdir(parents=True, exist_ok=True)
sys.stdout = Tee(str(OUT / "pensacola_anisotropy_power_log.txt"))

REGION = sys.argv[1] if len(sys.argv) > 1 else "Pensacola"
SRC = Path(OUTPUT_BASE_PATH)

def _find(sub):
    """Both tree layouts: flat <root>/<sub>/ and per-region <root>/<region>/<sub>/."""
    found = list(SRC.glob(f"{sub}/*.csv")) or list(SRC.glob(f"*/{sub}/*.csv"))
    hits = [p for p in found if REGION.lower() in p.name.lower()]
    if not hits: sys.exit(f"no CSV matching {REGION!r} in {sub} under {SRC}")
    return hits[0]

def prep(path):
    """Match plot_anisotropy: drop nan θ/β, drop transition, compute production weights."""
    df = pd.read_csv(path).dropna(subset=["incidence_deg", "beta"])
    if "is_transition" in df: df = df[~df["is_transition"].astype(bool)].copy()
    th = df["incidence_deg"].to_numpy(float)
    b = df["beta"].to_numpy(float)
    spd = df["measures_speed_mean"].to_numpy(float) if "measures_speed_mean" in df else None
    w = flow_weight(df["flow_error_mean"].to_numpy(float), speed=spd)
    return th, b, w

def ess(w):
    w = w[w > 0]; return (w.sum() ** 2 / np.sum(w ** 2)) if w.size else 0.0

def characterise(name, th, b, w):
    m = w > 0                                    # points the fit actually uses
    c2 = np.cos(np.radians(th)) ** 2             # the fit's regressor
    wm = w[m]; thm = th[m]; c2m = c2[m]
    def wstd(x):                                 # weighted std over used points
        if wm.sum() == 0: return np.nan
        mu = np.average(x, weights=wm); return np.sqrt(np.average((x - mu) ** 2, weights=wm))
    par = int(np.sum(m & (th < 30)))             # parallel arm, at weight
    perp = int(np.sum(m & (th > 60)))            # perpendicular arm, at weight
    print(f"\n── {name} ──")
    print(f"  n rows (post-filter)      : {len(th)}")
    print(f"  n at weight>0             : {int(m.sum())}")
    print(f"  effective sample size ESS : {ess(w):.1f}")
    print(f"  θ range (used)            : {thm.min():.1f}–{thm.max():.1f}°   weighted std {wstd(thm):.1f}°")
    print(f"  cos²θ spread (used, wstd) : {wstd(c2m):.3f}   [0=no lever, ~0.35=full]")
    print(f"  parallel arm  θ<30, w>0   : {par}")
    print(f"  perp arm      θ>60, w>0   : {perp}")
    fit_w = fit_cos2(thm, b[m], weights=wm)
    fit_u = fit_cos2(thm, b[m], weights=None)
    for tag, f in [("weighted", fit_w), ("unweighted", fit_u)]:
        if f: print(f"  Δβ {tag:10s}          : {f['delta']:+.3f} ± {f['delta_se']:.3f}   "
                    f"(β∥={f['beta_par']:.2f} β⊥={f['beta_perp']:.2f}, R²={f['r2']:.3f})")
    return dict(n=int(m.sum()), ess=ess(w), c2=wstd(c2m), par=par, perp=perp, fit=fit_w)

print(f"{'='*66}\n#1 anisotropy power/coverage diagnostic — {REGION}\n{'='*66}")
wth, wb, ww = prep(_find("window_csvs")); W = characterise("WINDOW level", wth, wb, ww)
sth, sb, sw = prep(_find("segment_csvs")); S = characterise("SEGMENT level", sth, sb, sw)

# Secondary: does the window signal survive at segment n? Subsample windows to the
# segment used-count and refit B times. If Δβ stays significant, n alone isn't the
# killer. CAVEAT: windows overlap 50% and are spatially correlated, so this OVERSTATES
# the surviving signal (independent-n is lower); read it as an upper bound on power.
mW = ww > 0; thW, bW, wW = wth[mW], wb[mW], ww[mW]
rng = np.random.default_rng(0); B = 2000; nS = S["n"]; deltas = []
if nS >= 5 and nS <= mW.sum():
    for _ in range(B):                            # point-estimate only, no nested bootstrap
        idx = rng.choice(len(thW), size=nS, replace=False)
        try:
            popt, _ = _do_curve_fit(thW[idx], bW[idx], wW[idx],
                                    p0=[bW[idx].mean(), bW[idx].mean()])
            deltas.append(popt[1] - popt[0])     # β∥ - β⊥
        except Exception: pass
    deltas = np.array(deltas)
    frac_aniso = float(np.mean(deltas <= -0.20))  # stays as negative as the window signal
    frac_null = float(np.mean(deltas >= -0.10))   # regresses to the segment-like null
    print(f"\n── window signal down-sampled (no replacement) to segment n={nS} ──")
    print(f"  (isolates whether n/arm-thinning alone kills it; UPPER bound — window")
    print(f"   points overlap 50% so true independent-n is lower)")
    print(f"  median Δβ over {len(deltas)} refits : {np.median(deltas):+.3f}  "
          f"[p16,p84]=[{np.percentile(deltas,16):+.3f},{np.percentile(deltas,84):+.3f}]")
    print(f"  fraction staying ≤ −0.20 (window-like) : {frac_aniso*100:.0f}%")
    print(f"  fraction ≥ −0.10 (segment-like null)   : {frac_null*100:.0f}%")

# Verdict heuristic
print(f"\n{'='*66}\nREAD\n{'='*66}")
flags = []
if S["perp"] < 5 or S["par"] < 5: flags.append(f"one θ arm thin at segment scale (∥={S['par']}, ⊥={S['perp']})")
if W["c2"] > 0 and S["c2"] < 0.6 * W["c2"]: flags.append(f"cos²θ lever shrinks {W['c2']:.3f}→{S['c2']:.3f} window→segment")
if S["ess"] < 10: flags.append(f"segment ESS low ({S['ess']:.1f})")
if flags:
    print("  Leans POWER/COVERAGE — the segment null may be uninformative:")
    for f in flags: print(f"    - {f}")
else:
    print("  Leans GENUINE SCALE DIFFERENCE — segment keeps n, both arms, and lever,")
    print("  yet Δβ collapses. The anisotropy weakens on averaging to segment scale.")
print("\n  (heuristic only — final interpretation is a judgement call on these numbers.)")
