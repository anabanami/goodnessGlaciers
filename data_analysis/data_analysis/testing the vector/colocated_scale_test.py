"""Which vector elements the transect actually contributes, at two window sizes.

Two questions, both about whether an element is worth measuring on a line rather than
reading off a grid.

1. REPRODUCTION. r between co-located 30 km and 50 km windows within one trajectory. An
   element that survives a change of window length is measuring the bed and not the
   window. The smooth elements (bed_elev_mean, measures_speed_mean) are long-wavelength
   fields and should sit near 1.00, which is the control rather than a result. Only velocity
   is borrowed, from MEaSUREs; bed_elev_mean is a radar pick off the same profiles.
   >>> The fit band's upper edge tracks WINDOW_SIZE, so beta is fitted 250 m-30 km in one
   run and 250 m-50 km in the other. beta's r is a lower bound and is not directly
   comparable to eta, which uses a fixed band.

2. SPATIAL STRUCTURE. mean|delta| / (1.128 sd) between window pairs on one trajectory,
   binned by centre separation. 1.0 is uncorrelated, since E|X-Y| = 1.128 sd for
   independent Gaussian draws. An element that reaches 1.0 within one window length cannot
   be interpolated between flight lines.

Crossing the two gives the claim: reliable AND fast-decorrelating is what a transect
contributes; reliable AND slow-decorrelating is already carried by a gridded product.

The 30 km side is a separate run tree and is not regenerated here. Default points at the
window-size sweep; pass a second argument to point elsewhere.

    python colocated_scale_test.py [individual_region_TEST] [30km_run_tree]
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee, WINDOW_SIZE, STEP_SIZE

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / 'individual_region_TEST')
ALT = sys.argv[2] if len(sys.argv) > 2 else str(
    ROOT_DIR / 'v23' / 'window_size' / 'runs' / 'window_csvs')

ALT_TOKEN = 'w30km'
PROD_TOKEN = f'w{WINDOW_SIZE // 1000}km'
MATCH_TOL_KM = 7.5          # half the 30 km run's step, so each 50 km centre takes one partner
TEXTURE = ['beta', 'A_1km', 'rms_roughness', 'eta_wavelength_m',
           'hill_count', 'relief_m', 'xi_band', 'skewness']
SMOOTH = ['bed_elev_mean', 'measures_speed_mean']
ELEMENTS = TEXTURE + SMOOTH
# Overlapping neighbours sit one step apart, the next distinct window one window apart.
BINS = [1, 1.5 * STEP_SIZE / 1e3, 1.2 * WINDOW_SIZE / 1e3, 200, 1e9]
LABELS = ['overlapping', 'one-window-apart', f'{int(1.2 * WINDOW_SIZE / 1e3)}-200km', '>200km']


def load(root, per_region_folders=True):
    """Non-transition windows, one frame per region, keyed by the region folder name."""
    pat = os.path.join(root, '*', 'window_csvs', '*_window_stats.csv') if per_region_folders \
        else os.path.join(root, '*_window_stats.csv')
    out = {}
    for f in sorted(glob.glob(pat)):
        d = pd.read_csv(f)
        missing = [e for e in ELEMENTS if e not in d.columns]
        if missing:
            raise SystemExit(f"{f} is missing {missing}. Both runs must carry the same "
                             f"column names, so re-run the one that is on the old names.")
        d = d[~d.is_transition].copy()
        key = os.path.basename(os.path.dirname(os.path.dirname(f))) if per_region_folders \
            else os.path.basename(f)
        out[key] = (d, os.path.basename(f))
    return out


def pair_scales(prod, alt):
    """Nearest 30 km centre to each 50 km centre, within trajectory and MATCH_TOL_KM."""
    rows = []
    for traj, g in prod.groupby('trajectory'):
        h = alt[alt.trajectory == traj]
        if not len(h):
            continue
        for _, w in g.iterrows():
            d = np.hypot(h.center_x - w.center_x, h.center_y - w.center_y) / 1e3
            if d.min() > MATCH_TOL_KM:
                continue
            m = h.loc[d.idxmin()]
            rows.append({f'{e}_50': w.get(e, np.nan) for e in ELEMENTS} |
                        {f'{e}_30': m.get(e, np.nan) for e in ELEMENTS} |
                        {'trajectory': traj, 'sep_km': float(d.min())})
    return pd.DataFrame(rows)


def madogram(d):
    """mean|delta| / 1.128 sd per element, binned by centre separation. 1.0 = uncorrelated."""
    rows = []
    for _, g in d.groupby('trajectory'):
        i, j = np.triu_indices(len(g), k=1)
        if not len(i):
            continue
        rec = {'dist': np.hypot(g.center_x.values[i] - g.center_x.values[j],
                                g.center_y.values[i] - g.center_y.values[j]) / 1e3}
        for e in ELEMENTS:
            v = pd.to_numeric(g[e], errors='coerce').to_numpy(float) if e in g else None
            rec[e] = np.abs(v[i] - v[j]) if v is not None else np.full(len(i), np.nan)
        rows.append(pd.DataFrame(rec))
    if not rows:
        return None
    p = pd.concat(rows, ignore_index=True)
    # Region sd, so a between-region offset cannot inflate the ratio.
    sd = {e: pd.to_numeric(d[e], errors='coerce').std(ddof=1) if e in d else np.nan
          for e in ELEMENTS}
    for e in ELEMENTS:
        p[e] = p[e] / (1.128 * sd[e]) if np.isfinite(sd.get(e, np.nan)) and sd[e] > 0 else np.nan
    p['bin'] = pd.cut(p.dist, BINS, right=False, labels=LABELS)
    return p


# Its madogram sits below 1.0 at every lag including the far bin, so the normalisation
# misbehaves on a heavy tail. Drawn greyed rather than dropped.
MADOGRAM_UNRELIABLE = ['xi_band']


def _stack(y, gap, lo=0.0, hi=1.0):
    """Label positions in axes fraction, ordered as y and at least gap apart."""
    y = np.asarray(y, float)
    order = np.argsort(y)
    v = y[order].copy()
    for i in range(1, v.size):
        v[i] = max(v[i], v[i - 1] + gap)
    v -= v.mean() - y.mean()
    v -= max(v[-1] - hi, 0.0)
    v += max(lo - v[0], 0.0)
    out = np.empty_like(v)
    out[order] = v
    return out


TITLE = 'The elements a transect contributes are the ones that resist gridding'

CAPTION = (
    'Blue circles: texture measured along the profile. Red squares: long-wavelength '
    'fields, of which only velocity is borrowed (MEaSUREs). Top right of the right panel '
    'is reproducible and unmappable, which is what a transect adds over a gridded product.'
)


def write_metadata(png):
    """Sidecar JSON holding the caption, written next to the figure it describes."""
    out = os.path.splitext(png)[0] + '.json'
    meta = {'figure': os.path.basename(png), 'title': TITLE, 'caption': CAPTION}
    with open(out, 'w') as f:
        json.dump(meta, f, indent=2)
    return out


def plot_crossing(med, tbl, out_path):
    """Left: how fast each element forgets itself along the line. Right: the crossing,
    reproduces well against decorrelates fast. Texture sits top right, smooth bottom right."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.4, 5.6))
    x = np.arange(len(LABELS))
    LABEL_X = x[-1] + 0.28

    a1.axhline(1.0, color='0.35', lw=1.0, ls=':', zorder=1)
    a1.annotate('uncorrelated', xy=(0.02, 1.0), xytext=(0, 4), xycoords=('axes fraction', 'data'),
                textcoords='offset points', fontsize=7.5, color='0.35')
    drawn = [e for e in ELEMENTS if e in tbl]
    for e in drawn:
        grey = e in MADOGRAM_UNRELIABLE
        smooth = e in SMOOTH
        a1.plot(x, tbl[e].to_numpy(float),
                color='0.72' if grey else ('#b22222' if smooth else '#1a6faf'),
                ls=':' if grey else ('--' if smooth else '-'),
                lw=1.0 if grey else (2.0 if e in ('beta', 'eta_wavelength_m') else 1.3),
                marker='o', ms=3.5, zorder=2, label=e)
    a1.set_xticks(x); a1.set_xticklabels(LABELS, fontsize=8)
    a1.set_xlim(-0.15, LABEL_X + 1.35)
    lo, hi = a1.get_ylim()
    a1.set_ylim(lo, hi)

    # The far-bin values are too close together to label in place, so the labels are
    # stacked in a column at LABEL_X and joined to their lines by leaders.
    ends = np.array([tbl[e].iloc[-1] for e in drawn], float)
    for e, y0, f in zip(drawn, ends, _stack((ends - lo) / (hi - lo), 0.046, 0.01, 0.99)):
        grey = e in MADOGRAM_UNRELIABLE
        colour = '0.72' if grey else ('#b22222' if e in SMOOTH else '#1a6faf')
        y1 = lo + f * (hi - lo)
        a1.plot([x[-1], LABEL_X], [y0, y1], color=colour, lw=0.6, alpha=0.55, zorder=1)
        a1.annotate(e + (' (normalisation fails)' if grey else ''),
                    xy=(LABEL_X, y1), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color='0.55' if grey else '0.25')

    a1.set_ylabel('mean|Δ| / 1.128 sd      (1.0 = uncorrelated)')
    a1.set_xlabel('separation along one trajectory')
    a1.set_title('Texture is forgotten within one window; the smooth fields are not',
                 fontsize=10)
    a1.grid(alpha=0.25, lw=0.6)

    pts = [(e, med[e], tbl[e].iloc[1]) for e in ELEMENTS
           if e in med.index and e in tbl and np.isfinite(med[e])]
    for e, mx, my in pts:
        smooth = e in SMOOTH
        a2.scatter(mx, my, s=58, zorder=3,
                   color='#b22222' if smooth else '#1a6faf',
                   marker='s' if smooth else 'o',
                   edgecolor='0.15', linewidth=0.6)
    a2.margins(x=0.13, y=0.12)
    # Labels go above the point, and flip below or inwards where that would collide
    # with an already placed label or overrun the right edge.
    x0, x1 = a2.get_xlim(); y0, y1 = a2.get_ylim()
    placed = []
    for e, mx, my in sorted(pts, key=lambda p: -p[2]):
        above = not any(abs(mx - px) < 0.05 * (x1 - x0) and abs(my - py) < 0.13 * (y1 - y0)
                        for px, py in placed)
        inward = mx > x1 - 0.22 * (x1 - x0)
        a2.annotate(e, xy=(mx, my), xytext=(-6 if inward else 0, 9 if above else -9),
                    textcoords='offset points', ha='right' if inward else 'center',
                    va='bottom' if above else 'top', fontsize=7.5, color='0.25')
        placed.append((mx, my))
    a2.axhline(1.0, color='0.35', lw=1.0, ls=':')
    a2.set_xlabel('reproduces between 30 km and 50 km windows  (median r)')
    a2.set_ylabel(f'decorrelation at {LABELS[1]}')
    a2.set_title('Reliable and fast: what only a transect gives you', fontsize=10)
    a2.grid(alpha=0.25, lw=0.6)

    fig.suptitle(TITLE, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    meta = write_metadata(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {meta}")


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'colocated_scale_test_log.txt'))
    prod, alt = load(ROOT), load(ALT, per_region_folders=False)
    if not alt:
        raise SystemExit(f"No {ALT_TOKEN} window CSVs under {ALT}. Regenerate the reference "
                         f"with v23/run_window_size_sweep.py, or pass its tree as argument 2.")
    alt = {os.path.basename(f).replace(f'_{ALT_TOKEN}_window_stats.csv', ''): d
           for (d, f) in alt.values()}
    print(f"production: {ROOT}\n{ALT_TOKEN} tree : {ALT}\n")

    t1, mado, n_rows = {}, [], []
    for region, (d, fname) in prod.items():
        stem = fname.replace(f'_{PROD_TOKEN}_window_stats.csv', '')
        if stem not in alt:
            print(f"  ! {region}: no {ALT_TOKEN} partner for {stem}")
            continue
        pairs = pair_scales(d, alt[stem])
        n_rows.append({'region': region, f'n_{PROD_TOKEN}': len(d),
                       f'n_{ALT_TOKEN}': len(alt[stem]), 'n_paired': len(pairs)})
        t1[region] = {e: pairs[f'{e}_50'].corr(pairs[f'{e}_30']) for e in ELEMENTS} \
            if len(pairs) > 2 else {e: np.nan for e in ELEMENTS}
        m = madogram(d)
        if m is not None:
            mado.append(m)

    print("window counts and co-located pairs\n" +
          pd.DataFrame(n_rows).to_string(index=False) + "\n")

    r = pd.DataFrame(t1).T
    r.loc['MEDIAN'] = r.median()
    print(f"=== 1. REPRODUCTION: r between co-located {ALT_TOKEN} and {PROD_TOKEN} windows ===")
    print(r.round(3).to_string(), "\n")
    print("  median over regions, ranked:")
    med = r.loc['MEDIAN'].sort_values(ascending=False)
    print("    " + " | ".join(f"{e} {v:.2f}{'*' if e in SMOOTH else ''}"
                              for e, v in med.items()) + "\n    * long-wavelength\n")

    p = pd.concat(mado, ignore_index=True)
    tbl = p.groupby('bin', observed=True).agg(n=('dist', 'size'),
                                              **{e: (e, 'mean') for e in ELEMENTS})
    print("=== 2. SPATIAL STRUCTURE: mean|delta| / 1.128 sd, same trajectory (1.0 = uncorrelated) ===")
    print(tbl.T.round(2).to_string(), "\n")

    print("=== CROSSING ===")
    near = tbl.T[LABELS[1]]
    for e in ELEMENTS:
        tag = 'smooth' if e in SMOOTH else 'texture'
        print(f"  {e:20s} {tag:9s} r={med.get(e, np.nan):.2f}  "
              f"one-window-apart={near.get(e, np.nan):.2f}")
    print("\n  Contributed by the transect: reliable r AND at 1.0 by one window length.")
    print("  Long-wavelength, a grid carries these: reliable r AND below 1.0 at 200 km.")

    plot_crossing(med, tbl, os.path.join(ROOT, 'colocated_scale_test.png'))

    r.to_csv(os.path.join(ROOT, 'colocated_scale_reproduction.csv'))
    tbl.T.to_csv(os.path.join(ROOT, 'colocated_scale_madogram.csv'))
    print(f"\n  Saved: {os.path.join(ROOT, 'colocated_scale_reproduction.csv')}")
    print(f"  Saved: {os.path.join(ROOT, 'colocated_scale_madogram.csv')}")
