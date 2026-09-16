"""Which arm recovers the true beta on a bed of known beta: the cut piece or the uncut segment?

    python "testing segmentation/uncut_vs_cut_synthetic.py" [output_root]
    python "testing segmentation/uncut_vs_cut_synthetic.py" --region HD --reps 10
    python "testing segmentation/uncut_vs_cut_synthetic.py" --no-window-mask
    python "testing segmentation/uncut_vs_cut_synthetic.py" --figure-only     # redraw only

uncut_vs_cut_beta.py measures the difference between the two arms and cannot say which is
closer to the bed: in the truncated cell the arms do not cover the same ground, so it reads
what removing split_by_landscape would DO, not which reading is right. This puts a bed of
known beta under the same real geometry and scores both arms against truth.

Synthesis. Two fBm fields share one phase draw, so the two regimes are the same landscape
at two roughnesses rather than two unrelated beds, and every crossing is offset-matched so
the profile stays continuous: the seam is a kink, which is what a landscape boundary is,
not a jump, which would inject broadband power and manufacture the answer. Regimes
alternate across the pieces split_by_landscape returns, switching at the midpoint of each
transition zone, and the field is interpolated onto the parent's own sample positions, so
the sampling is production's including its gaps. fbm() and the synthesis constants are
v23/beta_ceiling_real_geometry.py's and self-check against that source.

Scoring. Both arms are scored against the truth of the ground the classification is trying
to describe: for each cut piece, its own beta and the beta of the uncut window covering it
are both compared to that piece's true beta. The win rate is the fraction of pairs where
the cut arm is closer. DELTA_TRUE = 0 is the method control, a bed with no contrast at all,
where any arm difference is the estimator rather than the boundary.

Window positions depend only on the sample positions, so the pairing is geometric and is
computed once per parent and reused across every realisation.

Piece length drives two things at once. It sets the top of the fit band, which is the
truncation this investigation is about, and it sets how many windows the segment-averaged
PSD is built from, which sets how noisy the peak mask is: a one-window piece masks off a
single periodogram and find_peaks then fires on noise. The two are confounded in
production. --no-window-mask reruns everything with window-level masking off, into its own
folder, and the difference between the two runs is the mask's share.

Read-only apart from its own output folder. Writes pairs_synthetic.csv, cell_summary.csv,
delta_summary.csv, uncut_vs_cut_synthetic.png and one log into
<output_root>/tests-results/uncut_vs_cut_synthetic/.
"""
import glob, io, os, re, sys, warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(HERE))

from config import Tee, WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, WINDOW_MASK, GRADIENT_THRESHOLD
from loading import load_datasets, OUTPUT_BASE_PATH as _REGION_BASE
from segmentation import split_into_segments, split_by_landscape

_argv, sys.argv = sys.argv, sys.argv[:1]
from transition_cut_truncation import pieces, _gradient, _tkey, _dsname   # noqa: E402
from transition_cut_truncation import PROD_MERGE_KM                       # noqa: E402
from uncut_vs_cut_beta import _windows, MIN_OVERLAP_FRAC                  # noqa: E402
sys.argv = _argv

FIGURE_ONLY = '--figure-only' in sys.argv[1:]
# Peak masking is derived from the segment-averaged PSD, so a one-window piece masks off
# a single noisy periodogram while a many-window parent masks off a smooth average. That
# co-varies with piece length exactly as band truncation does, so the two are confounded
# in production and this flag is the only way to separate them.
NO_MASK = '--no-window-mask' in sys.argv[1:]
if NO_MASK:
    import bed_analysis as _ba
    _ba.WINDOW_MASK = False
_flag = lambda k, d: (type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d)
ONLY_REGION = _flag('--region', '') or None
N_REPS = _flag('--reps', 15)
_used = {sys.argv[sys.argv.index(k) + 1] for k in ('--region', '--reps') if k in sys.argv}
_pos = [a for a in sys.argv[1:] if not a.startswith('-') and a not in _used]
ROOT = _pos[0] if _pos else _REGION_BASE
OUT = os.path.join(ROOT, 'tests-results',
                   'uncut_vs_cut_synthetic' + ('_nomask' if NO_MASK else ''))

# The true contrast across a landscape boundary, in beta. 0.0 is the method control.
DELTA_TRUE = (0.00, 0.25, 0.50, 0.75, 1.00)
BETA_BASE = 2.05          # regime A, the level the existing synthetic seeds are built at
SEED = 20260818

# Mirrored from v23/beta_ceiling_real_geometry.py; checked against that source below.
SLICE_MULT = 32
SAMPLES_PER_BAND_FLOOR = 25.0
GRID_N_CAP = 2 ** 21
BAND_MIN = 250.0

_V23 = ROOT_DIR / 'v23' / 'beta_ceiling_real_geometry.py'


def check_mirror():
    """C: the synthesis constants and the fBm exponent must still match their source."""
    print(f"\nCONTROL C - synthesis mirrored from {_V23.name}")
    if not _V23.exists():
        print("  source absent; mirror unchecked")
        return False
    src = _V23.read_text()
    ok = True
    for name, mine, pat in (('SLICE_MULT', SLICE_MULT, r"SLICE_MULT\s*=\s*(\d+)"),
                            ('SAMPLES_PER_BAND_FLOOR', SAMPLES_PER_BAND_FLOOR,
                             r"SAMPLES_PER_BAND_FLOOR\s*=\s*([\d.]+)"),
                            ('GRID_N_CAP', GRID_N_CAP, r"GRID_N_CAP\s*=\s*2\s*\*\*\s*(\d+)")):
        m = re.search(pat, src)
        theirs = None if m else None
        if m:
            theirs = 2 ** int(m.group(1)) if name == 'GRID_N_CAP' else float(m.group(1))
        same = theirs is not None and float(theirs) == float(mine)
        ok &= same
        print(f"  {name:<24s} here {mine!s:<10s} source {theirs!s:<10s} "
              f"{'ok' if same else 'DRIFTED'}")
    exp_ok = "f ** (-(2 * H + 1) / 2.0)" in src
    ok &= exp_ok
    print(f"  fbm exponent f**(-(2H+1)/2)      {'ok' if exp_ok else 'DRIFTED'}")
    print(f"  -> {'PASS' if ok else 'WARNING - the mirror has drifted, reconcile before reading'}")
    return ok


def draw_phases(n, domain, rng):
    """The frequency axis and one phase draw, in beta_ceiling_real_geometry.fbm's order,
    so _field below reproduces that function exactly for a single H."""
    f = np.fft.rfftfreq(n, d=domain / n); f[0] = f[1]
    ph = rng.uniform(0, 2 * np.pi, len(f)); ph[0] = 0.0
    return f, np.exp(1j * ph)


def _field(f, e, H, n):
    z = np.fft.irfft(f ** (-(2 * H + 1) / 2.0) * e, n)
    s = np.std(z)
    return z / s * 100.0 if s > 0 else z


def piecewise(dpos, switches_m, zA, zB, xg, s0):
    """Sample the two fields onto dpos, alternating at switches, continuous at every seam.

    Returns the profile and the offsets applied, which say how big the seams would have
    been. An offset is a DC shift per run and every window is detrended, so it adds no
    power of its own."""
    vA = np.interp(dpos, xg - s0, zA)
    vB = np.interp(dpos, xg - s0, zB)
    run = np.searchsorted(switches_m, dpos)
    z = np.empty(len(dpos))
    offs, prev_last, off = [], None, 0.0
    for k in range(int(run.max()) + 1):
        m = run == k
        if not m.any():
            continue
        v = vA[m] if k % 2 == 0 else vB[m]
        if prev_last is not None:
            off = prev_last - v[0]
            offs.append(abs(off))
        z[m] = v + off
        prev_last = z[m][-1]
    return z, offs


# ------------------------------------------------------------------------------- geometry

def geometry(regions_wanted, transformer):
    """One pass for the parents the real test qualifies, with their pairing precomputed."""
    out = []
    for bundle in load_datasets():
        reg = regions_wanted.get(bundle['name'])
        if reg is None:
            continue
        df = bundle['data']
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]

        for traj_id in valid['trajectory_id'].unique():
            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20:
                continue
            x, y = transformer.transform(line['longitude (degree_east)'].values,
                                         line['latitude (degree_north)'].values)
            dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))])
            with redirect_stdout(io.StringIO()):
                gap_segments = split_into_segments(line, dist)

            for pi, (seg_data, seg_dist) in enumerate(gap_segments):
                d = np.asarray(seg_dist, float)
                p_len = float(d[-1] - d[0])
                if p_len < WINDOW_SIZE:
                    continue
                with redirect_stdout(io.StringIO()):
                    subs = split_by_landscape(seg_data, seg_dist)
                kept, dropped, _ = pieces(d, _gradient(seg_data['bedrock_altitude (m)'].values, d),
                                          GRADIENT_THRESHOLD, PROD_MERGE_KM)
                zones = [(float(d[a] - d[0]), float(d[b - 1] - d[0])) for a, b, t in
                         list(kept) + list(dropped) if t]
                if len(subs) < 2 or not zones:
                    continue                      # qualifying = the split made a cut here

                dpos = d - d[0]
                switches = np.array(sorted((a + b) / 2 for a, b in zones))

                # Pieces come from the mirror rather than split_by_landscape because it
                # returns index bounds, and the sibling's control A holds the two equal.
                cuts = []
                for si, (a, b, is_tr) in enumerate(kept):
                    sd = dpos[a:b]
                    s_len = float(sd.max() - sd.min())
                    ws = s_len if s_len < WINDOW_SIZE else WINDOW_SIZE
                    ss = s_len if s_len < WINDOW_SIZE else STEP_SIZE
                    mid = (sd.max() + sd.min()) / 2
                    cuts.append(dict(piece=si, lo=a, hi=b, dpos=sd,
                                     ws=ws, ss=ss, is_transition=bool(is_tr),
                                     piece_length_km=s_len / 1000,
                                     truncated=s_len < WINDOW_SIZE,
                                     regime=int(np.searchsorted(switches, mid)) % 2))

                out.append(dict(region=reg, dataset=bundle['name'], trajectory=_tkey(traj_id),
                                parent=pi, dpos=dpos, parent_length_km=p_len / 1000,
                                switches=switches, zones=zones, cuts=cuts,
                                n_pieces=len(subs), n_zones=len(zones)))
        print(f"  {reg:<8s} parents carried: "
              f"{sum(1 for o in out if o['region'] == reg)}")
    return out


def pair_geometry(P, cut_wins, unc_wins):
    """Pairs are fixed by the sample positions, so they are built once and reused."""
    pairs = []
    us = np.array([w['start_km'] for w in unc_wins])
    ue = np.array([w['end_km'] for w in unc_wins])
    for ci, c in enumerate(cut_wins):
        if not len(us):
            break
        ov = np.maximum(0.0, np.minimum(ue, c['end_km']) - np.maximum(us, c['start_km']))
        frac = ov / max(c['end_km'] - c['start_km'], 1e-9)
        j = int(np.argmax(frac))
        if frac[j] < MIN_OVERLAP_FRAC:
            continue
        zov = sum(max(0.0, min(unc_wins[j]['end_km'], b / 1000)
                      - max(unc_wins[j]['start_km'], a / 1000)) for a, b in P['zones'])
        pairs.append(dict(ci=ci, ui=j, overlap_frac=float(frac[j]),
                          piece=c['piece'], piece_length_km=c['piece_length_km'],
                          truncated=c['truncated'], is_transition=c['is_transition'],
                          regime=c['regime'],
                          straddles=bool(zov > 0), trans_overlap_frac=zov / (WINDOW_SIZE / 1000),
                          shared_frac_uncut=float(ov[j] / max(ue[j] - us[j], 1e-9))))
    return pairs


# ------------------------------------------------------------------------------- the run

def run_parent(P, rng):
    """All deltas x reps on one parent. Returns paired rows averaged over realisations."""
    dpos = P['dpos']
    seg_len = float(dpos[-1] - dpos[0])
    domain = SLICE_MULT * seg_len
    grid_n = int(min(GRID_N_CAP, 2 ** np.ceil(np.log2(
        max(4096, domain * SAMPLES_PER_BAND_FLOOR / BAND_MIN)))))
    xg = np.linspace(0.0, domain, grid_n)

    HA = (BETA_BASE - 1) / 2
    acc, seams = {}, []
    for _ in range(N_REPS):
        f, e = draw_phases(grid_n, domain, rng)
        zA = _field(f, e, HA, grid_n)
        s0 = rng.uniform(0.05 * domain, 0.95 * domain - seg_len)
        for dt in DELTA_TRUE:
            HB = (BETA_BASE + dt - 1) / 2
            zB = zA if dt == 0 else _field(f, e, HB, grid_n)
            elev, offs = piecewise(dpos, P['switches'], zA, zB, xg, s0)
            seams += [o / 100.0 for o in offs]

            cw, uw = [], []
            for c in P['cuts']:
                for w in _windows(c['dpos'], elev[c['lo']:c['hi']], c['ws'], c['ss']):
                    # the piece attributes ride on the window row: pair_geometry reads
                    # them off the cut side to label the 2x2 cells
                    cw.append(dict(start_km=w['start_km'], end_km=w['end_km'],
                                   beta=w['window_beta'],
                                   **{k: c[k] for k in ('piece', 'regime', 'piece_length_km',
                                                        'truncated', 'is_transition')}))
            for w in _windows(dpos, elev, WINDOW_SIZE, STEP_SIZE):
                uw.append(dict(start_km=w['start_km'], end_km=w['end_km'],
                               beta=w['window_beta']))
            if 'pairs' not in P:
                P['pairs'] = pair_geometry(P, cw, uw)
            for p in P['pairs']:
                if p['ci'] >= len(cw) or p['ui'] >= len(uw):
                    continue
                k = (dt, p['ci'])
                a = acc.setdefault(k, dict(p=p, dt=dt, bc=[], bu=[]))
                a['bc'].append(cw[p['ci']]['beta'])
                a['bu'].append(uw[p['ui']]['beta'])

    rows = []
    for (dt, _ci), a in acc.items():
        p = a['p']
        bt = BETA_BASE + (dt if p['regime'] else 0.0)
        bc = np.nanmedian(a['bc']); bu = np.nanmedian(a['bu'])
        rows.append(dict(region=P['region'], dataset=P['dataset'],
                         trajectory=P['trajectory'], parent=P['parent'],
                         delta_true=dt, beta_true=bt, n_reps=len(a['bc']),
                         beta_cut=bc, beta_uncut=bu,
                         bias_cut=bc - bt, bias_uncut=bu - bt,
                         cut_closer=abs(bc - bt) < abs(bu - bt),
                         **{k: p[k] for k in ('piece_length_km', 'truncated', 'is_transition',
                                              'straddles', 'trans_overlap_frac',
                                              'shared_frac_uncut', 'overlap_frac')}))
    return rows, seams


# ------------------------------------------------------------------------------- controls

def control_a(P_all, ref_path):
    """A: the geometry must be the same parents the real test qualified, piece for piece."""
    print(f"\nCONTROL A - qualifying parents vs {os.path.basename(ref_path)}")
    if not os.path.exists(ref_path):
        print("  reference absent; run uncut_vs_cut_beta.py first. Control skipped.")
        return False
    ref = pd.read_csv(ref_path, dtype={'trajectory': str})
    ref = ref[ref.qualifies]
    if ONLY_REGION:
        ref = ref[ref.region == ONLY_REGION]
    mine = pd.DataFrame([{k: P[k] for k in ('region', 'dataset', 'trajectory', 'parent',
                                            'parent_length_km', 'n_pieces')} for P in P_all])
    m = ref.merge(mine, on=['dataset', 'trajectory', 'parent'], how='outer',
                  suffixes=('_ref', '_mine'), indicator=True)
    both = m[m['_merge'] == 'both']
    dl = (both['parent_length_km_ref'] - both['parent_length_km_mine']).abs()
    dp = (both['n_pieces_ref'] - both['n_pieces_mine']).abs()
    ok = (len(both) == len(ref) and len(both) == len(mine)
          and (dl.max() if len(dl) else 0) < 1e-9 and (dp.max() if len(dp) else 0) == 0)
    print(f"  {len(both)} matched of {len(ref)} reference and {len(mine)} here, "
          f"max |length diff| {dl.max() if len(dl) else float('nan'):.2e} km, "
          f"max |piece-count diff| {int(dp.max()) if len(dp) else 0}")
    print(f"  -> {'PASS - same parents, same pieces' if ok else 'FAIL - geometry differs'}")
    return ok


# ------------------------------------------------------------------------------- measures

def cell_of(r):
    if not r['truncated']:
        return 'B mixing only' if r['straddles'] else 'A matched ground'
    return 'D truncated + mixing' if r['straddles'] else 'C truncation only'


def report(df):
    out = []
    for dt in sorted(df.delta_true.unique()):
        g = df[(df.delta_true == dt) & (~df.is_transition)]
        print(f"\n  true contrast across the boundary = {dt:.2f} in beta"
              + ("   <- method control, no contrast" if dt == 0 else ""))
        print(f"    {'cell':<24s} {'n':>5s} {'med bias cut':>13s} {'med bias uncut':>15s} "
              f"{'|cut|':>7s} {'|uncut|':>8s} {'cut closer':>11s}")
        for cell in ('A matched ground', 'B mixing only', 'C truncation only',
                     'D truncated + mixing', 'ALL'):
            s = g if cell == 'ALL' else g[g.cell == cell]
            if not len(s):
                continue
            print(f"    {cell:<24s} {len(s):5d} {s.bias_cut.median():+13.3f} "
                  f"{s.bias_uncut.median():+15.3f} {s.bias_cut.abs().median():7.3f} "
                  f"{s.bias_uncut.abs().median():8.3f} {s.cut_closer.mean():10.0%}")
            out.append(dict(delta_true=dt, cell=cell, n=len(s),
                            median_bias_cut=s.bias_cut.median(),
                            median_bias_uncut=s.bias_uncut.median(),
                            median_abs_bias_cut=s.bias_cut.abs().median(),
                            median_abs_bias_uncut=s.bias_uncut.abs().median(),
                            cut_closer_frac=s.cut_closer.mean()))
    return out


def figure(df, path):
    dts = sorted(df.delta_true.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    g = df[~df.is_transition]
    for arm, c, lab in (('bias_cut', 'C3', 'cut piece'), ('bias_uncut', 'C0', 'uncut segment')):
        med = [g[g.delta_true == d][arm].median() for d in dts]
        q1 = [g[g.delta_true == d][arm].quantile(.25) for d in dts]
        q3 = [g[g.delta_true == d][arm].quantile(.75) for d in dts]
        axes[0].plot(dts, med, 'o-', color=c, label=lab)
        axes[0].fill_between(dts, q1, q3, color=c, alpha=0.18, lw=0)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_xlabel(r'true $\Delta\beta$ across the boundary')
    axes[0].set_ylabel(r'measured $-$ true $\beta$ of the piece')
    axes[0].set_title('bias against known truth', fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)

    for cell, c in (('A matched ground', 'C2'), ('B mixing only', 'C1'),
                    ('D truncated + mixing', 'C3')):
        s = g[g.cell == cell]
        if not len(s):
            continue
        axes[1].plot(dts, [s[s.delta_true == d].cut_closer.mean() for d in dts],
                     'o-', color=c, label=cell)
    axes[1].axhline(0.5, color='k', ls='--', lw=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r'true $\Delta\beta$ across the boundary')
    axes[1].set_ylabel('fraction where the cut arm is closer')
    axes[1].set_title('which arm wins', fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  figure -> {path}")


# ----------------------------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    if FIGURE_ONLY:
        figure(pd.read_csv(os.path.join(OUT, 'pairs_synthetic.csv')),
               os.path.join(OUT, 'uncut_vs_cut_synthetic.png'))
        return
    sys.stdout = Tee(os.path.join(OUT, 'uncut_vs_cut_synthetic_log.txt'))

    region_of = {}
    for f in sorted(glob.glob(os.path.join(ROOT, '*', 'window_csvs', '*_window_stats.csv'))):
        reg = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if ONLY_REGION and reg != ONLY_REGION:
            continue
        region_of[_dsname(f)] = reg

    print(f"Regions: {sorted(set(region_of.values()))}")
    # the effective value, not config's: --no-window-mask patches it after import
    import bed_analysis as _bam
    print(f"WINDOW_SIZE={WINDOW_SIZE / 1000:.0f} km, WINDOW_TYPE={WINDOW_TYPE}, "
          f"WINDOW_MASK={_bam.WINDOW_MASK} (config says {WINDOW_MASK}), "
          f"pair overlap >= {MIN_OVERLAP_FRAC:.0%}")
    print(f"beta_base={BETA_BASE}, contrasts={DELTA_TRUE}, reps={N_REPS}, seed={SEED}")
    if NO_MASK:
        print("\n*** --no-window-mask: window-level peak masking is OFF for this run. ***")
        print("This is NOT production. It exists to separate the band-truncation effect")
        print("from the single-window peak-mask effect, which piece length drives both of.")
        print("Compare against the masked run in ../uncut_vs_cut_synthetic/.\n")
    else:
        print("")

    check_mirror()
    P_all = geometry(region_of, Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True))
    control_a(P_all, os.path.join(ROOT, 'tests-results', 'uncut_vs_cut_beta',
                                  'parent_summary.csv'))

    rng = np.random.default_rng(SEED)
    rows, seams = [], []
    for i, P in enumerate(P_all):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r, s = run_parent(P, rng)
        rows += r; seams += s
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(P_all)} parents synthesised")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo pairs formed. Nothing to report.")
        return
    df['cell'] = df.apply(cell_of, axis=1)
    df.to_csv(os.path.join(OUT, 'pairs_synthetic.csv'), index=False)

    if seams:
        print(f"\nseam size, |offset| as a fraction of the field sd: "
              f"median {np.median(seams):.3f}, p90 {np.percentile(seams, 90):.3f} "
              f"over {len(seams)} crossings")
    else:
        print("\nno seam crossed a sample, so every profile is single-regime here")
    print("A seam is a DC shift per run and every window is detrended, so it carries no")
    print("power of its own; this line is here so the construction can be checked.")

    print("\n=== bias against known truth, uncut minus truth and cut minus truth ===")
    print("Both arms are scored against the true beta of the CUT PIECE, which is the")
    print("ground the classification is describing. Negative bias means reading too")
    print("shallow. 'cut closer' above 50% means the split is helping.")
    cells = report(df)
    pd.DataFrame(cells).to_csv(os.path.join(OUT, 'cell_summary.csv'), index=False)

    print("\n=== per region, pooled over cells, non-transition ===")
    print(f"{'region':<8s} {'delta':>6s} {'n':>5s} {'|bias| cut':>11s} {'|bias| uncut':>13s} "
          f"{'cut closer':>11s}")
    reg_rows = []
    for reg in sorted(df.region.unique()):
        for dt in sorted(df.delta_true.unique()):
            s = df[(df.region == reg) & (df.delta_true == dt) & (~df.is_transition)]
            if not len(s):
                continue
            print(f"{reg:<8s} {dt:6.2f} {len(s):5d} {s.bias_cut.abs().median():11.3f} "
                  f"{s.bias_uncut.abs().median():13.3f} {s.cut_closer.mean():10.0%}")
            reg_rows.append(dict(region=reg, delta_true=dt, n=len(s),
                                 median_abs_bias_cut=s.bias_cut.abs().median(),
                                 median_abs_bias_uncut=s.bias_uncut.abs().median(),
                                 cut_closer_frac=s.cut_closer.mean()))
    pd.DataFrame(reg_rows).to_csv(os.path.join(OUT, 'delta_summary.csv'), index=False)

    figure(df, os.path.join(OUT, 'uncut_vs_cut_synthetic.png'))
    print(f"\nCSVs and log -> {OUT}")


if __name__ == '__main__':
    main()
