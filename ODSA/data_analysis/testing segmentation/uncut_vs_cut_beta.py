"""Is beta from a cut sub-50 km piece worse than beta from the uncut segment it came out of?

    python "testing segmentation/uncut_vs_cut_beta.py" [output_root]
    python "testing segmentation/uncut_vs_cut_beta.py" --region HD
    python "testing segmentation/uncut_vs_cut_beta.py" --figure-only     # redraw only

The test named in 'ODSA - open questions.md' (CLOSED: the truncation offset is not worth
changing the segmentation for). Removing split_by_landscape trades a measured, bounded,
single-signed truncation offset for an unmeasured mixed-spectrum one, and only the first
half of that trade has ever been measured. This measures the second half.

Two arms over the same ground, both through production's own analyse_sliding_windows:
  cut    the pieces split_by_landscape returns, at production's window-size fallback
  uncut  the whole gap-parent in one call, no landscape split, full 50 km band

Read-only. It calls loading, segmentation and bed_analysis and writes only into its own
output folder. No production CSV is touched, no beta is fed back, nothing is corrected.
The cut arm is recomputed from raw rather than read from the CSVs, so control A can hold
it against production bit for bit and the uncut arm is then the only new computation.

Pairs each cut window to the uncut window over the same ground and reads the pairs as a
2x2 in whether the cut piece is shorter than one window (truncation) and whether the uncut
window straddles a transition zone (mixing). Cell A, a full-band cut piece against an
uncut window with no boundary in it, is the baseline and should sit at zero; B adds
mixing, D adds truncation on top of mixing. Transition pieces are dropped from the arms
because the classification drops them, and are reported on their own line. The two effects
are only separable where they do not coincide, so read the n column before the contrasts:
at a heavily cut region almost every pair lands in D and only D is measured.

Writes window_pairs.csv, windows_cut.csv, windows_uncut.csv, parent_summary.csv,
arm_summary.csv, region_summary.csv, uncut_vs_cut_beta.png and one log into
<output_root>/tests-results/uncut_vs_cut_beta/.
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

from config import (Tee, WINDOW_SIZE, STEP_SIZE, WINDOW_TYPE, WINDOW_MASK, FIT_BAND_M,
                    GRADIENT_THRESHOLD)
from loading import load_datasets, OUTPUT_BASE_PATH as _REGION_BASE
from segmentation import split_into_segments, split_by_landscape
from bed_analysis import analyse_sliding_windows
from bed_character import BED_CLASSES

# The sibling script's split_by_landscape mirror, which its control A holds against the
# real function piece for piece. Reused rather than written again so there is one mirror
# in the folder, not two. It reads sys.argv at import, so import it under a clean one.
_argv, sys.argv = sys.argv, sys.argv[:1]
from transition_cut_truncation import (pieces, _gradient, _tkey, _dsname,   # noqa: E402
                                       PROD_MERGE_KM)
sys.argv = _argv

FIGURE_ONLY = '--figure-only' in sys.argv[1:]
ONLY_REGION = None
if '--region' in sys.argv:
    ONLY_REGION = sys.argv[sys.argv.index('--region') + 1]
_pos = [a for a in sys.argv[1:] if not a.startswith('-') and a != ONLY_REGION]
ROOT = _pos[0] if _pos else _REGION_BASE
OUT = os.path.join(ROOT, 'tests-results', 'uncut_vs_cut_beta')

MIN_OVERLAP_FRAC = 0.80   # a pair must share this much of the cut window's ground
N_BOOT = 2000
BOOT_SEED = 0


def med_ci(v, n=N_BOOT, seed=BOOT_SEED):
    """Median and its bootstrap 95% interval; nan-safe on an empty arm."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    b = np.median(np.random.default_rng(seed).choice(v, (n, v.size)), axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), v.size


def _in_keys(df, keys):
    """Boolean mask of rows whose (dataset, trajectory, parent) is in keys; empty-safe."""
    if df.empty:
        return pd.Series([], dtype=bool)
    return pd.Series(list(zip(df['dataset'], df['trajectory'], df['parent'])),
                     index=df.index).isin(keys)


def class_of(beta):
    for name, lo, hi in BED_CLASSES:
        if lo <= beta < hi:
            return name
    return 'nan'


def band_top_m(window_size):
    """Longest wavelength the fit can see: no wavelength longer than one window."""
    return min(FIT_BAND_M[1], window_size)


def _windows(dist, elev, window_size, step_size):
    """analyse_sliding_windows for beta only. Incidence is NaN because beta never reads it."""
    inc = np.full(len(dist), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                       # all-NaN incidence slices
        with redirect_stdout(io.StringIO()):
            _, _, feats, _, _ = analyse_sliding_windows(
                dist, elev, inc, window_size=window_size, step_size=step_size)
    return feats


def _row(f, **extra):
    return dict(window_id=f['window_id'], start_km=f['start_km'], end_km=f['end_km'],
                beta=f['window_beta'], beta_uncertainty=f['window_beta_uncertainty'],
                relief_m=f['local_relief_m'], rms_roughness=f['roughness_rms'],
                self_affine_valid=f['window_self_affine_valid'], **extra)


# ------------------------------------------------------------------------- the two arms

def run_region(name, df, region, transformer):
    """Both arms over one region. Returns (cut rows, uncut rows, parent rows)."""
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
    cut, uncut, parents = [], [], []

    for traj_id in valid['trajectory_id'].unique():
        line = valid[valid['trajectory_id'] == traj_id].copy()
        if len(line) < 20:
            continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))])
        traj = _tkey(traj_id)

        with redirect_stdout(io.StringIO()):                  # both are chatty
            gap_segments = split_into_segments(line, dist)

        # Production numbers segments over the flattened piece list of the whole
        # trajectory, so the parent loop has to carry one running index.
        seg_no = 0
        for pi, (seg_data, seg_dist) in enumerate(gap_segments):
            p_elev = seg_data['bedrock_altitude (m)'].values
            p_len = float(seg_dist[-1] - seg_dist[0])
            with redirect_stdout(io.StringIO()):
                subs = split_by_landscape(seg_data, seg_dist)

            # Transition ground, dropped zones included, from the validated mirror.
            pd_arr = np.asarray(seg_dist, float)
            kept, dropped, fell_back = pieces(pd_arr, _gradient(p_elev, pd_arr),
                                              GRADIENT_THRESHOLD, PROD_MERGE_KM)
            zones = [(float(seg_dist[s]) / 1000, float(seg_dist[e - 1]) / 1000)
                     for s, e, t in list(kept) + list(dropped) if t]

            # cut arm: production's own pieces at production's window-size fallback
            n_cut_w = 0
            for sub_data, sub_dist, is_tr in subs:
                seg_no += 1
                s_len = float(sub_dist.max() - sub_dist.min())
                ws = s_len if s_len < WINDOW_SIZE else WINDOW_SIZE
                ss = s_len if s_len < WINDOW_SIZE else STEP_SIZE
                if ws <= 0:
                    continue
                for f in _windows(np.asarray(sub_dist, float),
                                  sub_data['bedrock_altitude (m)'].values, ws, ss):
                    cut.append(_row(f, region=region, dataset=name, trajectory=traj,
                                    segment=seg_no, parent=pi,
                                    is_transition=bool(is_tr),
                                    piece_length_km=s_len / 1000,
                                    truncated=s_len < WINDOW_SIZE,
                                    band_top_m=band_top_m(ws)))
                    n_cut_w += 1

            # uncut arm: the whole parent, one call, only where a 50 km window fits
            n_unc_w = 0
            if p_len >= WINDOW_SIZE:
                for f in _windows(np.asarray(seg_dist, float), p_elev, WINDOW_SIZE, STEP_SIZE):
                    ov = sum(max(0.0, min(f['end_km'], b) - max(f['start_km'], a))
                             for a, b in zones)
                    uncut.append(_row(f, region=region, dataset=name, trajectory=traj,
                                      parent=pi, parent_length_km=p_len / 1000,
                                      trans_overlap_frac=ov / (WINDOW_SIZE / 1000),
                                      band_top_m=band_top_m(WINDOW_SIZE)))
                    n_unc_w += 1

            parents.append(dict(
                region=region, dataset=name, trajectory=traj, parent=pi,
                parent_length_km=p_len / 1000, n_pieces=len(subs),
                n_transition_zones=len(zones), split_fell_back=bool(fell_back),
                n_pieces_dropped=len(dropped),
                n_pieces_truncated=sum(1 for _, sd, _ in subs
                                       if float(sd.max() - sd.min()) < WINDOW_SIZE),
                qualifies=bool(p_len >= WINDOW_SIZE and len(zones) > 0 and len(subs) > 1),
                control_b=bool(p_len >= WINDOW_SIZE and len(subs) == 1
                               and len(subs[0][1]) == len(seg_dist)),
                n_windows_cut=n_cut_w, n_windows_uncut=n_unc_w))

    return cut, uncut, parents


def pair_windows(cut, uncut):
    """Each cut window against the uncut window it shares the most ground with."""
    rows = []
    if cut.empty or uncut.empty:
        return pd.DataFrame(rows)
    u_by = {k: g for k, g in uncut.groupby(['dataset', 'trajectory', 'parent'])}
    for key, g in cut.groupby(['dataset', 'trajectory', 'parent']):
        u = u_by.get(key)
        if u is None or u.empty:
            continue
        us, ue = u['start_km'].values, u['end_km'].values
        for _, c in g.iterrows():
            ov = np.maximum(0.0, np.minimum(ue, c['end_km']) - np.maximum(us, c['start_km']))
            frac = ov / max(c['end_km'] - c['start_km'], 1e-9)
            j = int(np.argmax(frac))
            if frac[j] < MIN_OVERLAP_FRAC:
                continue
            m = u.iloc[j]
            rows.append(dict(
                region=c['region'], dataset=c['dataset'], trajectory=c['trajectory'],
                parent=c['parent'], segment=c['segment'],
                cut_window_id=c['window_id'], uncut_window_id=m['window_id'],
                overlap_frac=float(frac[j]),
                shared_frac_uncut=float(ov[j] / max(m['end_km'] - m['start_km'], 1e-9)),
                cut_start_km=c['start_km'], cut_end_km=c['end_km'],
                uncut_start_km=m['start_km'], uncut_end_km=m['end_km'],
                beta_cut=c['beta'], beta_uncut=m['beta'],
                delta_beta=m['beta'] - c['beta'],
                beta_unc_cut=c['beta_uncertainty'], beta_unc_uncut=m['beta_uncertainty'],
                piece_length_km=c['piece_length_km'], truncated=bool(c['truncated']),
                is_transition=bool(c['is_transition']),
                straddles=bool(m['trans_overlap_frac'] > 0),
                trans_overlap_frac=float(m['trans_overlap_frac']),
                band_top_cut_m=c['band_top_m'], band_top_uncut_m=m['band_top_m'],
                relief_cut_m=c['relief_m'], relief_uncut_m=m['relief_m']))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ controls

def control_a(cut, win):
    """The recomputed cut arm must reproduce production's window_stats.csv beta exactly."""
    print("\nCONTROL A - recomputed cut arm vs production window_stats.csv")
    print("  (production additionally drops segments under 20% valid ice thickness, so")
    print("   replay-only rows are expected; cached-only rows are not)")
    ok = True
    for reg in sorted(win):
        w = win[reg][['trajectory', 'segment', 'window_id', 'beta']].copy()
        w['trajectory'] = w['trajectory'].map(_tkey)
        c = cut[cut.region == reg][['trajectory', 'segment', 'window_id', 'beta']]
        m = w.merge(c, on=['trajectory', 'segment', 'window_id'], how='outer',
                    suffixes=('_prod', '_replay'), indicator=True)
        both = m[m['_merge'] == 'both']
        d = (both['beta_prod'] - both['beta_replay']).abs()
        n_off = int((d > 1e-9).sum())
        n_cached_only = int((m['_merge'] == 'left_only').sum())
        ok &= n_off == 0 and n_cached_only == 0
        print(f"  {reg:<8s} {len(both):5d} matched, {n_cached_only:4d} cached-only, "
              f"{int((m['_merge'] == 'right_only').sum()):4d} replay-only, "
              f"max |dbeta| = {d.max() if len(d) else float('nan'):.2e}, {n_off} over 1e-9")
    print(f"  -> {'PASS - the cut arm is production' if ok else 'FAIL - do not read the table below'}")
    return ok


def control_b(pairs, parents):
    """On parents the split never cut, the two arms are the same call, so dbeta must be 0."""
    print("\nCONTROL B - uncut arm on parents the split left whole (dbeta must be exactly 0)")
    keys = set(map(tuple, parents.loc[parents.control_b,
                                      ['dataset', 'trajectory', 'parent']].values))
    if not keys or pairs.empty:
        print("  no parent survived the split whole, so this control cannot run here.")
        print("  Control A still exercises _windows at both window-size settings, but the")
        print("  parent-assembly path is unchecked until a less-cut region is run.")
        return False
    k = _in_keys(pairs, keys)
    d = pairs.loc[k, 'delta_beta'].abs()
    ok = len(d) > 0 and float(d.max()) <= 1e-12
    print(f"  {len(d)} paired windows over {len(keys)} whole parents, "
          f"max |dbeta| = {d.max() if len(d) else float('nan'):.2e}")
    print(f"  -> {'PASS - the uncut code path is the production path' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------------------- measures

def _line(name, g, out, region):
    med, lo, hi, n = med_ci(g['delta_beta'].values)
    v = g['delta_beta'].dropna().values
    p25, p75 = (np.percentile(v, [25, 75]) if v.size else (np.nan, np.nan))
    print(f"    {name:<42s} {n:5d} {med:+10.3f} [{lo:+7.3f},{hi:+7.3f}] "
          f"{p25:+7.3f} {p75:+7.3f}")
    out.append(dict(region=region, arm=name.strip(), n=n, median_delta_beta=med,
                    ci_lo=lo, ci_hi=hi, p25=p25, p75=p75))
    return med


def arm_table(pairs, label, region):
    """Paired dbeta = beta_uncut - beta_cut over the 2x2 that separates the two effects.

    Truncation and mixing are only separable where they do not coincide, so the cells
    matter more than the marginals: read the n column before any contrast below."""
    q = pairs[~pairs.is_transition]        # production classifies non-transition only
    cells = [
        ('A full-band cut, uncut clean', q[(~q.truncated) & (~q.straddles)]),
        ('B full-band cut, uncut straddles', q[(~q.truncated) & (q.straddles)]),
        ('C truncated cut, uncut clean', q[(q.truncated) & (~q.straddles)]),
        ('D truncated cut, uncut straddles', q[(q.truncated) & (q.straddles)]),
        ('  marginal: cut piece >= 50 km', q[~q.truncated]),
        ('  marginal: cut piece < 50 km', q[q.truncated]),
        ('  ALL non-transition pairs', q),
        ('  (transition pieces, not classified)', pairs[pairs.is_transition]),
    ]
    print(f"\n  {label}   [{len(q)} of {len(pairs)} pairs are non-transition]")
    print(f"    {'cell':<42s} {'n':>5s} {'med dbeta':>10s} {'95% CI':>18s} {'p25':>7s} {'p75':>7s}")
    out, med = [], {}
    for name, g in cells:
        med[name[0]] = _line(name, g, out, region)
    print(f"    contrast, truncation (D - B): {med['D'] - med['B']:+.3f}   "
          f"mixing (B - A): {med['B'] - med['A']:+.3f}   "
          f"(differences of medians on the n above, not fitted effects)")

    print(f"    {'dose-response, non-transition':<30s} {'n':>4s} {'med dbeta':>10s} {'med beta_cut':>13s}")
    for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 50), (50, np.inf)):
        s = q[(q.piece_length_km >= lo) & (q.piece_length_km < hi)]
        if len(s):
            print(f"      piece {lo:3.0f}-{hi:5.0f} km{'':<11s} {len(s):4d} "
                  f"{s.delta_beta.median():+10.3f} {s.beta_cut.median():13.3f}")
    if len(q) > 4:
        r = q[['piece_length_km', 'delta_beta']].corr(method='spearman').iloc[0, 1]
        print(f"      Spearman piece length vs dbeta: {r:+.3f} over {len(q)} pairs")

    # Cell B can only ever hold a sliver of boundary: a cut piece of 50 km or more that
    # the uncut window covers to 80% leaves the uncut window almost inside one piece.
    # Cell D is the only cell spanning real contamination, so the mixing dose-response
    # is read there, and shared_frac_uncut says how far the two arms stop sharing ground.
    d = q[q.truncated]
    if len(d) > 4:
        print(f"    {'mixing dose-response, cell D':<30s} {'n':>4s} {'med dbeta':>10s} "
              f"{'med piece':>10s} {'med shared':>11s}")
        for lo, hi in ((0, .10), (.10, .25), (.25, .45), (.45, 1.01)):
            s_ = d[(d.trans_overlap_frac >= lo) & (d.trans_overlap_frac < hi)]
            if len(s_):
                print(f"      uncut zone overlap {lo:.2f}-{hi:.2f}{'':<3s} {len(s_):4d} "
                      f"{s_.delta_beta.median():+10.3f} {s_.piece_length_km.median():9.1f}k "
                      f"{s_.shared_frac_uncut.median():11.2f}")
        r = d[['trans_overlap_frac', 'delta_beta']].corr(method='spearman').iloc[0, 1]
        print(f"      Spearman zone overlap vs dbeta: {r:+.3f} over {len(d)} pairs")
    return out


def figure(pairs, path):
    regs = sorted(pairs['region'].unique())
    fig, axes = plt.subplots(len(regs), 2, figsize=(9.5, 2.0 * len(regs)), squeeze=False)
    edges = [hi for _, _, hi in BED_CLASSES[:-1]]
    lim = (0.5, 4.2)
    for row, r in zip(axes, regs):
        g = pairs[pairs.region == r]
        a, b = row
        for sel, c, lab in ((g[g.truncated], 'C3', 'cut piece < 50 km'),
                            (g[~g.truncated], 'C0', 'cut piece >= 50 km')):
            a.scatter(sel['beta_cut'], sel['beta_uncut'], s=9, alpha=0.55, color=c,
                      edgecolors='none', label=lab)
        a.plot(lim, lim, 'k-', lw=0.8)
        for e in edges:
            a.axvline(e, color='0.8', lw=0.7, zorder=0)
            a.axhline(e, color='0.8', lw=0.7, zorder=0)
        a.set_xlim(*lim); a.set_ylim(*lim)
        a.set_ylabel(r, fontsize=9)

        bins = np.linspace(-1.5, 1.5, 31)
        b.hist(g.loc[g.straddles, 'delta_beta'], bins=bins, color='0.75',
               label='uncut straddles a zone')
        b.hist(g.loc[~g.straddles, 'delta_beta'], bins=bins, histtype='step',
               color='C2', lw=1.3, label='uncut inside one piece')
        b.axvline(0, color='k', lw=0.8)
        b.axvline(float(np.nanmedian(g['delta_beta'])), color='C3', ls='--', lw=1.0)

    axes[0][0].set_title(r'$\beta$ uncut vs $\beta$ cut', fontsize=9)
    axes[0][1].set_title(r'$\Delta\beta$ = uncut $-$ cut, dashed = median', fontsize=9)
    axes[0][0].legend(fontsize=7, frameon=False)
    axes[0][1].legend(fontsize=7, frameon=False)
    axes[-1][0].set_xlabel(r'$\beta$ from the cut piece')
    axes[-1][1].set_xlabel(r'$\Delta\beta$')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  figure -> {path}")


# ----------------------------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    if FIGURE_ONLY:
        # Redraw from the written CSVs. No Tee: the log belongs to the measuring run.
        figure(pd.read_csv(os.path.join(OUT, 'window_pairs.csv')),
               os.path.join(OUT, 'uncut_vs_cut_beta.png'))
        return
    sys.stdout = Tee(os.path.join(OUT, 'uncut_vs_cut_beta_log.txt'))

    win, region_of = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, '*', 'window_csvs', '*_window_stats.csv'))):
        reg = os.path.basename(os.path.dirname(os.path.dirname(f)))
        if ONLY_REGION and reg != ONLY_REGION:
            continue
        region_of[_dsname(f)] = reg
        win[reg] = pd.read_csv(f, dtype={'trajectory': str}).dropna(subset=['beta'])

    print(f"Production window CSVs under {ROOT}: {len(win)} regions"
          + (f" (restricted to {ONLY_REGION})" if ONLY_REGION else ""))
    print(f"WINDOW_SIZE={WINDOW_SIZE / 1000:.0f} km, STEP={STEP_SIZE / 1000:.0f} km, "
          f"WINDOW_TYPE={WINDOW_TYPE}, WINDOW_MASK={WINDOW_MASK}, "
          f"FIT_BAND_M={FIT_BAND_M}, pair overlap >= {MIN_OVERLAP_FRAC:.0%}\n")

    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    cut, uncut, parents = [], [], []
    for bundle in load_datasets():
        reg = region_of.get(bundle['name'])
        if reg is None:
            continue
        c, u, p = run_region(bundle['name'], bundle['data'], reg, tf)
        cut += c; uncut += u; parents += p
        print(f"  {reg:<8s} {len(p):4d} parents, {len(c):5d} cut windows, "
              f"{len(u):5d} uncut windows")

    cut, uncut, parents = pd.DataFrame(cut), pd.DataFrame(uncut), pd.DataFrame(parents)
    if cut.empty:
        print("\nNo windows recomputed. Check that ROOT holds the production region "
              "folders and that --region matches one of them.")
        return
    pairs = pair_windows(cut, uncut)
    cut.to_csv(os.path.join(OUT, 'windows_cut.csv'), index=False)
    uncut.to_csv(os.path.join(OUT, 'windows_uncut.csv'), index=False)
    parents.to_csv(os.path.join(OUT, 'parent_summary.csv'), index=False)
    pairs.to_csv(os.path.join(OUT, 'window_pairs.csv'), index=False)

    control_a(cut, win)
    control_b(pairs, parents)

    print("\n=== what qualifies: parents at least one window long that the split cut ===")
    print(f"{'region':<8s} {'parents':>8s} {'>=50km':>7s} {'cut':>5s} {'zones':>6s} "
          f"{'pieces':>7s} {'trunc':>6s} {'pairs':>6s}")
    for reg, g in parents.groupby('region'):
        q = g[g.qualifies]
        pr = pairs[pairs.region == reg]
        print(f"{reg:<8s} {len(g):8d} {int((g.parent_length_km * 1000 >= WINDOW_SIZE).sum()):7d} "
              f"{len(q):5d} {int(q.n_transition_zones.sum()):6d} "
              f"{int(q.n_pieces.sum()):7d} {int(q.n_pieces_truncated.sum()):6d} {len(pr):6d}")

    summary = []
    qkeys = set(map(tuple, parents.loc[parents.qualifies,
                                       ['dataset', 'trajectory', 'parent']].values))
    qpairs = pairs[_in_keys(pairs, qkeys)]

    print("\n=== paired dbeta, uncut minus cut ===")
    print("Negative means the cut piece reads steeper, which is the direction band")
    print("truncation biases. Cell A is the baseline: same band both arms, no boundary")
    print("in the uncut window, so it should sit at zero. B adds mixing, D adds")
    print("truncation on top of mixing. Transition pieces are excluded because the")
    print("classification excludes them; they are reported on their own line.")
    print("The pairing is one-sided: 80% of the CUT window must sit inside the uncut one,")
    print("but the uncut window is always 50 km, so wherever the cut piece is short the")
    print("uncut arm also covers ground the cut arm never saw. Read shared_frac_uncut.")
    print("Cell A is ground-matched and cell D is not, so D measures what removing the")
    print("split would DO, not which of the two readings is closer to the bed.")
    print(f"\n  2x2 counts, non-transition pairs, all regions")
    q_all = qpairs[~qpairs.is_transition]
    print(pd.crosstab(q_all.truncated, q_all.straddles).to_string())
    summary += arm_table(qpairs, 'ALL REGIONS, qualifying parents only', 'ALL')
    for reg in sorted(qpairs['region'].unique()):
        summary += arm_table(qpairs[qpairs.region == reg], reg, reg)

    print("\n=== does it move a class label? region median beta, retained windows only ===")
    print("NOT like for like, and not a class move: the cut basis is production's own")
    print("non-transition pieces, while the uncut basis has no transition label to drop,")
    print("so it also holds the ground the split discards. Read the paired arms for the")
    print("offset; this row only shows what the two bases would report.")
    print(f"{'region':<8s} {'n_cut':>6s} {'med_cut':>8s} {'class':>13s} "
          f"{'n_unc':>6s} {'med_unc':>8s} {'class':>13s} {'delta':>7s}")
    creg = []
    for reg in sorted(cut['region'].unique()):
        c = cut[(cut.region == reg) & (~cut.is_transition)]['beta'].dropna()
        u = uncut[uncut.region == reg]['beta'].dropna()
        mc, mu = float(np.median(c)) if len(c) else np.nan, float(np.median(u)) if len(u) else np.nan
        print(f"{reg:<8s} {len(c):6d} {mc:8.3f} {class_of(mc):>13s} "
              f"{len(u):6d} {mu:8.3f} {class_of(mu):>13s} {mu - mc:+7.3f}")
        creg.append(dict(region=reg, n_windows_cut=len(c), median_beta_cut=mc,
                         class_cut=class_of(mc), n_windows_uncut=len(u),
                         median_beta_uncut=mu, class_uncut=class_of(mu),
                         delta_median_beta=mu - mc,
                         class_changes=class_of(mc) != class_of(mu)))
    pd.DataFrame(creg).to_csv(os.path.join(OUT, 'region_summary.csv'), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'arm_summary.csv'), index=False)

    print("\n=== band bookkeeping, qualifying parents ===")
    print("The uncut arm always fits 250 m to 50 km. The cut arm's top edge is its own")
    print("piece length wherever that is shorter, which is the truncation the split makes.")
    for reg in sorted(qpairs['region'].unique()):
        g = qpairs[qpairs.region == reg]
        print(f"  {reg:<8s} cut band top: median {g['band_top_cut_m'].median() / 1000:5.1f} km, "
              f"min {g['band_top_cut_m'].min() / 1000:5.1f} km, "
              f"{g['truncated'].mean():4.0%} of pairs truncated, "
              f"{g['straddles'].mean():4.0%} straddling")

    if len(qpairs):
        figure(qpairs[~qpairs.is_transition], os.path.join(OUT, 'uncut_vs_cut_beta.png'))
    else:
        print("\n  no qualifying pairs, figure skipped")
    print(f"\nCSVs and log -> {OUT}")


if __name__ == '__main__':
    main()
