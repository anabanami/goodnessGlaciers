"""How much of the band-truncation problem is caused by split_by_landscape cutting?

    python "testing segmentation/transition_cut_truncation.py" [output_root]
    python "testing segmentation/transition_cut_truncation.py" --figure-only   # redraw only

Geometry only: mirrors split_into_segments (as gap_split) and split_by_landscape (as
pieces) over loading.py's production input and measures segment lengths, boundary
provenance and window counts. The mirrors expose index bounds and open up merge_gap_km,
which the real functions do not; split_by_landscape itself is called only as control A's
reference. No beta is computed or written, and nothing in the pipeline is touched.

Per region it reports the segment-length distribution with the landscape split on
(controlled against production's segment_lengths.csv) and off, whether each sub-50 km
segment is bounded by a transition cut or by a data gap, the counterfactual truncated
fraction with the split off, and a GRADIENT_THRESHOLD x zone-merge-gap sweep of the
truncation / excluded-window trade.

Writes segments_production.csv, segments_split_off.csv, region_summary.csv,
gradient_merge_sweep.csv, segment_length_distributions.png and one log into
<output_root>/tests-results/transition_cut_truncation/.
"""
import glob, io, os, re, sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.ndimage import uniform_filter1d

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee, WINDOW_SIZE, STEP_SIZE, SMOOTHING_LENGTH, GRADIENT_THRESHOLD
from loading import load_datasets, OUTPUT_BASE_PATH as _REGION_BASE
from segmentation import split_by_landscape

FIGURE_ONLY = '--figure-only' in sys.argv[1:]
_pos = [a for a in sys.argv[1:] if not a.startswith('-')]
ROOT = _pos[0] if _pos else _REGION_BASE
OUT = os.path.join(ROOT, 'tests-results', 'transition_cut_truncation')

PROD_GRADIENT = GRADIENT_THRESHOLD   # 15 m/km
PROD_MERGE_KM = 5.0                  # hardcoded inside split_by_landscape
MIN_SEG_KM, MIN_SEG_PTS = 10, 50     # split_by_landscape defaults
SWEEP_GRADIENT = (5, 8, 10, 12, 15, 20, 25, 30, 40)
SWEEP_MERGE_KM = (2.0, 5.0, 10.0)


def _tkey(t):
    s = str(t)
    return s[:-2] if s.endswith('.0') else s


def _dsname(csv_path):
    return re.sub(r'_w\d+km_window_stats\.csv$', '', os.path.basename(csv_path))


# ---------------------------------------------------------------- replay internals

def _gradient(elev, dist, smoothing_length=SMOOTHING_LENGTH):
    """Verbatim gradient stanza of split_by_landscape; gt/merge-gap independent, so cached."""
    _diffs = np.diff(dist)
    _pos = _diffs[_diffs > 0]
    min_step = float(np.median(_pos)) if _pos.size else 15.0
    grad_dist = dist.copy()
    for i in range(1, len(grad_dist)):
        if grad_dist[i] <= grad_dist[i - 1]:
            grad_dist[i] = grad_dist[i - 1] + min_step
    kernel_pts = int(smoothing_length / min_step)
    kernel_pts = max(3, kernel_pts if kernel_pts % 2 == 1 else kernel_pts + 1)
    smoothed = uniform_filter1d(elev, size=kernel_pts, mode='nearest')
    return np.gradient(smoothed, grad_dist / 1000)


def pieces(dist, grad, gradient_threshold, merge_gap_km,
           min_segment_km=MIN_SEG_KM, min_segment_pts=MIN_SEG_PTS):
    """split_by_landscape mirrored to expose index bounds, with merge_gap_km opened up.

    Returns (kept, dropped, fell_back). kept is [(s, e, is_transition)] into dist.
    Validated against the real function at production settings by control A.
    """
    n = len(dist)
    if n < 2:
        return [(0, n, False)], [], True

    in_transition = np.abs(grad) > gradient_threshold
    if not np.any(in_transition):
        return [(0, n, False)], [], True

    changes = np.diff(in_transition.astype(int))
    t_starts = np.where(changes == 1)[0] + 1
    t_ends = np.where(changes == -1)[0] + 1
    if in_transition[0]:
        t_starts = np.concatenate([[0], t_starts])
    if in_transition[-1]:
        t_ends = np.concatenate([t_ends, [len(in_transition)]])

    merged_starts, merged_ends = [t_starts[0]], [t_ends[0]]
    for s, e in zip(t_starts[1:], t_ends[1:]):
        if (dist[s] - dist[merged_ends[-1]]) / 1000 < merge_gap_km:
            merged_ends[-1] = e
        else:
            merged_starts.append(s)
            merged_ends.append(e)

    tset = {(int(s), int(e)) for s, e in zip(merged_starts, merged_ends)}
    boundaries = sorted({0, n} | {s for s, _ in tset} | {e for _, e in tset})

    kept, dropped = [], []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            continue
        rec = (s, e, (s, e) in tset)
        if e - s >= min_segment_pts and (dist[e - 1] - dist[s]) / 1000 >= min_segment_km:
            kept.append(rec)
        else:
            dropped.append(rec)

    if not kept:
        return [(0, n, False)], dropped, True
    return kept, dropped, False


def n_windows(dist, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Windows analyse_sliding_windows would emit, including its short-segment fallback."""
    lo, hi = float(dist.min()), float(dist.max())
    if hi - lo < window_size:
        window_size = step_size = hi - lo
    if window_size <= 0:
        return 0
    n, cur = 0, lo
    while cur + window_size <= hi + 1e-6:
        if int(np.sum((dist >= cur) & (dist <= cur + window_size))) > 50:
            n += 1
        cur += step_size
    return n


def gap_split(line, dist):
    """split_into_segments' index ranges, with each end labelled gap vs trajectory end."""
    gap_idx = np.where(np.diff(dist) > 2000)[0]
    pts = [0]
    for g in gap_idx:
        pts += [g + 1, g + 1]
    pts.append(len(dist))

    out = []
    for i in range(0, len(pts) - 1, 2):
        s, e = pts[i], pts[i + 1]
        if e - s >= 50 and (dist[e - 1] - dist[s]) / 1000 >= 10:
            out.append((s, e, 'traj_start' if s == 0 else 'gap',
                        'traj_end' if e == len(dist) else 'gap'))
    return out


# ---------------------------------------------------------------------- the replay

def replay():
    """One pass over the raw data: gap segments, cached gradients, production pieces."""
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    regions = []

    for bundle in load_datasets():
        name, df = bundle['name'], bundle['data']
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
        trajs = []

        for traj_id in valid['trajectory_id'].unique():
            line = valid[valid['trajectory_id'] == traj_id].copy()
            if len(line) < 20:
                continue
            x, y = tf.transform(line['longitude (degree_east)'].values,
                                line['latitude (degree_north)'].values)
            dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))])
            elev = line['bedrock_altitude (m)'].values

            parents = []
            for pi, (s, e, lb, rb) in enumerate(gap_split(line, dist)):
                d, ev = dist[s:e], elev[s:e]
                parents.append({'idx': pi, 'lo': s, 'hi': e, 'dist': d, 'elev': ev,
                                'left': lb, 'right': rb, 'grad': _gradient(ev, d)})
            if parents:
                trajs.append({'traj': _tkey(traj_id), 'line': line, 'dist': dist,
                              'parents': parents})

        regions.append({'dataset': name, 'trajs': trajs})
        print(f"  {name}: {len(trajs)} trajectories, "
              f"{sum(len(t['parents']) for t in trajs)} gap segments")

    return regions


def production_rows(regions, region_of):
    """Sub-segments under the production split, numbered as bed_analysis numbers them."""
    rows = []
    for R in regions:
        for T in R['trajs']:
            seg_no = 0
            for P in T['parents']:
                kept, dropped, fell_back = pieces(P['dist'], P['grad'],
                                                  PROD_GRADIENT, PROD_MERGE_KM)
                plen_km = (P['dist'][-1] - P['dist'][0]) / 1000
                for (s, e, is_tr) in kept:
                    seg_no += 1
                    d = P['dist'][s:e]
                    lkm = (d[-1] - d[0]) / 1000
                    left = 'transition' if s > 0 else P['left']
                    right = 'transition' if e < len(P['dist']) else P['right']
                    cut = (left == 'transition') or (right == 'transition')
                    rows.append({
                        'region': region_of.get(R['dataset'], '?'), 'dataset': R['dataset'],
                        'trajectory': T['traj'], 'segment': seg_no,
                        'length_km': lkm, 'n_points': e - s, 'is_transition': is_tr,
                        'parent': P['idx'], 'parent_length_km': plen_km,
                        'parent_left': P['left'], 'parent_right': P['right'],
                        'left_bound': left, 'right_bound': right,
                        'n_transition_bounds': int(left == 'transition') + int(right == 'transition'),
                        'parent_fallback': fell_back,
                        'n_pieces_dropped_in_parent': len(dropped),
                        'km_dropped_in_parent': sum((P['dist'][de - 1] - P['dist'][ds]) / 1000
                                                    for ds, de, _ in dropped),
                        'n_windows': n_windows(d),
                        'truncated': lkm * 1000 < WINDOW_SIZE,
                        'cut_created': bool(cut and plen_km * 1000 >= WINDOW_SIZE),
                    })
    return pd.DataFrame(rows)


def split_off_rows(regions, region_of):
    """Gap segments only: what the segment set would be with the landscape split off."""
    rows = []
    for R in regions:
        for T in R['trajs']:
            for P in T['parents']:
                lkm = (P['dist'][-1] - P['dist'][0]) / 1000
                rows.append({'region': region_of.get(R['dataset'], '?'), 'dataset': R['dataset'],
                             'trajectory': T['traj'], 'segment': P['idx'] + 1,
                             'length_km': lkm, 'n_points': len(P['dist']),
                             'left_bound': P['left'], 'right_bound': P['right'],
                             'n_windows': n_windows(P['dist']),
                             'truncated': lkm * 1000 < WINDOW_SIZE})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ controls

def control_mirror(regions):
    """A: the mirror must reproduce split_by_landscape piece for piece at (15 m/km, 5 km)."""
    print("\nCONTROL A - mirror vs segmentation.split_by_landscape at production settings")
    ok = True
    for R in regions:
        n_seg, n_bad = 0, 0
        for T in R['trajs']:
            for P in T['parents']:
                sub = T['line'].iloc[P['lo']:P['hi']]
                with redirect_stdout(io.StringIO()):        # the real function is chatty
                    ref = split_by_landscape(sub, P['dist'])
                mine = pieces(P['dist'], P['grad'], PROD_GRADIENT, PROD_MERGE_KM)[0]
                n_seg += len(ref)
                if len(ref) != len(mine):
                    n_bad += 1
                    continue
                for (_, rd, rt), (s, e, mt) in zip(ref, mine):
                    if len(rd) != e - s or rt != mt or \
                            abs((rd[-1] - rd[0]) - (P['dist'][e - 1] - P['dist'][s])) > 1e-6:
                        n_bad += 1
                        break
        ok &= n_bad == 0
        print(f"  {R['dataset']:<48s} {n_seg:5d} pieces, {n_bad} mismatched parents")
    print(f"  -> {'PASS' if ok else 'FAIL - the sweep below is not trustworthy'}")
    return ok


def control_lengths(prod, ref_path):
    """B: replayed lengths must equal production's cached segment_lengths.csv exactly."""
    print(f"\nCONTROL B - replayed segment lengths vs {ref_path}")
    if not os.path.exists(ref_path):
        print("  cached file absent; control skipped (run bed_character once to write it)")
        return False
    ref = pd.read_csv(ref_path, dtype={'trajectory': str})
    ref['trajectory'] = ref['trajectory'].map(_tkey)
    m = ref.merge(prod[['dataset', 'trajectory', 'segment', 'length_km']],
                  on=['dataset', 'trajectory', 'segment'], how='outer', indicator=True)
    both = m[m['_merge'] == 'both']
    d = (both['length_m'] - both['length_km'] * 1000).abs()
    n_off = int((d > 1e-6).sum())
    print(f"  {len(both)} of {len(ref)} cached rows matched, "
          f"{int((m['_merge'] == 'left_only').sum())} cached-only, "
          f"{int((m['_merge'] == 'right_only').sum())} replay-only")
    print(f"  max |length difference| = {d.max() if len(d) else float('nan'):.3e} m, "
          f"{n_off} rows over 1e-6 m")
    ok = n_off == 0 and (m['_merge'] == 'both').all()
    print(f"  -> {'PASS - the replay reproduces production segmentation exactly' if ok else 'FAIL'}")
    return ok


def control_windows(prod, win):
    """C: geometric window counts vs the production CSVs, which also carry the REMA filters."""
    print("\nCONTROL C - geometric window count vs production window_stats.csv")
    print("  (production additionally drops segments with <20% valid ice thickness and")
    print("   windows whose beta is NaN, so a shortfall here is expected and is quantified)")
    print(f"  {'region':<8s} {'geom':>6s} {'prod':>6s} {'delta':>7s}  {'geom trunc':>10s} {'prod trunc':>10s}")
    for reg, g in prod.groupby('region'):
        w = win.get(reg)
        if w is None:
            continue
        gk = g[~g['is_transition']]
        pk = w[~w['is_transition'].astype(bool)]
        gt = gk.loc[gk['truncated'], 'n_windows'].sum()
        L = dict(zip(zip(g['trajectory'], g['segment']), g['length_km']))
        pl = np.array([L.get((_tkey(t), int(s)), np.nan) for t, s in
                       zip(pk['trajectory'], pk['segment'])], float)
        print(f"  {reg:<8s} {gk['n_windows'].sum():6d} {len(pk):6d} "
              f"{gk['n_windows'].sum() - len(pk):+7d}  "
              f"{gt / max(gk['n_windows'].sum(), 1):9.0%} "
              f"{np.mean(pl[np.isfinite(pl)] * 1000 < WINDOW_SIZE):9.0%}")


# ------------------------------------------------------------------------ measures

def describe(v, label):
    q = np.percentile(v, [10, 25, 50, 75, 90]) if len(v) else [np.nan] * 5
    print(f"    {label:<10s} n={len(v):4d}  min={min(v, default=np.nan):6.1f}  "
          f"p10={q[0]:6.1f}  p25={q[1]:6.1f}  med={q[2]:6.1f}  p75={q[3]:6.1f}  "
          f"p90={q[4]:6.1f}  max={max(v, default=np.nan):7.1f} km  "
          f"<50km: {np.mean(np.asarray(v) < 50) if len(v) else np.nan:.0%}")


def sweep_row(regions, region_of, gt, mg):
    """Truncation and exclusion at one (gradient threshold, merge gap) setting."""
    acc = {}
    for R in regions:
        reg = region_of.get(R['dataset'], R['dataset'])
        a = acc.setdefault(reg, dict(n_seg=0, w_tot=0, w_trans=0, w_ret=0, w_trunc=0,
                                     km_dropped=0.0, n_dropped=0, n_zones=0))
        for T in R['trajs']:
            for P in T['parents']:
                kept, dropped, _ = pieces(P['dist'], P['grad'], gt, mg)
                a['n_dropped'] += len(dropped)
                a['km_dropped'] += sum((P['dist'][de - 1] - P['dist'][ds]) / 1000
                                       for ds, de, _ in dropped)
                a['n_zones'] += sum(1 for *_, t in kept if t) + sum(1 for *_, t in dropped if t)
                for s, e, is_tr in kept:
                    d = P['dist'][s:e]
                    nw = n_windows(d)
                    a['n_seg'] += 1
                    a['w_tot'] += nw
                    if is_tr:
                        a['w_trans'] += nw
                    else:
                        a['w_ret'] += nw
                        if (d[-1] - d[0]) < WINDOW_SIZE:
                            a['w_trunc'] += nw
    return [{'region': r, 'gradient_threshold': gt, 'merge_gap_km': mg,
             'n_segments': v['n_seg'], 'n_transition_zones': v['n_zones'],
             'n_windows_total': v['w_tot'], 'n_windows_transition': v['w_trans'],
             'n_windows_retained': v['w_ret'], 'n_windows_truncated': v['w_trunc'],
             'truncated_frac': v['w_trunc'] / max(v['w_ret'], 1),
             'excluded_frac': v['w_trans'] / max(v['w_tot'], 1),
             'n_pieces_dropped': v['n_dropped'], 'km_dropped': v['km_dropped']}
            for r, v in acc.items()]


def figure(prod, off, path):
    """Left column counts segments, right column weights each by the windows it carries.

    The truncated fractions are window-weighted, so the right column is the one that
    matches them; the left shows where the length mass sits regardless of window yield.
    """
    regs = sorted(prod['region'].unique())
    hi = max(prod['length_km'].max(), off['length_km'].max())
    bins = np.logspace(np.log10(MIN_SEG_KM * 0.8), np.log10(hi * 1.1), 30)

    fig, axes = plt.subplots(len(regs), 2, figsize=(10, 1.7 * len(regs)),
                             sharex=True, squeeze=False)
    for row, r in zip(axes, regs):
        p, o = prod[prod.region == r], off[off.region == r]
        for ax, wp, wo in ((row[0], None, None),
                           (row[1], p['n_windows'], o['n_windows'])):
            ax.hist(o['length_km'], bins=bins, weights=wo, color='0.75',
                    label='split off (gaps only)')
            ax.hist(p['length_km'], bins=bins, weights=wp, histtype='step',
                    color='C3', lw=1.4, label='production (split on)')
            ax.axvline(WINDOW_SIZE / 1000, color='k', ls='--', lw=0.9)
            ax.set_xscale('log')
        row[0].set_ylabel(r, fontsize=9)

    axes[0][0].set_title('segments per length bin', fontsize=9)
    axes[0][1].set_title('windows per length bin', fontsize=9)
    axes[0][1].legend(fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel('segment length (km), dashed line = 50 km window')
    fig.supylabel('count', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  figure -> {path}")


# ---------------------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    if FIGURE_ONLY:
        # Redraw from the written CSVs. No Tee: the log belongs to the measuring run.
        figure(pd.read_csv(os.path.join(OUT, 'segments_production.csv')),
               pd.read_csv(os.path.join(OUT, 'segments_split_off.csv')),
               os.path.join(OUT, 'segment_length_distributions.png'))
        return
    sys.stdout = Tee(os.path.join(OUT, 'transition_cut_truncation_log.txt'))

    win, region_of = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, '*', 'window_csvs', '*_window_stats.csv'))):
        reg = os.path.basename(os.path.dirname(os.path.dirname(f)))
        region_of[_dsname(f)] = reg
        win[reg] = pd.read_csv(f, dtype={'trajectory': str}).dropna(subset=['beta'])

    print(f"Production window CSVs under {ROOT}: {len(win)} regions")
    print(f"Replaying segmentation at GRADIENT_THRESHOLD={PROD_GRADIENT} m/km, "
          f"merge gap={PROD_MERGE_KM} km, WINDOW_SIZE={WINDOW_SIZE / 1000:.0f} km\n")

    regions = replay()
    prod = production_rows(regions, region_of)
    off = split_off_rows(regions, region_of)
    prod.to_csv(os.path.join(OUT, 'segments_production.csv'), index=False)
    off.to_csv(os.path.join(OUT, 'segments_split_off.csv'), index=False)

    control_mirror(regions)
    control_lengths(prod, os.path.join(ROOT, 'segment_lengths.csv'))
    control_windows(prod, win)

    summary = []
    for reg in sorted(prod['region'].unique()):
        p, o = prod[prod.region == reg], off[off.region == reg]
        pk = p[~p['is_transition']]
        short = pk[pk['truncated']]

        print(f"\n=== {reg} ===")
        print("  segment-length distribution")
        describe(p['length_km'].values, 'split on')
        describe(o['length_km'].values, 'split off')

        print("  what bounds each sub-50 km retained segment")
        n_s = len(short)
        for k, lab in [(2, 'transition both ends'), (1, 'transition one end'),
                       (0, 'gap / trajectory end only')]:
            sel = short[short['n_transition_bounds'] == k]
            print(f"    {lab:<26s} {len(sel):4d} segs ({len(sel) / max(n_s, 1):4.0%}), "
                  f"{sel['n_windows'].sum():4d} windows")
        cut = short[short['cut_created']]
        print(f"    of which cut from a parent already >=50 km (cut-created): "
              f"{len(cut)} segs ({len(cut) / max(n_s, 1):.0%}), {cut['n_windows'].sum()} windows")
        print(f"    inherited from a parent already <50 km: "
              f"{len(short) - len(cut)} segs, "
              f"{short['n_windows'].sum() - cut['n_windows'].sum()} windows")

        w = win.get(reg)
        prod_frac = np.nan
        if w is not None:
            wk = w[~w['is_transition'].astype(bool)]
            L = dict(zip(zip(p['trajectory'], p['segment']), p['length_km']))
            Lv = np.array([L.get((_tkey(t), int(s)), np.nan) for t, s in
                           zip(wk['trajectory'], wk['segment'])], float)
            k = np.isfinite(Lv)
            prod_frac = float(np.mean(Lv[k] * 1000 < WINDOW_SIZE))

        on_ret = pk['n_windows'].sum()
        on_trunc = short['n_windows'].sum()
        on_all, on_all_tr = p['n_windows'].sum(), p.loc[p['truncated'], 'n_windows'].sum()
        off_all, off_tr = o['n_windows'].sum(), o.loc[o['truncated'], 'n_windows'].sum()

        print("  truncated window fraction")
        print(f"    production CSV, retained basis   {prod_frac:5.0%}   (the published value)")
        print(f"    replay, split on, retained basis  {on_trunc / max(on_ret, 1):4.0%}   "
              f"({on_trunc} of {on_ret} windows)")
        print(f"    replay, split on, all windows     {on_all_tr / max(on_all, 1):4.0%}   "
              f"({on_all_tr} of {on_all}, {p['n_windows'].sum() - on_ret} in transition zones)")
        print(f"    replay, split OFF, all windows    {off_tr / max(off_all, 1):4.0%}   "
              f"({off_tr} of {off_all}) <- counterfactual")
        km_lost = p.groupby(['trajectory', 'parent'])['km_dropped_in_parent'].first().sum()
        print(f"    track length lost to the min-length gates under the split: {km_lost:.0f} km")

        summary.append({
            'region': reg, 'n_segments_on': len(p), 'n_segments_off': len(o),
            'median_length_on_km': p['length_km'].median(),
            'median_length_off_km': o['length_km'].median(),
            'frac_segments_sub50_on': float((p['length_km'] < 50).mean()),
            'frac_segments_sub50_off': float((o['length_km'] < 50).mean()),
            'trunc_frac_production_csv': prod_frac,
            'trunc_frac_replay_on_retained': on_trunc / max(on_ret, 1),
            'trunc_frac_replay_on_all': on_all_tr / max(on_all, 1),
            'trunc_frac_replay_off_all': off_tr / max(off_all, 1),
            'n_short_segments': n_s,
            'n_short_cut_created': len(cut),
            'frac_short_cut_created': len(cut) / max(n_s, 1),
            'frac_short_windows_cut_created': cut['n_windows'].sum() / max(on_trunc, 1),
            'n_windows_excluded_transition': int(on_all - on_ret),
        })

    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(OUT, 'region_summary.csv'), index=False)

    print("\n=== summary: is the cut a material cause of truncation? ===")
    print(f"{'region':<8s} {'prod':>6s} {'on(all)':>8s} {'off(all)':>9s} {'delta':>7s} "
          f"{'cut-created windows':>20s}")
    for r in summary:
        print(f"{r['region']:<8s} {r['trunc_frac_production_csv']:6.0%} "
              f"{r['trunc_frac_replay_on_all']:8.0%} {r['trunc_frac_replay_off_all']:9.0%} "
              f"{r['trunc_frac_replay_off_all'] - r['trunc_frac_replay_on_all']:+7.0%} "
              f"{r['frac_short_windows_cut_created']:20.0%}")

    print("\n=== GRADIENT_THRESHOLD x zone-merge-gap sweep ===")
    print("truncated_frac is over retained windows; excluded_frac is transition windows"
          " over all windows.")
    rows = []
    for gt in SWEEP_GRADIENT:
        for mg in SWEEP_MERGE_KM:
            rows += sweep_row(regions, region_of, gt, mg)
    sw = pd.DataFrame(rows)
    base = []
    for reg, g in off.groupby('region'):
        tot, tr = g['n_windows'].sum(), g.loc[g['truncated'], 'n_windows'].sum()
        base.append({'region': reg, 'gradient_threshold': np.inf, 'merge_gap_km': np.nan,
                     'n_segments': len(g), 'n_transition_zones': 0, 'n_windows_total': tot,
                     'n_windows_transition': 0, 'n_windows_retained': tot,
                     'n_windows_truncated': tr, 'truncated_frac': tr / max(tot, 1),
                     'excluded_frac': 0.0, 'n_pieces_dropped': 0, 'km_dropped': 0.0})
    sw = pd.concat([sw, pd.DataFrame(base)], ignore_index=True)
    sw.to_csv(os.path.join(OUT, 'gradient_merge_sweep.csv'), index=False)

    for reg in sorted(sw['region'].unique()):
        g = sw[sw.region == reg]
        print(f"\n  {reg}   (gradient=inf is the split-off baseline)")
        print(f"    {'grad':>5s} {'merge':>6s} {'segs':>5s} {'zones':>6s} {'w_tot':>6s} "
              f"{'w_excl':>7s} {'w_trunc':>8s} {'trunc%':>7s} {'excl%':>6s} {'km_lost':>8s}")
        for _, r in g.sort_values(['gradient_threshold', 'merge_gap_km']).iterrows():
            print(f"    {r.gradient_threshold:5.0f} {r.merge_gap_km:6.1f} {r.n_segments:5.0f} "
                  f"{r.n_transition_zones:6.0f} {r.n_windows_total:6.0f} "
                  f"{r.n_windows_transition:7.0f} {r.n_windows_truncated:8.0f} "
                  f"{r.truncated_frac:7.0%} {r.excluded_frac:6.0%} {r.km_dropped:8.0f}")

    figure(prod, off, os.path.join(OUT, 'segment_length_distributions.png'))
    print(f"\nCSVs and log -> {OUT}")


if __name__ == '__main__':
    main()
