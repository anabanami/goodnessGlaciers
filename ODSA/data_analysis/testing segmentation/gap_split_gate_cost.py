"""How much track do the segment gates discard at the gap split, before any landscape cut?

    python "testing segmentation/gap_split_gate_cost.py" [output_root]

Geometry only: replays loading.py's production input through split_into_segments' own
splitting rule and records every piece it produces, including the pieces that the 10 km
and 50-point gates discard, which the function itself drops silently. No beta is computed
and nothing in the pipeline is touched.

This is the gap-split half of the segment-gate cost. transition_cut_truncation.py measures
the track the same two gates take after split_by_landscape has cut a parent, so the two
partition the loss: the pieces here are the ones that never reach the landscape split.

Per region it reports the track presented to the gates, the track retained, the track each
gate discards, the track inside the >2 km flight gaps, and the trajectories dropped by
bed_analysis' 20-point floor. The retained count is controlled against split_into_segments
itself, trajectory by trajectory.

Writes region_summary.csv, pieces.csv and one log into
<output_root>/tests-results/gap_split_gate_cost/.
"""
import glob, inspect, io, os, re, sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee, MIN_SEGMENT_POINTS
from loading import load_datasets, OUTPUT_BASE_PATH as _REGION_BASE
from segmentation import split_into_segments

_pos = [a for a in sys.argv[1:] if not a.startswith('-')]
ROOT = _pos[0] if _pos else _REGION_BASE
OUT = os.path.join(ROOT, 'tests-results', 'gap_split_gate_cost')

GAP_THRESHOLD_M = 2000   # split_into_segments' gap_threshold default
MIN_SEG_KM = 10          # split_into_segments' min_segment_km default
MIN_TRAJ_PTS = 20        # bed_analysis.analyse_bedrock's trajectory floor


def check_mirrors():
    """Warn if a mirrored default has drifted from split_into_segments."""
    d = {k: v.default for k, v in inspect.signature(split_into_segments).parameters.items()
         if v.default is not inspect.Parameter.empty}
    for name, mine, theirs in (('gap_threshold', GAP_THRESHOLD_M, d.get('gap_threshold')),
                               ('min_segment_km', MIN_SEG_KM, d.get('min_segment_km')),
                               ('min_segment_length', MIN_SEGMENT_POINTS, d.get('min_segment_length'))):
        if mine != theirs:
            print(f"  WARNING: {name} mirror is {mine}, split_into_segments uses {theirs}")


def pieces_of(dist):
    """split_into_segments' index ranges, before the gates are applied."""
    pts = [0]
    for g in np.where(np.diff(dist) > GAP_THRESHOLD_M)[0]:
        pts += [g + 1, g + 1]
    pts.append(len(dist))
    return [(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]


def verdict(n_pts, length_km):
    pts_ok, km_ok = n_pts >= MIN_SEGMENT_POINTS, length_km >= MIN_SEG_KM
    if pts_ok and km_ok:
        return 'kept'
    if not pts_ok and not km_ok:
        return 'dropped_both'
    return 'dropped_few_points' if not pts_ok else 'dropped_short'


def region_codes():
    """Dataset name to region folder, read from the run tree's window CSVs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, '*', 'window_csvs', '*_window_stats.csv'))):
        ds = re.sub(r'_w\d+km_window_stats\.csv$', '', os.path.basename(f))
        out[ds] = os.path.basename(os.path.dirname(os.path.dirname(f)))
    return out


def replay():
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    code = region_codes()
    rows, traj_rows, mismatches = [], [], 0

    for bundle in load_datasets():
        name, df = bundle['name'], bundle['data']
        region = code.get(name, name)
        valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]

        for traj_id in valid['trajectory_id'].unique():
            line = valid[valid['trajectory_id'] == traj_id].copy()
            x, y = tf.transform(line['longitude (degree_east)'].values,
                                line['latitude (degree_north)'].values)
            dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            span_km = (dist[-1] - dist[0]) / 1000 if len(dist) > 1 else 0.0

            if len(line) < MIN_TRAJ_PTS:
                traj_rows.append({'region': region, 'trajectory': str(traj_id),
                                  'n_points': len(line), 'span_km': span_km})
                continue

            kept = 0
            for pi, (s, e) in enumerate(pieces_of(dist)):
                length_km = (dist[e - 1] - dist[s]) / 1000
                v = verdict(e - s, length_km)
                kept += v == 'kept'
                rows.append({'region': region, 'dataset': name, 'trajectory': str(traj_id),
                             'piece': pi, 'n_points': e - s, 'length_km': length_km,
                             'verdict': v, 'span_km': span_km})

            with redirect_stdout(io.StringIO()):
                n_real = len(split_into_segments(line, dist))
            mismatches += n_real != kept

    return pd.DataFrame(rows), pd.DataFrame(traj_rows), mismatches


def summarise(pieces, trajs):
    out = []
    for region, g in pieces.groupby('region'):
        km = g.groupby('verdict')['length_km'].sum()
        n = g['verdict'].value_counts()
        t = trajs[trajs.region == region] if len(trajs) else trajs
        span = g.drop_duplicates(['trajectory'])['span_km'].sum()
        dropped_km = float(km.drop('kept', errors='ignore').sum())
        out.append({
            'region': region,
            'n_trajectories': g['trajectory'].nunique(),
            'n_traj_under_20_pts': len(t),
            'km_traj_under_20_pts': float(t['span_km'].sum()) if len(t) else 0.0,
            'n_pieces': len(g),
            'n_kept': int(n.get('kept', 0)),
            'n_dropped_short': int(n.get('dropped_short', 0)),
            'n_dropped_few_points': int(n.get('dropped_few_points', 0)),
            'n_dropped_both': int(n.get('dropped_both', 0)),
            'km_trajectory_span': float(span),
            'km_in_gaps': float(span - g['length_km'].sum()),
            'km_kept': float(km.get('kept', 0.0)),
            'km_dropped_short': float(km.get('dropped_short', 0.0)),
            'km_dropped_few_points': float(km.get('dropped_few_points', 0.0)),
            'km_dropped_both': float(km.get('dropped_both', 0.0)),
            'km_dropped_total': dropped_km,
            'frac_track_dropped': dropped_km / g['length_km'].sum() if g['length_km'].sum() else np.nan,
        })
    return pd.DataFrame(out).sort_values('km_dropped_total', ascending=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUT, 'gap_split_gate_cost_log.txt'))

    print(f"Run tree: {ROOT}")
    print(f"Gates: >= {MIN_SEG_KM} km and >= {MIN_SEGMENT_POINTS} points, "
          f"on pieces split at gaps > {GAP_THRESHOLD_M / 1000:.0f} km\n")
    check_mirrors()

    pieces, trajs, mismatches = replay()
    summary = summarise(pieces, trajs)

    pieces.to_csv(os.path.join(OUT, 'pieces.csv'), index=False)
    summary.to_csv(os.path.join(OUT, 'region_summary.csv'), index=False)

    print(f"\n{'region':10s} {'pieces':>7s} {'kept':>5s} {'km kept':>9s} "
          f"{'km dropped':>11s} {'% dropped':>10s} {'km in gaps':>11s}")
    for _, r in summary.iterrows():
        print(f"{r['region']:10s} {r['n_pieces']:7.0f} {r['n_kept']:5.0f} {r['km_kept']:9.1f} "
              f"{r['km_dropped_total']:11.1f} {100 * r['frac_track_dropped']:9.1f}% "
              f"{r['km_in_gaps']:11.1f}")

    print(f"\n{summary['km_dropped_total'].sum():.0f} km discarded by the gates at the gap split, "
          f"over {summary['n_pieces'].sum():.0f} pieces in {len(summary)} regions.")
    print(f"By cause: {summary['km_dropped_short'].sum():.0f} km under {MIN_SEG_KM} km only, "
          f"{summary['km_dropped_few_points'].sum():.0f} km under {MIN_SEGMENT_POINTS} points only, "
          f"{summary['km_dropped_both'].sum():.0f} km under both.")
    print(f"{summary['n_traj_under_20_pts'].sum():.0f} trajectories dropped by the "
          f"{MIN_TRAJ_PTS}-point floor, carrying {summary['km_traj_under_20_pts'].sum():.1f} km.")
    print(f"\nControl: {mismatches} trajectories where the retained count disagrees with "
          f"split_into_segments.")
    print(f"Written to {OUT}")


if __name__ == '__main__':
    main()
