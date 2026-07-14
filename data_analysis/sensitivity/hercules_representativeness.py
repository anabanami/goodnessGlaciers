"""Hercules representativeness: partition gap-free surveyed bed length into
homogeneous basin (enters regional beta) vs flank/transition (TZ) vs dropped
below the 10 km / 50 pt gate. Uses the real split_by_landscape with the gate
disabled to get every pre-gate piece + its is_transition flag, then applies the
gate to lengths. Survivor counts are cross-checked against the gated call.

Run from v23/; writes results to v23/TESTING_LANDSCAPE_SPLITTING/."""
import numpy as np
from pyproj import Transformer
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))            # .../v23
ODSA = os.path.dirname(HERE)                                 # .../ODSA — current codebase
OUT = os.path.join(HERE, "TESTING_LANDSCAPE_SPLITTING")      # this script's results folder
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import split_into_segments, split_by_landscape
from config import Tee
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "hercules_representativeness_log.txt"))

MIN_KM, MIN_PTS, HERC = 10, 50, 'POLARGAP_2015_Fig2C_Hercules_Dome'
tf = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

acc = dict(gapfree=0.0, basin_keep=0.0, flank_keep=0.0, basin_drop=0.0, flank_drop=0.0)
chk = {True: 0, False: 0}   # survivors via my gate
real = {True: 0, False: 0}  # survivors via real gated call

for d in load_datasets():
    if d['name'] != HERC: continue
    v = d['data'][(d['data']['bedrock_altitude (m)'] != -9999) &
                  (d['data']['trajectory_id'] != -9999)]
    for tid in v['trajectory_id'].unique():
        line = v[v['trajectory_id'] == tid].copy()
        if len(line) < 20: continue
        x, y = tf.transform(line['longitude (degree_east)'].values,
                            line['latitude (degree_north)'].values)
        dist = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        for sd, sdist in split_into_segments(line.copy(), dist):
            acc['gapfree'] += sdist.max() - sdist.min()
            for pdat, pdist, is_t in split_by_landscape(sd, sdist, min_segment_km=0, min_segment_pts=0):
                Lm = pdist.max() - pdist.min()
                keep = (len(pdist) >= MIN_PTS) and (Lm/1000 >= MIN_KM)
                acc[('flank' if is_t else 'basin') + ('_keep' if keep else '_drop')] += Lm
                if keep: chk[is_t] += 1
            for _, _, is_t in split_by_landscape(sd, sdist):
                real[is_t] += 1

km, G = lambda m: m/1000, acc['gapfree']
print(f"\nHercules gap-free surveyed length:      {km(G):7.1f} km  (100%)")
print(f"  homogeneous basin, ANALYSED (->beta): {km(acc['basin_keep']):7.1f} km  ({acc['basin_keep']/G*100:4.1f}%)")
print(f"  flank/transition, retained inventory: {km(acc['flank_keep']):7.1f} km  ({acc['flank_keep']/G*100:4.1f}%)")
print(f"  basin dropped (<10km between TZs):     {km(acc['basin_drop']):7.1f} km  ({acc['basin_drop']/G*100:4.1f}%)")
print(f"  flank dropped (narrow escarpment):    {km(acc['flank_drop']):7.1f} km  ({acc['flank_drop']/G*100:4.1f}%)")
analysable = acc['basin_keep'] + acc['flank_keep']
print(f"\n  of ANALYSABLE (kept) length: basin {acc['basin_keep']/analysable*100:.1f}% | flank {acc['flank_keep']/analysable*100:.1f}%")
tot_flank = acc['flank_keep'] + acc['flank_drop']
print(f"  of ALL detected flank length: retained {acc['flank_keep']/tot_flank*100:.1f}% | dropped {acc['flank_drop']/tot_flank*100:.1f}%")
print(f"\ncross-check survivor counts  mine={chk}  real={real}  MATCH={chk==real}")
