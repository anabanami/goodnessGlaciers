"""Regression check for the single-window fix (landscape_splitting_findings §code-change 2).

Invariant: every single-window segment (n_windows == 1) must report a segment-CSV
beta that is BYTE-IDENTICAL to the beta of its one window in the window CSV. This
proves the segment fit bypasses the degenerate two-pass and copies the window
first-pass OLS verbatim. Compares raw CSV strings (no float parse, no tolerance);
joins on (trajectory, segment) and requires a unique window match. Exits nonzero on
any violation so it can run as a test."""
import csv, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Tee
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "Ockenden-regions")  # results to verify live beside this script
sys.stdout = Tee(os.path.join(HERE, "verify_single_window_invariant_log.txt"))

stem = lambda p: os.path.basename(p).replace("_segment_stats.csv", "").replace("_window_stats.csv", "")
seg_csvs = sorted(glob.glob(os.path.join(RESULTS, "segment_csvs", "*_segment_stats.csv")))
wmap = {stem(p): p for p in glob.glob(os.path.join(RESULTS, "window_csvs", "*_window_stats.csv"))}

tot_sw = tot_mm = 0
for sp in seg_csvs:
    region = stem(sp)
    wbeta = {}  # (traj, segment) -> [raw beta strings]
    with open(wmap[region]) as f:
        r = csv.reader(f); h = next(r)
        ti, si, bi = h.index("trajectory"), h.index("segment"), h.index("beta")
        for row in r: wbeta.setdefault((row[ti], row[si]), []).append(row[bi])
    sw = mm = 0; examples = []
    with open(sp) as f:
        r = csv.reader(f); h = next(r)
        ti, si, bi, ni, tr = (h.index(c) for c in
                              ("trajectory", "segment", "beta", "n_windows", "is_transition"))
        for row in r:
            if row[ni].strip() != "1": continue
            sw += 1
            wb = wbeta.get((row[ti], row[si]), [])
            if not (len(wb) == 1 and wb[0] == row[bi]):  # unique + byte-identical
                mm += 1
                if len(examples) < 5: examples.append((row[ti], row[si], row[bi], wb, row[tr]))
    tot_sw += sw; tot_mm += mm
    print(f"{region}: single-window segs={sw}, mismatches={mm}  {'OK' if mm == 0 else 'FAIL'}")
    for e in examples: print("   seg_beta!=win_beta:", e)

print(f"\nTOTAL single-window segs={tot_sw}, mismatches={tot_mm}  ->  "
      f"{'INVARIANT HOLDS' if tot_mm == 0 else 'INVARIANT VIOLATED'}")
sys.exit(1 if tot_mm else 0)
