"""Regression check for the single-window fix (landscape_splitting_findings §code-change 2).

Invariant: every single-window segment (n_windows == 1) must report a segment-CSV
beta equal to the beta of its one window in the window CSV to within a tight numeric
tolerance (1e-9). This proves the segment fit bypasses the degenerate two-pass and
uses the window first-pass OLS — the degeneracy artifact was +0.17 to +0.48, so 1e-9
catches it by ~8 orders of magnitude. The two CSVs can differ at the 1-ULP (~1e-16)
level from downstream float plumbing (segment β and window CSV β both trace to the
same window_beta but reach the CSVs by different paths); that is not a violation.
Joins on (trajectory, segment) and requires a unique window match. Exits nonzero on
any violation so it can run as a test.

Run from v23/; writes results to v23/TESTING_LANDSCAPE_SPLITTING/."""
import csv, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))            # .../v23
ODSA = os.path.dirname(HERE)                                 # .../ODSA — current codebase + results
OUT = os.path.join(HERE, "TESTING_LANDSCAPE_SPLITTING")      # this script's results folder
sys.path.insert(0, ODSA)
from config import Tee
from loading import OUTPUT_BASE_PATH
RESULTS = OUTPUT_BASE_PATH
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "verify_single_window_invariant_log.txt"))

TOL = 1e-9  # numeric tolerance — immune to 1-ULP CSV-plumbing noise, still catches the degeneracy


def beta_match(seg_s, win_list):
    """Unique window match, segment β == window β to within TOL (exact-string fallback for non-numeric)."""
    if len(win_list) != 1:
        return False
    try:
        return abs(float(seg_s) - float(win_list[0])) < TOL
    except ValueError:
        return seg_s == win_list[0]


stem = lambda p: os.path.basename(p).replace("_segment_stats.csv", "").replace("_window_stats.csv", "")


def csvs(sub, pattern):
    """Both tree layouts: flat <root>/<sub>/ and per-region <root>/<region>/<sub>/."""
    return (sorted(glob.glob(os.path.join(RESULTS, sub, pattern))) or
            sorted(glob.glob(os.path.join(RESULTS, '*', sub, pattern))))


seg_csvs = csvs("segment_csvs", "*_segment_stats.csv")
wmap = {stem(p): p for p in csvs("window_csvs", "*_window_stats.csv")}
# An empty run reports zero violations and exits 0, which reads as a pass. Refuse instead.
if not seg_csvs:
    sys.exit(f"No *_segment_stats.csv under {RESULTS}: nothing to check, so this is not a pass.")

tot_sw = tot_mm = 0
for sp in seg_csvs:
    region = stem(sp)
    if region not in wmap:
        sys.exit(f"{region}: segment CSV with no matching window CSV under {RESULTS}.")
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
            if not beta_match(row[bi], wb):  # unique match + equal within TOL
                mm += 1
                if len(examples) < 5: examples.append((row[ti], row[si], row[bi], wb, row[tr]))
    tot_sw += sw; tot_mm += mm
    print(f"{region}: single-window segs={sw}, mismatches={mm}  {'OK' if mm == 0 else 'FAIL'}")
    for e in examples: print("   seg_beta!=win_beta:", e)

print(f"\nTOTAL single-window segs={tot_sw}, mismatches={tot_mm}  ->  "
      f"{'INVARIANT HOLDS' if tot_mm == 0 else 'INVARIANT VIOLATED'}")
sys.exit(1 if tot_mm else 0)
