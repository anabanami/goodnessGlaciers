"""§5 spatial test, 9-region reproduction: do low-beta NO-split windows overlap a
transition zone? Footprint overlap (window i spans [i*STEP, i*STEP+WINDOW] along its
gap-segment) vs the pre-gate merged TZ extents from split_by_landscape's gradient detector."""
import numpy as np, pandas as pd, glob, os, sys
from scipy.ndimage import uniform_filter1d
from pyproj import Transformer
HERE = os.path.dirname(os.path.abspath(__file__))
ODSA = os.path.dirname(os.path.dirname(HERE))   # .../ODSA — current codebase
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import split_into_segments
from config import WINDOW_SIZE, STEP_SIZE, SMOOTHING_LENGTH, GRADIENT_THRESHOLD, Tee

# s5 tests a NO-landscape-splitting run; this is a separate test artifact, NOT the
# standard ODSA/Ockenden-regions (which IS landscape-split). Lives beside the script.
NOSPLIT = os.path.join(HERE, "Ockenden-regions-No_Landscape_splitting-TEST", "window_csvs")
sys.stdout = Tee(os.path.join(HERE, "s5_tz_overlap_log.txt"))
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
W, STEP = WINDOW_SIZE/1000, STEP_SIZE/1000  # km


def tz_extents_km(seg_data, seg_dist):
    """Pre-gate merged transition zones, as (start_km, end_km) relative to segment start.
    Replicates split_by_landscape's detector verbatim (5 km merge)."""
    elev = seg_data['bedrock_altitude (m)'].values
    dist = seg_dist.copy().astype(float)
    for i in range(1, len(dist)):
        if dist[i] <= dist[i-1]:
            dist[i] = dist[i-1] + 1e-3
    if len(dist) < 2:
        return []
    dx = np.median(np.diff(dist)) or 15.0
    kp = int(SMOOTHING_LENGTH/dx); kp = max(3, kp if kp % 2 == 1 else kp+1)
    grad = np.gradient(uniform_filter1d(elev, size=kp, mode='nearest'), dist/1000)
    intz = np.abs(grad) > GRADIENT_THRESHOLD
    if not np.any(intz):
        return []
    ch = np.diff(intz.astype(int))
    ts = np.where(ch == 1)[0] + 1
    te = np.where(ch == -1)[0] + 1
    if intz[0]:  ts = np.concatenate([[0], ts])
    if intz[-1]: te = np.concatenate([te, [len(intz)]])
    ms, me = [ts[0]], [te[0]]
    for s, e in zip(ts[1:], te[1:]):
        if (dist[s] - dist[me[-1]])/1000 < 5.0:
            me[-1] = e
        else:
            ms.append(s); me.append(e)
    d0 = dist[0]
    return [((dist[s]-d0)/1000, (dist[min(e, len(dist)-1)]-d0)/1000) for s, e in zip(ms, me)]


def overlap_km(a0, a1, extents):
    return sum(max(0, min(a1, e1) - max(a0, e0)) for e0, e1 in extents)


# region label -> reproduced {seg_num: tz_extents}
def region_tz(df):
    valid = df[(df['bedrock_altitude (m)'] != -9999) & (df['trajectory_id'] != -9999)]
    out = {}  # traj_id -> {seg_num: extents}
    for tid in valid['trajectory_id'].unique():
        line = valid[valid['trajectory_id'] == tid].copy()
        if len(line) < 20:
            continue
        x, y = transformer.transform(line['longitude (degree_east)'].values,
                                     line['latitude (degree_north)'].values)
        dist = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
        segs = split_into_segments(line, dist)
        out[str(tid)] = {i+1: tz_extents_km(sd, dd) for i, (sd, dd) in enumerate(segs)}
    return out


def csv_for(name):
    for f in glob.glob(os.path.join(NOSPLIT, "*.csv")):
        if name in os.path.basename(f):
            return f
    return None


tz_by_region = {}
for d in load_datasets():
    tz_by_region[d['name']] = region_tz(d['data'])

import numpy as np
BETA_T, RELIEF_T = 2.0, 800.0   # cliff signature: low beta AND mountainous relief (codebase RELIEF_CLASSES: mountainous >= 800 m)
rows = []  # (region, has_tz, beta, relief, overlap_frac)
for name, tzr in tz_by_region.items():
    f = csv_for(name)
    if f is None:
        continue
    df = pd.read_csv(f)
    has_tz = any(any(ext for ext in segs.values()) for segs in tzr.values())
    short = name.split('_')[-2][:8] + '/' + name.split('_')[-1][:6]
    for _, r in df.iterrows():
        extents = tzr.get(str(r['trajectory']), {}).get(int(r['segment']), [])
        a0 = int(r['window_id'])*STEP
        ov = overlap_km(a0, a0+W, extents)/W
        rows.append((short, has_tz, float(r['beta']), float(r['relief_m']), ov))

R = pd.DataFrame(rows, columns=['region', 'has_tz', 'beta', 'relief', 'ov'])
T = R[R.has_tz]  # TZ-bearing regions only

def rate(sub):
    return (sub.ov > 0).sum(), len(sub)

print(f"\n{'region':16} {'lowβ&hiRelief':>13} {'overlap':>8} {'%':>5} | {'lowβ only':>9} {'%':>5}")
for reg in T.region.unique():
    s = T[T.region == reg]
    cliff = s[(s.beta < BETA_T) & (s.relief > RELIEF_T)]
    low = s[s.beta < BETA_T]
    co, ct = rate(cliff); lo, lt = rate(low)
    print(f"{reg:16} {ct:13d} {co:8d} {100*co/ct if ct else float('nan'):5.0f} | "
          f"{lt:9d} {100*lo/lt if lt else float('nan'):5.0f}")

cliffT = T[(T.beta < BETA_T) & (T.relief > RELIEF_T)]
lowT = T[T.beta < BETA_T]
print(f"\nTZ-bearing regions, pooled:")
print(f"  low-β(<2.0) & high-relief(>{RELIEF_T:.0f} m): {rate(cliffT)[0]}/{rate(cliffT)[1]} "
      f"= {100*rate(cliffT)[0]/rate(cliffT)[1]:.1f}% overlap a TZ")
print(f"  low-β(<2.0) only:                       {rate(lowT)[0]}/{rate(lowT)[1]} "
      f"= {100*rate(lowT)[0]/rate(lowT)[1]:.1f}% overlap a TZ")
print(f"  median relief, low-β windows: overlap={lowT[lowT.ov>0].relief.median():.0f} m  "
      f"no-overlap={lowT[lowT.ov==0].relief.median():.0f} m")
print("(flat regions Maud/Wilhelm/Aurora excluded: 0 TZ, splitting is a no-op there)")
