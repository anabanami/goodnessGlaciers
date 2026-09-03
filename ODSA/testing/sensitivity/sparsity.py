"""Per-region band-grid sparsity for single-window (truncated) segments.
N_band = # of geomspace(1/L, 1/(2dx), 500) points with wavelength in [250,50000].
±5-bin peak buffer removes up to (2*5+1)=11 bins -> fraction 11/N_band per peak.

Run from v23/; writes results to v23/TESTING_LANDSCAPE_SPLITTING/."""
import numpy as np, pandas as pd, os, re
from pyproj import Transformer
import sys
HERE = os.path.dirname(os.path.abspath(__file__))            # .../v23
ODSA = os.path.dirname(HERE)                                 # .../ODSA — current codebase + results
OUT = os.path.join(HERE, "TESTING_LANDSCAPE_SPLITTING")      # this script's results folder
sys.path.insert(0, ODSA)
from loading import load_datasets
from segmentation import split_into_segments, split_by_landscape
from config import WINDOW_SIZE, FIT_BAND_M, Tee
RESULTS = os.path.join(ODSA, "Ockenden-regions")  # most recent codebase run
os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, "sparsity_log.txt"))

# --- Drift guard. nband() below reproduces the grid of analyse_sliding_windows
# (bed_analysis.py); the band comes from config, the grid bins and dx floor are still
# inline literals there, so read those two from source and warn (non-fatal) on drift.
BAND_MIN, BAND_MAX = FIT_BAND_M
def _prodval(pat, cast=float):
    with open(os.path.join(ODSA, "bed_analysis.py")) as _f: src = _f.read()
    m = re.search(pat, src); return cast(m.group(1)) if m else None
for _label, _mine, _prod in [("grid bins", 500,   _prodval(r"geomspace\([^)]*num=(\d+)", int)),
                             ("dx floor",  15.0,  _prodval(r"max\(dx_median,\s*([\d.]+)\)"))]:
    if _prod is None: print(f"NOTE: could not cross-check {_label} against bed_analysis.py.")
    elif _mine != _prod: print(f"WARNING: nband {_label}={_mine} but bed_analysis.py={_prod} — nband is STALE.")

tf = Transformer.from_crs("EPSG:4326","EPSG:3031",always_xy=True)
# per-segment (L, dx_median)
info={}
for d in load_datasets():
    name,df=d['name'],d['data']
    valid=df[(df['bedrock_altitude (m)']!=-9999)&(df['trajectory_id']!=-9999)]
    for t in valid['trajectory_id'].unique():
        line=valid[valid['trajectory_id']==t].copy()
        if len(line)<20: continue
        x,y=tf.transform(line['longitude (degree_east)'].values,line['latitude (degree_north)'].values)
        dist=np.concatenate([[0],np.cumsum(np.sqrt(np.diff(x)**2+np.diff(y)**2))])
        segs=[]
        for sd,sdist in split_into_segments(line.copy(),dist):
            segs.extend(split_by_landscape(sd,sdist))
        for i,(sdata,sdist,is_t) in enumerate(segs):
            L=sdist.max()-sdist.min()
            dx=np.median(np.diff(sdist)) if len(sdist)>1 else 100
            info[(name,str(t),i+1)]=(L,dx,is_t)

def nband(L,dx):
    ws=L if L<WINDOW_SIZE else WINDOW_SIZE
    mn,mx=1/ws,1/(2*max(dx,15.0))
    f=np.geomspace(mn,mx,500); wl=1/f
    return int(((wl>=BAND_MIN)&(wl<=BAND_MAX)).sum())

regions={'Aurora':'ASB_ICECAP_2010_Fig4C_Aurora_SB_lowrelief',
         'Pensacola':'POLARGAP_2015_Pensacola_Pole',
         'Hercules':'POLARGAP_2015_Fig2C_Hercules_Dome'}
for reg,dn in regions.items():
    seg=pd.read_csv(os.path.join(RESULTS,'segment_csvs',dn+'_w50km_segment_stats.csv'))
    single=seg[seg['n_windows']==1]
    Ls,dxs,nbs=[],[],[]
    for _,r in single.iterrows():
        key=(dn,str(r['trajectory']),int(r['segment']))
        if key in info:
            L,dx,_=info[key]; Ls.append(L); dxs.append(dx); nbs.append(nband(L,dx))
    Ls,dxs,nbs=map(np.array,(Ls,dxs,nbs))
    print(f"\n=== {reg}: {len(single)} single-win segs, matched {len(Ls)} ===")
    print(f"  L (km):  median {np.median(Ls)/1000:.1f}  range {Ls.min()/1000:.1f}-{Ls.max()/1000:.1f}")
    print(f"  dx (m):  median {np.median(dxs):.1f}")
    print(f"  N_band:  median {int(np.median(nbs))}  range {nbs.min()}-{nbs.max()}")
    print(f"  11-bin mask fraction: median {11/np.median(nbs)*100:.1f}%  worst {11/nbs.min()*100:.1f}%")
