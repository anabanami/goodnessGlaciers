"""Single-track version of flow_plots.py — same figures, one trajectory instead of the region."""
# USAGE
# python flow_plots_track.py <region_label> [trajectory_id]    # <region_label> as per loading.py
# example:
# python flow_plots_track.py BM3_DML_3E_sq_ICECAP IR2HI2_2011042_TRL_JKB2d_EH1TROa


import os
import sys
import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
import flow_plots as fp
from config import Tee
from loading import load_datasets

# Tighter than the region default (50 km): a single track is a much smaller frame.
fp.ARROW_SPACING_KM = 15.0
fp.OUTPUT_BASE_PATH = os.path.join(fp.OUTPUT_BASE_PATH, 'single_track/')


def track_dataset(region_label, traj_id):
    ds = next((d for d in load_datasets() if d['name'] == region_label), None)
    if ds is None:
        raise SystemExit(f"No region named {region_label!r} (check the active entries in loading.py)")

    df = ds['data']
    sub = df[df['trajectory_id'].astype(str) == str(traj_id)].copy()
    if sub.empty:
        avail = sorted(map(str, df['trajectory_id'].unique()))
        raise SystemExit(f"No trajectory {traj_id!r} in {region_label}. Available:\n  " + "\n  ".join(avail))

    return {'name': f'{region_label}_{traj_id}', 'data': sub}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python flow_plots_track.py <region_label> [trajectory_id]")

    region_label = sys.argv[1]
    if len(sys.argv) < 3:  # no track given -> just list what's in the region
        ds = next((d for d in load_datasets() if d['name'] == region_label), None)
        if ds is None:
            raise SystemExit(f"No region named {region_label!r}")
        for tid in sorted(map(str, ds['data']['trajectory_id'].unique())):
            print(tid)
        raise SystemExit(0)

    traj_id = sys.argv[2]
    os.makedirs(fp.OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(os.path.join(fp.OUTPUT_BASE_PATH, f'flow_plots_{region_label}_{traj_id}_log.txt'))

    ds = track_dataset(region_label, traj_id)
    fp.main(ds)
    fp.plot_flow_confidence(ds)
