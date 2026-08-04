"""Single-track version of hillshade-region_plots.py — same figure, one trajectory."""
import importlib.util
import os
import sys
from config import Tee
from loading import load_datasets

# Hyphen in the filename means it can't be imported by name.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'hillshade_region_plots', os.path.join(_HERE, 'hillshade-region_plots.py'))
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)

hs.OUTPUT_BASE_PATH = os.path.join(hs.OUTPUT_BASE_PATH, 'single_track/')


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
        raise SystemExit("usage: python hillshade_track_plots.py <region_label> [trajectory_id]")

    region_label = sys.argv[1]
    if len(sys.argv) < 3:  # no track given -> just list what's in the region
        ds = next((d for d in load_datasets() if d['name'] == region_label), None)
        if ds is None:
            raise SystemExit(f"No region named {region_label!r}")
        for tid in sorted(map(str, ds['data']['trajectory_id'].unique())):
            print(tid)
        raise SystemExit(0)

    traj_id = sys.argv[2]
    os.makedirs(hs.OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(os.path.join(hs.OUTPUT_BASE_PATH,
                                 f'hillshade_{region_label}_{traj_id}_log.txt'))

    hs.main(track_dataset(region_label, traj_id))
