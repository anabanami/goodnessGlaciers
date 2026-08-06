"""Single-track version of map_flightlines.py — same maps, one trajectory instead of the region."""
import os
import sys
import map_flightlines as mf
from config import Tee
from loading import load_datasets

# No padding override: frame_bounds_ps scales with the track's own span.
mf.OUTPUT_BASE_PATH = os.path.join(mf.OUTPUT_BASE_PATH, 'single_track/')


def track_dataset(region_label, traj_id):
    ds = next((d for d in load_datasets() if d['name'] == region_label), None)
    if ds is None:
        raise SystemExit(f"No region named {region_label!r} (check the active entries in loading.py)")

    df = ds['data']
    sub = df[df['trajectory_id'].astype(str) == str(traj_id)].copy()
    if sub.empty:
        avail = sorted(map(str, df['trajectory_id'].unique()))
        raise SystemExit(f"No trajectory {traj_id!r} in {region_label}. Available:\n  " + "\n  ".join(avail))

    # tier_key keeps the coverage-tier lookup pointed at the parent region
    return {'name': f'{region_label}_{traj_id}', 'data': sub, 'tier_key': region_label}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python map_flightlines_track.py <region_label> [trajectory_id]")

    region_label = sys.argv[1]
    if len(sys.argv) < 3:  # no track given -> just list what's in the region
        ds = next((d for d in load_datasets() if d['name'] == region_label), None)
        if ds is None:
            raise SystemExit(f"No region named {region_label!r}")
        for tid in sorted(map(str, ds['data']['trajectory_id'].unique())):
            print(tid)
        raise SystemExit(0)

    traj_id = sys.argv[2]
    os.makedirs(mf.OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(os.path.join(mf.OUTPUT_BASE_PATH,
                                 f'map_flightlines_{region_label}_{traj_id}_log.txt'))

    mf.main([track_dataset(region_label, traj_id)], prefix=f'{region_label}_{traj_id}_')
