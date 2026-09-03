"""Sample count per 50 km window, for seeds raked under a site mask.

The rake keeps a window if it holds more than 50 samples, out of the 5000 a full
window holds at 10 m spacing. This reports the distribution of that count.

    python window_coverage.py masked_seeds/site_f
    python window_coverage.py masked_seeds/site_f --seeds 5

"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from azimuth_rake import chord, offsets, sample
from config import WINDOW_SIZE, STEP_SIZE

HERE = Path(__file__).resolve().parent


def window_counts(seed_dir, spacing_m=5000.0, step_m=10.0):
    """Sample count of every window the rake would cut from this seed."""
    name = seed_dir.name
    with rasterio.open(seed_dir / f'{name}_dem.tif') as src:
        arr = src.read(1).astype(np.float32)
        bad = ~np.isfinite(arr)
        if src.nodata is not None:
            bad |= arr == src.nodata
        water = seed_dir / f'{name}_water.tif'
        if water.exists():
            with rasterio.open(water) as w:
                bad |= w.read(1) > 0
        bad = bad.astype(np.uint8) if bad.any() else None
        counts = []
        for bearing in np.arange(0, 180, 5.0):
            for off in offsets(src.bounds, bearing, spacing_m):
                ends = chord(src.bounds, bearing, off)
                if ends is None:
                    continue
                dist, _ = sample(src, arr, bad, *ends, step_m)
                if len(dist) < 50:
                    continue
                start = dist.min()
                while start + WINDOW_SIZE <= dist.max() + 1e-6:
                    m = (dist >= start) & (dist <= start + WINDOW_SIZE)
                    counts.append(int(m.sum()))
                    start += STEP_SIZE
    return np.array(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('batch', help='a directory of seeds, e.g. masked_seeds/site_f')
    p.add_argument('--seeds', type=int, default=3, help='how many seeds to read')
    a = p.parse_args()

    seeds = sorted(d for d in (HERE / a.batch).iterdir() if d.is_dir())[:a.seeds]
    counts = np.concatenate([window_counts(d) for d in seeds])

    kept = counts[counts > 50]
    full = WINDOW_SIZE / 10.0
    print(f'{len(seeds)} seeds, {len(counts)} windows cut, {len(kept)} kept')
    print(f'coverage of a kept window, as a fraction of {full:.0f} samples')
    for q in (0, 1, 5, 25, 50):
        print(f'  p{q:<3d} {np.percentile(kept, q) / full:6.1%}   '
              f'{np.percentile(kept, q):8.0f} samples')
    for edge in (51, 100, 250, 500):
        print(f'  under {edge:4d} samples   {(kept < edge).sum():5d} windows '
              f'({(kept < edge).mean():.1%})')


if __name__ == '__main__':
    main()
