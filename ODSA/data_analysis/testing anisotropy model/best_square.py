"""Find the 102.4 km square holding the most usable ground in a prepped site.

The null beds are 102.4 km square, so a site raked on the same square feeds the rake the
same transect count and the same chord lengths. What then differs between a site and the
null is the mask and the fabric rather than the domain shape.

Scores candidate positions on surviving cells, and reports the fabric each one holds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

SIDE_M = 102400.0


def coarse_good(mask_path, factor):
    """Surviving cells per coarse block, read in stripes so the mask is never held whole."""
    with rasterio.open(mask_path) as src:
        h, w = src.height // factor, src.width // factor
        out = np.zeros((h, w), np.int32)
        for r in range(h):
            a = src.read(1, window=Window(0, r * factor, w * factor, factor))
            out[r] = (a == 0).reshape(factor, w, factor).sum(axis=(0, 2))
        return out, src.transform, src.crs


def integral(a):
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("site_dir", help="prepped folder holding <name>_water.tif")
    p.add_argument("--name")
    p.add_argument("--factor", type=int, default=64, help="mask cells per coarse block")
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args()

    d = Path(a.site_dir)
    name = a.name or d.name
    good, transform, crs = coarse_good(d / f"{name}_water.tif", a.factor)
    res = transform.a * a.factor
    side = int(round(SIDE_M / res))
    if side > min(good.shape):
        raise SystemExit(f"{name} is smaller than {SIDE_M/1e3:.1f} km")

    ii = integral(good)
    total = (ii[side:, side:] - ii[:-side, side:] - ii[side:, :-side] + ii[:-side, :-side])
    km2 = total * (transform.a ** 2) / 1e6

    fabric = pd.read_csv(d / f"{name}_fabric.csv")
    print(f"{name}: {side} x {side} blocks of {res:.0f} m, "
          f"{km2.max():.0f} km2 best of {SIDE_M**2/1e6:.0f}")

    taken = np.zeros_like(km2, bool)
    for k in range(a.top):
        flat = np.where(taken, -1.0, km2)
        r, c = np.unravel_index(np.argmax(flat), flat.shape)
        if flat[r, c] <= 0:
            break
        left = transform.c + c * res
        top = transform.f - r * res
        box = (left, top - SIDE_M, left + SIDE_M, top)
        m = ((fabric.x >= box[0]) & (fabric.x <= box[2])
             & (fabric.y >= box[1]) & (fabric.y <= box[3]))
        f = fabric[m]
        t = np.deg2rad(2 * f.bearing_deg.values)
        s, cc = np.sin(t).mean(), np.cos(t).mean()
        print(f"  {k+1}. bbox {box[0]/1e3:.1f} {box[1]/1e3:.1f} {box[2]/1e3:.1f} "
              f"{box[3]/1e3:.1f}   {km2[r, c]:.0f} km2 "
              f"({km2[r, c]/(SIDE_M**2/1e6):.0%})   {len(f)} nodes, "
              f"site R {np.hypot(s, cc):.3f}, median cell R {f.R.median():.3f}")
        # Suppress positions overlapping this one by more than half a side
        r0, r1 = max(0, r - side // 2), min(km2.shape[0], r + side // 2)
        c0, c1 = max(0, c - side // 2), min(km2.shape[1], c + side // 2)
        taken[r0:r1, c0:c1] = True


if __name__ == "__main__":
    main()
