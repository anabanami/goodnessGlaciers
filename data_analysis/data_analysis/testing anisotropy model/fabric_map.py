"""Map a fabric lattice as one oriented segment per cell, coloured by bearing.

Bearings are axes mod 180, so the colour scale is cyclic and a domain that shares a
direction reads as one colour block whichever way its segments point.
"""

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def tile_boxes(pattern):
    """Bounds of each raster matching the pattern, in km."""
    import rasterio
    out = []
    for path in sorted(glob.glob(pattern)):
        with rasterio.open(path) as src:
            b = src.bounds
        out.append((b.left / 1e3, b.bottom / 1e3, b.right / 1e3, b.top / 1e3))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fabric", help="lattice CSV with x, y, n, bearing_deg, R")
    p.add_argument("out")
    p.add_argument("--tiles", help="glob of DEM rasters to outline")
    p.add_argument("--reference", help="vector file to outline by its extent")
    p.add_argument("--step", type=float, default=5000.0)
    p.add_argument("--length", type=float, default=0.8,
                   help="segment length as a fraction of the lattice step")
    a = p.parse_args()

    d = pd.read_csv(a.fabric)
    x, y, b = d.x.values / 1e3, d.y.values / 1e3, d.bearing_deg.values
    half = 0.5 * a.length * a.step / 1e3
    dx, dy = half * np.sin(np.deg2rad(b)), half * np.cos(np.deg2rad(b))
    segments = np.stack([np.column_stack([x - dx, y - dy]),
                         np.column_stack([x + dx, y + dy])], axis=1)

    fig, ax = plt.subplots(figsize=(9, 9))
    lc = LineCollection(segments, array=b, cmap="twilight", clim=(0, 180),
                        linewidths=1.0 + 1.6 * d.R.values)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.7, ticks=[0, 45, 90, 135, 180],
                 label="bearing, degrees clockwise from north")

    for left, bottom, right, top in tile_boxes(a.tiles) if a.tiles else []:
        ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom,
                                   fill=False, ec="0.6", lw=0.8, zorder=0))
    if a.reference:
        import geopandas as gpd
        e = gpd.read_file(a.reference).to_crs("EPSG:3413").total_bounds / 1e3
        ax.add_patch(plt.Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1],
                                   fill=False, ec="crimson", lw=1.2, ls="--"))

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("EPSG:3413 x, km")
    ax.set_ylabel("EPSG:3413 y, km")
    ax.set_title(f"{a.fabric}, {len(d)} cells")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
