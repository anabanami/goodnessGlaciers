"""Plot a prepped site's DEM over its AOI square, as elevation and as hillshade.

The hillshade panel is the one to compare against a satellite image, since it shows
the bedforms the fabric is mapped from. The mask in <name>_water.tif covers the ground
the rake does not see.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource
from pyproj import Transformer

CELL_M = 10.0


def read_decimated(path, stride):
    """Read a raster at one cell in `stride`, with its bounds in km."""
    with rasterio.open(path) as src:
        arr = src.read(1, out_shape=(src.height // stride, src.width // stride))
        b, nodata = src.bounds, src.nodata
    return arr, [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3], nodata


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("site_dir", help="folder holding <name>_dem.tif")
    p.add_argument("out")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--vert-exag", type=float, default=5.0)
    p.add_argument("--azdeg", type=float, default=315.0)
    a = p.parse_args()

    d = Path(a.site_dir)
    name = d.name
    dem, extent, nodata = read_decimated(d / f"{name}_dem.tif", a.stride)
    dem = dem.astype(np.float32)
    if nodata is not None:
        dem[dem == nodata] = np.nan
    mask, _, _ = read_decimated(d / f"{name}_water.tif", a.stride)
    masked = mask > 0

    good = dem[~masked & np.isfinite(dem)]
    lo, hi = np.percentile(good, [1, 99])

    # The light source needs a finite surface, so gaps take the mean elevation
    filled = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    cell = CELL_M * a.stride
    shade = LightSource(azdeg=a.azdeg, altdeg=45).hillshade(
        filled, vert_exag=a.vert_exag, dx=cell, dy=cell)

    grey = np.zeros(masked.shape + (4,))
    grey[masked] = [0.45, 0.45, 0.45, 0.85]

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
    im = axes[0].imshow(dem, extent=extent, cmap="terrain", vmin=lo, vmax=hi,
                        origin="upper", interpolation="nearest")
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="elevation, m")
    axes[0].set_title("elevation")
    axes[1].imshow(shade, extent=extent, cmap="gray", vmin=0, vmax=1,
                   origin="upper", interpolation="nearest")
    axes[1].set_title(f"hillshade, light from {a.azdeg:.0f} deg, "
                      f"vertical exaggeration {a.vert_exag:g}")

    x0, x1, y0, y1 = extent
    lon, lat = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True).transform(
        500.0 * (x0 + x1), 500.0 * (y0 + y1))
    for ax in axes:
        ax.imshow(grey, extent=extent, origin="upper", interpolation="nearest")
        ax.set_xlabel("EPSG:3413 x, km")
        ax.set_aspect("equal")
    axes[0].set_ylabel("EPSG:3413 y, km")
    fig.suptitle(f"{name}, {x1 - x0:.1f} km square, centre {lat:.4f} N {lon:.4f} E, "
                 f"{100 * masked.mean():.1f} % masked")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
