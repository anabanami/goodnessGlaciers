"""Reimplementation of McKenzie's TPI_SemiAutomatedID for positive relief features.

Per annulus: tpi = Int(dem - focalmean(dem) + 0.5), standardised by the whole raster
mean and standard deviation, kept where the standardised value reaches 1. The four
annuli are unioned, labelled, and each blob gets a minimum rotated rectangle.
"""

import argparse
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from scipy.signal import oaconvolve
from shapely.geometry import MultiPoint

# Annulus radii in cells, from TPI_SemiAutomatedTool.tbx.
ANNULI = [(35, 41), (47, 53), (72, 78), (85, 91)]

# A cell needs this fraction of its annulus to be valid ground for a focal mean.
MIN_ANNULUS_COVERAGE = 0.10


def annulus_kernel(r_in, r_out):
    n = 2 * r_out + 1
    y, x = np.ogrid[-r_out:r_out + 1, -r_out:r_out + 1]
    d2 = x * x + y * y
    return ((d2 >= r_in * r_in) & (d2 <= r_out * r_out)).astype(np.float32).reshape(n, n)


def focal_mean(dem, valid, kernel):
    """Mean of the valid cells under the kernel, as ArcGIS FocalStatistics with DATA."""
    total = oaconvolve(np.where(valid, dem, 0.0).astype(np.float32), kernel, mode="same")
    count = oaconvolve(valid.astype(np.float32), kernel, mode="same")
    enough = count >= MIN_ANNULUS_COVERAGE * kernel.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(enough, total / np.maximum(count, 1.0), np.nan)
    return mean, enough


def scale_mask(dem, valid, r_in, r_out, log):
    """Cells at or above one standard deviation of the standardised TPI for one annulus."""
    t0 = time.time()
    mean, enough = focal_mean(dem, valid, annulus_kernel(r_in, r_out))
    tpi = np.trunc(dem - mean + 0.5)
    good = valid & enough & np.isfinite(tpi)
    v = tpi[good]
    mu, sd = float(v.mean()), float(v.std())
    mask = good & ((tpi - mu) >= sd)
    log(f"  annulus {r_in}-{r_out} cells: tpi mean {mu:.3f} sd {sd:.3f}, "
        f"{mask.sum() / good.sum():.2%} of ground kept, {time.time() - t0:.0f} s")
    return mask


def water_mask(paths, shape, transform, crs, bounds):
    """Cells inside a water polygon, from one or more vector files clipped to the raster."""
    import geopandas as gpd
    from shapely.geometry import box
    bbox = gpd.GeoSeries([box(*bounds)], crs=crs)
    out = np.zeros(shape, bool)
    for path in paths:
        g = gpd.read_file(path, bbox=bbox).to_crs(crs)
        g = g[g.geometry.notna() & ~g.geometry.is_empty]
        if g.empty:
            continue
        out |= rasterize(g.geometry, out_shape=shape, transform=transform,
                         fill=0, default_value=1, dtype="uint8").astype(bool)
    return out


def features(mask, scales, transform, dem, min_cells):
    """One minimum rotated rectangle per connected blob, with its bearing and size."""
    lab, n = ndimage.label(mask)
    counts = np.bincount(lab.ravel())
    keep = np.flatnonzero(counts >= min_cells)
    keep = keep[keep > 0]
    objects = ndimage.find_objects(lab)
    px = abs(transform.a)
    rows = []
    for i in keep:
        sl = objects[i - 1]
        sub = lab[sl] == i
        edge = sub & ~ndimage.binary_erosion(sub)
        ry, rx = np.nonzero(edge)
        xs = transform.c + (sl[1].start + rx + 0.5) * transform.a
        ys = transform.f + (sl[0].start + ry + 0.5) * transform.e
        rect = MultiPoint(np.column_stack([xs, ys])).minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            continue
        cx, cy = np.asarray(rect.exterior.coords[:4]).T
        e = np.hypot(np.diff(cx, append=cx[0]), np.diff(cy, append=cy[0]))
        long_i = int(np.argmax(e[:2]))
        dx = cx[long_i + 1] - cx[long_i]
        dy = cy[long_i + 1] - cy[long_i]
        length, width = e[long_i] + px, e[1 - long_i] + px
        z = dem[sl][sub]
        rows.append((int(i), xs.mean(), ys.mean(), counts[i] * px * px,
                     length, width, length / width,
                     np.degrees(np.arctan2(dx, dy)) % 180.0,
                     float(z.mean()), float(z.max() - z.min()),
                     int(ndimage.maximum(scales[sl], sub, 1))))
    return pd.DataFrame(rows, columns=["fid", "x", "y", "area_m2", "mbg_length",
                                       "mbg_width", "eratio", "bearing_deg",
                                       "z_mean", "deltaZ", "scales"]), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dem")
    p.add_argument("out")
    p.add_argument("--count", help="ArcticDEM count raster; cells at 0 are dropped")
    p.add_argument("--water", action="append", default=[],
                   help="vector file of water to drop before TPI; repeatable")
    p.add_argument("--min-cells", type=int, default=12)
    p.add_argument("--save-mask", help="write the unioned positive relief mask here")
    a = p.parse_args()

    def log(m):
        print(m, flush=True)

    t0 = time.time()
    with rasterio.open(a.dem) as src:
        dem = src.read(1).astype(np.float32)
        profile, transform, crs = src.profile, src.transform, src.crs
        valid = dem != (src.nodata if src.nodata is not None else -9999.0)
        bounds = tuple(src.bounds)
    log(f"{a.dem}: {dem.shape[1]} x {dem.shape[0]}, {valid.mean():.2%} not nodata")

    if a.count:
        with rasterio.open(a.count) as src:
            valid &= src.read(1) > 0
        log(f"  after the count raster: {valid.mean():.2%}")
    if a.water:
        valid &= ~water_mask(a.water, dem.shape, transform, crs, bounds)
        log(f"  after water: {valid.mean():.2%}")

    union = np.zeros(dem.shape, bool)
    scales = np.zeros(dem.shape, np.uint8)
    for bit, (r_in, r_out) in enumerate(ANNULI):
        m = scale_mask(dem, valid, r_in, r_out, log)
        union |= m
        scales |= (m * (1 << bit)).astype(np.uint8)
    log(f"union: {union.sum() / valid.sum():.2%} of ground, {time.time() - t0:.0f} s")

    if a.save_mask:
        profile.update(dtype="uint8", nodata=0, count=1, compress="deflate")
        with rasterio.open(a.save_mask, "w", **profile) as dst:
            dst.write(scales, 1)

    df, n = features(union, scales, transform, dem, a.min_cells)
    log(f"{n} blobs, {len(df)} at or above {a.min_cells} cells, {time.time() - t0:.0f} s")
    df.to_csv(a.out, index=False)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
