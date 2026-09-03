"""Turn a site's downloads into the three files azimuth_rake.py reads.

Mosaics the DEM tiles onto one grid clipped to the AOI, builds the validity mask from
the strip count, the water polygons and the AOI outline, and clips the fabric lattice to
the same ground. Writes <name>_dem.tif, <name>_water.tif, <name>_fabric.csv and a QC
JSON into the output folder.

Tiles are read one at a time into the output window they cover, so peak memory is one
tile plus the mask rather than the whole mosaic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window

NODATA = -9999.0
STRIPE = 4096


def output_grid(tiles, bbox_m):
    """Grid covering the AOI, aligned to the first tile and clipped to the tile union."""
    with rasterio.open(tiles[0]) as src:
        res, crs, ref = src.res[0], src.crs, src.transform
    b = []
    for t in tiles:
        with rasterio.open(t) as src:
            b.append(src.bounds)
    left = max(min(x.left for x in b), bbox_m[0])
    bottom = max(min(x.bottom for x in b), bbox_m[1])
    right = min(max(x.right for x in b), bbox_m[2])
    top = min(max(x.top for x in b), bbox_m[3])
    if right <= left or top <= bottom:
        raise SystemExit("the AOI does not overlap the tiles")
    left = ref.c + round((left - ref.c) / res) * res
    top = ref.f - round((ref.f - top) / res) * res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    return from_origin(left, top, res, res), width, height, crs


def overlap(src, transform, width, height):
    """Matching windows in a source raster and in the output grid, or None."""
    grid = windows.bounds(Window(0, 0, width, height), transform)
    b = src.bounds
    left, bottom = max(b.left, grid[0]), max(b.bottom, grid[1])
    right, top = min(b.right, grid[2]), min(b.top, grid[3])
    if right <= left or top <= bottom:
        return None
    w_src = windows.from_bounds(left, bottom, right, top, src.transform)
    w_dst = windows.from_bounds(left, bottom, right, top, transform)
    w_src = w_src.round_offsets().round_lengths()
    w_dst = w_dst.round_offsets().round_lengths()
    rows = min(w_src.height, w_dst.height)
    cols = min(w_src.width, w_dst.width)
    return (Window(w_src.col_off, w_src.row_off, cols, rows),
            Window(w_dst.col_off, w_dst.row_off, cols, rows))


def write_dem(tiles, out, transform, width, height, crs, mask, log):
    """Mosaic the tiles onto the grid, clearing the mask wherever a tile has data."""
    profile = dict(driver="GTiff", dtype="float32", count=1, nodata=NODATA,
                   width=width, height=height, transform=transform, crs=crs,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(out, "w", **profile) as dst:
        for r0 in range(0, height, STRIPE):
            rows = min(STRIPE, height - r0)
            dst.write(np.full((rows, width), NODATA, "float32"), 1,
                      window=Window(0, r0, width, rows))
        for path in tiles:
            with rasterio.open(path) as src:
                w = overlap(src, transform, width, height)
                if w is None:
                    continue
                a = src.read(1, window=w[0]).astype("float32")
                nd = src.nodata if src.nodata is not None else NODATA
                good = np.isfinite(a) & (a != nd)
                dst.write(np.where(good, a, NODATA), 1, window=w[1])
                s = (slice(int(w[1].row_off), int(w[1].row_off + w[1].height)),
                     slice(int(w[1].col_off), int(w[1].col_off + w[1].width)))
                mask[s] &= ~good
            log(f"  {Path(path).name}: {good.mean():.2%} of its overlap has data")


def write_count(counts, out, transform, width, height, crs, mask, min_count, log):
    """Mosaic the strip count onto the grid and mask cells below min_count."""
    with rasterio.open(counts[0]) as src:
        dtype = src.dtypes[0]
    profile = dict(driver="GTiff", dtype=dtype, count=1, nodata=0, width=width,
                   height=height, transform=transform, crs=crs, compress="deflate",
                   tiled=True, blockxsize=512, blockysize=512)
    low = np.zeros((height, width), bool)
    with rasterio.open(out, "w", **profile) as dst:
        for path in counts:
            with rasterio.open(path) as src:
                w = overlap(src, transform, width, height)
                if w is None:
                    continue
                a = src.read(1, window=w[0])
                dst.write(a, 1, window=w[1])
                s = (slice(int(w[1].row_off), int(w[1].row_off + w[1].height)),
                     slice(int(w[1].col_off), int(w[1].col_off + w[1].width)))
                low[s] |= a < min_count
            log(f"  {Path(path).name}: {(a < min_count).mean():.2%} below "
                f"{min_count} strips")
    mask |= low
    return float(low.mean())


def read_vector(spec, crs, bbox=None):
    """Read a vector file, keeping one attribute value where the spec asks for it.

    A spec is a path, or a path and a selector as `path::FIELD=VALUE`.
    """
    import geopandas as gpd
    path, _, where = spec.partition("::")
    g = gpd.read_file(path, bbox=bbox)
    if where:
        field, _, value = where.partition("=")
        g = g[g[field].astype(str) == value]
    g = g.to_crs(crs)
    return g[g.geometry.notna() & ~g.geometry.is_empty]


def burn_vector(specs, transform, crs, mask, outside=False):
    """Mask cells inside the polygons, or outside them when outside is set."""
    import geopandas as gpd
    from shapely.geometry import box
    height, width = mask.shape
    bounds = windows.bounds(Window(0, 0, width, height), transform)
    bbox = gpd.GeoSeries([box(*bounds)], crs=crs)
    geoms = []
    for spec in specs:
        geoms.extend(read_vector(spec, crs, bbox).geometry.values)
    if not geoms:
        return 0.0
    hit = 0
    for r0 in range(0, height, STRIPE):
        rows = min(STRIPE, height - r0)
        t = windows.transform(Window(0, r0, width, rows), transform)
        burned = rasterize(geoms, out_shape=(rows, width), transform=t,
                           fill=0, default_value=1, dtype="uint8").astype(bool)
        if outside:
            burned = ~burned
        mask[r0:r0 + rows] |= burned
        hit += int(burned.sum())
    return hit / (height * width)


def dem_stats(path, mask):
    """Elevation statistics over the surviving cells."""
    n, total, total2 = 0, 0.0, 0.0
    lo, hi = np.inf, -np.inf
    with rasterio.open(path) as src:
        for r0 in range(0, src.height, STRIPE):
            rows = min(STRIPE, src.height - r0)
            a = src.read(1, window=Window(0, r0, src.width, rows))
            v = a[~mask[r0:r0 + rows]]
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            n += v.size
            total += float(v.sum())
            total2 += float(np.square(v.astype(np.float64)).sum())
            lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
    if not n:
        return {}
    mean = total / n
    return dict(n=n, mean=mean, std=float(np.sqrt(max(total2 / n - mean * mean, 0.0))),
                min=lo, max=hi)


def reproject_fabric(d, src_crs, crs):
    """Move the lattice to the DEM's CRS, taking each bearing from a short segment
    through its node so that the grid convergence between the two is carried."""
    from pyproj import Transformer
    t = Transformer.from_crs(src_crs, crs, always_xy=True)
    step = 100.0
    a = np.deg2rad(d.bearing_deg.values)
    dx, dy = step * np.sin(a), step * np.cos(a)
    x0, y0 = t.transform(d.x.values - dx, d.y.values - dy)
    x1, y1 = t.transform(d.x.values + dx, d.y.values + dy)
    x, y = t.transform(d.x.values, d.y.values)
    out = d.copy()
    out["x"], out["y"] = x, y
    out["bearing_deg"] = np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 180.0
    return out


def clip_fabric(path, out, bbox_m, aoi_paths, crs, src_crs=None):
    """Keep the lattice cells inside the AOI, and summarise what is left."""
    import pandas as pd
    d = pd.read_csv(path)
    if src_crs and str(src_crs) != str(crs):
        d = reproject_fabric(d, src_crs, crs)
    d = d[(d.x >= bbox_m[0]) & (d.x <= bbox_m[2])
          & (d.y >= bbox_m[1]) & (d.y <= bbox_m[3])]
    if aoi_paths:
        import geopandas as gpd
        area = None
        for spec in aoi_paths:
            g = read_vector(spec, crs)
            area = g.union_all() if area is None else area.intersection(g.union_all())
        pts = gpd.GeoSeries(gpd.points_from_xy(d.x, d.y), crs=crs)
        d = d[pts.within(area).values]
    d.to_csv(out, index=False)
    t = np.deg2rad(2 * d.bearing_deg.values)
    s, c = float(np.sin(t).mean()), float(np.cos(t).mean())
    return dict(cells=len(d), median_n=float(d.n.median()), median_R=float(d.R.median()),
                site_bearing_deg=float(np.rad2deg(np.arctan2(s, c)) / 2 % 180),
                site_R=float(np.hypot(s, c)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--name", help="file stem the rake reads; defaults to the folder name")
    p.add_argument("--dem", nargs="+", required=True)
    p.add_argument("--count", nargs="*", default=[])
    p.add_argument("--water", action="append", default=[])
    p.add_argument("--aoi", action="append", default=[],
                   help="polygon file bounding the AOI, or path::FIELD=VALUE to keep one "
                        "row; repeatable and intersected")
    p.add_argument("--bbox", type=float, nargs=4, required=True,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="in km")
    p.add_argument("--fabric", help="lattice CSV to clip to the AOI")
    p.add_argument("--fabric-crs", help="CRS of the lattice CSV, if not the DEM's")
    p.add_argument("--min-count", type=int, default=1)
    a = p.parse_args()

    def log(m):
        print(m, flush=True)

    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    name = a.name or d.name
    bbox_m = [v * 1e3 for v in a.bbox]

    transform, width, height, crs = output_grid(a.dem, bbox_m)
    log(f"{name}: {width} x {height} cells at {transform.a:.0f} m, {crs}")
    mask = np.ones((height, width), bool)

    write_dem(a.dem, d / f"{name}_dem.tif", transform, width, height, crs, mask, log)
    cumulative = {"nodata": float(mask.mean())}
    alone = dict(cumulative)
    log(f"nodata and ground no tile covers: {cumulative['nodata']:.2%}")

    if a.count:
        alone["count"] = write_count(a.count, d / f"{name}_count.tif", transform, width,
                                     height, crs, mask, a.min_count, log)
        cumulative["count"] = float(mask.mean())
        log(f"after the strip count: {cumulative['count']:.2%} masked")
    if a.water:
        alone["water"] = burn_vector(a.water, transform, crs, mask)
        cumulative["water"] = float(mask.mean())
        log(f"after water: {cumulative['water']:.2%} masked")
    if a.aoi:
        outside = np.zeros_like(mask)
        for path in a.aoi:
            burn_vector([path], transform, crs, outside, outside=True)
        mask |= outside
        alone["aoi"] = float(outside.mean())
        cumulative["aoi"] = float(mask.mean())
        log(f"after the AOI outline: {cumulative['aoi']:.2%} masked")
    qc = {"masked_cumulative": cumulative, "masked_by_cause": alone}

    profile = dict(driver="GTiff", dtype="uint8", count=1, nodata=0, width=width,
                   height=height, transform=transform, crs=crs, compress="deflate",
                   tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(d / f"{name}_water.tif", "w", **profile) as dst:
        for r0 in range(0, height, STRIPE):
            rows = min(STRIPE, height - r0)
            dst.write(mask[r0:r0 + rows].astype("uint8"), 1,
                      window=Window(0, r0, width, rows))

    res = transform.a
    qc.update(name=name, crs=str(crs), width=width, height=height, res_m=res,
              bounds=list(windows.bounds(Window(0, 0, width, height), transform)),
              masked=float(mask.mean()), good_km2=float((~mask).sum() * res * res / 1e6),
              min_count=a.min_count,
              dem=dem_stats(d / f"{name}_dem.tif", mask))
    log(f"kept {qc['good_km2']:.0f} km2, {1 - qc['masked']:.2%} of the grid")

    if a.fabric:
        qc["fabric"] = clip_fabric(a.fabric, d / f"{name}_fabric.csv", bbox_m, a.aoi,
                                   crs, a.fabric_crs)
        log(f"fabric: {qc['fabric']['cells']} cells, site R "
            f"{qc['fabric']['site_R']:.3f}")

    with open(d / f"{name}_qc.json", "w") as f:
        json.dump(qc, f, indent=2)
    log(f"wrote {d}/{name}_dem.tif, _water.tif, _count.tif, _fabric.csv, _qc.json")


if __name__ == "__main__":
    main()
