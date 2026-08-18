"""Build analysis-ready grids and fabric fields for the validation sites.

Merges each site's 10 m ArcticDEM mosaic tiles into a VRT, clips DEM/mad/count to the mapped
bedform extent, rasterises a water mask, and reduces MBG_Orient to a reference bearing on a
regular grid. Bedforms and mosaic are both EPSG:3413, so nothing is reprojected but CanVec.
Sites with no mapped fabric (Dubawnt) get the VRT and QC only — Stage A needs a θ=0.

    python prep_sites.py                      # extent = bedform bounding box  -> prep/
    python prep_sites.py --hull               # extent = buffered bedform union -> prep_hull/
    python prep_sites.py --site "Site F Nunavut"

--hull writes alongside rather than over the bbox run, so the two extents can be compared.
It matters on coasts: M'Clintock's bbox reached into the channel and one connected water body
covered 25% of that grid, so masking followed azimuth. The hull excludes it by construction,
since bedforms are only mapped on land. That site is dropped and archived, but the flag stays
because any coastal site has the same problem.
"""
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import pandas as pd
import rasterio
from rasterio import features, windows
from scipy import spatial

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

# (kind, path). 'bedform' pools per-bedform MBG_Orient from the McKenzie shapefiles;
# 'flowline' takes tangents off mapped ice-stream flowlines, which is a different and more
# independent theta=0, since it is not derived from the surface being fitted.
FABRIC = {# "Site D M'Clintock": ('bedform', 'MClintockChannel_Canada.zip'),  # archive/README.md
          'Site E Prince of Wales': ('bedform', 'PrinceofWalesIsland_Canada.zip'),
          'Site F Nunavut': ('bedform', 'Nunavut_Canada.zip'),
          'Dubawnt': ('flowline', "Dubawnt/1033955/IS shp.zip/IS_flowline.shp")}

BANDS = ['dem', 'mad', 'count']
PAD_KM = 5.0            # margin around the mapped bedform extent
HULL_BUFFER_KM = 3.0    # --hull: each bedform buffered by this, then dissolved
GRID_KM = 5.0           # fabric node spacing
RADIUS_KM = 10.0        # bedforms pooled per fabric node
MIN_FABRIC_N = 5
MAD_MAX = 5.0           # m; stereo disagreement above this is treated as unreliable
MAX_AREA_KM2 = 20000    # beyond this, leave the VRT and skip materialising
FLOWLINE_BOX_KM = 100.0 # flowline sites: box sized to hold 50 km profiles at every azimuth
# Flowlines are sparse generalised lines, not a dense bedform cloud, so the local bearing
# comes from the nearest line rather than an average over many. Wider radius, no count gate.
FLOWLINE_RADIUS_KM = 25.0
FLOWLINE_MIN_N = 1
CRS = 3413              # ArcticDEM; McKenzie is already in it, Margold is EPSG:3978


def build_vrt(site, band, outdir):
    """Lazy mosaic of a site's tiles — cheap, and openable in QGIS."""
    tiles = sorted((site / 'mosaic_10m').glob(f'*_{band}.tif'))
    out = outdir / f'{site.name}_{band}.vrt'
    out.parent.mkdir(exist_ok=True)
    subprocess.run(['gdalbuildvrt', '-q', '-overwrite', str(out), *map(str, tiles)], check=True)
    return out, len(tiles)


def clip(vrt, bounds, dest):
    with rasterio.open(vrt) as src:
        w = windows.from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        w = w.intersection(windows.Window(0, 0, src.width, src.height))
        a = src.read(1, window=w)
        prof = src.profile | dict(height=w.height, width=w.width, driver='GTiff',
                                  transform=src.window_transform(w),
                                  compress='lzw', tiled=True, BIGTIFF='IF_SAFER')
    with rasterio.open(dest, 'w', **prof) as dst:
        dst.write(a, 1)
    return a, prof


def water_mask(bounds, prof, count, mad, dest, hull=None):
    """CanVec lakes, pixels the mosaic cannot vouch for, and anything outside the hull."""
    import pyproj
    tf = pyproj.Transformer.from_crs(3413, 4326, always_xy=True)
    lon, lat = zip(*[tf.transform(x, y) for x in bounds[::2] for y in bounds[1::2]])
    bbox = (min(lon), min(lat), max(lon), max(lat))

    polys = []
    for terr in ['NU', 'NT']:
        z = DATA / 'hydro' / f'canvec_250K_{terr}_Hydro_shp.zip'
        if not z.exists():
            continue
        src = f'/vsizip/{z}/canvec_250K_{terr}_Hydro/waterbody_2.shp'
        try:
            g = gpd.read_file(src, bbox=bbox)
        except Exception as e:                                  # noqa: BLE001
            print(f'    canvec {terr} unreadable: {e}')
            continue
        if len(g):
            polys.append(g.to_crs(3413))

    m = np.zeros((prof['height'], prof['width']), dtype=bool)
    n_poly = 0
    if polys:
        g = pd.concat(polys, ignore_index=True)
        n_poly = len(g)
        m = features.rasterize(g.geometry, out_shape=m.shape, transform=prof['transform'],
                               fill=0, default_value=1, dtype='uint8').astype(bool)

    unverified = (count < 1) | (mad > MAD_MAX)
    outside = np.zeros_like(m)
    if hull is not None:
        inside = features.rasterize([hull], out_shape=m.shape, transform=prof['transform'],
                                    fill=0, default_value=1, dtype='uint8').astype(bool)
        outside = ~inside
    mask = m | unverified | outside
    with rasterio.open(dest, 'w', **(prof | dict(dtype='uint8', nodata=None, count=1))) as dst:
        dst.write(mask.astype('uint8'), 1)
    return mask, m, unverified, outside, n_poly


def best_box(cx, cy, vb):
    """Sub-box holding the most fabric segments. Sized for 50 km profiles at every
    azimuth rather than to fill the memory cap."""
    side = min(FLOWLINE_BOX_KM * 1000, vb[2] - vb[0], vb[3] - vb[1])
    best, top = None, -1
    for x in np.linspace(vb[0], vb[2] - side, 25):
        for y in np.linspace(vb[1], vb[3] - side, 25):
            n = np.count_nonzero((cx >= x) & (cx < x + side) & (cy >= y) & (cy < y + side))
            if n > top:
                best, top = (x, y, x + side, y + side), n
    return np.array(best)


def bedform_fabric(zipname):
    """Per-bedform long axes from a McKenzie shapefile."""
    g = gpd.read_file(DATA / 'pangaea' / zipname).to_crs(CRS)
    c = g.geometry.centroid
    return (c.x.values, c.y.values, g.MBG_Orient.values.astype(float),
            g.MBG_Length.values.astype(float), g)


def flowline_fabric(rel, extent=None):
    """Ice-stream flowlines cut into segments: midpoint, tangent bearing, length."""
    g = gpd.read_file(f'/vsizip/{DATA / rel}').to_crs(CRS)
    if extent is not None:
        g = g[g.intersects(shapely.box(*extent))]
    cx, cy, ang, wgt = [], [], [], []
    for geom in g.geometry:
        for part in (geom.geoms if geom.geom_type.startswith('Multi') else [geom]):
            p = np.asarray(part.coords)[:, :2]
            d, m = np.diff(p, axis=0), (p[:-1] + p[1:]) / 2
            L = np.hypot(d[:, 0], d[:, 1])
            k = L > 0
            cx.append(m[k, 0]); cy.append(m[k, 1]); wgt.append(L[k])
            ang.append(np.degrees(np.arctan2(d[k, 0], d[k, 1])) % 180)
    if not cx:
        return np.array([]), np.array([]), np.array([]), np.array([]), g
    return (np.concatenate(cx), np.concatenate(cy), np.concatenate(ang),
            np.concatenate(wgt), g)


def fabric(cx, cy, ang_deg, wgt, bounds, dest, radius_km=RADIUS_KM, min_n=MIN_FABRIC_N):
    """Length-weighted axial mean on a regular grid — the local θ=0."""
    ang = np.radians(ang_deg) * 2                               # axial: double, average, halve
    r = radius_km * 1000

    xs = np.arange(bounds[0], bounds[2], GRID_KM * 1000)
    ys = np.arange(bounds[1], bounds[3], GRID_KM * 1000)
    rows = []
    for x in xs:
        for y in ys:
            sel = (np.abs(cx - x) < r) & (np.abs(cy - y) < r)
            sel &= (cx - x) ** 2 + (cy - y) ** 2 < r ** 2
            n = int(sel.sum())
            if n < min_n:
                continue
            w = wgt[sel]
            s, c = np.average(np.sin(ang[sel]), weights=w), np.average(np.cos(ang[sel]), weights=w)
            rows.append(dict(x=x, y=y, n=n, bearing_deg=np.degrees(np.arctan2(s, c)) / 2 % 180,
                             R=float(np.hypot(s, c))))
    d = pd.DataFrame(rows)
    d.to_csv(dest, index=False)
    return d


def run(site, use_hull=False):
    outdir = site / ('prep_hull' if use_hull else 'prep')
    outdir.mkdir(exist_ok=True)
    print(f'\n{site.name}')
    qc = {'site': site.name, 'extent_mode': 'hull' if use_hull else 'bbox'}

    vrts = {}
    for band in BANDS:
        vrts[band], n = build_vrt(site, band, outdir)
        qc[f'n_tiles_{band}'] = n
    print(f"  vrt      {qc['n_tiles_dem']} tiles per band")

    spec = FABRIC.get(site.name)
    if spec is None:
        with rasterio.open(vrts['dem']) as src:
            qc['vrt_bounds'] = list(src.bounds)
        qc['status'] = 'vrt only — no mapped fabric, Stage A cannot define theta=0'
        print(f"  SKIP     {qc['status']}")
        (outdir / 'qc.json').write_text(json.dumps(qc, indent=1))
        return qc

    kind, path = spec
    hull = None
    with rasterio.open(vrts['dem']) as src:
        vb = np.array(src.bounds)

    if kind == 'bedform':
        cx, cy, ang, wgt, g = bedform_fabric(path)
        if use_hull:
            # Buffer each bedform and dissolve: follows the mapped area instead of boxing it,
            # so coastal water outside the mapping is never in the grid to begin with.
            hull = g.geometry.buffer(HULL_BUFFER_KM * 1000).union_all()
            b = np.array(hull.bounds) + np.array([-1, -1, 1, 1]) * PAD_KM * 1000
            qc['hull_area_km2'] = round(hull.area / 1e6, 1)
        else:
            b = g.total_bounds + np.array([-1, -1, 1, 1]) * PAD_KM * 1000
        qc['n_bedforms'] = len(g)
    else:
        # Flowlines span the whole ice stream, so the mosaic is the binding extent, and it
        # is far over the cap. Take the sub-box holding the most flowline segments.
        cx, cy, ang, wgt, g = flowline_fabric(path, vb)
        if not len(cx):
            qc['status'] = 'no flowlines intersect the mosaic'
            print(f"  SKIP     {qc['status']}")
            (outdir / 'qc.json').write_text(json.dumps(qc, indent=1))
            return qc
        b = best_box(cx, cy, vb)
        qc |= {'n_flowlines': len(g), 'n_segments': int(len(cx)),
               'flowline_bounds': [round(float(v), 0) for v in
                                   (cx.min(), cy.min(), cx.max(), cy.max())]}
        print(f'  fabric   {len(g)} flowlines, {len(cx)} segments')

    area = (b[2] - b[0]) * (b[3] - b[1]) / 1e6
    qc |= {'fabric_kind': kind, 'bounds': list(map(float, b)), 'area_km2': round(area, 1)}
    extra = f", hull {qc['hull_area_km2']:.0f} km2" if use_hull else ''
    n_src = qc.get('n_bedforms', qc.get('n_segments'))
    print(f'  extent   {(b[2]-b[0])/1000:.0f} x {(b[3]-b[1])/1000:.0f} km, {n_src} sources{extra}')

    if area > MAX_AREA_KM2:
        qc['status'] = f'extent {area:.0f} km2 over cap — vrt only'
        print(f"  SKIP     {qc['status']}")
        (outdir / 'qc.json').write_text(json.dumps(qc, indent=1))
        return qc

    arrs = {}
    for band in BANDS:
        arrs[band], prof = clip(vrts[band], b, outdir / f'{site.name}_{band}.tif')
    dem = arrs['dem']
    nodata = dem == prof.get('nodata', -9999)
    print(f"  clipped  {prof['width']} x {prof['height']} px, {nodata.mean()*100:.1f}% nodata")

    mask, lakes, unver, outside, n_poly = water_mask(
        b, prof, arrs['count'], arrs['mad'], outdir / f'{site.name}_water.tif', hull)
    qc |= {'px': [prof['width'], prof['height']], 'nodata_frac': round(float(nodata.mean()), 4),
           'canvec_polys': n_poly, 'lake_frac': round(float(lakes.mean()), 4),
           'unverified_frac': round(float(unver.mean()), 4),
           'outside_hull_frac': round(float(outside.mean()), 4),
           'masked_frac': round(float((mask | nodata).mean()), 4)}
    good = arrs['mad'][~nodata]
    qc |= {'mad_median': round(float(np.median(good)), 3),
           'mad_p95': round(float(np.percentile(good, 95)), 3)}
    print(f"  water    {n_poly} lakes, masked {qc['masked_frac']*100:.1f}% "
          f"(mad median {qc['mad_median']} m, p95 {qc['mad_p95']} m)")

    rad, minn = ((FLOWLINE_RADIUS_KM, FLOWLINE_MIN_N) if kind == 'flowline'
                 else (RADIUS_KM, MIN_FABRIC_N))
    f = fabric(cx, cy, ang, wgt, b, outdir / f'{site.name}_fabric.csv', rad, minn)
    qc |= {'pool_radius_km': rad, 'pool_min_n': minn}
    qc |= {'fabric_nodes': len(f),
           'fabric_R_median': round(float(f.R.median()), 3) if len(f) else None}

    # Coverage: how far a node sits from its nearest fabric source. Beyond the pooling
    # radius the bearing is extrapolated, which is the failure mode that killed buffering
    # Site F. Reported rather than gated, since a smooth field can survive some of it.
    if len(f):
        gx = np.arange(b[0], b[2], GRID_KM * 1000)
        gy = np.arange(b[1], b[3], GRID_KM * 1000)
        nodes = np.array([(x, y) for x in gx for y in gy])
        d, _ = spatial.cKDTree(np.c_[cx, cy]).query(nodes)
        qc |= {'grid_nodes': len(nodes),
               'node_coverage_frac': round(len(f) / len(nodes), 3),
               'dist_to_source_km_median': round(float(np.median(d)) / 1000, 1),
               'dist_to_source_km_p95': round(float(np.percentile(d, 95)) / 1000, 1)}
        print(f"  coverage {qc['node_coverage_frac']*100:.0f}% of nodes, source distance "
              f"median {qc['dist_to_source_km_median']} km, p95 "
              f"{qc['dist_to_source_km_p95']} km (pooling radius {rad:.0f} km)")
    print(f"  fabric   {len(f)} nodes, median R {qc['fabric_R_median']}")

    qc['status'] = 'ready'
    (outdir / 'qc.json').write_text(json.dumps(qc, indent=1))
    return qc


if __name__ == '__main__':
    args = sys.argv[1:]
    want = args[args.index('--site') + 1] if '--site' in args else None
    use_hull = '--hull' in args
    sites = [s for s in sorted(DATA.iterdir())
             if (s / 'mosaic_10m').is_dir() and (want is None or s.name == want)]
    if not sites:
        sys.exit(f'no sites under {DATA}' + (f' matching {want!r}' if want else ''))

    rows = [run(s, use_hull) for s in sites]
    out = HERE / ('prep_summary_hull.csv' if use_hull else 'prep_summary.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{sum(r['status'] == 'ready' for r in rows)}/{len(rows)} ready -> {out}")
