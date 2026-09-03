"""Build a fabric lattice from mapped landform lines.

Each line gives one bearing, taken from its endpoints after reprojection, and the lines
are binned onto the lattice the rake reads and averaged as axes. The gate matches
fabric_from_tpi.py, so the three sites share one lattice definition.
"""

import argparse

import numpy as np
import pandas as pd

FEATURES = ["Streamlined landform", "Crag-and-tail landform"]


def endpoints(geom):
    """First and last coordinate of a line, joining the parts of a multipart line."""
    if geom.geom_type == "LineString":
        c = list(geom.coords)
    else:
        c = [p for part in geom.geoms for p in part.coords]
    return c[0], c[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lines", help="vector file of landform lines")
    p.add_argument("out")
    p.add_argument("--crs", default="EPSG:3413", help="CRS of the DEM the rake reads")
    p.add_argument("--feature-field", default="Feature")
    p.add_argument("--feature", action="append", default=[],
                   help=f"value to keep; repeatable, defaults to {FEATURES}")
    p.add_argument("--min-length", type=float, default=0.0)
    p.add_argument("--step", type=float, default=5000.0)
    p.add_argument("--min-n", type=int, default=2, help="lines needed to fill a cell")
    p.add_argument("--min-r", type=float, default=0.7,
                   help="concentration a cell's own bearings must reach")
    a = p.parse_args()

    import geopandas as gpd
    g = gpd.read_file(a.lines).to_crs(a.crs)
    print(f"{len(g)} features in {a.lines}")

    keep = a.feature or FEATURES
    g = g[g[a.feature_field].isin(keep)]
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    print(f"{len(g)} after {a.feature_field} in {keep}")

    ends = np.array([np.concatenate(endpoints(geom)) for geom in g.geometry])
    dx, dy = ends[:, 2] - ends[:, 0], ends[:, 3] - ends[:, 1]
    length = np.hypot(dx, dy)
    m = length > a.min_length
    ends, dx, dy, length = ends[m], dx[m], dy[m], length[m]
    print(f"{m.sum()} after end to end length > {a.min_length}, "
          f"median {np.median(length):.0f} m")

    x = 0.5 * (ends[:, 0] + ends[:, 2])
    y = 0.5 * (ends[:, 1] + ends[:, 3])
    bearing = np.degrees(np.arctan2(dx, dy)) % 180.0

    t = np.deg2rad(2 * bearing)
    grp = pd.DataFrame({"i": np.floor(x / a.step).astype(np.int64),
                        "j": np.floor(y / a.step).astype(np.int64),
                        "s": np.sin(t), "c": np.cos(t)}).groupby(["i", "j"])
    f = pd.DataFrame({"s": grp.s.mean(), "c": grp.c.mean(),
                      "n": grp.size()}).reset_index()
    f = f[f.n >= a.min_n]
    f = f[np.hypot(f.s, f.c) >= a.min_r]

    out = pd.DataFrame({
        "x": (f.i + 0.5) * a.step,
        "y": (f.j + 0.5) * a.step,
        "n": f.n.astype(int),
        "bearing_deg": np.rad2deg(np.arctan2(f.s, f.c)) / 2 % 180,
        "R": np.hypot(f.s, f.c),
    }).sort_values(["x", "y"])
    out.to_csv(a.out, index=False)
    print(f"{len(out)} lattice cells, median n {out.n.median():.0f}, "
          f"median R {out.R.median():.3f}, {(out.R < 0.5).mean():.1%} below R 0.5")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
