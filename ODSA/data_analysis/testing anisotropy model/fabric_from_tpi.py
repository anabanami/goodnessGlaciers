"""Build a fabric lattice from the per tile output of tpi_bedforms.py.

Features are filtered on size and elongation, binned onto the lattice the rake reads,
and averaged as axes, giving one bearing and one concentration per occupied cell.
"""

import argparse
import glob

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("features", help="directory of per tile CSVs, or a glob")
    p.add_argument("out")
    p.add_argument("--step", type=float, default=5000.0)
    p.add_argument("--min-length", type=float, default=300.0)
    p.add_argument("--max-width", type=float, default=900.0)
    p.add_argument("--min-eratio", type=float, default=2.0)
    p.add_argument("--min-n", type=int, default=2, help="features needed to fill a cell")
    p.add_argument("--min-r", type=float, default=0.7,
                   help="concentration a cell's own bearings must reach")
    a = p.parse_args()

    paths = sorted(glob.glob(a.features if "*" in a.features else f"{a.features}/*.csv"))
    d = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    print(f"{len(paths)} files, {len(d)} features")

    # Adjacent ArcticDEM tiles overlap by 200 m, so the same feature can appear twice.
    d = d.drop_duplicates(subset=["x", "y", "area_m2"])

    keep = ((d.mbg_length > a.min_length) & (d.mbg_width < a.max_width)
            & (d.eratio > a.min_eratio))
    d = d[keep]
    print(f"{len(d)} after length > {a.min_length}, width < {a.max_width}, "
          f"elongation > {a.min_eratio}")

    i = np.floor(d.x.values / a.step).astype(np.int64)
    j = np.floor(d.y.values / a.step).astype(np.int64)
    t = np.deg2rad(2 * d.bearing_deg.values)
    g = pd.DataFrame({"i": i, "j": j, "s": np.sin(t), "c": np.cos(t)}).groupby(["i", "j"])
    f = pd.DataFrame({"s": g.s.mean(), "c": g.c.mean(), "n": g.size()}).reset_index()
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
