"""Check a fabric lattice: its coherence, how far a bearing transfers, and how well it
agrees with an independently mapped fabric binned onto the same lattice."""

import argparse

import numpy as np
import pandas as pd

SEPARATION_BINS = [(5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 200)]


def axis_diff(a, b):
    """Acute angle between two axes, in degrees."""
    d = np.abs(a - b) % 180.0
    return np.minimum(d, 180.0 - d)


def axis_mean(bearings):
    t = np.deg2rad(2 * np.asarray(bearings, float))
    s, c = np.sin(t).mean(), np.cos(t).mean()
    return np.rad2deg(np.arctan2(s, c)) / 2 % 180.0, float(np.hypot(s, c))


def lattice(x, y, bearing, step, min_n, min_r):
    """Bin bearings onto the lattice and average them as axes."""
    t = np.deg2rad(2 * np.asarray(bearing, float))
    g = pd.DataFrame({"i": np.floor(np.asarray(x) / step).astype(np.int64),
                      "j": np.floor(np.asarray(y) / step).astype(np.int64),
                      "s": np.sin(t), "c": np.cos(t)}).groupby(["i", "j"])
    f = pd.DataFrame({"s": g.s.mean(), "c": g.c.mean(), "n": g.size()}).reset_index()
    f = f[(f.n >= min_n) & (np.hypot(f.s, f.c) >= min_r)]
    return pd.DataFrame({
        "i": f.i, "j": f.j, "n": f.n.astype(int),
        "bearing_deg": np.rad2deg(np.arctan2(f.s, f.c)) / 2 % 180.0,
        "R": np.hypot(f.s, f.c),
    })


def separation_table(x, y, bearing, rng):
    """Median and p90 bearing difference against node separation, and against a
    shuffled pairing that carries no spatial information."""
    n = len(x)
    i, j = np.triu_indices(n, 1)
    sep = np.hypot(x[i] - x[j], y[i] - y[j]) / 1000.0
    diff = axis_diff(bearing[i], bearing[j])
    print(f"{'separation':>14} {'n pairs':>9} {'median':>8} {'p90':>7}")
    for lo, hi in SEPARATION_BINS:
        m = (sep >= lo) & (sep < hi)
        if m.sum() < 20:
            continue
        print(f"{lo:>7} to {hi:<4} {m.sum():>9} {np.median(diff[m]):>7.1f} "
              f"{np.percentile(diff[m], 90):>7.1f}")
    shuffled = axis_diff(bearing[i], rng.permutation(bearing)[j])
    print(f"{'unrelated':>14} {len(shuffled):>9} {np.median(shuffled):>7.1f} "
          f"{np.percentile(shuffled, 90):>7.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fabric", help="lattice CSV with x, y, n, bearing_deg, R")
    p.add_argument("--reference", help="vector file of an independently mapped fabric")
    p.add_argument("--bearing-field", default="MBG_Orient")
    p.add_argument("--step", type=float, default=5000.0)
    p.add_argument("--min-n", type=int, default=2)
    p.add_argument("--min-r", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="keep cells inside this box, in km")
    a = p.parse_args()

    d = pd.read_csv(a.fabric)
    if a.bbox:
        xmin, ymin, xmax, ymax = (v * 1e3 for v in a.bbox)
        d = d[(d.x >= xmin) & (d.x <= xmax) & (d.y >= ymin) & (d.y <= ymax)]
    x, y, b = d.x.values, d.y.values, d.bearing_deg.values
    mean, r = axis_mean(b)
    print(f"{len(d)} cells, x {x.min()/1e3:.0f} to {x.max()/1e3:.0f} km, "
          f"y {y.min()/1e3:.0f} to {y.max()/1e3:.0f} km")
    print(f"site mean {mean:.1f} deg, site R {r:.3f}, median cell R {d.R.median():.3f}")
    print(f"node against the site mean: median {np.median(axis_diff(b, mean)):.1f} deg, "
          f"p90 {np.percentile(axis_diff(b, mean), 90):.1f} deg")
    print()
    separation_table(x, y, b, np.random.default_rng(a.seed))

    if not a.reference:
        return
    import geopandas as gpd
    g = gpd.read_file(a.reference).to_crs("EPSG:3413")
    c = g.geometry.centroid
    ref = lattice(c.x.values, c.y.values, g[a.bearing_field].values,
                  a.step, a.min_n, a.min_r)
    own = lattice(x, y, b, a.step, 1, 0.0)
    m = own.merge(ref, on=["i", "j"], suffixes=("", "_ref"))
    print()
    print(f"\n{a.reference}: {len(g)} features, {len(ref)} lattice cells, "
          f"{len(m)} shared with the fabric")
    if m.empty:
        return
    diff = axis_diff(m.bearing_deg.values, m.bearing_deg_ref.values)
    amp = np.cos(np.deg2rad(2 * diff))
    print(f"bearing difference: median {np.median(diff):.1f} deg, "
          f"p90 {np.percentile(diff, 90):.1f} deg")
    print(f"cos2 amplitude retained: {amp.mean():.3f} over all shared cells")
    tight = m.R_ref >= 0.9
    if tight.any():
        print(f"on the {tight.sum()} cells at reference R 0.9 or better: median "
              f"{np.median(diff[tight]):.1f} deg, "
              f"amplitude {amp[tight.values].mean():.3f}")


if __name__ == "__main__":
    main()
