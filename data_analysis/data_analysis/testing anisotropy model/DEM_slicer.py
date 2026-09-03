"""Take perpendicular slices through a DEM and fit beta on each.

Every SPACING_M of raster row gives a full-width slice along x at azimuth 90, and every
SPACING_M of column gives a full-height slice along y at azimuth 0. Water and nodata
samples are dropped. theta is the angle between a slice and the bearing of the nearest
node in the fabric.csv.

A bed folder holds <name>_dem.tif, <name>_water.tif and <name>_fabric.csv. The output
holds the columns weighted_anisotropy.py reads: center_x, center_y, incidence_deg, beta,
flow_error_mean.

    python DEM_slicer.py                            # the three sites -> window_csvs/
    python DEM_slicer.py --site Dubawnt
    python DEM_slicer.py --dir null_seeds/seed_001
    python DEM_slicer.py --spacing-km 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import signal, spatial

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'window_csvs'

SITES = ['Site E Prince of Wales', 'Site F Nunavut', 'Dubawnt']

SPACING_M = 3000.0      # between slices
BAND_M = (250.0, 50000.0)
MIN_WAVELENGTH_M = 30.0
N_FREQ = 500


def beta_of(dist, elev, length):
    """Detrend, taper, Lomb-Scargle, and fit a power law over the band.
    Returns beta, its sigma, and the profile rms."""
    freqs = np.geomspace(1 / length, 1 / MIN_WAVELENGTH_M, N_FREQ)
    wl = 1 / freqs
    band = (wl >= BAND_M[0]) & (wl <= min(BAND_M[1], length))
    d = signal.detrend(elev)
    rms = float(np.sqrt(np.mean(d ** 2)))

    pgram = signal.lombscargle(dist, d * signal.windows.hann(len(d)),
                               freqs * 2 * np.pi, normalize=False)
    if band.sum() < 3 or not np.all(pgram[band] > 0):
        return np.nan, np.nan, rms

    (slope, _), cov = np.polyfit(np.log10(freqs[band]), np.log10(pgram[band]), 1, cov=True)
    return -slope, float(np.sqrt(cov[0, 0])), rms


def bed_files(folder):
    """(name, dem, water, fabric) for a folder holding one bed."""
    folder = Path(folder)
    dem = next(folder.glob('*_dem.tif'))
    name = dem.name[:-len('_dem.tif')]
    return name, dem, folder / f'{name}_water.tif', folder / f'{name}_fabric.csv'


def slice_bed(folder, spacing_m=SPACING_M):
    """Slice one bed along x and y. Returns (name, window statistics)."""
    name, dem_path, water_path, fabric_path = bed_files(folder)

    with rasterio.open(dem_path) as src:
        dem, tr, nod = src.read(1).astype('float32'), src.transform, src.nodata
    with rasterio.open(water_path) as src:
        mask = src.read(1) > 0
    mask |= dem == (nod if nod is not None else -9999)

    h, w = dem.shape
    px, x0, y0 = tr.a, tr.c, tr.f
    step = int(round(spacing_m / px))
    fab = pd.read_csv(fabric_path)
    tree = spatial.cKDTree(fab[['x', 'y']].values)

    slices = ([(90.0, w, dem[r, :], ~mask[r, :], x0 + w * px / 2, y0 - (r + 0.5) * px)
               for r in range(0, h, step)] +
              [(0.0, h, dem[:, c], ~mask[:, c], x0 + (c + 0.5) * px, y0 - h * px / 2)
               for c in range(0, w, step)])

    rows = []
    for az, n, z, k, cx, cy in slices:
        length = n * px
        beta, se, rms = beta_of(np.arange(n)[k] * px, z[k], length)
        bearing = float(fab.bearing_deg.values[tree.query([cx, cy])[1]])
        th = abs(az - bearing) % 180
        rows.append(dict(center_x=cx, center_y=cy, incidence_deg=min(th, 180 - th),
                         beta=beta, flow_error_mean=0.0, beta_uncertainty=se,
                         azimuth_deg=az, bearing_deg=bearing, rms=rms,
                         length_km=length / 1000, masked_frac=round(1 - k.mean(), 4),
                         n_samples=int(k.sum())))

    return name, pd.DataFrame(rows).dropna(subset=['beta'])


def run(folder, spacing_m, outdir):
    name, d = slice_bed(folder, spacing_m)
    outdir.mkdir(exist_ok=True)
    path = outdir / f'{name}_window_stats.csv'
    d.to_csv(path, index=False)

    n_x = int((d.azimuth_deg == 90).sum())
    print(f'\n{name}')
    print(f'  {len(d)} slices, {n_x} along x and {len(d) - n_x} along y')
    print(f'  theta {d.incidence_deg.min():.1f} to {d.incidence_deg.max():.1f} deg, '
          f'beta median {d.beta.median():.3f}, masked median {d.masked_frac.median():.3f}')
    print(f'  -> {path}')
    return d


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--site')
    p.add_argument('--dir')
    p.add_argument('--spacing-km', type=float, default=SPACING_M / 1000)
    a = p.parse_args()

    if a.dir:
        run(Path(a.dir), a.spacing_km * 1000, Path(a.dir))
    else:
        sites = [s for s in SITES if a.site is None or a.site.lower() in s.lower()]
        if not sites:
            raise SystemExit(f'no site matching {a.site!r} in {SITES}')
        for s in sites:
            run(DATA / s / 'prep', a.spacing_km * 1000, OUT)
