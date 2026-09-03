"""Masked null beds. Stage each isotropic bed under a site's mask and fabric.

A masked bed is an isotropic seed carrying a site's validity mask and the site's
own fabric lattice, so its Delta_beta is still zero by construction while the
window loss and the theta scatter are the site's. The DEM, the mask and the
fabric are symlinked, so a batch costs no disk.

`--only` stages one of the two factors into its own folder, which separates the
contribution of the mask from that of the fabric.

    python masked_seeds.py                              # all three sites, all 100 seeds
    python masked_seeds.py --site site_e                # one site
    python masked_seeds.py --n 5                        # seeds 1-5, for a trial
    python masked_seeds.py --site site_f --only mask    # site mask, seed fabric
    python masked_seeds.py --site site_f --only fabric  # site fabric, no mask

"""
import argparse
import csv
import os
from pathlib import Path

import rasterio

HERE = Path(__file__).resolve().parent
SEEDS = HERE / 'null_seeds'
WORK = HERE / 'masked_seeds'
PREPPED = HERE / 'data/prepped'

SITES = {'site_e': 'Site E Prince of Wales',
         'site_f': 'Site F Nunavut',
         'dubawnt': 'Dubawnt'}


def grid(path):
    """Raster shape and the map coordinates of its top left corner."""
    with rasterio.open(path) as r:
        return r.shape, r.transform.c, r.transform.f, r.res


def translate_fabric(src_csv, out_csv, dx, dy):
    """The site's lattice on the seed's coordinates, bearings unchanged.

    Both grids are north up at 10 m, so a bearing means the same pixel
    direction on either and only the node positions move.
    """
    with open(src_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['x'] = f"{float(r['x']) + dx:.1f}"
        r['y'] = f"{float(r['y']) + dy:.1f}"
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def link(target, name):
    """Relative symlink, replaced if it already points somewhere."""
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(os.path.relpath(target, name.parent))


def stage_site(slug, seeds, only=None):
    """Stage one site. `only='mask'` keeps the seed's fabric, `only='fabric'` drops the mask."""
    site = SITES[slug]
    prep = PREPPED / site
    mask = prep / f'{site}_water.tif'
    ref = SEEDS / 'seed_001/seed_001_dem.tif'

    seed_shape, seed_x, seed_y, seed_res = grid(ref)
    site_shape, site_x, site_y, site_res = grid(prep / f'{site}_dem.tif')
    if seed_shape != site_shape or seed_res != site_res:
        raise SystemExit(f'{site}: grid is {site_shape} at {site_res}, '
                         f'the beds are {seed_shape} at {seed_res}')
    mask_shape, _, _, _ = grid(mask)
    if mask_shape != seed_shape:
        raise SystemExit(f'{site}: mask is {mask_shape}, the beds are {seed_shape}')

    out = WORK / (slug if only is None else f'{slug}_{only}')
    out.mkdir(parents=True, exist_ok=True)

    fabric = None
    if only == 'mask':
        seed_lattice = SEEDS / 'seed_001/seed_001_fabric.csv'
        n_nodes = sum(1 for _ in open(seed_lattice)) - 1
    else:
        fabric = out / f'{slug}_fabric.csv'
        n_nodes = translate_fabric(prep / f'{site}_fabric.csv', fabric,
                                   seed_x - site_x, seed_y - site_y)

    for seed in seeds:
        name = f'seed_{seed:03d}'
        dem = SEEDS / name / f'{name}_dem.tif'
        if not dem.exists():
            print(f'  {name}: no bed, skipping')
            continue
        d = out / name
        d.mkdir(exist_ok=True)
        link(dem, d / f'{name}_dem.tif')
        if only != 'fabric':
            link(mask, d / f'{name}_water.tif')
        src = fabric if fabric else SEEDS / name / f'{name}_fabric.csv'
        link(src, d / f'{name}_fabric.csv')

    carries = {None: 'site mask and site fabric',
               'mask': 'site mask, seed fabric',
               'fabric': 'site fabric, no mask'}[only]
    print(f'{out.name}: {len(seeds)} beds under {site}, {carries}, {n_nodes} fabric nodes')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--site', choices=sorted(SITES), action='append',
                   help='repeatable, default all three')
    p.add_argument('--only', choices=['mask', 'fabric'],
                   help='stage one factor, into <site>_mask or <site>_fabric')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--start', type=int, default=1)
    a = p.parse_args()

    seeds = range(a.start, a.start + a.n)
    for slug in a.site or sorted(SITES):
        stage_site(slug, seeds, a.only)


if __name__ == '__main__':
    main()
