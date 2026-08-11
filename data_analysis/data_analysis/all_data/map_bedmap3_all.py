"""
Continent-wide flight-track overview from the Bedmap Results CSVs.

Same map as map_flightlines.plot_antarctica_overview, but over whole releases
instead of one region: 84 campaign files for Bedmap3 (~6.8 GB), 66 for Bedmap2
(~2.7 GB), 1 compiled file for Bedmap1 (~157 MB). Lon/lat are parsed once per
release into a subsampled .npz cache; plotting reads only the cache.

USAGE: run from the ODSA root:

  python map_bedmap3_all.py             # Bedmap3 only, coloured by institution
  python map_bedmap3_all.py             # again: replots from cache in seconds

  # more than one release -> use --by-release, or the institution legend
  # sprawls across every campaign in all of them
  python map_bedmap3_all.py --generations 1 2 3 --by-release   # Bedmap1/2/3 legend
  python map_bedmap3_all.py --generations 2 3 --by-release     # any subset

  # --stride applies to every release named by --generations, but only bites when
  # a cache is (re)parsed. Caches record their stride, so any release still held
  # at a different one is rebuilt automatically — no mixed-density maps.
  python map_bedmap3_all.py --rebuild --stride 50                    # BM3 only (default gens)
  python map_bedmap3_all.py --generations 1 2 3 --stride 50 --by-release   # all three at 50
  python map_bedmap3_all.py --generations 1 --stride 50          # only gen1 at 50

"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer

# Run from anywhere: put ODSA/ on sys.path and make it the working dir, so the
# relative paths below resolve against the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from config import Tee

RESULTS_DIRS = {1: 'all_data/bedmap3_data/bedmap1/Results/',
                2: 'all_data/bedmap3_data/bedmap2/Results/',
                3: 'all_data/bedmap3_data/bedmap3/Results/'}
OUTPUT_BASE_PATH = 'all_data/Bedmap_track_plots/map_flightlines/'
# one cache per release, so adding BM1/BM2 never re-parses BM3's 6.8 GB
CACHE_TMPL = os.path.join(OUTPUT_BASE_PATH, 'bedmap{gen}_all_coords.npz')

STRIDE = 100        # keep every Nth trace; ~70M points -> ~700k plotted
CHUNK_ROWS = 2_000_000

ANTARCTIC_STEREO = ccrs.SouthPolarStereo(true_scale_latitude=-71)
_TO_PS = Transformer.from_crs('EPSG:4326', 'EPSG:3031', always_xy=True)
LON, LAT = 'longitude (degree_east)', 'latitude (degree_north)'


def institution_of(name):
    """Leading token of the Bedmap filename (AWI, BAS, RNRF, ...)."""
    return name.split('_')[0]


def generation_of(name):
    """Bedmap release from the _BM<n> suffix -> 'Bedmap1' / 'Bedmap2' / 'Bedmap3'."""
    return 'Bedmap' + name.rsplit('_BM', 1)[-1]


def _colour_dist(a, b):
    """Redmean approximation — good enough to reject near-duplicate hues."""
    rm = (a[0] + b[0]) / 2
    dr, dg, db = (x - y for x, y in zip(a[:3], b[:3]))
    return ((2 + rm) * dr ** 2 + 4 * dg ** 2 + (3 - rm) * db ** 2) ** 0.5


def palette(n, cmap_names=('tab10', 'Dark2')):
    """n colours: the first map in order, then the most distinct entries of the rest.

    The overflow maps are picked farthest-point rather than in order — tab10 and
    Dark2 share a near-identical orange and purple, and taking Dark2 straight
    through would put both on the map at once.
    """
    picked = [tuple(c[:3]) for c in plt.get_cmap(cmap_names[0]).colors][:n]
    for nm in cmap_names[1:]:
        cand = [tuple(c[:3]) for c in plt.get_cmap(nm).colors]
        while cand and len(picked) < n:
            c = max(cand, key=lambda c: min(_colour_dist(c, p) for p in picked))
            cand.remove(c)
            picked.append(c)
    return [picked[i % len(picked)] for i in range(n)]


def frame_radius_ps(coords, pad_frac=0.03):
    """Half-width of a pole-centred PS71 square holding every track, plus padding.

    Data-derived rather than a fixed parallel: the ICECAP ferry legs reach -56.7 N
    and were cropped by the old -60 frame.
    """
    xy = np.vstack([v for v in coords.values() if len(v)])
    x, y = _TO_PS.transform(xy[:, 0], xy[:, 1])
    return (1 + pad_frac) * max(np.abs(x).max(), np.abs(y).max())


def build_cache(gen=3, stride=STRIDE):
    """Parse lon/lat from every CSV of one Bedmap release, subsample, save one .npz."""
    results_dir, cache_path = RESULTS_DIRS[gen], CACHE_TMPL.format(gen=gen)
    files = sorted(glob.glob(os.path.join(results_dir, '*.csv')))
    print(f"Caching {len(files)} Bedmap{gen} CSVs (stride {stride}) -> {cache_path}")

    out, total_kept = {}, 0
    for i, path in enumerate(files, 1):
        name = os.path.basename(path).replace('.csv', '')
        parts, offset = [], 0
        # chunked so the multi-GB campaigns never sit in memory whole
        for chunk in pd.read_csv(path, usecols=[LON, LAT], comment='#',
                                 chunksize=CHUNK_ROWS):
            parts.append(chunk.values[(-offset) % stride::stride])
            offset = (offset + len(chunk)) % stride
        xy = np.vstack(parts) if parts else np.empty((0, 2))
        valid = (np.abs(xy[:, 0]) <= 180) & (xy[:, 1] < -55)
        out[name] = xy[valid].astype(np.float32)
        total_kept += len(out[name])
        print(f"  [{i:2d}/{len(files)}] {name}: {len(out[name])} points kept")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, _stride=np.int32(stride), **out)
    print(f"Cached {total_kept} points from {len(files)} datasets")
    return {k: v for k, v in out.items()}


def load_coords(gens=(3,), rebuild=False, stride=STRIDE):
    """Merged {dataset_name: xy} for the requested releases, building caches as needed.

    A cache built at a different stride is rebuilt rather than loaded, so mixing
    releases can never silently mix densities.
    """
    coords = {}
    for gen in gens:
        cache_path = CACHE_TMPL.format(gen=gen)
        stale = True
        if not rebuild and os.path.exists(cache_path):
            with np.load(cache_path) as z:
                cached = int(z['_stride']) if '_stride' in z.files else None
                stale = cached != stride
                if stale:
                    print(f"Cache {cache_path} is stride {cached}, want {stride} — re-parsing")
                else:
                    print(f"Loading cache {cache_path} (stride {stride}, --rebuild to re-parse)")
                    coords.update({k: z[k] for k in z.files if k != '_stride'})
        if stale:
            coords.update(build_cache(gen, stride))
    return coords


def plot_antarctica_overview(coords, output_path, cmap_names=('tab10', 'Dark2'),
                             point_size=1.0, alpha=0.5,
                             per_institution_legend=True, pad_frac=0.03,
                             frame_radius_m=None, legend_title=None,
                             order_by_size=True, title=None):
    """
    Full-continent track map, coloured by whatever the legend keys on.

    `per_institution_legend=True` colours and labels by institution (AWI, BAS, ...);
    False colours by Bedmap release instead, giving the three-entry Bedmap1/2/3
    legend that stays readable once BM1 and BM2 are loaded alongside BM3.

    The frame is sized to the data (`pad_frac` buffer); `frame_radius_m`
    overrides it with a fixed PS71 half-width.
    """
    group_of = institution_of if per_institution_legend else generation_of

    size = {}
    for name, xy in coords.items():
        size[group_of(name)] = size.get(group_of(name), 0) + len(xy)
    # institutions: biggest first, so they take the well-separated head of the
    # palette. Releases: chronological, so BM3 draws on top of the older ones.
    groups = (sorted(size, key=lambda g: -size[g])
              if order_by_size and per_institution_legend else sorted(size))
    color = dict(zip(groups, palette(len(groups), cmap_names)))

    gens = sorted({generation_of(n) for n in coords})
    label = gens[0] if len(gens) == 1 else 'Bedmap ' + '/'.join(g[-1] for g in gens)
    title = title or f'{label} Radar Flight Tracks — Antarctica'
    legend_title = legend_title or ('Institution' if per_institution_legend
                                    else 'Bedmap release')

    r = frame_radius_m if frame_radius_m is not None else frame_radius_ps(coords, pad_frac)
    print(f"Frame half-width: {r/1e3:.0f} km (PS71)")

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ANTARCTIC_STEREO)
    ax.set_extent([-r, r, -r, r], crs=ANTARCTIC_STEREO)

    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--', color='gray')

    # later groups draw on top: newest release, or smallest institution
    rank = {g: i for i, g in enumerate(groups)}
    for name, xy in sorted(coords.items(), key=lambda kv: rank[group_of(kv[0])]):
        if not len(xy):
            continue
        ax.scatter(xy[:, 0], xy[:, 1], c=[color[group_of(name)]],
                   s=point_size, alpha=alpha, linewidths=0,
                   transform=ccrs.PlateCarree())

    handles = [Line2D([], [], marker='o', ls='', color=color[g], label=g) for g in groups]
    ax.legend(handles=handles, loc='upper left', fontsize=8,
              ncol=2 if len(groups) > 6 else 1, framealpha=0.85,
              title=legend_title, title_fontsize=9, markerscale=1.4)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview map to {output_path}")
    plt.close()


def print_summary(coords, key=institution_of, heading='INSTITUTION'):
    print("\n" + "=" * 60)
    print(f"BEDMAP COVERAGE SUMMARY BY {heading}")
    print("=" * 60)
    tally = {}
    for name, xy in coords.items():
        tally.setdefault(key(name), [0, 0])
        tally[key(name)][0] += 1
        tally[key(name)][1] += len(xy)
    for k, (n_files, n_pts) in sorted(tally.items()):
        print(f"  {k:12s} {n_files:3d} files  {n_pts:9d} points")
    print(f"  {'TOTAL':12s} {len(coords):3d} files  "
          f"{sum(len(v) for v in coords.values()):9d} points")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild', action='store_true', help='re-parse the CSVs')
    p.add_argument('--stride', type=int, default=STRIDE)
    p.add_argument('--generations', type=int, nargs='+', default=[3], choices=[1, 2, 3],
                   help='Bedmap releases to include (default: 3)')
    p.add_argument('--by-release', action='store_true',
                   help='colour and label by Bedmap release instead of institution')
    args = p.parse_args()

    gens = sorted(args.generations)
    tag = ''.join(str(g) for g in gens)

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUTPUT_BASE_PATH, f'map_bedmap{tag}_all_log.txt'))

    coords = load_coords(gens, args.rebuild, args.stride)

    by_inst = not args.by_release
    print_summary(coords, institution_of if by_inst else generation_of,
                  'INSTITUTION' if by_inst else 'RELEASE')
    plot_antarctica_overview(
        coords,
        os.path.join(OUTPUT_BASE_PATH, f'bedmap{tag}_all_tracks_overview.png'),
        per_institution_legend=by_inst)
