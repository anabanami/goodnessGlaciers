"""
Bedmap flight tracks on the Ockenden et al. landscape classification.

The classification background of map_flightlines.plot_tracks_on_ockenden, with
the region's tracks replaced by whole Bedmap releases read from the subsampled
caches of all_data/map_bedmap3_all.py. Tracks are drawn in black; the legend
names every institution or release present.

USAGE: run from anywhere:

  # Releases. --generations takes any subset of 1 2 3; the default is Bedmap3.
  python map_bedmap_on_ockenden.py                                  # Bedmap3
  python map_bedmap_on_ockenden.py --generations 1                  # Bedmap1 only
  python map_bedmap_on_ockenden.py --generations 1 2 3              # all three

  # Legend. Institution by default, one entry per institution present.
  # --by-release gives the three-entry Bedmap1/2/3 legend instead.
  python map_bedmap_on_ockenden.py --generations 1 2 3 --by-release

  # Institutions. Filters the datasets themselves, so only these tracks are drawn.
  python map_bedmap_on_ockenden.py --institutions AWI               # AWI within Bedmap3
  python map_bedmap_on_ockenden.py --generations 2 3 --institutions AWI BAS
  python map_bedmap_on_ockenden.py --institutions AWI BAS --by-release

  # Point density. --spacing is the cell size in metres of the grid the points are
  # thinned onto: one point per occupied cell, so every campaign is drawn at the
  # same density and repeat coverage of the same ground collapses to a single
  # point. It applies to every release named by --generations, but only bites when
  # a cache is (re)parsed. Caches record their spacing, so any release still held
  # at a different one is rebuilt automatically.
  python map_bedmap_on_ockenden.py --spacing 5000                   # Bedmap3, sparser
  python map_bedmap_on_ockenden.py --generations 1 2 3 --spacing 500
  python map_bedmap_on_ockenden.py --rebuild                        # re-parse the CSVs

The caches are shared with all_data/map_bedmap3_all.py, so a release parsed by
either script plots in seconds from the other.

Output goes to all_data/Bedmap_track_plots/tracks_on_ockenden/, with the release
digits and any institution filter in the filename:
bedmap3_on_ockenden.png, bedmap123_on_ockenden.png, bedmap3_AWI-BAS_on_ockenden.png.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer

import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
sys.path.insert(0, os.path.join(_bootstrap.ROOT, 'all_data'))

from config import Tee
from map_flightlines import CONTINENT_XLIM, CONTINENT_YLIM, draw_ockenden
from map_bedmap3_all import SPACING_M, generation_of, institution_of, load_coords, print_summary

OUTPUT_BASE_PATH = 'all_data/Bedmap_track_plots/tracks_on_ockenden/'

_TO_PS = Transformer.from_crs('EPSG:4326', 'EPSG:3031', always_xy=True)


def plot_bedmap_on_ockenden(coords, output_path, group_of=institution_of,
                            legend_title='Institution', title=None,
                            track_ms=0.3, casing_ms=0, track_alpha=0.5):
    """
    Continental map of the cached Bedmap tracks over the landscape classification.

    Every track is black; `group_of` maps a dataset name to its legend entry, so
    the legend lists the institutions or the releases on the map. `casing_ms` sets
    the width of a white casing under each track, and is off by default.
    """
    groups = {}
    for name, xy in coords.items():
        if len(xy):
            groups.setdefault(group_of(name), []).append(xy)

    fig, ax = plt.subplots(figsize=(12, 10))
    _, class_handles = draw_ockenden(ax, CONTINENT_XLIM, CONTINENT_YLIM)

    track_handles = []
    for group in sorted(groups):
        lonlat = np.vstack(groups[group])
        x, y = _TO_PS.transform(lonlat[:, 0], lonlat[:, 1])
        if casing_ms:
            ax.plot(x, y, '.', color='white', ms=casing_ms, zorder=3)
        ax.plot(x, y, '.', color='black', ms=track_ms, alpha=track_alpha, zorder=4)
        track_handles.append(Line2D([], [], marker='.', ls='', ms=10, color='black',
                                    label=f'{group} ({len(lonlat)} points)'))

    ax.set_xlim(CONTINENT_XLIM); ax.set_ylim(CONTINENT_YLIM)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_artist(ax.legend(handles=class_handles, loc='upper left', fontsize=8,
                            framealpha=0.8, title='Landscape class', title_fontsize=9))
    ax.legend(handles=track_handles, loc='lower left', fontsize=8, framealpha=0.8,
              ncol=2 if len(track_handles) > 8 else 1,
              title=legend_title, title_fontsize=9)

    gens = sorted({generation_of(n) for n in coords})
    label = gens[0] if len(gens) == 1 else 'Bedmap ' + '/'.join(g[-1] for g in gens)
    ax.set_title(title or f'{label} flight tracks on Ockenden et al. '
                          'landscape classification', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overlay map to {output_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--rebuild', action='store_true', help='re-parse the CSVs')
    p.add_argument('--spacing', type=int, default=SPACING_M,
                   help='cell size in metres of the grid the points are thinned onto')
    p.add_argument('--generations', type=int, nargs='+', default=[3], choices=[1, 2, 3],
                   help='Bedmap releases to include (default: 3)')
    p.add_argument('--institutions', nargs='+',
                   help='keep only these institutions (AWI, BAS, ...)')
    p.add_argument('--by-release', action='store_true',
                   help='label by Bedmap release instead of institution')
    args = p.parse_args()

    gens = sorted(args.generations)
    tag = ''.join(str(g) for g in gens)
    if args.institutions:
        tag += '_' + '-'.join(sorted(args.institutions))

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    sys.stdout = Tee(os.path.join(OUTPUT_BASE_PATH, f'bedmap{tag}_on_ockenden_log.txt'))

    coords = load_coords(gens, args.rebuild, args.spacing)
    if args.institutions:
        keep = set(args.institutions)
        coords = {n: xy for n, xy in coords.items() if institution_of(n) in keep}
        if not coords:
            print(f"No datasets for institutions {sorted(keep)}. Nothing to plot.")
            sys.exit(1)

    by_inst = not args.by_release
    print_summary(coords, institution_of if by_inst else generation_of,
                  'INSTITUTION' if by_inst else 'RELEASE')
    plot_bedmap_on_ockenden(
        coords,
        os.path.join(OUTPUT_BASE_PATH, f'bedmap{tag}_on_ockenden.png'),
        group_of=institution_of if by_inst else generation_of,
        legend_title='Institution' if by_inst else 'Bedmap release')
