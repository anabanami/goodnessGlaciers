"""FIG 7c: per-window maps, one panel per region.

Every window is drawn at its step centre as its true 50 km along-track footprint, so the 50%
overlap is visible rather than hidden behind a dot. No thinning: halving the data to
non-overlapping windows still leaves neighbours agreeing on the class tuple at 4.1x chance,
so the overlap is kept and its cost is stated instead. Adjacent footprints share half their
data and the tuple only decorrelates at 200 km.

Three colourings of the same geometry, written to separate files:
  verdict     what the catalogue could conclude
  bed_class   the beta class, point estimate
  archetype   the entry, where exactly one survives

Reads the output tree directly, so it does not depend on which regions are live in loading.py.

    python fig7c_maps.py [individual_region_TEST] [scheme]
"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
from config import Tee

PS71 = ccrs.SouthPolarStereo(true_scale_latitude=-71)
WINDOW_M = 50_000
STEP_M = 25_000
DECORRELATION_KM = 200

# Backgrounds are drawn first so the named categories sit on top of them.
BACKGROUND = {'degenerate', 'out of catalogue', 'unclassified'}

VERDICT_COLORS = {'RESOLVED': '#1a7f37', 'RESOLVED-WITH-EXTERNAL': '#4493f8',
                  'DEGENERATE': '#bf8700', 'OUT-OF-CATALOGUE': '#cf222e'}
BED_COLORS = {'chaotic': '#8250df', 'hard': '#1a7f37',
              'transitional': '#bf8700', 'soft': '#cf222e'}
ARCHETYPE_COLORS = {
    'TRUNK': '#cf222e', 'TRUNK-HARD': '#a40e26', 'TRUNK-RELICT': '#fa7970',
    'ONSET': '#d4a72c', 'HIGHLAND': '#1a7f37', 'RIFT': '#8250df',
    'BASIN': '#0969da', 'BASIN-HIGH': '#54aeff', 'DISSECTED': '#bc4c00',
    'DIVIDE': '#3fb950', 'SHATTERED': '#57606a',
    'degenerate': '#d8dee4', 'out of catalogue': '#24292f'}


def load(root):
    """Window stats joined to the archetype report on window:<traj>|s<seg>|w<id>."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        region = os.path.basename(os.path.dirname(os.path.dirname(f)))
        d = pd.read_csv(f)
        d = d[~d.is_transition.astype(bool)].copy()
        d['region'] = region
        d['unit'] = ('window:' + d.trajectory.astype(str) + '|s' + d.segment.astype(str)
                     + '|w' + d.window_id.astype(str))
        rep = glob.glob(os.path.join(root, region, 'landscape_vector', '*_archetype_report.csv'))
        r = pd.read_csv(rep[0])
        r = r[r.level == 'window'][['unit', 'verdict', 'admissible', 'n_admissible']]
        out.append(d.merge(r, on='unit', how='left'))
    return pd.concat(out, ignore_index=True)


def archetype_label(d):
    """Only a singleton names an entry. Everything else says so rather than picking one."""
    return pd.Series(np.where(d.n_admissible == 1, d.admissible.astype(str),
                              np.where(d.n_admissible == 0, 'out of catalogue', 'degenerate')),
                     index=d.index)


SCHEMES = {
    'verdict': dict(
        label=lambda d: d.verdict.fillna('unclassified'), colors=VERDICT_COLORS,
        title='FIG 7c. Per-window archetype verdict', legend='window verdict',
        note='Read the map for where verdicts change, not for how much area each covers.'),
    'bed_class': dict(
        label=lambda d: d.bed_class.fillna('unclassified'), colors=BED_COLORS,
        title='Per-window $\\beta$ class', legend='bed_class',
        note='The point estimate. The catalogue classifies on the 2$\\sigma$ envelope, which is\n'
             'wider, so a window shown in one class is often admissible in its neighbour too.'),
    'archetype': dict(
        label=archetype_label, colors=ARCHETYPE_COLORS,
        title='Per-window archetype, where one survives', legend='admissible entry',
        note='Only windows with exactly one admissible entry are named. Grey is a degenerate\n'
             'set of two or more, and that is the common case, not a gap in the map.'),
}


def frame(d, pad_frac=0.18, pad_min_m=1.5 * WINDOW_M):
    """Padded PS71 box, never tighter than a window and a half so footprints are not clipped."""
    x, y = d.center_x.to_numpy(float), d.center_y.to_numpy(float)
    pad = max(pad_frac * max(np.ptp(x), np.ptp(y)), pad_min_m)
    return (x.min() - pad, x.max() + pad), (y.min() - pad, y.max() + pad)


def basemap(ax, xlim, ylim, gridlines=True):
    ax.set_extent([*xlim, *ylim], crs=PS71)
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff', alpha=0.5)
    ax.coastlines(resolution='10m', linewidth=0.8)
    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4,
                          linestyle='--', color='gray')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {'size': 7}


def footprints(ax, d, lab, colors, lw=2.4, alpha=0.55, missing='0.6'):
    """azimuth_deg is ccw from +x in PS71, not a compass bearing: checked against the
    step-to-step direction, median disagreement 0.01 deg."""
    t = np.radians(d.azimuth_deg.to_numpy(float))
    hx, hy = WINDOW_M / 2 * np.cos(t), WINDOW_M / 2 * np.sin(t)
    order = np.argsort([v not in BACKGROUND for v in lab], kind='stable')
    for i in order:
        x, y, dx, dy = d.center_x.iloc[i], d.center_y.iloc[i], hx[i], hy[i]
        v = lab.iloc[i]
        ax.plot([x - dx, x + dx], [y - dy, y + dy], transform=PS71,
                zorder=3 if v in BACKGROUND else 4,
                c=colors.get(v, missing), lw=lw, alpha=alpha, solid_capstyle='butt')


def scale_bar(ax, xlim, ylim, length_m=WINDOW_M, frac=(0.06, 0.08), lw=3):
    """A bar exactly one window long, so the reader can see the footprint against the frame."""
    x0 = xlim[0] + frac[0] * (xlim[1] - xlim[0])
    y0 = ylim[0] + frac[1] * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length_m], [y0, y0], transform=PS71, c='k', lw=lw,
            solid_capstyle='butt', zorder=5)
    ax.text(x0, y0 + 0.02 * (ylim[1] - ylim[0]), f"{length_m/1000:.0f} km window",
            transform=PS71, fontsize=7, zorder=5)


def panel(ax, d, lab, region, colors, **kw):
    xlim, ylim = frame(d)
    basemap(ax, xlim, ylim)
    footprints(ax, d, lab, colors, **kw)
    scale_bar(ax, xlim, ylim)
    top = lab.value_counts(normalize=True)
    lead = f"   {top.index[0]} {100 * top.iloc[0]:.0f}%" if len(top) else ''
    ax.set_title(f"{region}   n = {len(d)}{lead}", fontsize=9)


def locator(ax, d, box_pad_m=WINDOW_M):
    """Where the seven sit. Boxes, not points, because a region is 300 km across."""
    ax.set_extent([-180, 180, -90, -63], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', edgecolor='black', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='#cce5ff', alpha=0.5)
    for region, g in d.groupby('region'):
        x0, x1 = g.center_x.min() - box_pad_m, g.center_x.max() + box_pad_m
        y0, y1 = g.center_y.min() - box_pad_m, g.center_y.max() + box_pad_m
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], transform=PS71,
                c='#cf222e', lw=1.0, zorder=4)
        ax.text(x1, y1, region, transform=PS71, fontsize=6, c='#cf222e', zorder=5)
    ax.set_title('region locations', fontsize=9)


def legend_panel(ax, lab, spec):
    ax.axis('off')
    counts = lab.value_counts()
    handles = [Line2D([], [], c=c, lw=4,
                      label=f"{v}  {counts.get(v, 0)} ({100*counts.get(v, 0)/len(lab):.1f}%)")
               for v, c in spec['colors'].items() if counts.get(v, 0)]
    ax.legend(handles=handles, loc='upper center', frameon=False, fontsize=8,
              title=spec['legend'], title_fontsize=9)
    ax.text(0.5, 0.02, CAPTION + '\n' + spec['note'], ha='center', va='bottom',
            fontsize=7, transform=ax.transAxes)


CAPTION = (
    f"Each bar is one {WINDOW_M//1000} km along-track window drawn at its step centre.\n"
    f"Windows step every {STEP_M//1000} km, so adjacent bars overlap by half and are not\n"
    f"independent: the class tuple only decorrelates at {DECORRELATION_KM} km."
)


def render(d, regions, name, root, **kw):
    spec = SCHEMES[name]
    lab = spec['label'](d)
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), subplot_kw={'projection': PS71})
    flat = axes.ravel()
    for ax, region in zip(flat, regions):
        m = d.region == region
        panel(ax, d[m], lab[m], region, spec['colors'], **kw)
    locator(flat[len(regions)], d)
    for ax in flat[len(regions) + 1:]:
        ax.remove()
    legend_panel(fig.add_subplot(3, 3, 9), lab, spec)

    fig.suptitle(spec['title'], fontsize=14, y=0.92)
    out = os.path.join(root, 'landscape_vector', f'fig7c_{name}_map.png')
    fig.savefig(out, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"\n=== {name} ===")
    print(lab.value_counts().to_string())
    print(f"  Saved: {out}")


def main(root, only=None, **kw):
    d = load(root)
    regions = sorted(d.region.unique(), key=lambda r: -(d.region == r).sum())
    print(f"{len(d)} non-transition windows, {len(regions)} regions")
    for name in ([only] if only else SCHEMES):
        render(d, regions, name, root, **kw)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    root = args[0] if args and args[0] not in SCHEMES else 'individual_region_TEST'
    only = next((a for a in args if a in SCHEMES), None)
    os.makedirs(os.path.join(root, 'landscape_vector'), exist_ok=True)
    sys.stdout = Tee(os.path.join(root, 'landscape_vector', 'fig7c_maps_log.txt'))
    main(root, only)
