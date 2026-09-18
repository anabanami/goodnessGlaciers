"""The four classifying axes per window, grouped by [Ockenden_2026]'s published class.

Two figures, each with a sidecar JSON holding the title, caption and the counts it quotes.
  ockenden_overlap_classifiers   one panel per axis, titled with the class pairs that the
                                 axis separates (ockenden_concordance.csv)
  ockenden_combinations_matrix   the class pairs that each combination of the axes, the
                                 descriptors, and all elements separate
                                 (ockenden_combinations.csv)
Each is written as PDF and PNG. Both use all snapped windows, and classes with fewer than
MIN_N windows are drawn but not tested.

    python ockenden_overlap_figure.py [output_tree]

Needs ockenden_window_class.csv, ockenden_concordance.csv and ockenden_combinations.csv
(run the scripts in Ockenden comparison/ first). Writes into the output tree, which
defaults to OUTPUT_BASE_PATH in loading.py.
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
sys.path.insert(0, os.path.join(_bootstrap.ROOT, 'Ockenden comparison'))

from loading import OUTPUT_BASE_PATH as _REGION_BASE
from bed_character import BED_CLASSES, RELIEF_CLASSES, ELEVATION_CLASSES
from landscape_vector import VELOCITY_CLASSES
from ockenden_concordance import MIN_N, Z_MIN, D_MIN
from ockenden_combinations import AXES, SYMBOL, DESCRIPTORS, z_min, load_windows

ROOT = sys.argv[1] if len(sys.argv) > 1 else _REGION_BASE

# Order and colours of map_flightlines.ockenden_classes (Fig. 2.1).
CLASSES = [('low_relief', '#f3e738', 'Low relief'),
           ('sel_erosion_icestreams', '#4399bf', 'Selective erosion\n(ice streams)'),
           ('sel_erosion_relict', '#2f64b4', 'Selective erosion\n(relict)'),
           ('alpine_subglacial', '#ff9248', 'Alpine\n(subglacial)'),
           ('alpine_subaerial', '#e75921', 'Alpine\n(subaerial)')]

# element: (axis label, ODSA classes)
PANELS = {'beta': (r'$\beta$', BED_CLASSES),
          'relief_m': ('Relief (m)', RELIEF_CLASSES),
          'measures_speed_mean': ('Surface speed (m/yr)', VELOCITY_CLASSES),
          'bed_elev_mean': ('Mean bed elevation (m)', ELEVATION_CLASSES)}
NAME = {'beta': 'β', 'relief_m': 'relief', 'measures_speed_mean': 'surface speed',
        'bed_elev_mean': 'mean bed elevation'}
# Rows of the lower panel of the combinations figure: (symbol, label)
ELEMENT_ROWS = [(SYMBOL[a], NAME[a]) for a in AXES] + [('D', f'{len(DESCRIPTORS)} descriptors')]
GROUP_LABEL = {1: '1 axis', 2: '2 axes', 3: '3 axes', 4: '4\naxes',
               len(DESCRIPTORS): f'{len(DESCRIPTORS)}\ndescriptors',
               len(AXES) + len(DESCRIPTORS): f'all {len(AXES) + len(DESCRIPTORS)}\nelements'}


def read_tests(root, name, subset='all'):
    c = pd.read_csv(os.path.join(root, name))
    return c[c.subset == subset]


def untested_labels(d):
    return [lab.replace('\n', ' ') for cls, _, lab in CLASSES
            if (d.ockenden_class == cls).sum() < MIN_N]


def save_figure(fig, out, formats, dpi):
    """One figure per format, named off the .png path. PDF is vector, PNG for the docs."""
    base = os.path.splitext(out)[0]
    for ext in formats:
        path = f'{base}.{ext}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Wrote {path}')
    plt.close(fig)


def write_metadata(png, meta):
    out = os.path.splitext(png)[0] + '.json'
    with open(out, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'Wrote {out}')


def plot_classifiers(d, root, out, figsize=(11, 9), point_size=6, point_alpha=0.35,
                     jitter=0.18, box_width=0.55, break_color='0.45', untested_hatch='///',
                     log_axes=('measures_speed_mean',), formats=('png', 'pdf'), dpi=300):
    conc = read_tests(root, 'ockenden_concordance.csv')
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    rng = np.random.default_rng(0)
    for ax, (col, (ylabel, classes)) in zip(axes.ravel(), PANELS.items()):
        ns = []
        for i, (cls, color, _) in enumerate(CLASSES):
            v = d.loc[d.ockenden_class == cls, col].dropna().values
            ns.append(len(v))
            ax.scatter(i + rng.uniform(-jitter, jitter, len(v)), v, s=point_size,
                       color=color, alpha=point_alpha, edgecolor='none', zorder=2)
            bp = ax.boxplot(v, positions=[i], widths=box_width, showfliers=False,
                            patch_artist=True, zorder=3)
            bp['boxes'][0].set(facecolor='none', edgecolor='0.15', linewidth=1.2,
                               hatch=None if len(v) >= MIN_N else untested_hatch)
            for k in ('whiskers', 'caps', 'medians'):
                for line in bp[k]:
                    line.set(color='0.15', linewidth=1.2)
        if col in log_axes:
            ax.set_yscale('log')
        for (lo_name, _, y), (hi_name, _, _) in zip(classes[:-1], classes[1:]):
            ax.axhline(y, color=break_color, ls=':', lw=1, zorder=1)
            ax.annotate(f'{lo_name} |\n{hi_name}', (1, y), xycoords=('axes fraction', 'data'),
                        xytext=(3, 0), textcoords='offset points', ha='left', va='center',
                        fontsize=7, color=break_color)
        c = conc[conc.element == col]
        ax.set_title(f'{ylabel}: {int(c.separates.sum())} of {len(c)} class pairs separated',
                     fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([f'{lab}\nn = {n}' for (_, _, lab), n in zip(CLASSES, ns)],
                           fontsize=7)
        ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Patch(facecolor='none', edgecolor='0.15', hatch=untested_hatch,
                              label=f'fewer than {MIN_N} windows, not tested')],
               loc='lower center', frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, out, formats, dpi)

    per_element = conc.groupby('element').separates.agg(['sum', 'size'])
    fail = conc[~conc.separates]
    caption = (
        f"β, relief, surface speed and mean bed elevation of {len(d)} analysed windows, "
        f"grouped by the landscape class of the nearest cell in the Ockenden et al. "
        f"classification. Boxes show the median and interquartile range, and points are "
        f"individual windows. Dotted lines are the ODSA class thresholds (Table 4.1). "
        f"Surface speed is on a logarithmic axis. A class pair is separated when the medians "
        f"differ by at least {Z_MIN:g} standard errors and by at least {D_MIN:g} pooled "
        f"standard deviation. Classes with fewer than {MIN_N} windows "
        f"({', '.join(untested_labels(d)) or 'none'}) are not tested.")
    write_metadata(out, {
        'figure': os.path.basename(out),
        'title': 'Class pairs of Ockenden et al. separated by each classifying axis',
        'caption': caption, 'output_tree': root, 'windows': len(d),
        'windows_per_class': {cls: int((d.ockenden_class == cls).sum()) for cls, _, _ in CLASSES},
        'min_windows_tested': MIN_N,
        'pairs_separated': {k: {'separated': int(r['sum']), 'tested': int(r['size'])}
                            for k, r in per_element.iterrows()},
        'max_pairs_separated_by_any_element': int(per_element['sum'].max()),
        'non_separating_tests': len(fail),
        'non_separating_failing_both_criteria':
            int(((fail.z.abs() < Z_MIN) & (fail.d.abs() < D_MIN)).sum()),
        'sources': ['ockenden_window_class.csv', 'ockenden_concordance.csv (subset all)',
                    '*/window_csvs/*_window_stats.csv']})


def pair_name(a, b):
    order = [cls for cls, _, _ in CLASSES]
    lab = {cls: l.replace('\n', ' ') for cls, _, l in CLASSES}
    a, b = sorted((a, b), key=order.index)
    return f'{lab[a]} vs {lab[b]}'


def plot_combinations(d, root, out, figsize=(11, 6.6), cell_size=170, dot_size=55,
                      group_gap=0.8, group_label_size=9, on_color='#1a6faf',
                      ns_color='#8ec1e0', all_color='#e0a458', off_color='0.88',
                      dot_color='0.15', formats=('png', 'pdf'), dpi=300):
    both = {s: read_tests(root, 'ockenden_combinations.csv', s).assign(
        pair=lambda r: [pair_name(a, b) for a, b in zip(r.a, r.b)])
        for s in ('all', 'non_straddling')}
    n_win = {'all': len(d), 'non_straddling': int(d.alt_agrees.astype(bool).sum())}
    hit = {s: r.set_index(['pair', 'axes']).separates for s, r in both.items()}
    sep = {s: r.groupby(['n_axes', 'axes'], sort=False).separates.sum() for s, r in both.items()}
    ns_by_axes = sep['non_straddling'].reset_index().set_index('axes').separates
    cols = (sep['all'].rename('separated').reset_index()
            .assign(separated_ns=lambda c: c['axes'].map(ns_by_axes))
            .sort_values(['n_axes', 'separated'], kind='stable').reset_index(drop=True))
    rows = (both['all'].groupby('pair').separates.sum()
            + both['non_straddling'].groupby('pair').separates.sum()
            ).sort_values(ascending=False, kind='stable').index
    n_pairs = len(rows)

    def state(p, axes):
        a, n = hit['all'].get((p, axes), False), hit['non_straddling'].get((p, axes), False)
        return on_color if a and n else ns_color if n else all_color if a else off_color

    fig, (top, bot) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={
        'height_ratios': [n_pairs + 1, len(ELEMENT_ROWS)], 'hspace': 0.05})
    # Column positions, with a gap between groups of different size.
    cols['x'] = cols.index + group_gap * cols.n_axes.rank(method='dense').sub(1)
    for _, c in cols.iterrows():
        j = c['x']
        for i, p in enumerate(rows):
            top.scatter(j, i, s=cell_size, marker='s', color=state(p, c['axes']))
        top.annotate(f"{int(c['separated'])} | {int(c['separated_ns'])}", (j, -1),
                     ha='center', va='center', fontsize=8)
        used = [k for k, (sym, _) in enumerate(ELEMENT_ROWS) if sym in c['axes'].split('+')]
        bot.scatter([j] * len(ELEMENT_ROWS), range(len(ELEMENT_ROWS)), s=dot_size,
                    color=off_color)
        bot.scatter([j] * len(used), used, s=dot_size, color=dot_color, zorder=3)
        bot.plot([j, j], [min(used), max(used)], color=dot_color, lw=1.5, zorder=2)

    # Vertical rules between combinations of different size, and the size under each group.
    edges = cols.groupby('n_axes').x.agg(['min', 'max'])
    for hi in edges['max'].iloc[:-1]:
        for ax in (top, bot):
            ax.axvline(hi + (1 + group_gap) / 2, color='0.7', lw=0.8)
    bot.set_xticks((edges['min'] + edges['max']) / 2)
    bot.set_xticklabels([GROUP_LABEL[k] for k in edges.index], fontsize=group_label_size)
    bot.set_xlabel('vector elements combined')

    top.set_yticks([-1] + list(range(n_pairs)))
    top.set_yticklabels([f"pairs separated (of {n_pairs}): {n_win['all']} windows | "
                         f"{n_win['non_straddling']} same-class-neighbour windows"]
                        + list(rows))
    top.set_ylim(n_pairs - 0.5, -1.6)
    bot.set_yticks(range(len(ELEMENT_ROWS)))
    bot.set_yticklabels([lab for _, lab in ELEMENT_ROWS])
    bot.set_ylim(len(ELEMENT_ROWS) - 0.5, -0.5)
    top.set_xlim(cols.x.min() - 0.6, cols.x.max() + 0.6)
    for ax in (top, bot):
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    top.legend(handles=[
        Patch(color=on_color, label='separated in both sets'),
        Patch(color=all_color, label=f"separated only in all {n_win['all']} windows"),
        Patch(color=ns_color, label=f"separated only in the {n_win['non_straddling']} windows "
                                   f"whose neighbouring cell carries the same class"),
        Patch(color=off_color, label='not separated')],
        loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    save_figure(fig, out, formats, dpi)

    caption = (
        f"Separation of the Ockenden et al. landscape classes by combinations of vector "
        f"elements. Each column is one combination of the four classifying axes, the "
        f"{len(DESCRIPTORS)} descriptors together ({', '.join(DESCRIPTORS)}), or all "
        f"{len(AXES) + len(DESCRIPTORS)} elements together, and the black dots in the lower "
        f"panel show the elements it uses. Each row in the upper panel is a pair of Ockenden "
        f"et al. classes. Each window is 50 km and so is her cell, so a window can straddle "
        f"two cells of different class: the test runs over all {n_win['all']} co-located "
        f"windows and again over the {n_win['non_straddling']} windows whose second-nearest "
        f"cell carries the same class as the nearest. Square colour gives the two outcomes, "
        f"and the numbers above each column are the counts of separated pairs in the two "
        f"sets. Two classes are separated when the vector of their median differences is at "
        f"least {D_MIN:g} pooled standard deviation long and at least a threshold number of "
        f"standard errors long, both measured with the covariance of the combined elements. "
        f"Classes with fewer than {MIN_N} windows "
        f"({', '.join(untested_labels(d)) or 'none'}) are not tested.")
    write_metadata(out, {
        'figure': os.path.basename(out),
        'title': 'Ockenden et al. classes separated by combined vector elements',
        'caption': caption, 'output_tree': root, 'windows': n_win,
        'min_windows_tested': MIN_N,
        'z_threshold': {c['axes']: round(z_min(int(c['n_axes'])), 3) for _, c in cols.iterrows()},
        'pairs_separated': {s: {c['axes']: [p for p in rows if hit[s].get((p, c['axes']), False)]
                                for _, c in cols.iterrows()} for s in hit},
        'sources': ['ockenden_window_class.csv', 'ockenden_combinations.csv',
                    '*/window_csvs/*_window_stats.csv']})


if __name__ == '__main__':
    d = load_windows(ROOT)
    plot_classifiers(d, ROOT, os.path.join(ROOT, 'ockenden_overlap_classifiers.png'))
    plot_combinations(d, ROOT, os.path.join(ROOT, 'ockenden_combinations_matrix.png'))
