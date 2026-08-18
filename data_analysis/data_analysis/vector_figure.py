import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from config import Tee, PROCESSING_FLAG_COLORS as _FLAG_COLOR
from bed_character import (BED_CLASSES, RELIEF_CLASSES, ELEVATION_CLASSES,
                           walk_tree, tree_region_name)
from landscape_vector import (ALL_AXES, AXIS_VALUES, VELOCITY_CLASSES, CATALOGUE,
                              MIGRATION_WIDENS_BETA, COMPOSITION_DECIMATE_KM,
                              output_dir_for)
from loading import OUTPUT_BASE_PATH as _REGION_BASE

"""
The vector doing the work: three panels on what a range of descriptors buys over one.

Renders only what landscape_vector.py already wrote — no classification, no fits, no
thresholds are applied here. The headline comparison figure shows one statistic per
region, which is the opposite of the paper's claim; this shows the axes separately,
what they collapse, and how little of a region any single label covers.

Every panel takes the level == 'region' row of *_archetype_report.csv. Segments and
windows are the same quantity at finer support and are used only inside composition,
which landscape_vector already aggregated over windows.

Usage:
  python vector_figure.py                        # walks OUTPUT_BASE_PATH
  python vector_figure.py individual_region_TEST  # walk a tree of region folders
"""

# ALL_AXES is alphabetical. The figure reads left to right from the axis carrying the
# survey's own measurement to the one carrying nothing, so delta_beta goes last.
AXIS_ORDER = ['beta_class', 'relief_class', 'elevation_class', 'velocity_band', 'delta_beta']
assert set(AXIS_ORDER) == set(ALL_AXES), f"axis list drifted from ALL_AXES: {ALL_AXES}"

# Bands in physical order, not alphabetical, so a cell reads as a range and not a set.
AXIS_BANDS = {
    'beta_class':      [n for n, _, _ in BED_CLASSES],
    'relief_class':    [n for n, _, _ in RELIEF_CLASSES],
    'elevation_class': [n for n, _, _ in ELEVATION_CLASSES],
    'velocity_band':   [n for n, _, _ in VELOCITY_CLASSES],
    'delta_beta':      ['neg_sig', 'zero', 'pos_sig'],
}
assert all(set(AXIS_BANDS[a]) == set(AXIS_VALUES[a]) for a in AXIS_ORDER)

AXIS_LABEL = {'beta_class': r'$\beta$ class', 'relief_class': 'relief',
              'elevation_class': 'elevation', 'velocity_band': 'velocity',
              'delta_beta': r'$\Delta\beta$'}

# The dead axis. Named once so panel 1 and the footnote cannot drift apart.
DEAD_AXIS = 'delta_beta'

# Four statuses, three things a reader needs to tell apart: a band set that came from a
# measurement (resolved / assumed-exact / ambiguous), one widened to every band because
# there was no usable measurement (unavailable), and a resolution resting on no error bar
# at all (assumed-exact). A widened axis constrains nothing, so it is drawn as absence.
STATUS_STYLE = {
    'resolved':      dict(face='#2b7bba', alpha=0.55, hatch=None, edge='0.15', lw=1.1),
    'assumed-exact': dict(face='#2b7bba', alpha=0.55, hatch='..',  edge='#b22222', lw=1.5),
    'ambiguous':     dict(face='#a8c8e0', alpha=0.50, hatch=None, edge='0.50', lw=0.8),
    'unavailable':   dict(face='0.90',    alpha=0.95, hatch='///', edge='0.65', lw=0.8),
}
STATUS_ORDER = ('resolved', 'ambiguous', 'assumed-exact', 'unavailable')
STATUS_NOTE = {'resolved': 'measured, one band',
               'assumed-exact': 'one band, no error bar',
               'ambiguous': 'measured, envelope crosses a break',
               'unavailable': 'widened to every band — constrains nothing'}

# Composition segments are coloured by how many archetypes the set holds, because that is
# the quantity the panel is about. Set identity is carried by the in-bar text instead.
CARD_COLORS = {1: '#1a6faf', 2: '#6aa8d8', 3: '#a9c9e4', 4: '#d3e3f0'}
NONE_COLOR, TAIL_COLOR = '#c86a6a', '0.85'


def discover(root):
    """(region, archetype report, composition) per region folder under a tree.

    Discovery goes through the window CSVs so the figure covers exactly the regions
    the pipeline processed, and through output_dir_for so the folder rule stays owned
    by landscape_vector.
    """
    out = []
    for csv in walk_tree(root):
        d = output_dir_for(csv)
        rep = sorted(glob.glob(os.path.join(d, '*_archetype_report.csv')))
        comp = sorted(glob.glob(os.path.join(d, '*_composition.csv')))
        if not rep:
            print(f"  no archetype report under {d}, skipped")
            continue
        out.append((tree_region_name(csv), rep[0], comp[0] if comp else None))
    return out


def load(root):
    """Region-level report rows and their compositions, sharing one region ordering."""
    rows, comps = [], {}
    for region, rpath, cpath in discover(root):
        rep = pd.read_csv(rpath)
        r = rep[rep.level == 'region']
        if not len(r):
            print(f"  {region}: no level == 'region' row, skipped")
            continue
        rows.append(r.iloc[0].to_dict() | {'region': region})
        if cpath:
            comps[region] = pd.read_csv(cpath)
    reg = pd.DataFrame(rows)
    # Ordering is shared by all three panels: fewest admissible first, so the matrix, the
    # collapse and the mixture are read down the same list of regions.
    reg = reg.sort_values(['n_admissible', 'region']).reset_index(drop=True)
    return reg, comps


def collapse_composition(g, cover):
    """Leading sets to `cover` of the region, then one tail row.

    Mirrors composition_table.py's rule rather than importing it, which would run that
    module's argv parsing. `(none)` is never collapsed: the unmatched share is the
    out-of-catalogue rate and has to stay visible.

    Fractions are landscape_vector's own 3-dp values, so a region's bar sums to 1 within
    0.005. They are not renormalised here; the figure shows what the CSV holds.
    """
    g = g.sort_values('fraction', ascending=False)
    keep = g.fraction.cumsum().shift(fill_value=0) < cover
    keep |= g.admissible == '(none)'
    head, tail = g[keep], g[~keep]
    rows = [dict(label=r.admissible, fraction=float(r.fraction),
                 card=0 if r.admissible == '(none)' else len(str(r.admissible).split('|')),
                 kind='none' if r.admissible == '(none)' else 'set')
            for r in head.itertuples()]
    if len(tail):
        rows.append(dict(label=f'{len(tail)} further sets', fraction=float(tail.fraction.sum()),
                         card=None, kind='tail'))
    return rows


def dead_axis_evidence(reg):
    """How the delta_beta column fails, per region, straight out of the report columns."""
    ev = reg[['region', 'delta_beta_coverage', 'delta_beta_n_indep', 'delta_beta_reason',
              f'status_{DEAD_AXIS}']].copy()
    ev['no_fit'] = ev.delta_beta_coverage == 0
    return ev


def short_codes(ids):
    """Unique short codes for catalogue ids, grown from initials until nothing collides.

    Set names are wider than most composition segments. The codes keep the segment text
    inside its own segment; the key is printed under the figure and in the log.
    """
    code = {i: ''.join(p[0] for p in i.split('-')) for i in ids}
    n = 1
    while len(set(code.values())) < len(code) and n < 12:
        n += 1
        seen = list(code.values())
        dup = {c for c in seen if seen.count(c) > 1}
        for i, c in code.items():
            if c in dup:
                parts = i.split('-')
                code[i] = parts[0][:n] + ''.join(q[0] for q in parts[1:])
    return code


CODES = short_codes([c['id'] for c in CATALOGUE])


def coded(label):
    """A set name in short codes; anything that is not a set is left alone."""
    parts = str(label).split('|')
    return '|'.join(CODES.get(q, q) for q in parts)


# ---------------------------------------------------------------------------
def panel_axis_matrix(ax, reg, ncat, cell_pad=0.06, fs=7.5, fs_frac=6.0, fs_count=10,
                      dead_wash='#ffffff', dead_wash_alpha=0.55, count_gap=0.30):
    """Regions x classifying axes, each cell the admissible band set and its status,
    with what the row's axes leave admissible as a numeric column at the right edge."""
    n, m = len(reg), len(AXIS_ORDER)
    drawn = set()
    for i, r in reg.iterrows():
        for j, a in enumerate(AXIS_ORDER):
            bands = [b for b in AXIS_BANDS[a] if b in str(r[f'axis_{a}']).split(',')]
            st = r[f'status_{a}']
            drawn.add(st)
            s = STATUS_STYLE[st]
            ax.add_patch(Rectangle((j + cell_pad, i + cell_pad),
                                   1 - 2 * cell_pad, 1 - 2 * cell_pad,
                                   facecolor=s['face'], alpha=s['alpha'], hatch=s['hatch'],
                                   edgecolor=s['edge'], linewidth=s['lw'], zorder=1))
            txt = ('all bands\nwidened' if st == 'unavailable'
                   else '\n'.join(bands))
            ax.text(j + 0.5, i + 0.52, txt, ha='center', va='center', fontsize=fs,
                    color='0.25' if st == 'unavailable' else '0.05',
                    style='italic' if st == 'unavailable' else 'normal', zorder=3)
            # k/N is the constraint the axis actually applies: 4/4 excludes nothing.
            ax.text(j + 1 - 2 * cell_pad - 0.02, i + cell_pad + 0.04,
                    f'{len(bands)}/{len(AXIS_BANDS[a])}', ha='right', va='bottom',
                    fontsize=fs_frac, color='0.45', zorder=3)

    # The dead column, washed out and struck through so it cannot be read as a fifth
    # working axis. Every cell in it is already 'unavailable'; this says it once more.
    j = AXIS_ORDER.index(DEAD_AXIS)
    ax.add_patch(Rectangle((j, 0), 1, n, facecolor=dead_wash, alpha=dead_wash_alpha,
                           edgecolor='#b22222', linewidth=1.6, linestyle=(0, (4, 2)),
                           zorder=4))
    ax.plot([j + 0.08, j + 0.92], [0.06, n - 0.06], color='#b22222', lw=1.2, alpha=0.7,
            zorder=5)
    ax.plot([j + 0.08, j + 0.92], [n - 0.06, 0.06], color='#b22222', lw=1.2, alpha=0.7,
            zorder=5)

    # The row's answer, in the same units panel 1 is arguing about: the four live axes
    # together leave this many of the catalogue standing. No bar, because the comparison
    # that would need one (widened vs unwidened) does not exist while widening is off.
    x0 = m + count_gap
    for i, r in reg.iterrows():
        ax.text(x0 + 0.5, i + 0.5, f'{int(r.n_admissible)}', ha='center', va='center',
                fontsize=fs_count, fontweight='bold', color='#1a6faf', zorder=3)
        ax.text(x0 + 0.5, i + 1 - cell_pad - 0.06, f'of {ncat}', ha='center', va='top',
                fontsize=fs_frac, color='0.45', zorder=3)

    ax.set_xlim(0, x0 + 1)
    ax.set_ylim(n, 0)
    ax.set_xticks(list(np.arange(m) + 0.5) + [x0 + 0.5])
    ax.set_xticklabels([AXIS_LABEL[a] + ('\n(dead)' if a == DEAD_AXIS else '')
                        for a in AXIS_ORDER] + ['admissible'], fontsize=9)
    for t, a in zip(ax.get_xticklabels(), AXIS_ORDER):
        if a == DEAD_AXIS:
            t.set_color('#b22222')
    ax.get_xticklabels()[-1].set_color('#1a6faf')
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(np.arange(len(reg)) + 0.5)
    ax.set_yticklabels(reg.region, fontsize=9)
    # Region label carries its own migration flag; beta's reliability differs by region.
    for t, f in zip(ax.get_yticklabels(), reg.processing_flag):
        t.set_color(_FLAG_COLOR.get(f, '0.2'))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('1. What each axis says, per region     '
                 '(cell = admissible bands, corner = bands kept / bands on the axis)',
                 fontsize=10, loc='left', pad=26)
    # Keyed off the cells actually drawn. assumed-exact is unreachable while relief and
    # elevation carry their nominal errors, so it earns no key here; set either error to
    # None for the sensitivity sweep and the status, and its key, come back on their own.
    ax.legend(handles=[Patch(facecolor=STATUS_STYLE[k]['face'], alpha=STATUS_STYLE[k]['alpha'],
                             hatch=STATUS_STYLE[k]['hatch'], edgecolor=STATUS_STYLE[k]['edge'],
                             label=f'{k} — {STATUS_NOTE[k]}')
                       for k in STATUS_ORDER if k in drawn],
              fontsize=7, loc='upper left', bbox_to_anchor=(0, -0.02), ncol=2,
              frameon=False, handlelength=1.6)


def panel_composition(ax, reg, comps, cover=0.80, bar_h=0.62, fs=8, fs_seg=6.8,
                      label_pad_px=4.0, legend_y=-0.10):
    """Per region, the admissible sets its windows take, largest first, tail collapsed.

    A label goes inside its own segment or not at all: the full set name is tried first,
    the short-code form second, and a segment too narrow for either is left to its colour.
    Measured against the rendered text, so nothing overruns into its neighbour.
    """
    y = np.arange(len(reg))
    pending = []
    for i, r in reg.iterrows():
        g = comps.get(r.region)
        if g is None or not len(g):
            ax.text(0.02, i, 'no composition written', va='center', fontsize=fs, color='0.5')
            continue
        left = 0.0
        for seg in collapse_composition(g, cover):
            c = (NONE_COLOR if seg['kind'] == 'none' else TAIL_COLOR if seg['kind'] == 'tail'
                 else CARD_COLORS.get(min(seg['card'], 4), CARD_COLORS[4]))
            ax.barh(i, seg['fraction'], left=left, height=bar_h, color=c,
                    edgecolor='white', linewidth=0.7,
                    hatch='///' if seg['kind'] == 'tail' else None, zorder=2)
            dark = seg['kind'] == 'none' or seg['card'] == 1
            pending.append((i, left, seg['fraction'], seg['label'],
                            'white' if dark else '0.15'))
            left += seg['fraction']
        n_w, n_eff, dec = int(g.n_windows_total.iloc[0]), int(g.n_eff.iloc[0]), g.decimate_km.iloc[0]
        ax.text(1.01, i, f'n={n_w}   n_eff={n_eff} at {dec:.0f} km', va='center',
                fontsize=fs - 0.5, color='0.35')

    ax.set_yticks(y)
    ax.set_yticklabels(reg.region, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('fraction of windows', fontsize=9)
    ax.set_title('One label does not cover a region', fontsize=10, loc='left')

    # Axis geometry has to be settled before a text width means anything.
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = lambda a, b, i: abs(ax.transData.transform((b, i))[0] - ax.transData.transform((a, i))[0])
    for i, left, frac, label, colour in pending:
        room = px(left, left + frac, i) - label_pad_px
        for text in (label, coded(label)):
            t = ax.text(left + frac / 2, i, text, ha='center', va='center',
                        fontsize=fs_seg, color=colour, zorder=3)
            if t.get_window_extent(renderer=rend).width <= room:
                break
            t.remove()

    # No interval is drawn: at these n_eff the SE on a fraction rivals the fraction.
    ax.legend(handles=[Patch(facecolor=CARD_COLORS[1], label='1 archetype'),
                       Patch(facecolor=CARD_COLORS[2], label='2'),
                       Patch(facecolor=CARD_COLORS[3], label='3'),
                       Patch(facecolor=CARD_COLORS[4], label='4+'),
                       Patch(facecolor=NONE_COLOR, label='(none) — out of catalogue'),
                       Patch(facecolor=TAIL_COLOR, hatch='///', label=f'tail beyond {cover:.0%}')],
              fontsize=7, ncol=6, loc='upper left', bbox_to_anchor=(0, legend_y),
              frameon=False, handlelength=1.4)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
def report_disagreement(reg, comps):
    """Where the region-level verdict and the window composition do not say the same thing.

    Reported, not reconciled: the two are different supports, and a region label that no
    window carries is a fact about the aggregation, not an error to patch here.
    """
    print("\n  REPORT vs COMPOSITION (region row against its own windows):")
    for _, r in reg.iterrows():
        g = comps.get(r.region)
        if g is None:
            print(f"    {r.region:8s} no composition file")
            continue
        S = set(str(r.admissible).split('|')) if isinstance(r.admissible, str) and r.admissible else set()
        eq = float(g[g.admissible == r.admissible].fraction.sum())
        ov = float(sum(row.fraction for row in g.itertuples()
                       if S & set(str(row.admissible).split('|'))))
        top = g.sort_values('fraction', ascending=False).iloc[0]
        n_mismatch = '' if int(g.n_windows_total.iloc[0]) == int(r.n_windows) else \
            f"  ** n_windows {int(r.n_windows)} vs {int(g.n_windows_total.iloc[0])} **"
        flags = []
        if not (S & set(str(top.admissible).split('|'))):
            flags.append(f"modal window set {top.admissible} ({top.fraction:.0%}) shares no "
                         f"archetype with the region verdict")
        if eq == 0:
            flags.append("no window carries the region's own set")
        print(f"    {r.region:8s} region={r.admissible or '(none)':38s} "
              f"exact={eq:5.1%} overlap={ov:5.1%}{n_mismatch}")
        for f in flags:
            print(f"             ** {f}")


def report_dead_axis(reg):
    """The delta_beta column, in the terms panel 1 marks it with."""
    ev = dead_axis_evidence(reg)
    n_nofit = int(ev.no_fit.sum())
    st = ', '.join(sorted(ev[f'status_{DEAD_AXIS}'].unique()))
    n_dead = int((ev[f'status_{DEAD_AXIS}'] == 'unavailable').sum())
    print(f"\n  DEAD AXIS ({DEAD_AXIS}): status {st} in {n_dead}/{len(ev)} regions, so it "
          f"never removed an archetype anywhere.")
    print(f"    {n_nofit} regions return no usable local fit (coverage 0, reason not_fitted); "
          f"the rest fit but resolve no direction.")
    for _, e in ev.iterrows():
        print(f"    {e.region:8s} coverage {e.delta_beta_coverage:5.1%}  "
              f"n_indep {int(e.delta_beta_n_indep):2d}  reason {e.delta_beta_reason}")


def main(root, cover=0.80, width=11.0, matrix_height=0.78, strip_height=0.95,
         matrix_kw=None, composition_kw=None):
    """Two figures off one load, one log. Panel styling that differs between variants goes
    in the *_kw dicts, so it can be tuned from the call site without a mode flag."""
    reg, comps = load(root)
    if not len(reg):
        print(f"No region-level archetype reports under {root}")
        return
    ncat = len(CATALOGUE)
    print(f"Regions: {len(reg)}  (level == 'region' row of each archetype report)")
    print(f"Catalogue: {ncat} entries;  MIGRATION_WIDENS_BETA = {MIGRATION_WIDENS_BETA};  "
          f"composition decimation {COMPOSITION_DECIMATE_KM:.0f} km")
    print("\n  ORDER (shared by both figures, fewest admissible first): "
          + ', '.join(reg.region))

    print("\n  COLLAPSE:")
    for _, r in reg.iterrows():
        print(f"    {r.region:8s} {int(r.n_admissible):2d}/{ncat} admissible  "
              f"({int(r.n_admissible_unwidened)} unwidened)  {r.verdict:11s} {r.admissible}")
    if (reg.n_admissible == reg.n_admissible_unwidened).all():
        print(f"    widening cost 0 archetypes in every region "
              f"(MIGRATION_WIDENS_BETA = {MIGRATION_WIDENS_BETA}), so the unwidened count is "
              f"stated under figure 1 rather than drawn")

    used = {st for a in AXIS_ORDER for st in reg[f'status_{a}'].unique()}
    print("\n  CELL STATUSES on figure 1: " + ', '.join(k for k in STATUS_ORDER if k in used)
          + ("" if set(STATUS_ORDER) <= used else
             "  (absent, so no key drawn: "
             + ', '.join(k for k in STATUS_ORDER if k not in used) + ")"))

    report_dead_axis(reg)
    report_disagreement(reg, comps)
    print("\n  SET CODES (figure 2 segment labels): "
          + ', '.join(f'{v}={k}' for k, v in CODES.items()))

    # Figure 1: the axes, and what they leave standing.
    fig, ax = plt.subplots(figsize=(width, matrix_height * len(reg) + 2.6))
    panel_axis_matrix(ax, reg, ncat, **(matrix_kw or {}))
    fig.suptitle('Landscape class is not identifiable from one statistic', fontsize=13)
    fig.text(0.5, 0.005,
             f'{DEAD_AXIS} is shown dead, not live: it is widened to every band in every '
             f'region, so it excludes nothing and is not a working fifth axis.\n'
             f'Migration widening is off (MIGRATION_WIDENS_BETA = {MIGRATION_WIDENS_BETA}), '
             f'so n_admissible_unwidened equals n_admissible in every region.',
             ha='center', va='top', fontsize=7.5, color='0.35')
    out1 = os.path.join(root, 'vector_figure_axes.png')
    fig.savefig(out1, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Figure 2: the same regions as mixtures, at a height the segment labels can use.
    fig, ax = plt.subplots(figsize=(width, strip_height * len(reg) + 2.2))
    panel_composition(ax, reg, comps, cover=cover, **(composition_kw or {}))
    fig.text(0.5, 0.005,
             'Set codes: ' + ', '.join(f'{v}={k}' for k, v in CODES.items()) + '.\n'
             'No confidence interval is drawn: at these n_eff the standard error on a '
             'fraction rivals the fraction.',
             ha='center', va='top', fontsize=7.5, color='0.35')
    out2 = os.path.join(root, 'vector_figure_composition.png')
    fig.savefig(out2, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Saved: {out1}")
    print(f"  Saved: {out2}")


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else _REGION_BASE
    sys.stdout = Tee(os.path.join(root, 'vector_figure_log.txt'))
    main(root)
