"""Landscape catalogue examples: one row per catalogue entry, each with one example window.

Columns: the entry's cartoon, the example window over its region's other windows, and the
entry's bounds beside the window's values. The ranked candidates for each entry are printed,
and any pick can be overridden at the call site.

    python catalogue_examples.py [output_tree]

The output tree defaults to OUTPUT_BASE_PATH in loading.py.
"""
import glob, os, sys, textwrap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401  (sets sys.path + cwd to ODSA/)
from config import Tee, element_label
from loading import OUTPUT_BASE_PATH as _REGION_BASE
from bed_character import (BED_CLASSES, RELIEF_CLASSES, ELEVATION_CLASSES, TRUNC_OFFSET,
                           dataset_name, segment_lengths, write_metadata, _tkey)
from landscape_vector import (CATALOGUE, VELOCITY_CLASSES, AXIS_VALUES, K_SIGMA, observe,
                              EXTERNAL, BETA_SYSTEMATIC_ERROR, RELIEF_ERROR_M,
                              ELEVATION_ERROR_M)
from fig6b_maps import load, ARCHETYPE_COLORS, PS71, WINDOW_M, frame, basemap, footprints

NAME = 'catalogue_examples'
TITLE = 'Landscape catalogue: an example window for each entry'
CASE = {c['id']: c for c in CATALOGUE}

CARTOON_DIR = os.path.join('plotting_post-process', 'Landscape archetypes')
CARTOONS = {
    'TRUNK': 'Ice-stream-trunk.png', 'TRUNK-HARD': 'Trunk-hard.png',
    'TRUNK-RELICT': 'Trunk-relict.png', 'ONSET': 'Ice-stream-onset.png',
    'HIGHLAND': 'Crystalline-highland.png', 'RIFT': 'Rift.png', 'BASIN': 'Sedimentary.png',
    'BASIN-HIGH': 'Basin-high.png', 'DISSECTED': 'Deeply-dissected-highland.png',
    'DIVIDE': 'Ice-divide.png', 'SHATTERED': 'Shattered.png',
}

# (catalogue axis, symbol, class table, unit, value format)
AXES = [
    ('beta_class', 'β', BED_CLASSES, '', '.2f'),
    ('relief_class', 'relief', RELIEF_CLASSES, ' m', '.0f'),
    ('velocity_band', 'speed', VELOCITY_CLASSES, ' m/yr', '.1f'),
    ('elevation_class', 'elevation', ELEVATION_CLASSES, ' m', '.0f'),
]

# (landscape vector element, unit, value format)
DESCRIPTORS = [
    ('A_1km', '', '.2f'), ('rms_roughness', ' m', '.0f'), ('eta_wavelength_m', ' m', '.0f'),
    ('hill_count', '', '.0f'), ('skewness', '', '.2f'), ('kurtosis', '', '.2f'),
    ('xi_band', '', '.3g'),
]


def load_all(root):
    """load() plus the report's axis columns and the segment length, and the window-level
    landscape vector indexed by (region, unit). Units repeat between regions, so both keys."""
    d = load(root)
    lengths = segment_lengths(root) or {}
    reps, vecs = [], []
    for region in sorted(d.region.unique()):
        lv = os.path.join(root, region, 'landscape_vector')
        r = pd.read_csv(glob.glob(os.path.join(lv, '*_archetype_report.csv'))[0])
        r = r[r.level == 'window'][['unit', 'needs_external']
                                   + [f'axis_{a[0]}' for a in AXES]]
        csv = glob.glob(os.path.join(root, region, 'window_csvs', '*_window_stats.csv'))[0]
        reps.append(r.assign(region=region, dataset=dataset_name(csv)))
        v = pd.read_csv(glob.glob(os.path.join(lv, '*_landscape_vector.csv'))[0])
        vecs.append(v[v.level == 'window'].assign(region=region))
    d = d.merge(pd.concat(reps), on=['region', 'unit'], how='left')
    d['admissible'] = d.admissible.fillna('')
    d['needs_external'] = d.needs_external.fillna('')
    d['key'] = d.region + '/' + d.unit.str.replace('window:', '', regex=False)
    d['segment_km'] = [lengths.get((ds, _tkey(t), int(s)), np.nan) / 1000
                       for ds, t, s in zip(d.dataset, d.trajectory, d.segment)]
    return d, pd.concat(vecs).set_index(['region', 'unit']).sort_index()


def observed(vec, w):
    """The window's landscape vector row and the classifier's own observation of it."""
    row = vec.loc[(w.region, w.unit)]
    return row, observe(row, row.get('processing_flag'))


def break_margin(case, obs):
    """Smallest distance from a class break on the entry's constrained axes, in sigma."""
    out = []
    for axis, sym, classes, _, _ in AXES:
        if axis not in case['c']:
            continue
        v, s = obs[axis]['value'], obs[axis]['sigma']
        breaks = [hi for _, _, hi in classes if np.isfinite(hi)]
        ok = np.isfinite(v) and np.isfinite(s) and s > 0
        out.append((min(abs(v - b) for b in breaks) / s if ok else 0.0, sym))
    return min(out)


def candidates(d, vec, entry, min_segment_km):
    """Windows admissible for the entry, restricted to the smallest admissible set size and
    ranked by segment length, then break margin."""
    c = d[d.admissible.str.split('|').apply(lambda s: entry in s)]
    c = c[c.n_admissible == c.n_admissible.min()].copy()
    margins = [break_margin(CASE[entry], observed(vec, w)[1]) for _, w in c.iterrows()]
    c['margin_sigma'] = [z for z, _ in margins]
    c['margin_axis'] = [a for _, a in margins]
    c['long'] = c.segment_km >= min_segment_km
    return c.sort_values(['long', 'margin_sigma'], ascending=False)


def choose(d, entry, cands, override):
    if override is None:
        return cands.iloc[0] if len(cands) else None
    hit = d[(d.key == override) | (d.key.str.split('/', n=1).str[1] == override)]
    if len(hit) != 1:
        raise ValueError(f"{entry}: override {override!r} matches {len(hit)} non-transition "
                         f"windows; give it as REGION/traj|sN|wN")
    w = hit.iloc[0]
    if entry not in w.admissible.split('|'):
        raise ValueError(f"{entry}: {w.key} is not admissible for {entry} "
                         f"(admissible: {w.admissible or 'none'})")
    return w


def print_candidates(entry, cands, w, n_print, min_segment_km):
    print(f"\n=== {entry} ===")
    if not len(cands):
        print("  no admissible window")
        return
    n = int(cands.n_admissible.iloc[0])
    pool = 'admissible for this entry alone' if n == 1 else f'with {n} admissible entries'
    print(f"  {len(cands)} windows {pool}, ranked by segment >= {min_segment_km:g} km, then "
          f"margin (distance from a class break on this entry's axes, in sigma)")
    top = cands.head(n_print)
    for i, r in enumerate(top.itertuples(), 1):
        mark = '*' if w is not None and r.key == w.key else ' '
        print(f"  {mark}{i:3d}  {r.key:48s} segment {r.segment_km:6.1f} km  "
              f"margin {r.margin_sigma:5.1f} {r.margin_axis:9s}  beta {r.beta:.2f}  "
              f"relief {r.relief_m:4.0f}  elev {r.bed_elev_mean:5.0f}  "
              f"speed {r.measures_speed_mean:6.1f}  [{r.admissible}]")
    if w is not None and w.key not in set(top.key):
        print(f"  * override {w.key}  segment {w.segment_km:.1f} km  [{w.admissible}]")


def check_report(vec, w):
    """The admitted classes come from the report and the envelope from observe(), so flag
    any axis on which the two disagree."""
    if w is None:
        return
    _, obs = observed(vec, w)
    for axis, *_ in AXES:
        rep = set(str(w[f'axis_{axis}']).split(','))
        if obs[axis]['set'] != rep:
            print(f"  WARNING {w.key} {axis}: report admits {sorted(rep)}, "
                  f"observe() admits {sorted(obs[axis]['set'])}")


def bound_text(case, axis, sym, classes, unit):
    """The entry's bound on one axis, from the class table."""
    allowed = case['c'].get(axis)
    names = [n for n, _, _ in classes]
    if allowed is None or set(names) <= allowed:
        return f"{sym} unconstrained"
    runs, run = [], []
    for n, lo, hi in classes:
        if n in allowed:
            run.append((lo, hi))
        elif run:
            runs.append(run)
            run = []
    runs += [run] if run else []
    parts = []
    for r in runs:
        lo, hi = r[0][0], r[-1][1]
        parts.append(f"{sym} < {hi:g}{unit}" if not np.isfinite(lo)
                     else f"{sym} ≥ {lo:g}{unit}" if not np.isfinite(hi)
                     else f"{lo:g} ≤ {sym} < {hi:g}{unit}")
    return ' or '.join(parts) + f" ({', '.join(n for n in names if n in allowed)})"


def value_text(o, fmt, unit):
    """Value with the classifier's K_SIGMA envelope. A window without MEaSUREs error coverage
    has no velocity envelope and admits every band."""
    v, s = o['value'], o['sigma']
    if not np.isfinite(v):
        return 'no value'
    if not np.isfinite(s):
        return f"{v:{fmt}}{unit}, no σ"
    return f"{v:{fmt}} ± {K_SIGMA * s:{fmt}}{unit}"


def draw_cartoon(ax, entry, st):
    ax.axis('off')
    if entry in CARTOONS:
        ax.imshow(plt.imread(os.path.join(CARTOON_DIR, CARTOONS[entry])))
    ax.set_title(textwrap.fill(f"{entry}: {CASE[entry]['name']}", st['name_wrap']),
                 loc='left', fontsize=st['name_fontsize'])


def draw_map(fig, cell, d, entry, w, st):
    if w is None:
        fig.add_subplot(cell).axis('off')
        return
    ax = fig.add_subplot(cell, projection=PS71)
    g = d[d.region == w.region]
    xlim, ylim = frame(g)
    basemap(ax, xlim, ylim, gridlines=st['gridlines'])
    rest, ex = g[g.unit != w.unit], g[g.unit == w.unit]
    footprints(ax, rest, pd.Series('other', index=rest.index), {'other': st['other_color']},
               lw=st['other_lw'], alpha=st['other_alpha'])
    footprints(ax, ex, pd.Series(entry, index=ex.index), ARCHETYPE_COLORS,
               lw=st['example_lw'], alpha=st['example_alpha'])
    ax.set_title(f"{w.region}   {w.key.split('/', 1)[1]}", fontsize=st['map_title_fontsize'])


TEXT_COLUMNS = ['bound', 'value', 'admitted', 'descriptor', 'measured']


def draw_text(ax, entry, w, row, obs, st):
    """Column 3 text, all at x = 0 until pack_columns() places it.
    Returns (text, text column, whether its width counts)."""
    ax.axis('off')
    texts = []

    def put(col, line, s, measure=True, **kw):
        t = ax.annotate(s, xy=(0, 1), xycoords='axes fraction',
                        xytext=(0, -line * st['line_pt']), textcoords='offset points',
                        ha='left', va='top', fontsize=st['text_fontsize'], **kw)
        texts.append((t, col, measure))

    head = dict(color=st['header_color'])
    if w is None:
        put('bound', 0, 'No window in these regions is admissible for this entry',
            measure=False, fontweight='bold')
    else:
        others = [e for e in w.admissible.split('|') if e != entry]
        ext = [e for e in str(w.needs_external).split(',') if e]
        put('bound', 0, 'A window admissible for this entry'
            + ('; also admissible: ' + ', '.join(others) if others else ', and for no other entry')
            + ('' if not ext else
               ('; separating them needs ' if others else '; reading it needs ')
               + ', '.join(ext)),
            measure=False, fontweight='bold')
        put('value', 1.5, f'window value ± {K_SIGMA:g}σ', **head)
        put('admitted', 1.5, 'classes admitted by the envelope', **head)
        put('descriptor', 1.5, 'measured values for this window', measure=False, **head)
    put('bound', 1.5, 'catalogue bound', **head)
    for i, (axis, sym, classes, unit, fmt) in enumerate(AXES):
        put('bound', 2.5 + i, bound_text(CASE[entry], axis, sym, classes, unit))
        if w is not None:
            admitted = set(str(w[f'axis_{axis}']).split(','))
            put('value', 2.5 + i, value_text(obs[axis], fmt, unit))
            put('admitted', 2.5 + i, ', '.join(n for n in AXIS_VALUES[axis] if n in admitted))
    if w is None:
        return texts
    for i, (col, unit, fmt) in enumerate(DESCRIPTORS):
        put('descriptor', 2.5 + i, element_label(col))
        put('measured', 2.5 + i,
            f"{row[col]:{fmt}}{unit}" if np.isfinite(row[col]) else 'no value')
    return texts


def pack_columns(fig, texts, st):
    """Set each text column's x to the widest text in the columns to its left, over all rows."""
    renderer = fig.canvas.get_renderer()
    width = dict.fromkeys(TEXT_COLUMNS, 0.0)
    for t, col, measure in texts:
        if measure:
            width[col] = max(width[col], t.get_window_extent(renderer).width * 72 / fig.dpi)
    x = 0.0
    for col in TEXT_COLUMNS:
        for t, c, _ in texts:
            if c == col:
                t.set_position((x, t.get_position()[1]))
        x += width[col] + (st['block_gap_pt'] if col == 'admitted' else st['col_gap_pt'])


def render(d, vec, entries, picks, out, st):
    n = len(entries)
    fig = plt.figure(figsize=(sum(st['col_widths']), st['row_height'] * n))
    gs = fig.add_gridspec(n, 3, width_ratios=st['col_widths'], left=0, right=1, bottom=0,
                          top=1, hspace=st['hspace'], wspace=st['wspace'])
    texts = []
    for i, entry in enumerate(entries):
        w = picks[entry]
        row, obs = observed(vec, w) if w is not None else (None, None)
        draw_cartoon(fig.add_subplot(gs[i, 0]), entry, st)
        draw_map(fig, gs[i, 1], d, entry, w, st)
        texts += draw_text(fig.add_subplot(gs[i, 2]), entry, w, row, obs, st)
    pack_columns(fig, texts, st)
    fig.savefig(out, dpi=st['dpi'], bbox_inches='tight')
    plt.close(fig)


def listed(ids):
    return ids[0] if len(ids) == 1 else ', '.join(ids[:-1]) + ' and ' + ids[-1]


def caption(d, entries, cands, picks, overrides, min_segment_km, root):
    shown = [e for e in entries if picks[e] is not None]
    hand = [e for e in shown if e in overrides]
    never_alone = [e for e in shown if cands[e].n_admissible.min() > 1]
    shared = [e for e in shown if picks[e].n_admissible > 1]
    short = [e for e in shown if picks[e].segment_km < min_segment_km]
    no_length = [e for e in shown if not np.isfinite(picks[e].segment_km)]
    no_window = [e for e in entries if picks[e] is None]
    no_cartoon = [e for e in entries if e not in CARTOONS]

    s = [
        f"Each catalogue entry is paired with a measured window, showing which entries the four "
        f"classifying axes isolate and which they leave degenerate.",
        f"One row per landscape catalogue entry. Each row shows a window admissible for this "
        f"entry, taken from the {len(d)} non-transition windows of the {d.region.nunique()} "
        f"regions in {os.path.basename(os.path.normpath(root))}. Admissibility does not claim "
        f"that the archetype is correct.",
        "Left: the entry's cartoon.",
        f"Centre: the region's windows as {WINDOW_M // 1000} km along-track footprints in grey, "
        f"and the example window in its Fig 6b archetype colour. The title gives the region and "
        f"the window's unit ID.",
        f"Right, first block: the entry's bound on each of the four classifying axes, the "
        f"window's value with its ±{K_SIGMA:g}σ envelope, and the classes admitted by that "
        f"envelope. Admissibility is decided on the envelope, so the class of the value itself "
        f"can fall outside the entry's bound. The β σ combines the formal fit error with a "
        f"systematic error of {BETA_SYSTEMATIC_ERROR:g}, the velocity σ is the sampled MEaSUREs "
        f"error, and relief and elevation carry the nominal Bedmap3 errors of "
        f"{RELIEF_ERROR_M:g} m and {ELEVATION_ERROR_M:g} m [Pritchard_2025].",
        "Right, second block: the window's measured values of the seven continuous descriptors. "
        "The catalogue sets no per-entry ranges on these, and a single window has no spread. "
        "The amplitude descriptors A_1km and ξ_band are not comparable between rows, because "
        "the example windows come from different surveys.",
    ]
    rule = (f"taken from the smallest admissible set that contains its entry, preferring "
            f"segments at least {min_segment_km:g} km long and then the largest distance from a "
            f"class break.")
    if not hand:
        s.append(f"Each example is {rule}")
    elif len(hand) < len(shown):
        s.append(f"The examples for {listed(hand)} were chosen by hand. Each other example is "
                 f"{rule}")
    else:
        s.append("Every example was chosen by hand.")
    if never_alone:
        s.append(f"No window in these regions is admissible for {listed(never_alone)} alone.")
    if shared:
        s.append("Rows with more than one admissible entry name the others.")
    if short:
        s.append(f"The examples for {listed(short)} come from segments shorter than "
                 f"{min_segment_km:g} km, so their β carries a band-truncation offset of about "
                 f"+{TRUNC_OFFSET:.1f}.")
    if no_length:
        s.append(f"The examples for {listed(no_length)} have no length in segment_lengths.csv.")
    if no_window:
        s.append(f"No window in these regions is admissible for {listed(no_window)}.")
    if no_cartoon:
        s.append(f"{listed(no_cartoon)} {'has' if len(no_cartoon) == 1 else 'have'} no cartoon.")
    if any(e for x in shown for e in str(picks[x].needs_external).split(',') if e):
        s.append("Externals named on a row are observables that ODSA cannot supply from RES; "
                 "the externals key gives what each one reads.")
    return ' '.join(s)


def externals_used(entries, picks):
    """The externals named on the rows, as observable -> what reading it requires."""
    used = sorted({e for x in entries if picks[x] is not None
                   for e in str(picks[x].needs_external).split(',') if e})
    return {e: EXTERNAL[e] for e in used}


def main(root, entries, overrides, min_segment_km, n_print, **st):
    unknown = [e for e in [*entries, *overrides] if e not in CASE]
    if unknown:
        raise ValueError(f"not catalogue entries: {unknown}")
    d, vec = load_all(root)
    print(f"{len(d)} non-transition windows, {d.region.nunique()} regions")

    cands, picks = {}, {}
    for e in entries:
        cands[e] = candidates(d, vec, e, min_segment_km)
        picks[e] = choose(d, e, cands[e], overrides.get(e))
        print_candidates(e, cands[e], picks[e], n_print, min_segment_km)
        check_report(vec, picks[e])

    print("\n=== picks ===")
    for e in entries:
        w = picks[e]
        if w is None:
            print(f"  {e:12s} none")
            continue
        print(f"  {e:12s} {w.key:48s} n_admissible {int(w.n_admissible)}  "
              f"segment {w.segment_km:.1f} km{'  (override)' if e in overrides else ''}")

    out = os.path.join(root, 'landscape_vector', f'{NAME}.png')
    render(d, vec, entries, picks, out, st)
    meta = write_metadata(out, TITLE,
                          caption(d, entries, cands, picks, overrides, min_segment_km, root),
                          externals=externals_used(entries, picks))
    print(f"\n  Saved: {out}")
    print(f"  Saved: {meta}")


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else _REGION_BASE
    os.makedirs(os.path.join(root, 'landscape_vector'), exist_ok=True)
    sys.stdout = Tee(os.path.join(root, 'landscape_vector', f'{NAME}_log.txt'))
    main(
        root,
        entries=['TRUNK', 'TRUNK-HARD', 'TRUNK-RELICT', 'ONSET', 'HIGHLAND', 'RIFT', 'BASIN',
                 'BASIN-HIGH', 'DISSECTED', 'DIVIDE', 'SHATTERED'],
        # entry -> 'REGION/traj|sN|wN', or the bare traj|sN|wN where it is unique
        overrides={},
        min_segment_km=50,
        n_print=10,
        col_widths=(3.0, 3.0, 8.0), row_height=2.3, hspace=0.35, wspace=0.08, dpi=450,
        name_fontsize=10, name_wrap=30, map_title_fontsize=8,
        text_fontsize=7.5, line_pt=10.5, header_color='0.4',
        col_gap_pt=12, block_gap_pt=30,
        other_color='0.7', other_lw=1.5, other_alpha=0.6,
        example_lw=3.5, example_alpha=1.0, gridlines=False,
    )
