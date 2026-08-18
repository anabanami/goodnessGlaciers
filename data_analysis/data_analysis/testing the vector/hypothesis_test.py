"""The paper's claim, measured rather than asserted.

Landscape class is not identifiable from a single measured statistic, and a transect bounds
the class rather than pinpointing it. This script measures the bound: how far the admissible
set falls as axes are added, and whether the joint structure does anything a marginals-
preserving null does not.

>>> Falsified here if the subset curve had not fallen, or if resolution had beaten the null.
It falls 11 -> ~2 and resolution sits below the null, so the bound is the result and the
resolution rate carries no inference. The claim's other falsifiers are tested elsewhere and
are stated in NEXT.md under HYPOTHESIS.

Three parts, because the preregistered single-statistic test cannot fail:

1. SINGLE AXIS. Runs the match-all scan with one axis live at a time. An axis that is not
   live is widened to every value it can take, so it constrains nothing. >>> Resolution here
   is structurally capped: every axis except beta floors at 2 or more admissible entries
   (printed below), because an entry that does not constrain the live axis always matches.
   Beta floors at 1 only because SHATTERED is the sole entry admitting `chaotic`, so its
   3.1% is that entry's uncontested footprint and not a bed measurement. That column
   re-derives the catalogue's shape and says nothing about the bed. Only the set-size
   column carries information.

2. SUBSET CURVE. Resolution rate over every subset of the constraining axes, which is what
   "a range of descriptors" actually claims — how many are needed, and which combination.
   beta_spread is not among them: ALL_AXES is derived from the catalogue and no entry
   constrains it, so it never enters this scan in the first place.

3. PERMUTATION NULL. Shuffles each axis's observed label set independently across units,
   preserving every marginal and destroying only the joint structure. This is the null the
   resolution rate has never had. If shuffled units resolve nearly as often, the rate is
   marginals plus catalogue sparsity, not identification. Pooled across all units, and again
   within region so regional marginals are held fixed.

Reports admissible-set size, not verdict: 0 = out of catalogue, 1 = resolved, >1 = degenerate.
Verdict adds RESOLVED-WITH-EXTERNAL, which turns on discriminator logic irrelevant here.

    python hypothesis_test.py [individual_region_TEST] [n_permutations]
"""
import glob, itertools, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee
import landscape_vector as lv
from landscape_vector import (load_region, build_vector, observe, match, units_from,
                              CATALOGUE, ALL_AXES, AXIS_VALUES)

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / 'individual_region_TEST')
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
LEVEL = 'window'
SUBSET_AXES = sorted(ALL_AXES)
ALL_OF = {a: frozenset(AXIS_VALUES[a]) for a in ALL_AXES}


def n_match(sets):
    return len(match({a: {'set': s} for a, s in sets.items()}))


def live_only(sets, live):
    return n_match({a: (s if a in live else ALL_OF[a]) for a, s in sets.items()})


def floors():
    """Smallest set a single axis could ever produce, over every value it can take."""
    out = {}
    for a in sorted(ALL_AXES):
        out[a] = min((live_only({x: (frozenset({v}) if x == a else ALL_OF[x])
                                 for x in ALL_AXES}, {a}), v) for v in AXIS_VALUES[a])
    return out


def observations(root, widen=True):
    """One dict of axis -> admissible label set per window unit, plus its region.

    widen=False switches the migration widening off, so beta carries only what was measured.
    observe() reads the flag from its own module at call time, hence the patch."""
    lv.MIGRATION_WIDENS_BETA = widen
    regions, units = [], []
    for f in sorted(glob.glob(os.path.join(root, '*', 'window_csvs', '*_window_stats.csv'))):
        region = os.path.basename(os.path.dirname(os.path.dirname(f)))
        df, pflag = load_region(f, quiet=True)
        if df is None:
            continue
        for unit, g in units_from(df, LEVEL):
            obs = observe(build_vector(g, f'{LEVEL}:{unit}', pflag), pflag)
            regions.append(region)
            units.append({a: frozenset(obs[a]['set']) for a in ALL_AXES})
    return np.array(regions), units


def stats(sizes):
    s = np.asarray(sizes)
    return dict(median=float(np.median(s)), mean=round(float(s.mean()), 2),
                resolved=round(100 * float((s == 1).mean()), 1),
                ooc=round(100 * float((s == 0).mean()), 1))


def permute(units, regions, rng, within_region):
    """Shuffle each axis independently, so marginals survive and the joint does not."""
    out = [dict(u) for u in units]
    blocks = ([np.where(regions == r)[0] for r in np.unique(regions)] if within_region
              else [np.arange(len(units))])
    for a in ALL_AXES:
        for idx in blocks:
            vals = [units[i][a] for i in idx]
            for i, v in zip(idx, rng.permutation(len(idx))):
                out[i][a] = vals[v]
    return out


def plot_subset_curve(curve, tag):
    """The claim in one panel: how far the admissible set falls as axes are added, and
    where resolution stops improving. The flat step is delta_beta contributing nothing."""
    x = curve['n_axes'].to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                 gridspec_kw={'height_ratios': [1.15, 1]})

    a1.plot(x, curve['mean'], 'o-', color='#1a6faf', lw=1.8, ms=6, label='mean')
    a1.plot(x, curve['median'], 's--', color='0.45', lw=1.2, ms=5, label='median')
    a1.set_ylabel(f'admissible archetypes\nof {len(CATALOGUE)}')
    a1.set_ylim(0, len(CATALOGUE) + 0.6)
    a1.legend(fontsize=8, frameon=False)

    a2.plot(x, curve['resolved'], 'o-', color='#2e7d32', lw=1.8, ms=6,
            label='resolved to one entry')
    a2.plot(x, curve['ooc'], '^-', color='#b22222', lw=1.4, ms=5,
            label='out of catalogue')
    a2.set_ylabel('% of units')
    a2.set_xlabel('axes live')
    a2.legend(fontsize=8, frameon=False)

    # The dead axis reads as a flat step: whichever k adds it changes nothing.
    flat = [k for k in range(1, len(curve))
            if np.isclose(curve['resolved'].iloc[k], curve['resolved'].iloc[k - 1])
            and np.isclose(curve['mean'].iloc[k], curve['mean'].iloc[k - 1])]
    for k in flat:
        for ax in (a1, a2):
            ax.axvspan(k - 1, k, color='0.85', alpha=0.55, zorder=0)
        a1.annotate('no change:\nthe added axis\nexcludes nothing',
                    xy=(k - 0.5, a1.get_ylim()[1] * 0.72), ha='center',
                    fontsize=7.5, color='0.35')

    # Initials, because the full combination names overrun the axes by n_axes = 4.
    sym = {'beta_class': 'B', 'relief_class': 'R', 'velocity_band': 'V',
           'elevation_class': 'E', 'delta_beta': 'D'}
    for _, r in curve.iterrows():
        lab = '+'.join(sym.get(a, a) for a in r['best_combination'].split('+')) \
            if r['best_combination'] != '(none)' else 'none'
        last = r['n_axes'] == curve['n_axes'].max()
        a1.annotate(lab, xy=(r['n_axes'], r['mean']), xytext=(-4 if last else 0, 10),
                    textcoords='offset points', ha='right' if last else 'center',
                    fontsize=7.5, color='0.30')

    for ax in (a1, a2):
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xticks(x)
    fig.suptitle('No single statistic identifies landscape class', fontsize=13)
    fig.text(0.5, 0.005,
             f'Best axis combination at each size, {LEVEL} units, '
             f'migration widening {"on" if tag == "widened" else "off"}.  '
             f'B beta, R relief, V velocity, E elevation, D delta-beta.\n'
             f'A set of one is a bound that excluded ten entries, not an identification.',
             ha='center', va='top', fontsize=7.5, color='0.35')
    plt.tight_layout()
    out = os.path.join(ROOT, f'hypothesis_test_subset_curve_{tag}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def run(regions, units, tag):
    full = [n_match(u) for u in units]

    print("=== 1. SINGLE AXIS (resolution is structurally impossible; read set size only) ===")
    fl = floors()
    rows = {}
    for a in sorted(ALL_AXES):
        rows[a] = {**stats([live_only(u, {a}) for u in units]),
                   'floor': fl[a][0], 'floor_at': fl[a][1]}
    rows['(none live)'] = {**stats([n_match({x: ALL_OF[x] for x in ALL_AXES})] * len(units)),
                           'floor': len(CATALOGUE), 'floor_at': '-'}
    rows['FULL VECTOR'] = {**stats(full), 'floor': '', 'floor_at': ''}
    print(pd.DataFrame(rows).T.to_string(), "\n")

    print("=== 2. SUBSET CURVE (best combination at each size) ===")
    curve = []
    for k in range(len(SUBSET_AXES) + 1):
        best = None
        for combo in itertools.combinations(SUBSET_AXES, k):
            st = stats([live_only(u, set(combo)) for u in units])
            if best is None or st['resolved'] > best[1]['resolved']:
                best = (combo, st)
        curve.append({'n_axes': k, 'best_combination': '+'.join(best[0]) or '(none)',
                      **best[1]})
    print(pd.DataFrame(curve).to_string(index=False), "\n")

    print(f"=== 3. PERMUTATION NULL ({NPERM} shuffles, marginals preserved) ===")
    obs_st = stats(full)
    null_rows = []
    for label, wr in [('pooled', False), ('within region', True)]:
        rng = np.random.default_rng(0)
        null = [stats([n_match(u) for u in permute(units, regions, rng, wr)])
                for _ in range(NPERM)]
        res = np.array([n['resolved'] for n in null])
        med = np.array([n['median'] for n in null])
        # Both tails: the vector resolving LESS often than shuffled data is a real outcome
        # and p_high alone would report it as a flat null result.
        p_high = (1 + int((res >= obs_st['resolved']).sum())) / (NPERM + 1)
        p_low = (1 + int((res <= obs_st['resolved']).sum())) / (NPERM + 1)
        z = (obs_st['resolved'] - res.mean()) / max(res.std(ddof=1), 1e-9)
        null_rows.append({'null': label, 'n_perm': NPERM,
                          'observed_resolved_%': obs_st['resolved'],
                          'null_resolved_mean_%': round(float(res.mean()), 2),
                          'null_resolved_sd': round(float(res.std(ddof=1)), 2),
                          'p_better_than_null': round(p_high, 4),
                          'p_worse_than_null': round(p_low, 4), 'z': round(float(z), 2),
                          'observed_median_set': obs_st['median'],
                          'null_median_set_mean': round(float(med.mean()), 2),
                          'null_median_set_sd': round(float(med.std(ddof=1)), 2)})
        print(f"  {label:14s} resolved: observed {obs_st['resolved']}%  null "
              f"{res.mean():.1f}% ± {res.std(ddof=1):.1f}  (z = {z:+.1f}, "
              f"p_better = {p_high:.3f}, p_worse = {p_low:.3f})")
        print(f"  {'':14s} median set size: observed {obs_st['median']:.0f}  null "
              f"{med.mean():.2f} ± {med.std(ddof=1):.2f}")

    plot_subset_curve(pd.DataFrame(curve), tag)

    # Every summary persisted: these are the numbers that get quoted, so they must not
    # live only in a terminal buffer.
    for name, obj in [('units', pd.DataFrame({'region': regions,
                                              'n_admissible_full': full})),
                      ('single_axis', pd.DataFrame(rows).T.rename_axis('live')),
                      ('subset_curve', pd.DataFrame(curve)),
                      ('null', pd.DataFrame(null_rows))]:
        path = os.path.join(ROOT, f'hypothesis_test_{name}_{tag}.csv')
        obj.to_csv(path, index=(name == 'single_axis'))
        print(f"  Saved: {path}")
    return full


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'hypothesis_test_log.txt'))
    out = {}
    for tag, widen in [('widened', True), ('unwidened', False)]:
        regions, units = observations(ROOT, widen=widen)
        print(f"\n{'#'*100}\n#  MIGRATION WIDENING {'ON' if widen else 'OFF'}  ({tag})\n"
              f"{'#'*100}")
        print(f"{len(units)} {LEVEL} units, {len(np.unique(regions))} regions, "
              f"{len(CATALOGUE)} catalogue entries\n")
        out[tag] = run(regions, units, tag)

    # What the correction is worth, per region, on the same units.
    print(f"\n{'='*100}\n=== WIDENING ON vs OFF, resolved % by region ===")
    cmp = pd.DataFrame({'region': regions,
                        'widened': [n == 1 for n in out['widened']],
                        'unwidened': [n == 1 for n in out['unwidened']],
                        'ooc_widened': [n == 0 for n in out['widened']],
                        'ooc_unwidened': [n == 0 for n in out['unwidened']]})
    g = cmp.groupby('region').agg(n=('widened', 'size'),
                                  res_wid=('widened', lambda s: round(100*s.mean(), 1)),
                                  res_unwid=('unwidened', lambda s: round(100*s.mean(), 1)),
                                  ooc_wid=('ooc_widened', lambda s: round(100*s.mean(), 1)),
                                  ooc_unwid=('ooc_unwidened', lambda s: round(100*s.mean(), 1)))
    print(g.to_string())
    print(f"\n  pooled: resolved {100*cmp.widened.mean():.1f}% -> "
          f"{100*cmp.unwidened.mean():.1f}%, out-of-catalogue "
          f"{100*cmp.ooc_widened.mean():.1f}% -> {100*cmp.ooc_unwidened.mean():.1f}%")
    cmp.to_csv(os.path.join(ROOT, 'hypothesis_test_widening_compare.csv'), index=False)
