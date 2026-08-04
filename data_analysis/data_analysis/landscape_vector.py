import os, sys, glob, itertools
import numpy as np
import pandas as pd
from config import Tee, PROCESSING_FLAG_NOTE as _FLAG_NOTE, processing_flag_of as region_flag
from bed_character import (BED_CLASSES, RELIEF_CLASSES, ELEVATION_CLASSES,
                           discover_window_csvs, select_region)
from loading import OUTPUT_BASE_PATH as _REGION_BASE

"""
Landscape vector reporting and degeneracy labelling.

Emits the classification vector as numbers (for a bed generator) and, separately,
the set of archetypes that vector admits. Where more than one archetype survives,
the unit is labelled degenerate and the reason is named; the scheme never picks a
single case by assumption.

Usage:
  python landscape_vector.py                        # interactive, discovers from window_csvs/
  python landscape_vector.py Recovery               # partial match
  python landscape_vector.py path/to_window_stats.csv
  python landscape_vector.py individual_region_TEST # walk a tree of region folders
"""

# Output goes beside the window CSV's own region folder, so a tree of per-region
# test outputs each gets its own landscape_vector/.
def output_dir_for(csv_path):
    d = os.path.dirname(os.path.abspath(csv_path))
    if os.path.basename(d) == 'window_csvs':
        d = os.path.dirname(d)
    return os.path.join(d, 'landscape_vector')

# Half-width of the uncertainty envelope used to decide whether a class boundary
# is resolved. 2 sigma; an axis whose envelope crosses a break returns both classes.
K_SIGMA = 2.0

# Anisotropy is not yet validated (no null run, no coverage requirement, cos2 model
# untested). While False, delta_beta is treated as unavailable everywhere, so every
# case constrained on it stays admissible instead of being silently excluded.
ANISOTROPY_TRUSTED = False

# No literature threshold separates a "wide" from a "narrow" within-unit beta spread,
# so the Case E spread constraint is never exercised. Set a value in beta units to
# switch it on once one is defended.
BETA_IQR_WIDE = None

# Nominal velocity error folded into the band assignment. The within-unit spread alone is
# the standard error of the median MEaSUREs value, not MEaSUREs' own accuracy, so on its
# own it lets the axis resolve a 6 m/yr region against the 5 m/yr break it cannot resolve.
# 5.0 is the width of the ramp weighted_anisotropy already treats as untrustworthy
# (zero weight below 5, full above 10); Rignot_2011 quotes 1 m/yr at divides to ~17 m/yr
# under ionospheric perturbation. None restores the over-confident behaviour.
VELOCITY_ERROR_M_YR = 5.0

# Same problem as velocity, one level down: a single-window unit has no across-window
# spread and no formal error, so relief and elevation resolve against their breaks with
# zero uncertainty. 80 of Pensacola's 115 segments are single-window and 29 of them sit
# within 100 m of the flat/subdued break. None invents nothing and keeps the current
# behaviour; the axes are reported as assumed-exact so the fragility stays visible.
RELIEF_ERROR_M = None
ELEVATION_ERROR_M = None

# Partial/unmigrated radar biases beta upward, so the true class may be the one below
# the measured one. True adds that neighbour to the admissible beta set.
MIGRATION_WIDENS_BETA = True

VELOCITY_CLASSES = [
    ('very_low', -np.inf, 5.0),
    ('low',       5.0,   10.0),
    ('moderate',  10.0,  50.0),
    ('fast',      50.0,  np.inf),
]

# ---------------------------------------------------------------------------
# Vector elements. thresholded elements drive classification; the rest are carried
# as continuous numbers only, because no literature threshold exists for them.
# (name, source column, formal-uncertainty column, classifying axis)
ELEMENTS = [
    ('beta',              'beta',               'beta_uncertainty',           'beta_class'),
    ('psd_amplitude_1km', 'psd_amplitude_1km',  'psd_amplitude_uncertainty',  None),
    ('beta_iqr',          None,                 None,                         'beta_spread'),
    ('velocity',          'measures_speed_mean', None,                        'velocity_band'),
    ('relief',            'relief_m',            None,                        'relief_class'),
    ('rms_roughness',     'rms_roughness',       None,                        None),
    ('eta_wavelength_m',  'eta_wavelength_m',    None,                        None),
    ('hill_count',        'hill_count',          None,                        None),
    ('delta_beta',        None,                  None,                        'delta_beta'),
    ('elevation',         'bed_elev_mean',       None,                        'elevation_class'),
    ('skewness',          'skewness',            None,                        None),
    ('kurtosis',          'kurtosis',            None,                        None),
    ('xi_band',           'xi_band',             None,                        None),
]

AXIS_VALUES = {
    'beta_class':      [n for n, _, _ in BED_CLASSES],
    'relief_class':    [n for n, _, _ in RELIEF_CLASSES],
    'elevation_class': [n for n, _, _ in ELEVATION_CLASSES],
    'velocity_band':   [n for n, _, _ in VELOCITY_CLASSES],
    'delta_beta':      ['pos_sig', 'neg_sig', 'zero', 'unreliable'],
    'beta_spread':     ['wide', 'narrow'],
}
NUMERIC_AXES = {'beta_class': BED_CLASSES, 'relief_class': RELIEF_CLASSES,
                'elevation_class': ELEVATION_CLASSES, 'velocity_band': VELOCITY_CLASSES}
MEASURABLE = list(NUMERIC_AXES)  # the axes ODSA can actually put a number on today

# Observables that would break a degeneracy but that ODSA cannot supply from RES.
EXTERNAL = {
    'reflectivity':         'ice-water vs ice-sediment interface (specularity)',
    'composition':          'gravity / magnetics (sediment vs smooth bedrock)',
    'thermal_state':        'frozen vs warm base',
    'flow_history':         'disrupted internal layers, dating (active vs relict)',
    'origin':               'erosional vs structural provenance of the trough-ridge contrast',
    'amplitude_anisotropy': "Cooper_2019 Omega on rms_roughness (not built)",
}

# ---------------------------------------------------------------------------
# Archetype fingerprints. An axis absent from `c` is a "variable" row in the
# catalogue tables: the case makes no claim on it and it can never exclude the case.
CATALOGUE = [
    dict(id='A', name='Ice stream trunk', evidence='Strong',
         c={'beta_class': {'soft'}, 'delta_beta': {'pos_sig'},
            'velocity_band': {'fast'}, 'relief_class': {'flat'}},
         ext=[]),
    dict(id='A-confined', name='Ice stream on confined hard bed (Cooper regime)',
         evidence='Moderate',
         c={'beta_class': {'hard', 'transitional'}, 'velocity_band': {'fast'},
            'relief_class': {'mountainous'}},
         ext=['amplitude_anisotropy']),
    # Streamlining survives shutdown then fades, so delta_beta runs from clearly
    # positive (recent, separable from D) to zero (erased, not separable).
    dict(id='A-relict', name='Relict ice stream (shut down, still smooth)',
         evidence='Moderate',
         c={'beta_class': {'soft'}, 'delta_beta': {'pos_sig', 'zero', 'unreliable'},
            'velocity_band': {'very_low', 'low'}, 'relief_class': {'flat'}},
         ext=['flow_history']),
    dict(id='B', name='Ice stream onset', evidence='Moderate',
         c={'beta_class': {'transitional'}, 'delta_beta': {'pos_sig', 'zero'},
            'velocity_band': {'moderate', 'fast'}, 'relief_class': {'flat', 'subdued'}},
         ext=[]),
    dict(id='C', name='Crystalline highland', evidence='Strong',
         c={'beta_class': {'hard'}, 'delta_beta': {'zero', 'unreliable'},
            'velocity_band': {'very_low', 'low'},
            'relief_class': {'subdued', 'mountainous'},
            'elevation_class': {'emerged', 'elevated'}},
         ext=[]),
    dict(id='C3', name='Rift / tectonic corridor', evidence='Moderate',
         c={'beta_class': {'hard', 'transitional'}, 'delta_beta': {'neg_sig'},
            'relief_class': {'mountainous'}},
         ext=[]),
    dict(id='D', name='Sedimentary basin (quiescent)', evidence='Moderate',
         c={'beta_class': {'soft'}, 'delta_beta': {'zero', 'unreliable'},
            'velocity_band': {'very_low', 'low'}, 'relief_class': {'flat'},
            'elevation_class': {'submerged'}},
         ext=[]),
    dict(id='D3', name='Candidate hard bed (smooth + elevated)', evidence='Moderate',
         c={'beta_class': {'soft'}, 'delta_beta': {'zero', 'unreliable'},
            'velocity_band': {'very_low'},
            'elevation_class': {'emerged', 'elevated'}},
         ext=['composition']),
    dict(id='D4', name='Candidate subglacial lake', evidence='Moderate',
         c={'beta_class': {'soft'}, 'delta_beta': {'zero', 'unreliable'},
            'relief_class': {'flat'}, 'elevation_class': {'submerged'}},
         ext=['reflectivity']),
    dict(id='E', name='Deeply dissected highland (trough-and-interfluve)',
         evidence='Moderate',
         c={'beta_spread': {'wide'}, 'delta_beta': {'pos_sig', 'unreliable'},
            'velocity_band': {'low', 'moderate'}, 'relief_class': {'mountainous'}},
         ext=['origin']),
    dict(id='F', name='Warm-base divide / plateau', evidence='Moderate',
         c={'beta_class': {'transitional'}, 'delta_beta': {'unreliable'},
            'velocity_band': {'very_low'}, 'relief_class': {'flat', 'subdued'},
            'elevation_class': {'submerged', 'emerged'}},
         ext=[]),
    dict(id='G1', name='Shattered / chaotic', evidence='Speculative',
         c={'beta_class': {'chaotic'}},
         ext=[]),
    dict(id='G2', name='Shattered bedrock with structural grain', evidence='Speculative',
         c={'beta_class': {'chaotic'}, 'delta_beta': {'neg_sig'},
            'relief_class': {'mountainous'}, 'elevation_class': {'elevated'}},
         ext=[]),
]

ALL_AXES = sorted({a for c in CATALOGUE for a in c['c']})


# ---------------------------------------------------------------------------
def _agg(g, col, unc_col):
    """Median, within-unit IQR, and a representative sigma combining formal error
    and the standard error of the median."""
    if col is None or col not in g.columns:
        return np.nan, np.nan, np.nan
    v = pd.to_numeric(g[col], errors='coerce').dropna().to_numpy(float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25)) if v.size > 1 else np.nan
    se = 1.253 * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan
    formal = np.nan
    if unc_col and unc_col in g.columns:
        u = pd.to_numeric(g[unc_col], errors='coerce').dropna().to_numpy(float)
        if u.size:
            formal = float(np.median(u))
    parts = [x for x in (se, formal) if np.isfinite(x)]
    sigma = float(np.sqrt(np.sum(np.square(parts)))) if parts else np.nan
    return med, iqr, sigma


def classify_set(value, sigma, classes):
    """Admissible class labels for value +/- K_SIGMA*sigma. NaN value -> every label,
    so a missing observable never excludes anything."""
    names = [n for n, _, _ in classes]
    if not np.isfinite(value):
        return set(names)
    half = K_SIGMA * sigma if np.isfinite(sigma) else 0.0
    lo, hi = value - half, value + half
    out = {n for n, l, h in classes if l <= hi and h > lo}
    return out or set(names)


def observe(vec, pflag):
    """Vector numbers -> admissible label set per classifying axis, with status."""
    obs = {}
    for axis, classes in NUMERIC_AXES.items():
        name = {'beta_class': 'beta', 'relief_class': 'relief',
                'elevation_class': 'elevation', 'velocity_band': 'velocity'}[axis]
        val, sig = vec[f'{name}'], vec[f'{name}_sigma']
        nominal = {'velocity_band': VELOCITY_ERROR_M_YR, 'relief_class': RELIEF_ERROR_M,
                   'elevation_class': ELEVATION_ERROR_M}.get(axis)
        if nominal:
            sig = np.hypot(sig if np.isfinite(sig) else 0.0, nominal)
        s = classify_set(val, sig, classes)
        obs[axis] = dict(set=s, value=val, sigma=sig, exact=not np.isfinite(sig))

    # Migration bias is one-directional (beta reads high), so widen downward only.
    if MIGRATION_WIDENS_BETA and pflag and pflag != 'migrated':
        order = AXIS_VALUES['beta_class']
        i = min(order.index(n) for n in obs['beta_class']['set'])
        if i > 0:
            obs['beta_class']['set'] |= {order[i - 1]}
            obs['beta_class']['widened'] = True

    # delta_beta: unavailable while the estimator is unvalidated.
    obs['delta_beta'] = dict(set=set(AXIS_VALUES['delta_beta']), value=np.nan, sigma=np.nan)

    # beta_spread: computable but unthresholded, so it cannot exclude Case E.
    iqr = vec['beta_iqr']
    if BETA_IQR_WIDE is None or not np.isfinite(iqr):
        s = set(AXIS_VALUES['beta_spread'])
    else:
        s = {'wide'} if iqr >= BETA_IQR_WIDE else {'narrow'}
    obs['beta_spread'] = dict(set=s, value=iqr, sigma=np.nan)

    # assumed-exact is a resolved axis carrying no uncertainty at all, so its class sits
    # on one number. It excludes archetypes as firmly as a measured one; it should not.
    for axis, o in obs.items():
        o['status'] = ('unavailable' if o['set'] == set(AXIS_VALUES[axis])
                       else 'ambiguous' if len(o['set']) > 1
                       else 'assumed-exact' if o.get('exact') else 'resolved')
    return obs


def match(obs):
    """Cases whose every constrained axis is compatible with the observation.
    Compatible means non-empty intersection, so an unresolved axis keeps the case in."""
    out = []
    for case in CATALOGUE:
        if all(obs[a]['set'] & allowed for a, allowed in case['c'].items()):
            exercised = [a for a, allowed in case['c'].items() if obs[a]['set'] <= allowed]
            out.append((case, exercised))
    return out


def resolvable_axes(cases, obs):
    """Axes that are currently unresolved and would exclude at least one admissible
    case if they were sharpened."""
    axes = []
    for a in ALL_AXES:
        if obs[a]['status'] == 'resolved':
            continue
        for case, _ in cases:
            allowed = case['c'].get(a)
            if allowed is not None and not (obs[a]['set'] <= allowed):
                axes.append(a)
                break
    return axes


def verdict(cases, obs):
    """RESOLVED / OUT-OF-CATALOGUE / one of three degeneracy kinds, plus the reason."""
    if not cases:
        return ('OUT-OF-CATALOGUE', '', 'vector matches no archetype fingerprint')
    if len(cases) == 1:
        case, ex = cases[0]
        if case['ext']:
            return ('RESOLVED-WITH-EXTERNAL', ','.join(case['ext']),
                    f"{case['id']} unique, but its own reading needs "
                    + '; '.join(EXTERNAL[e] for e in case['ext']))
        return ('RESOLVED', ','.join(ex), f"{case['id']} unique on {len(ex)} exercised axes")

    ra = resolvable_axes(cases, obs)
    if ra:
        detail = ', '.join(f"{a}[{obs[a]['status']}]" for a in ra)
        return ('DEGENERATE-UNMEASURED', ','.join(ra),
                f"{len(cases)} archetypes admissible; separable in principle on {detail}")
    ext = sorted({e for c, _ in cases for e in c['ext']})
    if ext:
        return ('DEGENERATE-IRREDUCIBLE', ','.join(ext),
                f"{len(cases)} archetypes admissible; separating them needs "
                + '; '.join(EXTERNAL[e] for e in ext) + ' — outside ODSA')
    return ('DEGENERATE-OVERLAP', '',
            f"{len(cases)} archetypes admissible on identical measured axes; "
            "their fingerprints overlap here, so the catalogue cannot separate them")


# ---------------------------------------------------------------------------
def build_vector(g, unit, pflag):
    row = {'unit': unit, 'n_windows': len(g), 'processing_flag': pflag}
    for name, col, unc, _ in ELEMENTS:
        v, iqr, sig = _agg(g, col, unc)
        row[name], row[f'{name}_iqr'], row[f'{name}_sigma'] = v, iqr, sig
    # beta_iqr is the within-unit spread of beta, undefined for a single window.
    row['beta_iqr'] = row['beta_iqr_iqr'] = row['beta_iqr_sigma'] = np.nan
    b = pd.to_numeric(g['beta'], errors='coerce').dropna()
    if len(b) > 1:
        row['beta_iqr'] = float(b.quantile(.75) - b.quantile(.25))
    return row


def units_from(df, level):
    if level == 'region':
        yield 'region', df
    elif level == 'segment':
        for (t, s), g in df.groupby(['trajectory', 'segment']):
            yield f'{t}|s{s:.0f}', g
    else:
        for i, r in df.iterrows():
            yield f"{r['trajectory']}|s{r['segment']:.0f}|w{r['window_id']:.0f}", df.loc[[i]]


def reachable_groups():
    """Catalogue entries that can never be the sole match, because some other entry
    admits everything they admit once the dead axes are removed. Subsumption, not
    equality: D4 allows any velocity where D allows only slow, so every D is a D4."""
    dead = {'delta_beta'} if not ANISOTROPY_TRUSTED else set()
    if BETA_IQR_WIDE is None:
        dead.add('beta_spread')
    live = [a for a in ALL_AXES if a not in dead]
    allow = {c['id']: {a: c['c'].get(a, set(AXIS_VALUES[a])) for a in live} for c in CATALOGUE}
    subsumed = {}
    for x in CATALOGUE:
        by = [y['id'] for y in CATALOGUE if y['id'] != x['id']
              and all(allow[x['id']][a] <= allow[y['id']][a] for a in live)]
        if by:
            subsumed[x['id']] = by
    return subsumed, dead


def collapse_pairs(vec_df, reports, z_min=2.0):
    """Unit pairs that no measured element separates at z_min, i.e. one vector for
    two places. Only elements carrying a sigma can enter."""
    cols = [n for n, _, _, _ in ELEMENTS
            if f'{n}_sigma' in vec_df.columns and vec_df[f'{n}_sigma'].notna().any()]
    rows = []
    for i, j in itertools.combinations(range(len(vec_df)), 2):
        a, b = vec_df.iloc[i], vec_df.iloc[j]
        zs = {}
        for c in cols:
            s = np.hypot(a[f'{c}_sigma'], b[f'{c}_sigma'])
            if np.isfinite(s) and s > 0 and np.isfinite(a[c]) and np.isfinite(b[c]):
                zs[c] = abs(a[c] - b[c]) / s
        if not zs:
            continue
        if max(zs.values()) < z_min:
            rows.append({'unit_a': a['unit'], 'unit_b': b['unit'],
                         'max_z': max(zs.values()),
                         'on': max(zs, key=zs.get),
                         'cases_a': reports[a['unit']], 'cases_b': reports[b['unit']]})
    return pd.DataFrame(rows).sort_values('max_z') if rows else pd.DataFrame()


def unthresholded_separation(vec_df, reports, z_min=2.0):
    """Pairs the catalogue gives the same answer for, but which are separated on an
    element carrying no threshold. Keyed on the admissible set rather than on the axes,
    because two units can share an answer while both are unresolved on an axis."""
    free = [n for n, _, _, a in ELEMENTS
            if not a and f'{n}_iqr' in vec_df.columns]
    rows = []
    for i, j in itertools.combinations(range(len(vec_df)), 2):
        a, b = vec_df.iloc[i], vec_df.iloc[j]
        ca, cb = reports[a['unit']], reports[b['unit']]
        if not ca or ca != cb:
            continue
        for c in free:
            s = np.hypot(a.get(f'{c}_sigma', np.nan), b.get(f'{c}_sigma', np.nan))
            if not (np.isfinite(s) and s > 0):
                # Fall back on the within-unit spread when no formal sigma exists.
                iqrs = [x for x in (a.get(f'{c}_iqr', np.nan), b.get(f'{c}_iqr', np.nan))
                        if np.isfinite(x)]
                s = np.mean(iqrs) / 1.349 if iqrs else np.nan
            if np.isfinite(s) and s > 0 and np.isfinite(a[c]) and np.isfinite(b[c]):
                z = abs(a[c] - b[c]) / s
                if z >= z_min:
                    rows.append({'unit_a': a['unit'], 'unit_b': b['unit'],
                                 'archetypes': ca, 'element': c, 'z': z})
    return pd.DataFrame(rows).sort_values('z', ascending=False) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
def process_region(region_name, csv_path, levels=('segment', 'region')):
    print(f"\n{'='*100}\n  LANDSCAPE VECTOR: {region_name}\n{'='*100}")
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    if len(df) == 0:
        print("  No valid data.")
        return
    pflag = region_flag(df)
    if pflag:
        print(f"  Processing: {_FLAG_NOTE.get(pflag, pflag)}")
    if 'is_transition' in df.columns and df['is_transition'].any():
        n = int(df['is_transition'].sum())
        df = df[~df['is_transition']].copy()
        print(f"  Excluded {n} transition windows ({len(df)} remain)")

    subsumed, dead = reachable_groups()
    live = len(CATALOGUE) - len(subsumed)
    print(f"\n  Catalogue: {live}/{len(CATALOGUE)} entries can ever be a sole match "
          f"({sorted(dead) or 'no'} axes unusable)")
    for x, by in sorted(subsumed.items()):
        print(f"    {x:11s} never alone: anything matching it also matches {', '.join(by)}")

    vec_rows, rep_rows, admissible_by_unit = [], [], {}
    for level in levels:
        for unit, g in units_from(df, level):
            v = build_vector(g, f'{level}:{unit}', pflag)
            obs = observe(v, pflag)
            cases = match(obs)
            kind, on, why = verdict(cases, obs)
            v['level'] = level
            vec_rows.append(v)
            ids = [c['id'] for c, _ in cases]
            admissible_by_unit[v['unit']] = '|'.join(ids)
            rep_rows.append({
                'unit': v['unit'], 'level': level, 'n_windows': v['n_windows'],
                'admissible': '|'.join(ids), 'n_admissible': len(ids),
                'verdict': kind, 'discriminator': on, 'why': why,
                'archetypes': '; '.join(f"{c['id']}: {c['name']}" for c, _ in cases),
                **{f'axis_{a}': ','.join(sorted(obs[a]['set'])) for a in ALL_AXES},
                **{f'status_{a}': obs[a]['status'] for a in ALL_AXES},
                'processing_flag': pflag,
            })

    vec = pd.DataFrame(vec_rows)
    rep = pd.DataFrame(rep_rows)

    out = output_dir_for(csv_path)
    os.makedirs(out, exist_ok=True)
    vpath = os.path.join(out, f'{region_name}_landscape_vector.csv')
    rpath = os.path.join(out, f'{region_name}_archetype_report.csv')
    vec.to_csv(vpath, index=False)
    rep.to_csv(rpath, index=False)

    reg = rep[rep.level == 'region'].iloc[0]
    print(f"\n  REGION VERDICT: {reg['verdict']}")
    print(f"    admissible : {reg['archetypes'] or '(none)'}")
    print(f"    reason     : {reg['why']}")
    if reg['discriminator']:
        print(f"    would need : {reg['discriminator']}")
    print("    axes       : " + ' | '.join(
        f"{a.replace('_class', '').replace('_band', '')}="
        f"{reg[f'axis_{a}']}({reg[f'status_{a}'][:3]})" for a in ALL_AXES))

    seg = rep[rep.level == 'segment']
    if len(seg):
        print(f"\n  SEGMENTS ({len(seg)}):")
        for k, n in seg['verdict'].value_counts().items():
            print(f"    {k:26s} {n:4d}  ({n/len(seg):.0%})")
        matched = seg[seg.n_admissible > 0]
        if len(matched):
            print(f"    median admissible archetypes, over the {len(matched)} segments "
                  f"that match anything: {matched['n_admissible'].median():.0f}")
        top = seg['admissible'].value_counts().head(6)
        print("    most common admissible sets:")
        for s, n in top.items():
            print(f"      {n:4d}  {s or '(none)'}")

        # Axes resolved off a single number, with no error of any kind behind them.
        ex = {a: (seg[f'status_{a}'] == 'assumed-exact').sum() for a in MEASURABLE}
        ex = {a: n for a, n in ex.items() if n}
        if ex:
            print("    assumed-exact axes (resolved with zero uncertainty): "
                  + ', '.join(f"{a.replace('_class', '').replace('_band', '')} {n}"
                              for a, n in ex.items()))
            near = 0
            for _, r in vec[vec.level == 'segment'].iterrows():
                for val, edges in ((r['relief'], [350, 800]), (r['elevation'], [0, 1000])):
                    if np.isfinite(val) and min(abs(val - e) for e in edges) < 100:
                        near += 1
                        break
            print(f"    of those, {near} segments sit within 100 m of a relief or "
                  f"elevation break, so the label turns on a number carrying no error bar")

    cp = collapse_pairs(vec[vec.level == 'segment'], admissible_by_unit)
    if len(cp):
        cp.to_csv(os.path.join(out, f'{region_name}_collapsed_pairs.csv'), index=False)
        both = cp[(cp.cases_a != '') & (cp.cases_b != '')]
        diff = both[both.cases_a != both.cases_b]
        print(f"\n  COLLAPSED PAIRS: {len(cp)} segment pairs separated by nothing at 2 sigma; "
              f"of the {len(both)} where both sides match something, {len(diff)} carry "
              f"different archetype sets")

    us = unthresholded_separation(vec[vec.level == 'segment'], admissible_by_unit)
    if len(us):
        us.to_csv(os.path.join(out, f'{region_name}_catalogue_blind.csv'), index=False)
        npairs = len(us.groupby(['unit_a', 'unit_b']))
        print(f"  CATALOGUE-BLIND: {npairs} segment pairs the catalogue answers identically "
              f"but which separate at 2 sigma on an unthresholded element")
        # As a fraction of pairs within each archetype group, since a raw count is n^2.
        for grp, n in seg[seg.n_admissible > 0]['admissible'].value_counts().head(4).items():
            tot = n * (n - 1) // 2
            if tot:
                hit = len(us[us.archetypes == grp].groupby(['unit_a', 'unit_b']))
                print(f"      {grp:24s} n={n:3d}  {hit}/{tot} pairs ({hit/tot:.0%}) "
                      f"are measurably different beds under one label")
        print(f"      separating elements: "
              f"{', '.join(f'{k} {v}' for k, v in us['element'].value_counts().head(4).items())}")

    print(f"\n  Vector saved : {vpath}")
    print(f"  Report saved : {rpath}")


def compare_regions(root, z_min=2.0):
    """Cross-region degeneracy: regions the catalogue answers identically, and what
    separates them anyway. Runs off already-written outputs, so it needs no re-processing."""
    vecs, reps = [], []
    for f in sorted(glob.glob(os.path.join(root, '*', 'landscape_vector',
                                           '*_landscape_vector.csv'))):
        v = pd.read_csv(f)
        v = v[v.level == 'region']
        r = pd.read_csv(f.replace('_landscape_vector.csv', '_archetype_report.csv'))
        r = r[r.level == 'region']
        if not len(v) or not len(r):
            continue
        row = v.iloc[0].copy()
        row['region'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        row['admissible'] = r.iloc[0]['admissible']
        row['verdict'] = r.iloc[0]['verdict']
        vecs.append(row)
    if len(vecs) < 2:
        print(f"\nCross-region: only {len(vecs)} region vector(s) under {root}, nothing to compare.")
        return
    d = pd.DataFrame(vecs)

    print(f"\n{'='*100}\n  CROSS-REGION DEGENERACY ({len(d)} regions)\n{'='*100}")
    for _, r in d.iterrows():
        print(f"  {r['region']:10s} {str(r['admissible']):24s} {r['verdict']}")

    cols = [n for n, _, _, _ in ELEMENTS if n in d.columns]
    rows = []
    for i, j in itertools.combinations(range(len(d)), 2):
        a, b = d.iloc[i], d.iloc[j]
        zs = {}
        for c in cols:
            s = np.hypot(a.get(f'{c}_sigma', np.nan), b.get(f'{c}_sigma', np.nan))
            if not (np.isfinite(s) and s > 0):
                iqrs = [x for x in (a.get(f'{c}_iqr', np.nan), b.get(f'{c}_iqr', np.nan))
                        if np.isfinite(x)]
                s = np.mean(iqrs) / 1.349 if iqrs else np.nan
            if np.isfinite(s) and s > 0 and np.isfinite(a[c]) and np.isfinite(b[c]):
                zs[c] = abs(a[c] - b[c]) / s
        same = a['admissible'] == b['admissible'] and bool(str(a['admissible']))
        sep = {k: v for k, v in zs.items() if v >= z_min}
        rows.append({'region_a': a['region'], 'region_b': b['region'],
                     'admissible_a': a['admissible'], 'admissible_b': b['admissible'],
                     'same_answer': same,
                     'n_elements_separating': len(sep),
                     'max_z': max(zs.values()) if zs else np.nan,
                     'separated_on': ','.join(f'{k}:{v:.1f}' for k, v in
                                              sorted(sep.items(), key=lambda kv: -kv[1])),
                     **{f'z_{k}': v for k, v in zs.items()}})
    out = pd.DataFrame(rows)
    path = os.path.join(root, 'cross_region_degeneracy.csv')
    out.to_csv(path, index=False)

    coll = out[out.same_answer]
    print(f"\n  {len(coll)} of {len(out)} region pairs receive the same archetype answer.")
    for _, r in coll.iterrows():
        print(f"    {r['region_a']} = {r['region_b']}  -> {r['admissible_a']}")
        if r['n_elements_separating']:
            print(f"      but separate at {z_min:.0f} sigma on {r['n_elements_separating']} "
                  f"elements: {r['separated_on']}")
            print(f"      DEGENERATE-COLLAPSE: one archetype label, two measurably "
                  f"different beds")
        else:
            print(f"      and nothing separates them: the label is honest here")
    print(f"\n  Saved: {path}")
    return out


def walk_tree(root):
    """Every *_window_stats.csv under a tree of region folders (individual_region_TEST/)."""
    return {os.path.basename(f).replace('_window_stats.csv', ''): f
            for f in sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'),
                                      recursive=True))}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Compare regions already processed, without reprocessing them.
    if arg == '--compare':
        root = sys.argv[2] if len(sys.argv) > 2 else 'individual_region_TEST'
        sys.stdout = Tee(os.path.join(root, 'cross_region_log.txt'))
        compare_regions(root)
        sys.exit(0)

    if arg and os.path.isdir(arg):
        found = walk_tree(arg)
        log = os.path.join(arg, 'landscape_vector_log.txt')
        sys.stdout = Tee(log)
        print(f"Walking {arg}: {len(found)} region CSVs")
        for r, f in found.items():
            process_region(r, f)
        compare_regions(arg)
        print(f"\nLog: {log}")
        sys.exit(0)

    os.makedirs(os.path.join(_REGION_BASE, 'landscape_vector'), exist_ok=True)
    sys.stdout = Tee(os.path.join(_REGION_BASE, 'landscape_vector',
                                  'landscape_vector_log.txt'))

    csvs = discover_window_csvs(os.path.join(_REGION_BASE, 'window_csvs')) \
        or discover_window_csvs(_REGION_BASE)

    if arg:
        if arg.endswith('.csv'):
            process_region(os.path.basename(arg).replace('_window_stats.csv', ''), arg)
        elif arg in csvs:
            process_region(arg, csvs[arg])
        else:
            m = [r for r in csvs if arg.lower() in r.lower()]
            if len(m) == 1:
                process_region(m[0], csvs[m[0]])
            else:
                print(f"'{arg}' matched {len(m)}. Available:")
                for r in sorted(csvs):
                    print(f"  - {r}")
    else:
        sel = select_region(csvs)
        if sel == 'ALL':
            for r in sorted(csvs):
                process_region(r, csvs[r])
        elif sel:
            process_region(sel, csvs[sel])
