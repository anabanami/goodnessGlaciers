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

# Output goes beside the window CSV's own region folder, so each region in a tree gets
# its own landscape_vector/.
def output_dir_for(csv_path):
    d = os.path.dirname(os.path.abspath(csv_path))
    if os.path.basename(d) == 'window_csvs':
        d = os.path.dirname(d)
    return os.path.join(d, 'landscape_vector')

# Half-width of the uncertainty envelope used to decide whether a class boundary is
# resolved, in sigma. An axis with an envelope that crosses a break returns both classes.
K_SIGMA = 2.0

# Lag at which the class tuple reaches chance agreement (window_atom_test.py). Set for the
# verdict.
COMPOSITION_DECIMATE_KM = 200.0

# Fallback only, for a region with no velocity sidecar. Where the sidecar exists, the
# sampled per-window MEaSUREs error replaces this entirely through velocity_sigma.
# This value is wrong wherever it fires: over-confident by 3x at PPB, which is in the InSAR
# pole hole at CNT 2 and has a sampled error of ~14 m/yr, and under-confident by 10x at MSB
# and ASB-LR. None drops the floor and leaves velocity_sigma as the across-window spread
# alone, which v23/velocity_error_sweep.py uses.
VELOCITY_ERROR_M_YR = 5.0

# Same problem as velocity, one level down: a single-window unit has no across-window
# spread and no formal error, so relief and elevation would resolve their class breaks with
# zero uncertainty. Both carry a nominal error from Pritchard_2025 (Bedmap3), the point data
# of which this pipeline reads. None makes both axes assumed-exact on a single-window unit,
# which v23/relief_elevation_error_sweep.py uses.
#
# [Pritchard_2025] sec. Uncertainty estimates.
# Bedmap3 quotes +/-20 m 1-sigma thickness uncertainty for single-measurement cells and
# +/-7 m on the surface. Relief is max(bed) - min(bed), a DIFFERENCE of two picks, so the
# surface term is common-mode over 50 km and cancels while the +/-20 m pick precision
# propagates as sqrt(2): 28 m, rounded to 30. Elevation is a MEAN over the window, so the
# common-mode surface error survives and the pick noise averages down: ~10 m.
RELIEF_ERROR_M = 30.0
ELEVATION_ERROR_M = 10.0

# Partial/unmigrated radar biases beta upward, so the true class may be the one below the
# measured one. True adds that neighbour to the admissible beta set, and the unwidened
# numbers stay available through beta_class_unwidened, widened_only and
# n_admissible_unwidened.
MIGRATION_WIDENS_BETA = False
# Beta's systematic as a symmetric envelope, in quadrature with the formal fit error. Set
# from the deviogram validation: levels agree to 0.02 in zeta, which is 0.04 in beta. The
# sweep is flat below 0.10, so nothing rests on the exact value. A floor rather than an
# estimate: per-window agreement between the two estimators is only r = 0.48 and is open.
BETA_SYSTEMATIC_ERROR = 0.05

VELOCITY_CLASSES = [
    ('very_low', -np.inf, 5.0),
    ('low',       5.0,   10.0),
    ('moderate',  10.0,  50.0),
    ('fast',      50.0,  np.inf),
]

# Above this the velocity envelope is wider than the `low` band itself, so no sharpening
# of anything else can separate ONSET from DIVIDE. Reported, never a gate.
SEAM_THRESHOLD_M_YR = (VELOCITY_CLASSES[1][2] - VELOCITY_CLASSES[1][1]) / (2 * K_SIGMA)

# ---------------------------------------------------------------------------
# Vector elements. thresholded elements drive classification; the rest are carried
# as continuous numbers only, because no literature threshold exists for them.
# (name, source column, formal-uncertainty column, axis tag)
# The tag is a classifying axis only when some CATALOGUE entry constrains it. beta_spread
# is tagged but unconstrained, so ALL_AXES excludes it.
ELEMENTS = [
    ('beta',              'beta',               'beta_uncertainty',           'beta_class'),
    ('A_1km', 'A_1km',  'A_1km_uncertainty',  None),
    ('beta_iqr',          None,                 None,                         'beta_spread'),
    ('velocity',          'measures_speed_mean', 'measures_err_m_yr',         'velocity_band'),
    ('relief',            'relief_m',            None,                        'relief_class'),
    ('rms_roughness',     'rms_roughness',       None,                        None),
    ('eta_wavelength_m',  'eta_wavelength_m',    None,                        None),
    ('hill_count',        'hill_count',          None,                        None),
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
    'beta_spread':     ['wide', 'narrow'],
}
NUMERIC_AXES = {'beta_class': BED_CLASSES, 'relief_class': RELIEF_CLASSES,
                'elevation_class': ELEVATION_CLASSES, 'velocity_band': VELOCITY_CLASSES}
MEASURABLE = list(NUMERIC_AXES)  # the axes ODSA can measure

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
# Ids and fingerprints follow papers/landscape_catalogue.md §2, which is the authority.
CATALOGUE = [
    dict(id='TRUNK', name='Ice stream trunk', evidence='Strong',
         c={'beta_class': {'soft'}, 'velocity_band': {'fast'},
            'relief_class': {'flat'}},
         ext=[]),
    dict(id='TRUNK-HARD', name='Ice stream on a confined hard bed (Cooper regime)',
         evidence='Moderate',
         c={'beta_class': {'hard', 'transitional'}, 'velocity_band': {'fast'},
            'relief_class': {'mountainous'}},
         ext=['amplitude_anisotropy']),
    dict(id='TRUNK-RELICT', name='Relict ice stream, shut down and still smooth',
         evidence='Moderate',
         c={'beta_class': {'soft'}, 'velocity_band': {'very_low', 'low'},
            'relief_class': {'flat'}},
         ext=['flow_history']),
    dict(id='ONSET', name='Ice stream onset', evidence='Moderate',
         c={'beta_class': {'transitional'}, 'velocity_band': {'moderate', 'fast'},
            'relief_class': {'flat', 'subdued'}},
         ext=[]),
    dict(id='HIGHLAND', name='Crystalline highland', evidence='Strong',
         c={'beta_class': {'hard'}, 'velocity_band': {'very_low', 'low'},
            'relief_class': {'subdued', 'mountainous'},
            'elevation_class': {'emerged', 'elevated'}},
         ext=[]),
    dict(id='RIFT', name='Rift / tectonic corridor', evidence='Moderate',
         c={'beta_class': {'hard', 'transitional'}, 'relief_class': {'mountainous'}},
         ext=[]),
    # Sediment and water are not separable here.
    dict(id='BASIN', name='Sedimentary basin or subglacial lake', evidence='Moderate',
         c={'beta_class': {'soft'}, 'relief_class': {'flat'},
            'elevation_class': {'submerged'}},
         ext=['reflectivity']),
    dict(id='BASIN-HIGH', name='Candidate hard bed, smooth and elevated', evidence='Moderate',
         c={'beta_class': {'soft'}, 'velocity_band': {'very_low'},
            'elevation_class': {'emerged', 'elevated'}},
         ext=['composition']),
    # beta_class excludes chaotic: a bed with no characteristic wavelength is not a
    # trough-and-interfluve landscape. The wide spread implied by the name is a descriptor
    # here and not a criterion.
    dict(id='DISSECTED', name='Deeply dissected highland, trough-and-interfluve',
         evidence='Moderate',
         c={'beta_class': {'hard', 'transitional', 'soft'},
            'velocity_band': {'low', 'moderate'}, 'relief_class': {'mountainous'}},
         ext=['origin']),
    # velocity covers very_low AND low: Siegert_2004's Group 2 defines this entry by the
    # absence of basal sliding, not by a speed, and interior ice deforming internally
    # reaches 5-10 m/yr with no sliding.
    dict(id='DIVIDE', name='Warm-base divide / plateau', evidence='Moderate',
         c={'beta_class': {'transitional'}, 'velocity_band': {'very_low', 'low'},
            'relief_class': {'flat', 'subdued'},
            'elevation_class': {'submerged', 'emerged'}},
         ext=[]),
    dict(id='SHATTERED', name='Shattered / chaotic', evidence='Speculative',
         c={'beta_class': {'chaotic'}},
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


def _independent_subset(xy, min_sep_km):
    """Greedy pick of rows at least min_sep_km apart, so no two neighbourhoods overlap."""
    keep = []
    for i in range(len(xy)):
        if all(np.hypot(*(xy[i] - xy[j])) / 1000.0 >= min_sep_km for j in keep):
            keep.append(i)
    return keep


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
                   'elevation_class': ELEVATION_ERROR_M,
                   'beta_class': BETA_SYSTEMATIC_ERROR}.get(axis)
        if axis == 'velocity_band':
            n_ok = vec.get('velocity_err_n_ok', np.nan)
            if np.isfinite(n_ok):
                # Sidecar ran, so the sampled error is already inside velocity_sigma and the
                # constant must not be added on top. No coverage widens to every band: a
                # failed measurement must never narrow the set, and `exact = not
                # isfinite(sig)` below would otherwise mark it assumed-exact.
                nominal = None
                if n_ok == 0 or not np.isfinite(sig):
                    obs[axis] = dict(set={n for n, _, _ in classes}, value=val,
                                     sigma=np.nan, exact=False)
                    continue
        if nominal:
            sig = np.hypot(sig if np.isfinite(sig) else 0.0, nominal)
        s = classify_set(val, sig, classes)
        obs[axis] = dict(set=s, value=val, sigma=sig, exact=not np.isfinite(sig))

    # Migration bias is one-directional (beta reads high), so widen downward only.
    # Keep the pre-widening set: a case admitted only by the correction is not a measurement.
    obs['beta_class'].update(unwidened=set(obs['beta_class']['set']), widened=False)
    if MIGRATION_WIDENS_BETA and pflag and pflag != 'migrated':
        order = AXIS_VALUES['beta_class']
        i = min(order.index(n) for n in obs['beta_class']['set'])
        if i > 0:
            obs['beta_class']['set'] |= {order[i - 1]}
            obs['beta_class']['widened'] = True

    # beta_iqr is a descriptor and not a classifying axis: no threshold is defensible, and
    # inventing one on these seven regions would repeat the borrowed-boundary problem.
    obs['beta_spread'] = dict(set=set(AXIS_VALUES['beta_spread']),
                              value=vec['beta_iqr'], sigma=np.nan)

    # assumed-exact is a resolved axis with no uncertainty behind it, so its class turns on
    # a single number while excluding archetypes as firmly as a measured axis does.
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


def widened_only(cases, obs):
    """Archetypes admitted only because migration widening added a beta class, i.e. cases
    resting on a one-directional bias correction rather than on a measured beta."""
    o = obs['beta_class']
    if not o['widened']:
        return []
    kept = {c['id'] for c, _ in match(dict(obs, beta_class=dict(o, set=o['unwidened'])))}
    return [c['id'] for c, _ in cases if c['id'] not in kept]


def narrowing_axes(cases, obs):
    """Axes that are unresolved and would exclude at least one admissible case if
    sharpened. Narrowing the set is weaker than resolving it: with a sparse catalogue this
    is almost always non-empty, so it cannot carry a verdict on its own."""
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


_POINT_MATCHES = {}


def _match_point(key):
    """Archetypes matching one fully-specified point. Cached, since units repeat points."""
    hit = _POINT_MATCHES.get(key)
    if hit is None:
        pt = dict(zip(ALL_AXES, key))
        hit = tuple(c['id'] for c in CATALOGUE
                    if all(pt[a] in allowed for a, allowed in c['c'].items()))
        _POINT_MATCHES[key] = hit
    return hit


def separability(cases, obs):
    """Sharpen every axis to a point, in every way this observation allows, and take the
    archetypes that remain. Returns whether any sharpening isolates a single archetype, and
    the pairs that coincide on at least one of those points. A coincident pair is not
    inseparable everywhere: it means that for a truth at such a point, nothing ODSA measures
    separates the two."""
    if len(cases) < 2:
        return True, []
    single, pairs = False, set()
    for key in itertools.product(*[sorted(obs[a]['set']) for a in ALL_AXES]):
        hit = _match_point(key)
        if len(hit) == 1:
            single = True
        elif len(hit) > 1:
            pairs.update(itertools.combinations(sorted(hit), 2))
    return single, sorted('+'.join(p) for p in pairs)


def verdict(cases, obs, sep=None):
    """RESOLVED / RESOLVED-WITH-EXTERNAL / OUT-OF-CATALOGUE / DEGENERATE, plus the reason.
    Degeneracy is reported as data: the axes that narrow it, and the pairs that nothing
    separates."""
    if not cases:
        return ('OUT-OF-CATALOGUE', '', 'vector matches no archetype fingerprint')
    if len(cases) == 1:
        case, ex = cases[0]
        if case['ext']:
            return ('RESOLVED-WITH-EXTERNAL', ','.join(case['ext']),
                    f"{case['id']} unique, but its own reading needs "
                    + '; '.join(EXTERNAL[e] for e in case['ext']))
        return ('RESOLVED', ','.join(ex), f"{case['id']} unique on {len(ex)} exercised axes")

    na = narrowing_axes(cases, obs)
    single, pairs = sep if sep is not None else separability(cases, obs)
    why = f"{len(cases)} archetypes admissible; "
    if single and na:
        why += "sharpening " + ', '.join(f"{a}[{obs[a]['status']}]" for a in na) + " isolates one"
    else:
        why += "no sharpening of the measured axes isolates one"
    if pairs:
        why += f"; {', '.join(pairs)} coincide inside the envelope"
    ext = sorted({e for c, _ in cases for e in c['ext']})
    if ext:
        why += f"; outside ODSA this needs " + '; '.join(EXTERNAL[e] for e in ext)
    return ('DEGENERATE', ','.join(na), why)


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
    # NaN = no velocity sidecar at all, 0 = the sidecar ran and MEaSUREs covers nothing here.
    # observe() must not treat those the same: the first falls back, the second widens.
    row['velocity_err_n_ok'] = np.nan
    row['velocity_err'] = row['velocity_cnt'] = np.nan
    if 'measures_err_m_yr' in g.columns:
        e = pd.to_numeric(g['measures_err_m_yr'], errors='coerce').dropna()
        row['velocity_err_n_ok'] = len(e)
        if len(e):
            row['velocity_err'] = float(e.median())
    if 'measures_cnt' in g.columns:
        c = pd.to_numeric(g['measures_cnt'], errors='coerce').dropna()
        if len(c):
            row['velocity_cnt'] = float(c.median())
    return row


def load_velocity_error(csv_path):
    """Per-window MEaSUREs speed error written by velocity_error_sidecar.py, which lives in
    the region's velocity/ folder beside window_csvs/."""
    region = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    p = os.path.join(region, 'velocity', os.path.basename(csv_path)
                     .replace('_window_stats.csv', '_velocity_error.csv'))
    return pd.read_csv(p) if os.path.exists(p) else None


def units_from(df, level):
    if level == 'region':
        yield 'region', df
    elif level == 'segment':
        for (t, s), g in df.groupby(['trajectory', 'segment']):
            yield f'{t}|s{s:.0f}', g
    else:
        for i, r in df.iterrows():
            yield f"{r['trajectory']}|s{r['segment']:.0f}|w{r['window_id']:.0f}", df.loc[[i]]


def composition(rep, df, region_name):
    """The region as a mixture of admissible sets over windows, not one label.

    The fraction is areal, so every window counts, overlapping ones included. Independence
    governs only the error bar, and n_independent is far too small to carry one, so the
    fraction is reported with n_independent beside it and no interval. See NEXT.md, queue
    item 5.
    """
    w = rep[rep.level == 'window']
    if not len(w):
        return None
    xy = df[['center_x', 'center_y']].to_numpy(float)
    n_independent = len(_independent_subset(xy, COMPOSITION_DECIMATE_KM)) if len(xy) else 0
    c = w['admissible'].fillna('').value_counts()
    return pd.DataFrame({'region': region_name, 'admissible': c.index.where(c.index != '', '(none)'),
                         'n_windows': c.values, 'fraction': (c.values / len(w)).round(3),
                         'n_windows_total': len(w), 'n_independent': n_independent,
                         'decimate_km': COMPOSITION_DECIMATE_KM})


def reachable_groups():
    """Catalogue entries that can never be the sole match, because some other entry
    admits everything they admit. Subsumption, not equality: an entry constraining no
    velocity subsumes one that allows only slow."""
    allow = {c['id']: {a: c['c'].get(a, set(AXIS_VALUES[a])) for a in ALL_AXES}
             for c in CATALOGUE}
    subsumed = {}
    for x in CATALOGUE:
        by = [y['id'] for y in CATALOGUE if y['id'] != x['id']
              and all(allow[x['id']][a] <= allow[y['id']][a] for a in ALL_AXES)]
        if by:
            subsumed[x['id']] = by
    return subsumed


# Two units differ on an element when their medians are distinguishable (z, from the
# standard error of the median) AND their window distributions differ (d, from the
# population spread). z alone shrinks as sqrt(n), so on z alone a large region would
# separate from everything.
# 1.0 population spread is a convention, not a literature value.
SEPARATION_D_MIN = 1.0


def _spread_se(row, col):
    """Population spread and standard error of the median for one element on one unit,
    both from the within-unit IQR so every element is measured the same way. Undefined
    for a single-window unit, which has no spread."""
    iqr, n = row.get(f'{col}_iqr', np.nan), row.get('n_windows', np.nan)
    sp = iqr / 1.349 if np.isfinite(iqr) else np.nan
    if not (np.isfinite(sp) and sp > 0 and np.isfinite(n) and n > 1):
        return np.nan, np.nan
    return sp, 1.253 * sp / np.sqrt(n)


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


def unthresholded_separation(vec_df, reports, z_min=2.0, d_min=SEPARATION_D_MIN):
    """Pairs for which the catalogue gives the same answer, but which differ on an element
    carrying no threshold. Keyed on the admissible set rather than on the axes, because two
    units can share an answer while both are unresolved on an axis.

    Every element uses the same scale, taken from the within-unit spread. Formal sigmas are
    not used: only beta and A_1km carry one, and mixing the two scales would
    rank amplitude by its error bar rather than by its separation. A single-window unit has
    no spread, so it cannot enter; the returned coverage is the number of pairs for which
    the question was answerable."""
    # beta_iqr is itself a within-unit spread, so it has no spread of its own and no d.
    free = [n for n, _, _, a in ELEMENTS
            if not a and f'{n}_iqr' in vec_df.columns]
    rows, n_same, n_answerable = [], 0, 0
    for i, j in itertools.combinations(range(len(vec_df)), 2):
        a, b = vec_df.iloc[i], vec_df.iloc[j]
        ca, cb = reports[a['unit']], reports[b['unit']]
        if not ca or ca != cb:
            continue
        n_same += 1
        answerable = False
        for c in free:
            (spa, sea), (spb, seb) = _spread_se(a, c), _spread_se(b, c)
            if not (np.isfinite(sea) and np.isfinite(seb)
                    and np.isfinite(a[c]) and np.isfinite(b[c])):
                continue
            answerable = True
            diff = abs(a[c] - b[c])
            z, d = diff / np.hypot(sea, seb), diff / np.mean([spa, spb])
            if z >= z_min and d >= d_min:
                rows.append({'unit_a': a['unit'], 'unit_b': b['unit'],
                             'archetypes': ca, 'element': c, 'z': z, 'd': d})
        n_answerable += answerable
    out = pd.DataFrame(rows).sort_values('z', ascending=False) if rows else pd.DataFrame()
    return out, n_same, n_answerable


# ---------------------------------------------------------------------------
def load_region(csv_path, quiet=False):
    """Window CSV plus both sidecars, transitions dropped. Returns (df, pflag), or
    (None, None) if there is nothing to classify. Separated so that a test sees exactly
    the frame the classifier reads."""
    say = (lambda *a: None) if quiet else print
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    if len(df) == 0:
        say("  No valid data.")
        return None, None
    pflag = region_flag(df)
    if pflag:
        say(f"  Processing: {_FLAG_NOTE.get(pflag, pflag)}")
    if 'is_transition' in df.columns and df['is_transition'].any():
        n = int(df['is_transition'].sum())
        df = df[~df['is_transition']].copy()
        say(f"  Excluded {n} transition windows ({len(df)} remain)")

    vel = load_velocity_error(csv_path)
    if vel is None:
        say(f"  Velocity error: no *_velocity_error.csv — falling back to the "
            f"VELOCITY_ERROR_M_YR = {VELOCITY_ERROR_M_YR} constant "
            f"(run velocity_error_sidecar.py)")
    else:
        keys = [c for c in ('trajectory', 'segment', 'window_id')
                if c in vel.columns and c in df.columns]
        df = df.merge(vel[keys + ['measures_err_m_yr', 'measures_cnt']], on=keys, how='left')
        e = pd.to_numeric(df['measures_err_m_yr'], errors='coerce')
        say(f"  Velocity error: sampled, median {e.median():.2f} m/yr, CNT median "
            f"{df['measures_cnt'].median():.0f}, {int((e > SEAM_THRESHOLD_M_YR).sum())}/"
            f"{len(df)} windows above the {SEAM_THRESHOLD_M_YR:.2f} m/yr seam threshold "
            f"(ONSET|DIVIDE inseparable), {int(e.isna().sum())} with no coverage")
    return df, pflag


def process_region(region_name, csv_path, levels=('window', 'segment', 'region')):
    print(f"\n{'='*100}\n  LANDSCAPE VECTOR: {region_name}\n{'='*100}")
    df, pflag = load_region(csv_path)
    if df is None:
        return

    subsumed = reachable_groups()
    live = len(CATALOGUE) - len(subsumed)
    print(f"\n  Catalogue: {live}/{len(CATALOGUE)} entries can ever be a sole match")
    for x, by in sorted(subsumed.items()):
        print(f"    {x:11s} never alone: anything matching it also matches {', '.join(by)}")

    vec_rows, rep_rows, admissible_by_unit = [], [], {}
    for level in levels:
        for unit, g in units_from(df, level):
            v = build_vector(g, f'{level}:{unit}', pflag)
            obs = observe(v, pflag)
            cases = match(obs)
            sep = separability(cases, obs)
            kind, on, why = verdict(cases, obs, sep)
            wo = widened_only(cases, obs)
            if wo:
                why += (f" — {','.join(wo)} admitted only by the migration widening, "
                        f"so that part of the answer rests on a bias correction")
            v['level'] = level
            vec_rows.append(v)
            ids = [c['id'] for c, _ in cases]
            admissible_by_unit[v['unit']] = '|'.join(ids)
            rep_rows.append({
                'unit': v['unit'], 'level': level, 'n_windows': v['n_windows'],
                'admissible': '|'.join(ids), 'n_admissible': len(ids),
                # Widening only ever adds beta classes, so the unwidened match set is a subset
                # and this needs no second match(). 0 here = OUT-OF-CATALOGUE without the
                # correction, which is the comparator for the preregistered ceiling.
                'n_admissible_unwidened': len(ids) - len(wo),
                'verdict': kind, 'discriminator': on, 'why': why,
                # The sampled velocity error drives the classification, so it belongs in the
                # output. n_ok separates "no coverage" (0, axis widened) from "no sidecar" (NaN).
                'measures_err_m_yr': v['velocity_err'],
                'measures_cnt': v['velocity_cnt'],
                'velocity_err_n_ok': v['velocity_err_n_ok'],
                'archetypes': '; '.join(f"{c['id']}: {c['name']}" for c, _ in cases),
                'beta_widened': obs['beta_class']['widened'],
                'beta_class_unwidened': ','.join(sorted(obs['beta_class']['unwidened'])),
                'widened_only': ','.join(wo),
                'separable': sep[0],
                'coincident_pairs': ','.join(sep[1]),
                'needs_external': ','.join(sorted({e for c, _ in cases for e in c['ext']})),
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
    if reg['widened_only']:
        print(f"    widening   : {reg['widened_only']} admissible ONLY via the migration "
              f"widening (beta unwidened = {reg['beta_class_unwidened']})")
    if reg['coincident_pairs']:
        print(f"    coincident : {reg['coincident_pairs']} coincide inside the envelope, so "
              f"nothing measured splits them where the truth lands there")
    if reg['needs_external']:
        print(f"    external   : {reg['needs_external']}")
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

        # Pairs coinciding inside the envelope: degeneracy the catalogue owns, not the survey.
        deg = seg[seg.n_admissible > 1]
        if len(deg):
            sp = deg['coincident_pairs'].str.split(',').explode()
            sp = sp[sp.astype(bool)].value_counts()
            if len(sp):
                print("    archetype pairs coinciding inside the envelope (nothing measured "
                      "splits them where the truth lands there):")
                for k, n in sp.items():
                    print(f"      {n:4d}  {k}")
            n_ns = int((~deg['separable']).sum())
            print(f"    of {len(deg)} degenerate segments, {n_ns} have no sharpening that "
                  f"isolates a single archetype")

        # Axes resolved from a single number, with no error behind them.
        ex = {a: (seg[f'status_{a}'] == 'assumed-exact').sum() for a in MEASURABLE}
        ex = {a: n for a, n in ex.items() if n}
        if ex:
            print("    assumed-exact axes (resolved with zero uncertainty): "
                  + ', '.join(f"{a.replace('_class', '').replace('_band', '')} {n}"
                              for a, n in ex.items()))
            # Only the axes carrying no error bar, each at its own breaks.
            breaks = {'relief_class': ('relief', RELIEF_CLASSES),
                      'elevation_class': ('elevation', ELEVATION_CLASSES)}
            edges = {a: (col, [h for _, _, h in cl if np.isfinite(h)])
                     for a, (col, cl) in breaks.items()}
            v = vec[vec.level == 'segment'].merge(
                seg[['unit'] + [f'status_{a}' for a in edges]], on='unit')
            on_edge = lambda r, a: (r[f'status_{a}'] == 'assumed-exact'
                                    and np.isfinite(r[edges[a][0]])
                                    and min(abs(r[edges[a][0]] - e) for e in edges[a][1]) < 100)
            n_ex = sum(any(r[f'status_{a}'] == 'assumed-exact' for a in edges)
                       for _, r in v.iterrows())
            near = sum(any(on_edge(r, a) for a in edges) for _, r in v.iterrows())
            print(f"    of the {n_ex} with relief or elevation assumed-exact, {near} sit within "
                  f"100 m of that axis's own break, so the label turns on a number carrying "
                  f"no error bar")

        # Beta widened downward for migration bias, and the archetypes it alone admits.
        nw = int(seg['beta_widened'].sum())
        if nw:
            print(f"    migration widening applied on {nw}/{len(seg)} segments "
                  f"({nw/len(seg):.0%})")
            wo = seg['widened_only'].str.split(',').explode()
            wo = wo[wo.astype(bool)].value_counts()
            if len(wo):
                print("    admitted ONLY by that widening: "
                      + ', '.join(f'{k} {n}' for k, n in wo.items()))
                sole = seg[(seg.n_admissible == 1) & (seg['widened_only'] != '')]
                print(f"      of which {len(sole)} segments have their ENTIRE verdict "
                      f"resting on it")

    # Widening can only fire where the radar was not fully migrated, so the rate is
    # structurally zero in a `migrated` region. Quoting one blended number across regions
    # hides that, so print it per region beside its flag and never pool the two.
    res = seg[seg.verdict == 'RESOLVED']
    if len(res):
        wo_res = int((res['widened_only'] != '').sum())
        print(f"    RESOLVED {len(res)}, of which {wo_res} ({wo_res/len(res):.0%}) rest "
              f"entirely on the migration widening  [processing_flag: {pflag}]")
        if wo_res == len(res):
            print("      ** every resolution in this region is widening-dependent; it has no "
                  "measured resolution at all **")

    # The region as a mixture rather than one label.
    comp = composition(rep, df, region_name)
    if comp is not None:
        comp.to_csv(os.path.join(out, f'{region_name}_composition.csv'), index=False)
        n_w, n_ind = comp['n_windows_total'].iloc[0], comp['n_independent'].iloc[0]
        print(f"\n  COMPOSITION ({n_w} windows, {n_ind} independent at "
              f"{COMPOSITION_DECIMATE_KM:.0f} km):")
        for _, r in comp.head(8).iterrows():
            print(f"    {r['fraction']:6.1%}  {r['n_windows']:4d}  {r['admissible']}")
        if len(comp) > 8:
            print(f"    ... {len(comp) - 8} further sets")
        print(f"    descriptive only: at {n_ind} independent a fraction of 0.28 carries an "
              f"SE of {np.sqrt(0.28 * 0.72 / max(n_ind, 1)):.2f}, so no interval is quoted "
              f"and fractions are not comparable between regions")

    cp = collapse_pairs(vec[vec.level == 'segment'], admissible_by_unit)
    if len(cp):
        cp.to_csv(os.path.join(out, f'{region_name}_collapsed_pairs.csv'), index=False)
        both = cp[(cp.cases_a != '') & (cp.cases_b != '')]
        diff = both[both.cases_a != both.cases_b]
        print(f"\n  COLLAPSED PAIRS: {len(cp)} segment pairs separated by nothing at 2 sigma; "
              f"of the {len(both)} where both sides match something, {len(diff)} carry "
              f"different archetype sets")

    us, n_same, n_ans = unthresholded_separation(vec[vec.level == 'segment'],
                                                 admissible_by_unit)
    if n_same:
        print(f"\n  CATALOGUE-BLIND: {n_same} segment pairs get the same archetype answer; "
              f"{n_ans} ({n_ans/n_same:.0%}) have a within-unit spread on at least one "
              f"unthresholded element, so the question is answerable for those only")
    if len(us):
        us.to_csv(os.path.join(out, f'{region_name}_catalogue_blind.csv'), index=False)
        npairs = len(us.groupby(['unit_a', 'unit_b']))
        print(f"      {npairs} of the {n_ans} differ at 2 sigma AND by a full population "
              f"spread on an unthresholded element")
        # As a fraction of the answerable pairs within each archetype group, not of all pairs.
        for grp, n in seg[seg.n_admissible > 0]['admissible'].value_counts().head(4).items():
            hit = len(us[us.archetypes == grp].groupby(['unit_a', 'unit_b']))
            if hit:
                print(f"      {grp:24s} n={n:3d}  {hit} pairs are different bed populations "
                      f"under one label")
        print(f"      separating elements: "
              f"{', '.join(f'{k} {v}' for k, v in us['element'].value_counts().head(4).items())}")
    elif n_ans:
        print(f"      none of the {n_ans} answerable pairs differ on both scales")

    print(f"\n  Vector saved : {vpath}")
    print(f"  Report saved : {rpath}")


def compare_regions(root, z_min=2.0):
    """Cross-region degeneracy: regions the catalogue answers identically, and what
    separates them anyway. Runs from already-written outputs, so it reprocesses nothing."""
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
        # z asks whether the medians are distinguishable and shrinks as sqrt(n_windows).
        # d asks whether the window distributions differ and does not shrink with n.
        zs, ds = {}, {}
        for c in cols:
            if not (np.isfinite(a[c]) and np.isfinite(b[c])):
                continue
            diff = abs(a[c] - b[c])
            se = np.hypot(a.get(f'{c}_sigma', np.nan), b.get(f'{c}_sigma', np.nan))
            iqrs = [x for x in (a.get(f'{c}_iqr', np.nan), b.get(f'{c}_iqr', np.nan))
                    if np.isfinite(x)]
            sp = np.mean(iqrs) / 1.349 if iqrs else np.nan
            if np.isfinite(se) and se > 0:
                zs[c] = diff / se
            if np.isfinite(sp) and sp > 0:
                ds[c] = diff / sp
        same = a['admissible'] == b['admissible'] and bool(str(a['admissible']))
        sep = {k: v for k, v in zs.items() if v >= z_min and ds.get(k, 0) >= SEPARATION_D_MIN}
        median_only = sorted(k for k, v in zs.items()
                             if v >= z_min and ds.get(k, 0) < SEPARATION_D_MIN)
        rows.append({'region_a': a['region'], 'region_b': b['region'],
                     'admissible_a': a['admissible'], 'admissible_b': b['admissible'],
                     'same_answer': same,
                     'n_elements_separating': len(sep),
                     'max_z': max(zs.values()) if zs else np.nan,
                     'max_d': max(ds.values()) if ds else np.nan,
                     'separated_on': ','.join(f'{k}:z{zs[k]:.1f}/d{ds[k]:.1f}' for k in
                                              sorted(sep, key=lambda k: -zs[k])),
                     'median_only': ','.join(median_only),
                     **{f'z_{k}': v for k, v in zs.items()},
                     **{f'd_{k}': v for k, v in ds.items()}})
    out = pd.DataFrame(rows)
    path = os.path.join(root, 'cross_region_degeneracy.csv')
    out.to_csv(path, index=False)

    coll = out[out.same_answer]
    print(f"\n  {len(coll)} of {len(out)} region pairs receive the same archetype answer.")
    for _, r in coll.iterrows():
        print(f"    {r['region_a']} = {r['region_b']}  -> {r['admissible_a']}")
        if r['n_elements_separating']:
            print(f"      distributions differ on {r['n_elements_separating']} elements: "
                  f"{r['separated_on']}")
            print(f"      DEGENERATE-COLLAPSE: one archetype label, two different bed "
                  f"populations")
        else:
            print(f"      no element separates both the medians and the distributions")
        if r['median_only']:
            print(f"      medians differ but the distributions overlap on: {r['median_only']} "
                  f"(precision, not a bed difference)")
    print(f"\n  Saved: {path}")
    return out


def walk_tree(root):
    """Every *_window_stats.csv under a tree of region folders (individual_region_TEST/)."""
    return {os.path.basename(f).replace('_window_stats.csv', ''): f
            for f in sorted(glob.glob(os.path.join(root, '**', '*_window_stats.csv'),
                                      recursive=True))}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Compare regions already processed.
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
