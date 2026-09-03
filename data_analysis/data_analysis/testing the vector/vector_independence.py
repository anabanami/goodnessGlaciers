import os, sys, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee, PROCESSING_FLAG_NOTE as _FLAG_NOTE, processing_flag_of as region_flag
from bed_character import (BED_CLASSES, RELIEF_CLASSES, ELEVATION_CLASSES, BED_COLORS,
                           CLASS_ORDER, walk_tree, tree_region_name, class_of,
                           classify_relief, classify_elevation)
from plotting import flag_title as _flag_title, flag_suptitle as _flag_suptitle
# landscape_vector imports bed_character, not this module, so there is no cycle to break
# here and the import stays at the top. bed_character.region_panel's lazy import is for the
# reverse direction only.
from landscape_vector import (ELEMENTS, VELOCITY_CLASSES, AXIS_VALUES,
                              COMPOSITION_DECIMATE_KM, _independent_subset)
from loading import OUTPUT_BASE_PATH as _REGION_BASE

"""
Dependence check across the landscape vector.

The archetype catalogue breaks degeneracies by having each axis constrain something
different. Two axes carrying the same information make the catalogue weaker than it looks,
so this measures the dependence between the vector's continuous elements and ranks the
classifying axes by how far they duplicate each other. It reports an ordering and no
verdict: no redundancy threshold is defensible on seven regions.

Usage:
  python vector_independence.py                        # walks OUTPUT_BASE_PATH
  python vector_independence.py individual_region_TEST # walk any tree of region folders
"""

DEFAULT_ROOT = _REGION_BASE

# The continuous variable behind each classifying axis, read off ELEMENTS rather than
# retyped: ELEMENTS is the definitive list of vector elements and its axis tag is the link.
# beta_spread drops out here because its element carries no source column, and it would drop
# out anyway: no CATALOGUE entry constrains that axis, so it classifies nothing and cannot
# make another axis redundant. beta_iqr, the number behind it, is a within-unit spread that
# exists per segment and per region and has NO window-level value at all, so it cannot enter
# a window-level matrix without mixing two scales. Left out; see the AXIS TABLE note.
AXIS_SOURCE = {axis: col for _, col, _, axis in ELEMENTS if axis and col}
DESCRIPTORS = [col for _, col, _, axis in ELEMENTS if col and not axis]
CONTINUOUS = list(AXIS_SOURCE.values()) + DESCRIPTORS

# Short labels for the matrix axes; the CSVs keep the full column names.
SHORT = {'beta': 'β', 'measures_speed_mean': 'speed', 'relief_m': 'relief',
         'bed_elev_mean': 'elevation', 'A_1km': 'A_1km',
         'rms_roughness': 'rms', 'eta_wavelength_m': 'η_wl', 'hill_count': 'hills',
         'skewness': 'skew', 'kurtosis': 'kurt', 'xi_band': 'ξ_band',
         'psd_intercept': 'intercept'}

# psd_intercept is NOT a landscape vector element and never enters CONTINUOUS or the matrix.
# The matrix is ELEMENTS, and keeping that true is what makes this script's scope readable.
# It is here only as the parametrisation diagnostic behind 2.1h: A_1km =
# psd_intercept + 3β, and the raw intercept is close to a restatement of β, which is the
# reason the vector carries the in-band amplitude instead. These numbers live in their own
# table and these pairs are drawn unconditionally, never ranked against the vector's own.
PARAMETRISATION_PAIRS = [
    ('beta', 'psd_intercept'),          # before: the raw intercept coordinate
    ('beta', 'A_1km'),      # after: the in-band coordinate the vector uses
    ('relief_m', 'beta'),               # how much of the slope axis relief already carries
    ('relief_m', 'A_1km'),  # and how much of the amplitude axis
]

# The classifying axes against each other: does each one earn its place in the vector?
# beta x relief_m is deliberately absent, already drawn under the parametrisation prefix.
CLASSIFIER_PAIRS = [
    ('beta', 'measures_speed_mean'),
    ('relief_m', 'measures_speed_mean'),
    ('bed_elev_mean', 'measures_speed_mean'),   # the one classifier pair that is not weak
    ('relief_m', 'bed_elev_mean'),
]

# By-construction pairs, so the ranking never picks them. Drawn anyway: they are the
# evidence for how few independent dimensions the descriptors really carry.
REDUNDANCY_PAIRS = [
    ('beta', 'eta_wavelength_m'),              # beta's strongest correlation with anything
    ('A_1km', 'xi_band'),          # strongest pair in the matrix
    ('relief_m', 'rms_roughness'),             # two space-domain amplitude measures
]

# Class labels are thresholds on the columns above, so they are derived here rather than
# read from the window CSV: the script then does not depend on bed_character having run.
LABEL_OF = {'beta_class': class_of, 'relief_class': classify_relief,
            'elevation_class': classify_elevation}

# Pairs whose relationship is fixed by how the numbers are made rather than measured.
# Marked in every output so a strong cell is never read as a finding. Two tiers:
#   identity - one is determined by the other (plus a third term or a fit residual)
#   shared   - both come off the same detrended window, so they share its realisation
#              noise even where neither determines the other
BY_CONSTRUCTION = {}


def _mark(a, b, tier, why):
    BY_CONSTRUCTION[frozenset((a, b))] = (tier, why)


_mark('beta', 'A_1km', 'identity',
      'algebraic: A_1km = psd_intercept + 3β')
for _c in ('beta', 'A_1km'):
    _mark(_c, 'rms_roughness', 'identity',
          'Parseval: rms_roughness is an integral of the PSD this is fit to')
    # xi_band is the same mechanism over two octaves instead of the whole band. Not in the
    # brief's list, added because a reader would mistake it for a measured relation.
    _mark(_c, 'xi_band', 'identity',
          'Parseval: xi_band is the bedform-band integral of that same PSD')
_mark('rms_roughness', 'xi_band', 'identity',
      'two integrals of one window PSD, full band vs bedform band')
_mark('relief_m', 'rms_roughness', 'shared',
      'two space-domain amplitude measures of one window')
for _a, _b in itertools.combinations(('hill_count', 'skewness', 'kurtosis'), 2):
    _mark(_a, _b, 'shared', 'same detrended profile (bed_analysis window block)')
for _a in ('hill_count', 'skewness', 'kurtosis'):
    for _b in ('beta', 'A_1km', 'rms_roughness'):
        _mark(_a, _b, 'shared',
              'same detrended profile as the PSD fit (bed_analysis _hill_counts / window block)')
# For an exact power law over a fixed band, eta is a function of beta and the band edges
# alone (bed_analysis: the amplitude divides out). The measured relation is therefore the
# departure from that, not an independent one. Marked, but as the weaker tier: unlike
# Parseval this is a model prediction and the residual is the informative part.
_mark('eta_wavelength_m', 'beta', 'shared',
      'η is a function of β alone for a pure power law over a fixed band')

# Floor on the independent count below which no p is printed. Not a significance rule and
# not a redundancy threshold: below five independent windows the p carries no information
# about anything, and most regions sit at 2-6 (see the log).
MIN_N_FOR_P = 5


# ---------------------------------------------------------------------------
def velocity_band(v):
    for name, lo, hi in VELOCITY_CLASSES:
        if lo <= v < hi:
            return name
    return 'unknown'


def load_windows(csv_path, region):
    """Window CSV with transitions dropped and labels derived.

    Transitions go first, matching bed_character and landscape_vector: a window spanning a
    landscape boundary samples neither side, and every other number in the project is quoted
    over the retained set.
    """
    df = pd.read_csv(csv_path).dropna(subset=['beta'])
    if not len(df):
        print("    no valid windows")
        return None
    pflag = region_flag(df)
    if 'is_transition' in df.columns and df['is_transition'].any():
        n = int(df['is_transition'].astype(bool).sum())
        df = df[~df['is_transition'].astype(bool)]
        print(f"    excluded {n} transition windows ({len(df)} remain)")
    df = df.reset_index(drop=True)
    df['region'] = region
    df['processing_flag'] = pflag

    for axis, fn in LABEL_OF.items():
        col = AXIS_SOURCE[axis]
        df[axis] = [fn(v) if np.isfinite(v) else None
                    for v in pd.to_numeric(df[col], errors='coerce')]
    df['velocity_band'] = [velocity_band(v) if np.isfinite(v) else None
                           for v in pd.to_numeric(df[AXIS_SOURCE['velocity_band']],
                                                  errors='coerce')]
    return df


# ---------------------------------------------------------------------------
def _indep_rows(d, min_sep_km=COMPOSITION_DECIMATE_KM):
    """Positional indices of spatially independent rows, decimated within each region.

    Windows overlap by half a window and are spatially clustered, so the window count is not
    a sample size. _independent_subset is imported rather than mirrored; it is greedy and
    walks rows in order, so this is a lower bound on independent units and not a canonical
    count. Decimation is per region because two regions are already thousands of km apart.
    """
    if not {'center_x', 'center_y'} <= set(d.columns):
        return []
    out, pos = [], {ix: i for i, ix in enumerate(d.index)}
    for _, g in d.groupby('region', sort=False):
        xy = g[['center_x', 'center_y']].to_numpy(float)
        ok = np.isfinite(xy).all(1)
        gi = g.index[ok]
        out += [pos[gi[k]] for k in _independent_subset(xy[ok], min_sep_km)]
    return sorted(out)


def _corr(d, x, y):
    """Pearson and Spearman for one pair, pairwise-complete, plus a Spearman p taken only
    over the spatially independent windows.

    Both coefficients are reported: several of these relations are monotone and nonlinear,
    where Pearson alone understates them. No p is quoted on the full window count.
    """
    out = dict(n=0, pearson=np.nan, spearman=np.nan,
               n_indep=0, spearman_indep=np.nan, p_indep=np.nan)
    if x not in d.columns or y not in d.columns:
        return out
    v = d[[x, y, 'region'] + [c for c in ('center_x', 'center_y') if c in d.columns]] \
        .dropna(subset=[x, y])
    out['n'] = len(v)
    if len(v) < 3 or v[x].nunique() < 2 or v[y].nunique() < 2:
        return out
    out['pearson'] = float(pearsonr(v[x], v[y])[0])
    out['spearman'] = float(spearmanr(v[x], v[y])[0])
    ii = _indep_rows(v)
    out['n_indep'] = len(ii)
    if len(ii) >= MIN_N_FOR_P:
        a, b = v[x].iloc[ii], v[y].iloc[ii]
        if a.nunique() > 1 and b.nunique() > 1:
            rho, p = spearmanr(a, b)
            out['spearman_indep'], out['p_indep'] = float(rho), float(p)
    return out


def pair_table(d, scope, cols=None):
    """Every pair of continuous elements, with its by-construction tier attached."""
    cols = cols or CONTINUOUS
    rows = []
    for x, y in itertools.combinations(cols, 2):
        tier, why = BY_CONSTRUCTION.get(frozenset((x, y)), ('', ''))
        rows.append({'scope': scope, 'a': x, 'b': y, **_corr(d, x, y),
                     'by_construction': tier, 'mechanism': why})
    return pd.DataFrame(rows)


def to_matrix(pairs, value='spearman', cols=None):
    """Long pair table -> symmetric frame, diagonal 1 for a coefficient and n for a count."""
    cols = cols or CONTINUOUS
    M = pd.DataFrame(np.nan, index=cols, columns=cols)
    for _, r in pairs.iterrows():
        M.loc[r['a'], r['b']] = M.loc[r['b'], r['a']] = r[value]
    for c in cols:
        M.loc[c, c] = 1.0 if value != 'n' else np.nan
    return M


# ---------------------------------------------------------------------------
def effective_dimensions(d, cols, scope):
    """How many dimensions the descriptor block really spans, from the eigen-spectrum of its
    Spearman matrix on listwise-complete rows.

    Rank correlations, not a covariance PCA: hill_count is a count and xi_band spans orders
    of magnitude, so a covariance PCA would mostly report their units. The headline number is
    the participation ratio (sum L)^2 / sum L^2, which is a continuous effective dimension
    and needs no variance cutoff to invent; n90 and the condition number are printed beside
    it because those are the familiar forms and each rests on a convention this one avoids.
    """
    v = d[[c for c in cols if c in d.columns]].dropna()
    p = v.shape[1]
    if len(v) < max(3 * p, 10):
        return {'scope': scope, 'n_listwise': len(v), 'n_elements': p,
                'participation_ratio': np.nan, 'n90': np.nan, 'condition_number': np.nan}
    R = v.corr(method='spearman').to_numpy()
    lam = np.sort(np.linalg.eigvalsh(R))[::-1]
    lam = np.clip(lam, 0, None)
    pr = lam.sum() ** 2 / np.square(lam).sum()
    cum = np.cumsum(lam) / lam.sum()
    return {'scope': scope, 'n_listwise': len(v), 'n_elements': p,
            'participation_ratio': float(pr),
            'n90': int(np.searchsorted(cum, 0.90) + 1),
            'condition_number': float(lam[0] / lam[-1]) if lam[-1] > 0 else np.inf,
            **{f'lambda_{i+1}': float(l) for i, l in enumerate(lam)}}


def cramers_v(a, b):
    """Label-level association for two classifying axes, kept out of the coefficient matrix.

    V and Cramer's bias-corrected V are both reported: with three or four levels and a few
    hundred overlapping windows the raw V is inflated. Neither carries a p, for the same
    reason no coefficient above does.
    """
    ct = pd.crosstab(a, b)
    n = int(ct.to_numpy().sum())
    if min(ct.shape) < 2 or n == 0:
        return np.nan, np.nan, n, ct.shape
    chi2 = chi2_contingency(ct, correction=False)[0]
    phi2, (r, k) = chi2 / n, ct.shape
    v = np.sqrt(phi2 / (min(r, k) - 1))
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / max(min(rc, kc) - 1, 1e-12))
    return float(v), float(vc), n, ct.shape


# ---------------------------------------------------------------------------
def parametrisation_table(all_df, root):
    """The (β, A_1km) parametrisation diagnostic, deliberately outside the matrix.

    psd_intercept appears in this table and its own figures and nowhere else: it is not a
    vector element, and the matrix stays ELEMENTS. Pooled and per-region side by side, same
    as every other table.
    """
    rows = []
    for x, y in PARAMETRISATION_PAIRS:
        rows.append({'a': x, 'b': y, 'scope': 'POOLED', **_corr(all_df, x, y)})
        for reg, g in all_df.groupby('region'):
            rows.append({'a': x, 'b': y, 'scope': reg, **_corr(g, x, y)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(root, 'vector_independence_parametrisation.csv'), index=False)

    print(f"\n  PARAMETRISATION DIAGNOSTIC (2.1h) — psd_intercept is not a vector element and "
          f"is not in the matrix")
    print(f"    {'pair':<26s} {'ρ_pool':>7s} {'r_pool':>7s} {'n':>5s} {'med|ρ|reg':>9s}  "
          f"per region")
    for x, y in PARAMETRISATION_PAIRS:
        p = t[(t.a == x) & (t.b == y)]
        pool = p[p.scope == 'POOLED'].iloc[0]
        per = p[p.scope != 'POOLED'].dropna(subset=['spearman'])
        flip = ((per.spearman > 0).any() and (per.spearman < 0).any())
        print(f"    {SHORT.get(x, x) + ' × ' + SHORT.get(y, y):<26s} "
              f"{pool['spearman']:>+7.2f} {pool['pearson']:>+7.2f} {pool['n']:>5.0f} "
              f"{per.spearman.abs().median():>9.2f}  "
              f"{', '.join(f'{r.scope} {r.spearman:+.2f}' for r in per.itertuples())}"
              f"{'  SIGN FLIP' if flip else ''}")
    print(f"    β × intercept is the evidence for the in-band coordinate: the raw intercept is "
          f"close to a restatement of β, which is why the vector carries A_1km. "
          f"Read the first two rows as before/after on the same windows.")
    return t


def plot_matrix(pairs, out_path, title, pflag=None, cols=None, cmap='RdBu_r',
                cell=0.92, fs=7, fs_n=5.5, hatch_identity='///', hatch_shared='...',
                grey='0.86', annotate_n=True):
    """Spearman heatmap with the by-construction cells hatched.

    Hatching rather than omission: the cell is still worth seeing (a Parseval pair that did
    NOT come out strong would say something about the fit), it just may not be counted as a
    measured relation. The two hatches separate an identity from a shared input.
    """
    cols = [c for c in (cols or CONTINUOUS)]
    M, N = to_matrix(pairs, 'spearman', cols), to_matrix(pairs, 'n', cols)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(cell * n + 2.2, cell * n + 1.6))
    ax.imshow(M.to_numpy(float), cmap=cmap, vmin=-1, vmax=1)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=grey, lw=0))
                continue
            tier = BY_CONSTRUCTION.get(frozenset((a, b)), ('', ''))[0]
            if tier:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor='none',
                                       edgecolor='0.35', lw=0.4,
                                       hatch=hatch_identity if tier == 'identity'
                                       else hatch_shared))
            # The numbers sit on a plate inside a hatched cell, otherwise the hatch runs
            # straight through them and the marked cells become the unreadable ones.
            tbox = (dict(boxstyle='square,pad=0.08', facecolor='white', alpha=0.8,
                         edgecolor='none') if tier else None)
            rho = M.iloc[i, j]
            if np.isfinite(rho):
                ax.text(j, i - (0.12 if annotate_n else 0), f'{rho:+.2f}', ha='center',
                        va='center', fontsize=fs, bbox=tbox,
                        color='white' if abs(rho) > 0.6 and not tier else '0.1')
                if annotate_n:
                    ax.text(j, i + 0.26, f'n={N.iloc[i, j]:.0f}', ha='center', va='center',
                            fontsize=fs_n, color='0.3', bbox=tbox)
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=fs, color='0.5')

    ticks = [SHORT.get(c, c) for c in cols]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks, fontsize=9)
    # The five classifying axes sit first; the rule marks where the descriptors start.
    k = len(AXIS_SOURCE) - 0.5
    ax.axvline(k, color='k', lw=1.2); ax.axhline(k, color='k', lw=1.2)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)

    keys = [Patch(facecolor='white', edgecolor='0.25', hatch=hatch_identity,
                  label='identity (algebraic / Parseval)'),
            Patch(facecolor='white', edgecolor='0.25', hatch=hatch_shared,
                  label='shared detrended window')]
    ax.legend(handles=keys, loc='upper center', bbox_to_anchor=(0.5, -0.13),
              ncol=2, fontsize=8, frameon=False)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.7, label='Spearman ρ')
    _flag_title(ax, f'{title}   (n per cell is pairwise-complete)', pflag, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    matrix figure: {out_path}")


def scatter_grid(all_df, x, y, out_path, color_by='beta_class', colors=None, order=None,
                 s=15, alpha=0.6, ncols_max=4, fs=10, vlines=(), hlines=(), note=None):
    """Per-region panels plus one pooled panel, coloured by bed class.

    Pooled and per-region are on the same sheet on purpose: a pooled coefficient across seven
    regions with different beta means measures the between-region contrast, not the
    within-window relation, and the panels are what shows the difference.
    """
    colors = colors or BED_COLORS
    order = order or CLASS_ORDER
    d = all_df.dropna(subset=[x, y])
    regions = list(dict.fromkeys(d['region']))
    ncols = min(len(regions) + 1, ncols_max)
    nrows = int(np.ceil((len(regions) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    flat = axes.flatten()

    def panel(ax, g, title, pflag, withhold_p=False):
        for cls in order:
            sub = g[g[color_by] == cls]
            if len(sub):
                ax.scatter(sub[x], sub[y], c=colors[cls], label=cls, s=s, alpha=alpha,
                           edgecolors='none')
        c = _corr(g, x, y)
        p = (f", {c['n_indep']} indep: p withheld, regions disagree on sign" if withhold_p
             else f", p={c['p_indep']:.3f} on {c['n_indep']} indep"
             if np.isfinite(c['p_indep']) else f", {c['n_indep']} indep: no p")
        _flag_title(ax, f"{title}\n(ρ={c['spearman']:.2f}, r={c['pearson']:.2f}, "
                        f"n={c['n']}{p})", pflag, fontsize=fs)
        for v in vlines:
            ax.axvline(v, color='0.5', ls='--', lw=0.8)
        for h in hlines:
            ax.axhline(h, color='0.5', ls='--', lw=0.8)
        ax.set_xlabel(SHORT.get(x, x))
        ax.set_ylabel(SHORT.get(y, y))
        ax.grid(True, alpha=0.3)

    signs = []
    for i, reg in enumerate(regions):
        g = d[d['region'] == reg]
        panel(flat[i], g, reg, region_flag(g))
        signs.append(np.sign(_corr(g, x, y)['spearman']))
    # No flag on the pooled panel: it mixes migration states, and a single tag would assert
    # one of them over the rest. Its p follows the same rule as the ranked table: withheld
    # where the regions disagree on a sign, because the pooled subset is then measuring the
    # between-region contrast.
    signs = [v for v in signs if np.isfinite(v) and v != 0]
    panel(flat[len(regions)], d, 'ALL REGIONS', None,
          withhold_p=any(v > 0 for v in signs) and any(v < 0 for v in signs))
    flat[len(regions)].legend(fontsize=8)
    for j in range(len(regions) + 1, len(flat)):
        flat[j].set_visible(False)

    tier, why = BY_CONSTRUCTION.get(frozenset((x, y)), ('', ''))
    cap = (f'  [{note}]' if note
           else f'  [{tier} by construction: {why}]' if tier else '')
    _flag_suptitle(fig, f'{SHORT.get(x, x)} vs {SHORT.get(y, y)}{cap}', None, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    scatter: {out_path}")


# ---------------------------------------------------------------------------
def print_axis_table():
    print(f"\n  AXIS TABLE — the classifying axes and the number behind each")
    print(f"    {'axis':<16s} {'continuous source':<22s} labels")
    for axis, col in AXIS_SOURCE.items():
        print(f"    {axis:<16s} {col:<22s} {','.join(AXIS_VALUES[axis])}")
    print(f"    beta_spread      (excluded)            no CATALOGUE entry constrains it, so it "
          f"classifies nothing")
    print(f"      and beta_iqr, the number behind it, is a within-unit spread with no "
          f"window-level value; correlating a per-segment spread against per-window columns "
          f"would mix two scales, so it is out of the matrix entirely.")
    print(f"    descriptors with no classifying axis ({len(DESCRIPTORS)}): "
          f"{', '.join(DESCRIPTORS)}")


def process_region(region, csv_path, out_dir):
    print(f"\n{'-'*100}\n  {region}\n{'-'*100}")
    df = load_windows(csv_path, region)
    if df is None:
        return None
    pflag = df['processing_flag'].iloc[0]
    if pflag:
        print(f"    processing: {_FLAG_NOTE.get(pflag, pflag)}")

    pairs = pair_table(df, region)
    os.makedirs(out_dir, exist_ok=True)
    pairs.to_csv(os.path.join(out_dir, f'{region}_pairs.csv'), index=False)
    to_matrix(pairs, 'spearman').to_csv(os.path.join(out_dir, f'{region}_spearman_matrix.csv'))
    to_matrix(pairs, 'n').to_csv(os.path.join(out_dir, f'{region}_n_matrix.csv'))
    plot_matrix(pairs, os.path.join(out_dir, f'{region}_spearman_matrix.png'),
                f'Vector element dependence — {region}', pflag)

    free = pairs[(pairs.by_construction == '') & pairs.spearman.notna()]
    top = free.reindex(free.spearman.abs().sort_values(ascending=False).index).head(5)
    print(f"    strongest measured pairs (by |ρ|, by-construction pairs excluded):")
    for _, r in top.iterrows():
        print(f"      {SHORT.get(r['a'], r['a']):>9s} × {SHORT.get(r['b'], r['b']):<9s} "
              f"ρ={r['spearman']:+.2f}  r={r['pearson']:+.2f}  n={r['n']:.0f}")
    print(f"    dimensionality of the {len(DESCRIPTORS)} descriptors: ", end='')
    ed = effective_dimensions(df, DESCRIPTORS, region)
    print(f"PR={ed['participation_ratio']:.2f}, n90={ed['n90']}, "
          f"κ={ed['condition_number']:.0f} (listwise n={ed['n_listwise']})"
          if np.isfinite(ed['participation_ratio'])
          else f"not computed, listwise n={ed['n_listwise']} too small for "
               f"{ed['n_elements']} elements")
    return {'df': df, 'pairs': pairs, 'dims': ed}


# ---------------------------------------------------------------------------
def cross_region(all_df, per_region, root, n_axis_scatter=4, n_scatter=4):
    print(f"\n{'='*100}\n  POOLED ACROSS {all_df['region'].nunique()} REGIONS\n{'='*100}")
    pooled = pair_table(all_df, 'POOLED')
    pooled.to_csv(os.path.join(root, 'vector_independence_pairs_pooled.csv'), index=False)
    to_matrix(pooled, 'spearman').to_csv(
        os.path.join(root, 'vector_independence_spearman_pooled.csv'))
    to_matrix(pooled, 'n').to_csv(os.path.join(root, 'vector_independence_n_pooled.csv'))
    plot_matrix(pooled, os.path.join(root, 'vector_independence_matrix.png'),
                'Vector element dependence — pooled')

    # Pooled and per-region side by side. A pooled coefficient across regions with different
    # beta means is a between-region contrast, so a pair whose regions disagree on sign is
    # flagged rather than summarised.
    reg = pd.concat([p['pairs'] for p in per_region.values()], ignore_index=True)
    reg.to_csv(os.path.join(root, 'vector_independence_pairs_by_region.csv'), index=False)
    wide = reg.pivot_table(index=['a', 'b'], columns='scope', values='spearman')
    side = pooled.set_index(['a', 'b']).join(wide, rsuffix='_reg')
    rcols = list(wide.columns)
    sign = np.sign(side[rcols])
    side['n_regions'] = side[rcols].notna().sum(1)
    side['n_pos'] = (sign > 0).sum(1)
    side['n_neg'] = (sign < 0).sum(1)
    side['median_abs_rho_region'] = side[rcols].abs().median(1)
    side['sign_flip'] = (side.n_pos > 0) & (side.n_neg > 0)
    side['pooled_sign_minority'] = np.sign(side['spearman']).where(
        side['spearman'].notna()) * (side.n_pos - side.n_neg) < 0
    # A pooled p is only interpretable where the regions agree on a sign. Where they do not,
    # the pooled independent subset (about three windows per region) is measuring the
    # between-region contrast, so the p is withheld rather than printed and quoted. This gates
    # the p on validity, not the coefficient on strength: no pair leaves the ranking.
    side['p_indep_interpretable'] = side['p_indep'].where(~side['sign_flip'])
    side.reset_index().to_csv(
        os.path.join(root, 'vector_independence_pairs_side_by_side.csv'), index=False)

    # The result that matters: how far the classifying axes duplicate each other. Ranked, with
    # no threshold — seven regions cannot support a cut, and the ordering is the claim.
    axis_cols = list(AXIS_SOURCE.values())
    ax_pairs = side.reset_index()
    ax_pairs = ax_pairs[ax_pairs.a.isin(axis_cols) & ax_pairs.b.isin(axis_cols)]
    ax_pairs = ax_pairs.reindex(ax_pairs.spearman.abs().sort_values(ascending=False).index)
    ax_pairs.to_csv(os.path.join(root, 'vector_independence_axis_pairs.csv'), index=False)
    inv = {v: k for k, v in AXIS_SOURCE.items()}

    print(f"\n  CLASSIFYING AXES, RANKED BY |ρ| (pooled; ordering only, no threshold)")
    print(f"    {'axis pair':<34s} {'ρ_pool':>7s} {'r_pool':>7s} {'n':>5s} {'ρ_ind':>6s} "
          f"{'p_ind':>7s} {'n_ind':>5s} {'med|ρ|reg':>9s} {'regions':>8s}  note")
    for _, r in ax_pairs.iterrows():
        name = f"{inv[r['a']]} × {inv[r['b']]}"
        pi = ('withheld' if r['sign_flip']
              else f"{r['p_indep']:.3f}" if np.isfinite(r['p_indep']) else '—')
        ri = f"{r['spearman_indep']:+.2f}" if np.isfinite(r['spearman_indep']) else '—'
        note = []
        if r['sign_flip']:
            note.append(f"SIGN FLIP ({int(r['n_pos'])}+/{int(r['n_neg'])}-)")
        if r['pooled_sign_minority']:
            note.append('pooled sign is the minority one')
        if r['by_construction']:
            note.append(f"{r['by_construction']} by construction")
        print(f"    {name:<34s} {r['spearman']:>+7.2f} {r['pearson']:>+7.2f} {r['n']:>5.0f} "
              f"{ri:>6s} {pi:>7s} {r['n_indep']:>5.0f} "
              f"{r['median_abs_rho_region']:>9.2f} {int(r['n_regions']):>8d}  "
              f"{'; '.join(note)}")
    # The two orderings answer different questions, so both are printed and neither is
    # promoted: pooled is the contrast the catalogue is applied across, within-region is the
    # relation a single window sits in.
    by_reg = ax_pairs.sort_values('median_abs_rho_region', ascending=False)
    print(f"    same pairs ordered by the median WITHIN-region |ρ| instead: "
          + ', '.join(f"{inv[r['a']]}×{inv[r['b']]} {r['median_abs_rho_region']:.2f}"
                      for _, r in by_reg.head(5).iterrows()))
    # Printed as its own line rather than left to the per-row tags: the pooled column is the
    # one that gets quoted, and on a flagged row it is a between-region contrast wearing the
    # notation of a within-window relation.
    nf, nm = int(ax_pairs.sign_flip.sum()), int(ax_pairs.pooled_sign_minority.sum())
    print(f"    ** {nf} of {len(ax_pairs)} axis pairs change sign between regions and {nm} "
          f"carry a pooled sign that is the minority one. Do not quote ρ_pool alone on a "
          f"flagged row; p_ind is withheld on all of them for the same reason. **")

    print(f"\n  STRONGEST MEASURED PAIRS OVERALL (by-construction pairs excluded)")
    free = side.reset_index()
    free = free[(free.by_construction == '') & free.spearman.notna()]
    free = free.reindex(free.spearman.abs().sort_values(ascending=False).index)
    for _, r in free.head(10).iterrows():
        flip = ' SIGN FLIP' if r['sign_flip'] else ''
        print(f"    {SHORT.get(r['a'], r['a']):>9s} × {SHORT.get(r['b'], r['b']):<9s} "
              f"ρ={r['spearman']:+.2f}  n={r['n']:.0f}  "
              f"med|ρ| within region {r['median_abs_rho_region']:.2f}{flip}")
    print(f"\n  BY-CONSTRUCTION PAIRS (labelled, not findings)")
    bc = side.reset_index()
    bc = bc[bc.by_construction != '']
    for _, r in bc.reindex(bc.spearman.abs().sort_values(ascending=False).index).iterrows():
        print(f"    {SHORT.get(r['a'], r['a']):>9s} × {SHORT.get(r['b'], r['b']):<9s} "
              f"ρ={r['spearman']:+.2f}  [{r['by_construction']}] {r['mechanism']}")

    # Dimensionality: pooled AND per region. Pooled alone would be the same between-region
    # contrast trap as a pooled coefficient — seven region means spread over a common axis
    # look like one shared dimension.
    dims = pd.DataFrame([effective_dimensions(all_df, DESCRIPTORS, 'POOLED')]
                        + [p['dims'] for p in per_region.values()])
    dims.to_csv(os.path.join(root, 'vector_independence_dimensionality.csv'), index=False)
    print(f"\n  EFFECTIVE DIMENSIONS OF THE {len(DESCRIPTORS)} DESCRIPTORS "
          f"(Spearman eigen-spectrum, listwise)")
    print(f"    {'scope':<10s} {'n':>5s} {'PR':>6s} {'n90':>4s} {'cond':>8s}")
    for _, r in dims.iterrows():
        pr = f"{r['participation_ratio']:.2f}" if np.isfinite(r['participation_ratio']) else '—'
        n90 = f"{r['n90']:.0f}" if np.isfinite(r['n90']) else '—'
        cd = f"{r['condition_number']:.0f}" if np.isfinite(r['condition_number']) else '—'
        print(f"    {r['scope']:<10s} {r['n_listwise']:>5.0f} {pr:>6s} {n90:>4s} {cd:>8s}")
    print(f"    PR is the participation ratio, a continuous effective dimension needing no "
          f"variance cutoff; n90 and cond are the conventional forms beside it. "
          f"{len(DESCRIPTORS)} would mean fully independent.")

    # Label-level association, kept in its own table: V and a coefficient are different
    # quantities and must not share a matrix.
    axes = list(AXIS_SOURCE)
    rows = []
    for a, b in itertools.combinations(axes, 2):
        d = all_df[[a, b]].dropna()
        v, vc, n, shape = cramers_v(d[a], d[b]) if len(d) else (np.nan, np.nan, 0, (0, 0))
        rows.append({'axis_a': a, 'axis_b': b, 'cramers_v': v, 'cramers_v_corrected': vc,
                     'n': n, 'shape': f'{shape[0]}x{shape[1]}'})
    cv = pd.DataFrame(rows).sort_values('cramers_v', ascending=False)
    cv.to_csv(os.path.join(root, 'vector_independence_cramers_v.csv'), index=False)
    print(f"\n  LABEL-LEVEL ASSOCIATION (Cramér's V on the crosstab, separate table by design)")
    print(f"    {'axis pair':<34s} {'V':>6s} {'V_corr':>7s} {'n':>5s}  table")
    for _, r in cv.iterrows():
        v = f"{r['cramers_v']:.2f}" if np.isfinite(r['cramers_v']) else '—'
        vc = f"{r['cramers_v_corrected']:.2f}" if np.isfinite(r['cramers_v_corrected']) else '—'
        print(f"    {r['axis_a'] + ' × ' + r['axis_b']:<34s} {v:>6s} {vc:>7s} "
              f"{r['n']:>5.0f}  {r['shape']}")

    breaks = {'relief_m': [h for _, _, h in RELIEF_CLASSES if np.isfinite(h)],
              'beta': [h for _, _, h in BED_CLASSES if np.isfinite(h)],
              'bed_elev_mean': [h for _, _, h in ELEVATION_CLASSES if np.isfinite(h)],
              'measures_speed_mean': [h for _, _, h in VELOCITY_CLASSES if np.isfinite(h)]}

    # The parametrisation pairs are drawn unconditionally and under their own prefix, because
    # they answer a fixed question and must not appear or vanish with the ranking. They are
    # then held out of the ranked set below so nothing is drawn twice; a pair that is both
    # (relief x A_1km) is drawn once, here.
    parametrisation_table(all_df, root)
    print(f"\n  PARAMETRISATION PANELS ({len(PARAMETRISATION_PAIRS)} pairs, always drawn)")
    for x, y in PARAMETRISATION_PAIRS:
        scatter_grid(all_df, x, y,
                     os.path.join(root, f'parametrisation_{x}_vs_{y}.png'),
                     vlines=breaks.get(x, ()), hlines=breaks.get(y, ()),
                     note='parametrisation diagnostic behind 2.1h, not a vector element')

    seen = {frozenset(p) for p in PARAMETRISATION_PAIRS}
    for label, pairs, prefix in [('CLASSIFIER', CLASSIFIER_PAIRS, 'classifier'),
                                 ('REDUNDANCY', REDUNDANCY_PAIRS, 'redundancy')]:
        todo = [(x, y) for x, y in pairs if frozenset((x, y)) not in seen]
        print(f"\n  {label} PANELS ({len(todo)} pairs, always drawn)")
        for x, y in todo:
            seen.add(frozenset((x, y)))
            scatter_grid(all_df, x, y, os.path.join(root, f'{prefix}_{x}_vs_{y}.png'),
                         vlines=breaks.get(x, ()), hlines=breaks.get(y, ()))

    # Scatters for the pairs the matrix picked out: the top axis pairs, then the top measured
    # pairs anywhere. Deduplicated, so a pair that is both is drawn once.
    draw = []
    for _, r in list(ax_pairs.head(n_axis_scatter).iterrows()) + \
            list(free.head(n_scatter).iterrows()):
        k = frozenset((r['a'], r['b']))
        if k not in seen and np.isfinite(r['spearman']):
            seen.add(k)
            draw.append((r['a'], r['b']))
    print(f"\n  SCATTER PANELS ({len(draw)} pairs)")
    for x, y in draw:
        scatter_grid(all_df, x, y,
                     os.path.join(root, f'vector_independence_scatter_{x}_vs_{y}.png'),
                     vlines=breaks.get(x, ()), hlines=breaks.get(y, ()))
    return pooled


def main(root=DEFAULT_ROOT):
    found = walk_tree(root)
    print(f"Walking {root}: {len(found)} region CSVs")
    print_axis_table()

    per_region, frames = {}, []
    for f in found:
        region = tree_region_name(f)
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(f))),
                               'vector_independence')
        res = process_region(region, f, out_dir)
        if res:
            per_region[region] = res
            frames.append(res['df'])
    if len(frames) < 2:
        print(f"\nOnly {len(frames)} usable region(s) under {root}, nothing to pool.")
        return
    all_df = pd.concat(frames, ignore_index=True)
    cross_region(all_df, per_region, root)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    sys.stdout = Tee(os.path.join(root, 'vector_independence_log.txt'))
    main(root)
