"""Queue item 6: does the catalogue exist in feature space?

Clusters the 651 non-transition windows on the standardised measured vector and asks three
things: is there cluster tendency at all, how many clusters, and do they map onto catalogue
entries or onto geography. Decimates to 200 km to answer the same questions at the honest n.
"""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import Tee
from landscape_vector import _independent_subset, COMPOSITION_DECIMATE_KM

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / 'individual_region_TEST')
ROBUST = '--robust' in sys.argv
KS = range(2, 11)
N_NULL = 200
SEED = 0

# Primary set is the texture elements only: what a transect contributes, no long-wavelength axes.
# rms_roughness (r=+0.906 with relief) and xi_band (tracks amplitude) are left out as
# near-duplicates that would double-weight the magnitude family under standardisation.
FEATURES = {
    'measured': ['beta', 'A_1km', 'relief_m', 'eta_wavelength_m', 'hill_count'],
    'catalogue': ['beta', 'relief_m', 'bed_elev_mean', 'measures_speed_mean'],
    'all': ['beta', 'A_1km', 'relief_m', 'bed_elev_mean', 'measures_speed_mean',
            'eta_wavelength_m', 'hill_count', 'skewness', 'kurtosis'],
}


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
        if rep:
            r = pd.read_csv(rep[0])
            r = r[r.level == 'window'][['unit', 'admissible', 'n_admissible', 'verdict']]
            d = d.merge(r, on='unit', how='left')
        out.append(d)
    return pd.concat(out, ignore_index=True)


def standardise(x):
    """z-score, or median/IQR under --robust. IQR is safer on the heavy tails but the doc
    says 'standardised vector', so z-score is primary."""
    if ROBUST:
        c = np.median(x, axis=0)
        s = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    else:
        c, s = x.mean(axis=0), x.std(axis=0, ddof=1)
    return (x - c) / np.where(s > 0, s, 1.0)


def hopkins(z, rng, frac=0.1):
    """Cluster tendency. 0.5 = no structure beyond a uniform cloud, ->1 = clustered.
    Known to be optimistic in high dimension, so read it beside the silhouette nulls."""
    n, d = z.shape
    m = max(5, int(n * frac))
    lo, hi = z.min(axis=0), z.max(axis=0)
    tree = cKDTree(z)
    idx = rng.choice(n, m, replace=False)
    # k=2 on real points so a point does not match itself.
    u = tree.query(rng.uniform(lo, hi, (m, d)), k=1)[0]
    w = tree.query(z[idx], k=2)[0][:, 1]
    return u.sum() / (u.sum() + w.sum())


def null_matrices(z, rng, kind):
    """marginal: shuffle each column independently, killing joint structure but keeping every
    marginal. gauss: matched mean and covariance, unimodal by construction. The second is the
    decisive one -- a correlated but single-blob cloud beats the marginal null on silhouette
    for free, because k-means scores elongated clouds well."""
    if kind == 'marginal':
        return np.column_stack([rng.permutation(z[:, j]) for j in range(z.shape[1])])
    return rng.multivariate_normal(z.mean(axis=0), np.cov(z, rowvar=False), size=len(z))


def sweep(z, ks=KS, seed=SEED):
    """Silhouette per k against both nulls. z-scores say whether the structure is real."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        if len(z) <= k + 1:
            continue
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(z)
        obs = silhouette_score(z, lab)
        rec = {'k': k, 'silhouette': obs}
        for kind in ('marginal', 'gauss'):
            s = []
            for _ in range(N_NULL):
                y = null_matrices(z, rng, kind)
                s.append(silhouette_score(y, KMeans(k, n_init=3, random_state=seed).fit_predict(y)))
            s = np.array(s)
            rec[f'null_{kind}'] = s.mean()
            rec[f'sd_{kind}'] = s.std(ddof=1)
            rec[f'z_{kind}'] = (obs - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def decimated_sweep(d, cols, n_rep=200, seed=SEED):
    """Greedy decimation is order-dependent, so randomise the order and repeat. Reports the
    spread of the silhouette z at the honest n rather than one arbitrary subset."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_rep):
        keep = []
        for _, g in d.groupby('region'):
            g = g.sample(frac=1, random_state=int(rng.integers(1 << 31)))
            xy = g[['center_x', 'center_y']].to_numpy(float)
            keep += g.index[_independent_subset(xy, COMPOSITION_DECIMATE_KM)].tolist()
        sub = d.loc[keep]
        if len(sub) < 8:
            continue
        z = standardise(sub[cols].to_numpy(float))
        s = sweep(z, ks=range(2, min(6, len(sub) - 1)), seed=rep)
        s['rep'], s['n'] = rep, len(sub)
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def against_catalogue(d, lab):
    """Do the clusters recover the catalogue's own answer, or the map?"""
    out = {}
    for name, truth in [('admissible', d.admissible), ('verdict', d.verdict),
                        ('region', d.region)]:
        m = truth.notna().to_numpy()
        if m.sum() < 2:
            continue
        t = truth[m].astype(str)
        out[name] = {'ari': adjusted_rand_score(t, lab[m]),
                     'nmi': normalized_mutual_info_score(t, lab[m]),
                     'n_levels': t.nunique()}
    return pd.DataFrame(out).T.round(3)


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'cluster_test_log.txt'))
    d = load(ROOT)
    print(f"{len(d)} non-transition windows, {d.region.nunique()} regions")
    print(f"scaling: {'median/IQR' if ROBUST else 'z-score'}, {N_NULL} null draws per k\n")

    diag, best = [], {}
    for name, cols in FEATURES.items():
        sub = d.dropna(subset=cols).copy()
        z = standardise(sub[cols].to_numpy(float))
        rng = np.random.default_rng(SEED)
        h = np.mean([hopkins(z, rng) for _ in range(20)])
        print(f"=== {name.upper()}  ({len(cols)} dims, {len(sub)} windows) ===")
        print(f"  {', '.join(cols)}")
        print(f"  Hopkins {h:.3f}  (0.5 = no tendency)")
        s = sweep(z)
        s.insert(0, 'features', name)
        print(s.round(3).to_string(index=False), "\n")
        diag.append(s)
        k = int(s.loc[s.z_gauss.idxmax(), 'k'])
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(z)
        best[name] = (sub, k, lab)
        print(f"  best k by gauss-null z: {k}")
        print(against_catalogue(sub, lab).to_string(), "\n")

    pd.concat(diag, ignore_index=True).to_csv(os.path.join(ROOT, 'cluster_diagnostics.csv'),
                                              index=False)
    sub, k, lab = best['measured']
    sub.assign(cluster=lab)[['unit', 'region', 'cluster', 'center_x', 'center_y',
                             'admissible', 'verdict']].to_csv(
        os.path.join(ROOT, 'cluster_labels.csv'), index=False)
    print("=== CLUSTER x REGION (measured, k=%d) ===" % k)
    print(pd.crosstab(pd.Series(lab, index=sub.index, name='cluster'), sub.region).to_string(), "\n")

    print(f"=== DECIMATED at {COMPOSITION_DECIMATE_KM:.0f} km (measured) ===")
    dec = decimated_sweep(d.dropna(subset=FEATURES['measured']).reset_index(drop=True),
                          FEATURES['measured'])
    if len(dec):
        g = dec.groupby('k').agg(n=('n', 'mean'), sil=('silhouette', 'mean'),
                                 z_marg=('z_marginal', 'mean'), z_gauss=('z_gauss', 'mean'),
                                 z_gauss_sd=('z_gauss', 'std'),
                                 frac_z2=('z_gauss', lambda v: (v >= 2).mean())).round(3)
        print(g.to_string())
        dec.to_csv(os.path.join(ROOT, 'cluster_decimated.csv'), index=False)
        print(f"\n  {dec.rep.nunique()} decimations, median n = {dec.n.median():.0f} windows")
    else:
        print("  too few independent windows to sweep")
