"""Queue item 5b Test B: does TRUNK-RELICT track Ockenden's relict selective erosion?

The one place the two schemes name the same thing. Her relict class and my TRUNK-RELICT
entry are the same idea in different words — an ice stream shut down and still smooth — so
if either scheme is picking up real relict ground they should co-occur.

The confound is admissible-set size: a window admitting five archetypes contains almost
anything, and her classes do not hold equal set sizes. So the test is not TRUNK-RELICT's
rate on its own. **Every one of the eleven entries is scored the same way**, and TRUNK-RELICT
has to stand out against the other ten. If all eleven lift together in her relict windows,
that is set size, not agreement.

Note her relict class is the residual of her mask chain, none-of-the-above, and holds half
my windows. A positive here is weaker than it looks; a negative is clean.

    python ockenden_relict_test.py [root] [--by-region]

`--by-region` controls the one thing the pooled run cannot: her relict class is 72% PPB, so a
pooled lift may be a region effect. It scores each region separately and pools again with PPB
dropped. **Read it as falsification only.** Dropping PPB takes n_eff on the relict side from 17
to about 9, so z falls by ~1.4x even if a lift holds exactly — a surviving lift is inconclusive
by construction, a vanishing one is informative. The evidence is the sign pattern across regions,
not any single region's z.

Needs ockenden_window_class.csv (run ockenden_class.py first).
Writes ockenden_relict_test.csv into the run tree (ROOT).
"""
import glob, os, sys
import numpy as np, pandas as pd
from config import Tee
from landscape_vector import CATALOGUE, _independent_subset, COMPOSITION_DECIMATE_KM

ROOT = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'individual_region_TEST'
BY_REGION = '--by-region' in sys.argv
ENTRIES = [e['id'] for e in CATALOGUE]
TARGET, HER = 'TRUNK-RELICT', 'sel_erosion_relict'
KEY = ['region', 'trajectory', 'segment', 'window_id']
MIN_N = 10   # a region needs this many windows on BOTH sides to be scored separately


def load(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'landscape_vector', '*_archetype_report.csv'))):
        r = pd.read_csv(f)
        r = r[r.level == 'window'].copy()
        r['region'] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        u = r.unit.str.split('|', expand=True)
        r['trajectory'] = u[0].str.split(':', n=1).str[1]
        r['segment'] = u[1].str[1:].astype(int)
        r['window_id'] = u[2].str[1:].astype(int)
        rows.append(r[KEY + ['admissible', 'n_admissible', 'verdict']])
    d = pd.concat(rows, ignore_index=True)

    cls = pd.read_csv(os.path.join(ROOT, 'ockenden_window_class.csv'))
    n = len(d)
    d = d.merge(cls[KEY + ['ockenden_class', 'alt_agrees', 'center_x', 'center_y']],
                on=KEY, how='left')
    assert len(d) == n, f'join fanned {n} rows to {len(d)}'
    d = d[d.ockenden_class.notna() & (d.ockenden_class != 'invalid_dunes')].copy()

    sets = d.admissible.fillna('').str.split('|')
    for e in ENTRIES:
        d[e] = sets.apply(lambda s, e=e: e in s)
    d['is_her_relict'] = d.ockenden_class == HER
    return d


def n_eff(g):
    return max(len(_independent_subset(g[['center_x', 'center_y']].values, COMPOSITION_DECIMATE_KM)), 1)


def contrast(d, col, by='is_her_relict'):
    """Presence rate inside vs outside the group, z on n_eff rather than on window count."""
    a, b = d[d[by]], d[~d[by]]
    if not len(a) or not len(b):
        return None
    pa, pb = a[col].mean(), b[col].mean()
    ea, eb = n_eff(a), n_eff(b)
    se = np.hypot(np.sqrt(max(pa * (1 - pa), 1e-9) / ea), np.sqrt(max(pb * (1 - pb), 1e-9) / eb))
    return {'entry': col, 'rate_in': pa, 'rate_out': pb, 'lift': pa - pb,
            'n_in': len(a), 'n_out': len(b), 'n_eff_in': ea, 'n_eff_out': eb,
            'z': (pa - pb) / se if se else np.nan}


def run(d, label):
    print(f"\n{'=' * 74}\n{label}: {len(d)} windows, {d.is_her_relict.sum()} in her relict class\n{'=' * 74}")
    print("median admissible-set size by her class (the confound):")
    print(d.groupby('ockenden_class').n_admissible.agg(['median', 'mean', 'size']).round(2).to_string())

    res = pd.DataFrame([r for e in ENTRIES if (r := contrast(d, e)) is not None])
    res = res.sort_values('lift', ascending=False)
    print(f"\nentry presence, her relict windows vs the rest (z on n_eff):")
    print(res.round(3).to_string(index=False))

    rank = list(res.entry).index(TARGET) + 1
    print(f"\n>>> {TARGET} ranks {rank} of {len(res)} on lift, z = {res[res.entry == TARGET].z.iloc[0]:.2f}")
    print(f"    entries with |z| >= 2: {(res.z.abs() >= 2).sum()} of {len(res)}"
          f"  — if many, this is set size, not agreement")
    return res


def by_region(d):
    """Lift per entry within each region, so a pooled result driven by PPB shows up as such."""
    out = []
    for reg, g in d.groupby('region'):
        a, b = g.is_her_relict.sum(), (~g.is_her_relict).sum()
        if min(a, b) < MIN_N:
            print(f"  skip {reg}: {a} relict / {b} rest")
            continue
        for e in ENTRIES:
            if (r := contrast(g, e)) is not None:
                out.append({**r, 'region': reg})
    return pd.DataFrame(out)


if __name__ == '__main__':
    sys.stdout = Tee(os.path.join(ROOT, 'ockenden_relict_by_region_log.txt' if BY_REGION
                                  else 'ockenden_relict_test_log.txt'))
    d = load(ROOT)
    res = run(d, 'ALL SNAPPED WINDOWS')
    res2 = run(d[d.alt_agrees], 'NON-STRADDLING ONLY')

    if BY_REGION:
        print(f"\n{'=' * 74}\nBY REGION — is the lift a PPB effect?\n{'=' * 74}")
        br = by_region(d)
        piv = br.pivot(index='entry', columns='region', values='lift')
        print("\nlift within region (her relict windows vs the rest of that region):")
        print(piv.round(3).to_string())
        nonppb = [c for c in piv.columns if c != 'PPB']
        print(f"\npositive lift across the {len(nonppb)} non-PPB regions {nonppb}:")
        print((piv[nonppb] > 0).sum(axis=1).sort_values(ascending=False).to_string())
        print("\n>>> A lift that only appears in the PPB column is a region effect, not agreement.")
        res3 = run(d[d.region != 'PPB'],
                   'POOLED, PPB EXCLUDED — underpowered, reads as falsification only')
        br.to_csv(os.path.join(ROOT, 'ockenden_relict_by_region.csv'), index=False)
        print(f"Wrote {os.path.join(ROOT, 'ockenden_relict_by_region.csv')}")

    # The original region-level formulation, kept because it is what the queue item asked.
    print(f"\n{TARGET} presence by region (region-level framing):")
    print((d.groupby('region')[TARGET].agg(['mean', 'size']).round(3)
           .sort_values('mean', ascending=False).to_string()))

    res['subset'], res2['subset'] = 'all', 'non_straddling'
    parts = [res, res2]
    if BY_REGION:
        res3['subset'] = 'ppb_excluded'
        parts.append(res3)
    pd.concat(parts, ignore_index=True).to_csv(
        os.path.join(ROOT, 'ockenden_relict_test.csv'), index=False)
    print(f"\nWrote {os.path.join(ROOT, 'ockenden_relict_test.csv')} ({len(parts)} subsets)")
