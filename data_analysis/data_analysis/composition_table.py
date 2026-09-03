"""Paper-facing composition table.

`composition()` writes every admissible set a region takes, 7 to 25 of them, which is too long
to print. This keeps each region's leading sets up to a coverage threshold, collapses the tail
into one row, and keeps `(none)` explicit wherever it lands.

Fractions are over every window, because the composition is an areal statement and every window
covers real ground whether or not it is independent of its neighbour. `n_independent` governs the
error bar only, and no CI is attached: at 2 to 6 independent windows the SE on a fraction rivals
the fraction itself.
For the same reason the table carries no cross-region comparison.

    python composition_table.py [individual_region_TEST] [--cover 0.8]
"""
import glob, os, sys
import pandas as pd

ROOT = next((a for a in sys.argv[1:] if not a.startswith('--')), 'individual_region_TEST')
COVER = float(next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--cover=')), 0.80))
ESCAPED_PIPE = '\\|'


def load(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'landscape_vector', '*_composition.csv'))):
        region = os.path.basename(os.path.dirname(os.path.dirname(f)))
        out.append(pd.read_csv(f).assign(region=region))
    return pd.concat(out, ignore_index=True)


def collapse(g, cover=COVER):
    """Leading sets to `cover` of the region, then one tail row. `(none)` is never collapsed:
    a region's unmatched share is the out-of-catalogue rate and has to stay readable."""
    g = g.sort_values('fraction', ascending=False)
    keep = g.fraction.cumsum().shift(fill_value=0) < cover
    keep |= g.admissible == '(none)'
    head, tail = g[keep], g[~keep]
    rows = head[['admissible', 'n_windows', 'fraction']].to_dict('records')
    if len(tail):
        rows.append({'admissible': f'{len(tail)} further sets',
                     'n_windows': int(tail.n_windows.sum()),
                     'fraction': round(float(tail.fraction.sum()), 3)})
    return rows


def main(root):
    d = load(root)
    lines = ['| region | n | n independent | admissible set | windows | fraction |',
             '| --- | --- | --- | --- | --- | --- |']
    flat = []
    for region, g in d.groupby('region'):
        n, n_ind, n_sets = int(g.n_windows_total.iloc[0]), int(g.n_independent.iloc[0]), len(g)
        rows = collapse(g)
        for i, r in enumerate(rows):
            # the set separator is itself a pipe, so it has to be escaped for the cell
            cell = r['admissible'].replace('|', ESCAPED_PIPE)
            lines.append(f"| {region if i == 0 else ''} | {n if i == 0 else ''} | "
                         f"{n_ind if i == 0 else ''} | {cell} | "
                         f"{r['n_windows']} | {r['fraction']:.3f} |")
            flat.append(dict(region=region, n_windows_total=n, n_independent=n_ind,
                             n_sets_total=n_sets, **r))
        print(f"{region:8s} {n:4d} windows, {n_ind} independent, {n_sets} distinct sets, "
              f"{len(rows)} rows printed")

    print(f"\ncoverage threshold {COVER:.0%}, `(none)` never collapsed\n")
    print('\n'.join(lines))
    out = os.path.join(root, 'composition_table.csv')
    pd.DataFrame(flat).to_csv(out, index=False)
    print(f"\n  Saved: {out}")


if __name__ == '__main__':
    from config import Tee
    sys.stdout = Tee(os.path.join(ROOT, 'composition_table_log.txt'))
    main(ROOT)
