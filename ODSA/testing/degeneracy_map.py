#!/usr/bin/env python3
"""
Re-derive the catalogue degeneracy structure (landscape_catalogue.md §4).

Enumerates every fully-specified point in observable space, counts how many entries each
admits, and reports the overlapping pairs. A property of the catalogue alone, no data:
it moves whenever the catalogue moves, same re-registration rule as the ceiling.

      python degeneracy_map.py
"""
import collections, itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import landscape_vector as lv                                          # noqa: E402


def main():
    axes = lv.ALL_AXES
    pts = list(itertools.product(*[lv.AXIS_VALUES[a] for a in axes]))
    hits = [[c['id'] for c in lv.CATALOGUE
             if all(dict(zip(axes, p))[a] in s for a, s in c['c'].items())] for p in pts]
    n = len(pts)

    print(' x '.join(f"{a}({len(lv.AXIS_VALUES[a])})" for a in axes) + f" = {n} points\n")
    for k, c in sorted(collections.Counter(map(len, hits)).items()):
        print(f"  {k} entries   {c:5d}   {c / n:6.1%}")

    pairs = collections.Counter()
    for h in hits:
        pairs.update(itertools.combinations(sorted(h), 2))
    print(f"\n  overlapping points: {sum(len(h) > 1 for h in hits)}")
    for (x, y), c in pairs.most_common():
        print(f"    {x} | {y}   {c}")

    # A pair listing splits a point carrying three or more entries across several rows,
    # so the combination itself is named here.
    wide = collections.Counter(tuple(sorted(h)) for h in hits if len(h) > 2)
    if wide:
        print("\n  points carrying more than two entries")
        for ids, c in wide.most_common():
            print(f"    {' | '.join(ids)}   {c}")

    # Overlapping points per entry, which is not the sum of its pair rows: a point with
    # three entries is counted once here and in three pair rows above.
    over = collections.Counter(i for h in hits if len(h) > 1 for i in h)
    print("\n  overlapping points each entry appears in")
    for c in lv.CATALOGUE:
        print(f"    {c['id']:14s} {over.get(c['id'], 0):4d}")

    # An entry that is never a sole match cannot be reported on its own, whatever the data.
    sole = collections.Counter(h[0] for h in hits if len(h) == 1)
    print("\n  points where each entry is the sole match")
    for c in lv.CATALOGUE:
        print(f"    {c['id']:14s} {sole.get(c['id'], 0):4d}")

    subsumed = lv.reachable_groups()
    print(f"\n  never a sole match: {sorted(subsumed) or 'none'}")
    print("\n  Re-register §4 and the ceiling together if the catalogue has changed.")


if __name__ == '__main__':
    main()
