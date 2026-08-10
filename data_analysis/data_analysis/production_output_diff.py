"""Diff every CSV a production run writes against a baseline run.

Open-ended sibling of v23/hill_count_additive_check.py: instead of one prescribed
schema it walks both output trees, pairs CSVs by relative path, and reports what
moved. Columns are discovered from the files, so a new output family needs no edit
here. Added columns never fail the check, since adding them is usually the point.
A dropped column, a changed value in a shared column, a row-key mismatch or a
missing file does fail.

Usage:  python production_output_diff.py [NEW BASELINE [NEW BASELINE ...]]
        Default pair is Ockenden-regions vs Ockenden-regions.bak.
        Paths resolve against the cwd or the ODSA root, so it runs from either.
   eg:  python production_output_diff.py
        python production_output_diff.py individual_region_TEST Ockenden-regions
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from config import Tee

ODSA = Path(__file__).resolve().parent
DEFAULT = ('Ockenden-regions', 'Ockenden-regions.bak')
LOG = ODSA / 'production_output_diff.log'

# Both sides are CSV round-trips, so decimal formatting alone moves values by ~1e-16
# relative. RTOL sits far above that and far below any real change.
RTOL = 1e-9

# bed_character.py adds these to the window CSVs in a later pass, so a fresh run that
# has not had it applied yet is short of them. That is expected, not a regression.
LATE_PASS_COLS = {'bed_class', 'class_confidence', 'p_chaotic', 'p_hard',
                  'p_transitional', 'p_soft', 'relief_class', 'elevation_class'}

# Tried in this order to build a row key. Only columns that identify a row belong here.
# A measured column must stay out: put wavelength_m in the key and a wavelength that
# shifted reads as a row that vanished, instead of a value that moved.
ID_HINTS = ['region', 'trajectory', 'trajectory_id', 'segment', 'window_id']


def resolve(p):
    q = Path(p)
    return q if q.exists() else ODSA / p


def pick_keys(N, B):
    """Greedily grow a key from ID_HINTS until it is unique on both sides."""
    keys = []
    for c in [c for c in ID_HINTS if c in N.columns and c in B.columns]:
        keys.append(c)
        if not N.duplicated(keys).any() and not B.duplicated(keys).any():
            return keys, 'unique'
    return keys, 'ambiguous' if keys else 'none'


def align(N, B, shared):
    """Return (n, b, how) indexed so row order cannot fake a difference, or (None, None, why)."""
    keys, kind = pick_keys(N, B)
    if kind == 'ambiguous':
        # Key repeats (eg several wavelength detections per window). Number the repeats
        # in file order, which the pipeline writes deterministically, so the nth
        # detection of a window meets the nth of the baseline.
        N, B = (df.sort_values(keys, kind='mergesort') for df in (N, B))
        N = N.assign(_dup=N.groupby(keys, dropna=False).cumcount())
        B = B.assign(_dup=B.groupby(keys, dropna=False).cumcount())
        keys = keys + ['_dup']
    elif kind == 'none':
        if len(N) != len(B):
            return None, None, f'no key column and {len(N)} vs {len(B)} rows'
        return N.reset_index(drop=True), B.reset_index(drop=True), 'row order'
    n, b = N.set_index(keys).sort_index(), B.set_index(keys).sort_index()
    how = ' + '.join(keys) if kind == 'unique' else ' + '.join(keys) + ' (key repeats)'
    return n, b, how


def compare(n, b, shared):
    """Shared columns whose values moved, as (column, max |diff|, rows changed).
    Key columns sit in the index by now and are equal by construction, so skip them."""
    moved, worst = [], 0.0
    for c in [c for c in shared if c in n.columns and c in b.columns]:
        x, y = n[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(float), y.to_numpy(float)
            with np.errstate(invalid='ignore', divide='ignore'):
                rel = np.abs(x - y) / np.maximum(np.abs(y), 1e-30)
            rel = np.where(np.isnan(x) & np.isnan(y), 0.0, rel)
            worst = max(worst, np.nanmax(rel) if rel.size else 0.0)
            if not np.allclose(x, y, rtol=RTOL, atol=RTOL, equal_nan=True):
                d = np.abs(x - y)
                moved.append((c, np.nanmax(d), int(np.nansum(d > RTOL))))
        else:
            x, y = x.astype(str).to_numpy(), y.astype(str).to_numpy()
            if (x != y).any():
                moved.append((c, np.nan, int((x != y).sum())))
    return moved, worst


args = sys.argv[1:] or list(DEFAULT)
if len(args) % 2:
    sys.exit(__doc__)
pairs = [(resolve(a), resolve(b)) for a, b in zip(args[::2], args[1::2])]

sys.stdout = Tee(LOG)

fail = 0
for new_dir, base_dir in pairs:
    print(f'\n{"=" * 78}\n{new_dir}  vs  {base_dir}\n{"=" * 78}')
    if not new_dir.is_dir() or not base_dir.is_dir():
        print(f'  MISSING DIRECTORY: {[d for d in (new_dir, base_dir) if not d.is_dir()]}')
        fail += 1
        continue

    newf = {f.relative_to(new_dir) for f in new_dir.rglob('*.csv')}
    basef = {f.relative_to(base_dir) for f in base_dir.rglob('*.csv')}
    for rel in sorted(newf - basef):
        print(f'  new file   : {rel}')
    for rel in sorted(basef - newf):
        print(f'  LOST FILE  : {rel}')
        fail += 1
    print(f'  {len(newf & basef)} CSV(s) in both, {len(newf - basef)} new, '
          f'{len(basef - newf)} only in baseline\n')

    for rel in sorted(newf & basef):
        N = pd.read_csv(new_dir / rel, low_memory=False)
        B = pd.read_csv(base_dir / rel, low_memory=False)
        added = sorted(set(N) - set(B))
        dropped = sorted(set(B) - set(N))
        pending = sorted(set(dropped) & LATE_PASS_COLS)
        lost = sorted(set(dropped) - LATE_PASS_COLS)
        shared = [c for c in N.columns if c in B.columns]
        bad = bool(lost)

        moved, note = [], []
        n, b, how = align(N, B, shared)
        if n is None:
            bad, tail = True, f'CANNOT ALIGN ({how})'
        else:
            extra = n.index.difference(b.index) if how != 'row order' else []
            gone = b.index.difference(n.index) if how != 'row order' else []
            if len(extra) or len(gone):
                bad = True
                note.append(f'{len(extra)} row(s) only in new, {len(gone)} only in baseline')
                common = n.index.intersection(b.index)
                n, b = n.loc[common], b.loc[common]
                tail = 'KEY MISMATCH, '
            else:
                tail = ''
            moved, worst = compare(n, b, shared)
            bad = bad or bool(moved)
            tail += (f'{len(moved)} column(s) MOVED' if moved else
                     f'all identical (worst {worst:.0e})')
        print(f'  {str(rel):<58s} {len(N):>6d} rows, {len(shared):>3d} shared, {tail}')

        for label, cols in (('added  ', added), ('absent ', pending), ('LOST   ', lost)):
            if cols:
                extra_note = '  (bed_character.py not run on the new output yet)' if cols is pending else ''
                print(f'      {label}: {cols}{extra_note}')
        for c, mx, cnt in moved:
            print(f'      MOVED  : {c:<34s} max |diff| {mx:<12.6g} rows {cnt}')
        if bad:
            note.append(f'aligned on {how}')
            print(f'      note   : {"; ".join(note)}')
        fail += bool(bad)

print('\n' + ('FAIL: see the flagged lines above' if fail else
             'PASS: every shared column matches the baseline; only additions differ'))

sys.stdout.flush()
sys.exit(1 if fail else 0)
