"""Per-campaign bed/trajectory audit across Bedmap1, 2 and 3.

Answers "is Bedmap1/2 usable at all in the ODSA pipeline?" — loading.py drops
rows with bedrock_altitude == -9999 and everything downstream groups by
trajectory_id, so a campaign is only usable if it has BOTH valid bed AND
trajectory ids that name flight lines rather than rows.

Writes scan_bed_validity.log + scan_bed_validity.csv.
"""

import glob
import os
import numpy as np
import pandas as pd

RESULTS = '/home/ana/Desktop/code/Data/ODSA/all_data/bedmap3_data/bedmap{gen}/Results/'
COLS = ['bedrock_altitude (m)', 'trajectory_id']
CHUNK = 2_000_000
HERE = os.path.dirname(os.path.abspath(__file__))

rows = []
for gen in (1, 2, 3):
    for path in sorted(glob.glob(RESULTS.format(gen=gen) + '*.csv')):
        name = os.path.basename(path)[:-4]
        n = nbed = 0
        ids = set()
        try:
            for ch in pd.read_csv(path, comment='#', usecols=COLS, chunksize=CHUNK,
                                  low_memory=False):
                n += len(ch)
                nbed += int((ch['bedrock_altitude (m)'] != -9999).sum())
                ids.update(ch.loc[ch['bedrock_altitude (m)'] != -9999, 'trajectory_id'].unique())
        except ValueError as e:          # column missing entirely
            print(f"  ! {name}: {e}")
            continue
        ntraj = len(ids)
        rows.append(dict(gen=gen, name=name, n=n, n_bed=nbed, n_traj=ntraj,
                         pts_per_traj=nbed / ntraj if ntraj else 0))
        print(f"BM{gen} {name:45s} rows={n:>9,} bed={nbed:>9,} "
              f"trajs={ntraj:>7,} pts/traj={rows[-1]['pts_per_traj']:>8.1f}")

df = pd.DataFrame(rows)
# A campaign is unusable if it has no bed, or if trajectory_id is effectively a
# row counter (one point per "track" leaves nothing to measure along-track).
df['verdict'] = np.where(df['n_bed'] == 0, 'no bed',
                  np.where(df['pts_per_traj'] < 5, 'degenerate trajectory_id', 'usable'))
df.to_csv(os.path.join(HERE, 'scan_bed_validity.csv'), index=False)

print('\n' + '=' * 78)
print('SUMMARY — campaigns by release and verdict')
print('=' * 78)
print(pd.crosstab(df['gen'], df['verdict']))
print('\nPoints (millions) with valid bed, by release and verdict:')
print(df.pivot_table(index='gen', columns='verdict', values='n_bed',
                     aggfunc='sum').fillna(0) / 1e6)
print('\nUsable campaigns per release:')
for g, sub in df[df['verdict'] == 'usable'].groupby('gen'):
    print(f"  BM{g}: {len(sub)} campaigns, {sub['n_bed'].sum()/1e6:.1f} M bed points")
print('\nUnusable, worst first:')
for _, r in df[df['verdict'] != 'usable'].sort_values('n', ascending=False).iterrows():
    print(f"  BM{r['gen']} {r['name']:45s} {r['verdict']:26s} rows={r['n']:>9,}")
