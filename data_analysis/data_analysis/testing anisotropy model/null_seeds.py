"""Null floor over seeds. Generate isotropic beds, run Stage A on each, keep the fits.

The floor has two components and one bed only gives the first. Sweeping the assumed bearing
on a single bed (`null_floor.py`) gives the sampling-geometry null: what the estimator returns
because of where the profiles fall. Seeds give the realisation null: the same geometry on a
different draw of the same process. Δβ_true is zero in every one, so the spread across seeds
is the floor a real Δβ has to clear.

Each DEM is 293 MB and is deleted once its profiles are fitted, so peak disk is one bed. The
generator holds ~4.4 GB, so seeds run one at a time.

    python null_seeds.py                    # seeds 1-20 into null_seeds/
    python null_seeds.py --n 5 --start 100
    python null_seeds.py --keep-dem         # 293 MB per seed, for spot checks
    python null_seeds.py -- --beta-1d 1.73  # everything after -- goes to the generator

Seed 0 is the bed already in data/synthetic_isotropic/ and is folded into the summary if its
Stage A profiles are present. Reruns skip any seed whose profile CSV already exists.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_a                                          # noqa: E402
from anisotropy_null import delta, sweep                # noqa: E402

FACTORY = Path.home() / 'Desktop/code/bedrock_factory/synthetic_isotropy'
GEN = FACTORY / 'synthetic_isotropic.py'
WORK = HERE / 'null_seeds'
SEED0 = HERE / 'data/synthetic_isotropic/prep/synthetic_isotropic_stage_a_profiles.csv'
GRIDS = ('dem.tif', 'water.tif', 'fabric.csv')


def generate(seed, name, passthru=()):
    """Run the generator in its own folder, then move the three grids into a site layout.
    Anything after -- on the command line is forwarded, e.g. -- --beta-1d 1.73."""
    subprocess.run([sys.executable, str(GEN), '--seed', str(seed), '--name', name, *passthru],
                   cwd=FACTORY, check=True)
    prep = WORK / name / 'prep'
    prep.mkdir(parents=True, exist_ok=True)
    for suffix in GRIDS:
        (FACTORY / f'{name}_{suffix}').rename(prep / f'{name}_{suffix}')
    for extra in ('constants.md', 'diagnostics.png'):
        src = FACTORY / f'{name}_{extra}'
        if src.exists():
            src.rename(WORK / name / f'{name}_{extra}')
    return prep


def floor_from(csv):
    """Sweep the assumed bearing on one bed: the sampling-geometry null for that seed."""
    d = pd.read_csv(csv).dropna(subset=['beta', 'azimuth_deg', 'incidence_deg'])
    _, null = sweep(d.azimuth_deg.values, d.beta.values, None)
    return dict(delta_obs=float(delta(d.incidence_deg.values, d.beta.values, None)),
                sweep_floor95=float(np.percentile(np.abs(null), 95)),
                sweep_sd=float(null.std()))


def row_for(seed, name, prep, qc=None):
    csv = prep / f'{name}_stage_a_profiles.csv'
    r = dict(seed=seed, name=name, **floor_from(csv))
    if qc:
        r |= {k: qc.get(k) for k in ('n_profiles', 'delta_beta_ref', 'delta_se', 'r2',
                                     'best_model', 'beta_median')}
    return r


if __name__ == '__main__':
    args = sys.argv[1:]
    passthru = args[args.index('--') + 1:] if '--' in args else []
    args = args[:args.index('--')] if '--' in args else args
    n = int(args[args.index('--n') + 1]) if '--n' in args else 20
    start = int(args[args.index('--start') + 1]) if '--start' in args else 1
    keep = '--keep-dem' in args

    rows = []
    if SEED0.exists():
        rows.append(row_for(0, 'synthetic_isotropic', SEED0.parent))

    for seed in range(start, start + n):
        name = f'synthetic_iso_s{seed:03d}'
        prep = WORK / name / 'prep'
        if (prep / f'{name}_stage_a_profiles.csv').exists():
            print(f'{name}: already fitted, skipping')
            rows.append(row_for(seed, name, prep))
            continue

        print(f'\n{"=" * 60}\nSEED {seed}  ({seed - start + 1} of {n})\n{"=" * 60}')
        if not (prep / f'{name}_dem.tif').exists():
            prep = generate(seed, name, passthru)
        qc = stage_a.run(WORK / name, prep, stage_a.AZ_STEP)
        if not keep:
            (prep / f'{name}_dem.tif').unlink()
        rows.append(row_for(seed, name, prep, qc))

    df = pd.DataFrame(rows).sort_values('seed')
    df.to_csv(HERE / 'null_floor_seeds.csv', index=False)

    d = df.delta_obs.values
    print(f'\n{"=" * 60}\nFLOOR over {len(d)} seeds, true delta_beta = 0')
    print(f'  delta_beta   mean {d.mean():+.4f}   sd {d.std(ddof=1):.4f}   '
          f'range {d.min():+.3f} to {d.max():+.3f}')
    print(f'  |delta_beta| p50 {np.percentile(abs(d), 50):.4f}   '
          f'p95 {np.percentile(abs(d), 95):.4f}   max {abs(d).max():.4f}')
    print(f'  within-seed sweep floor95, median over seeds '
          f'{df.sweep_floor95.median():.4f}')
    if 'delta_se' in df and df.delta_se.notna().any():
        print(f'  bootstrap SE, median over seeds {df.delta_se.median():.4f}  '
              f'(compare against the sd above)')
    print(f"\n-> {HERE / 'null_floor_seeds.csv'}")
