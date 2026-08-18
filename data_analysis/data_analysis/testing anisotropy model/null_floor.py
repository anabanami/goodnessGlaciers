"""Rotation-surrogate floor on Stage A profiles.

One bed fitted at one fabric bearing returns one Δβ, not a floor. Keep every β as measured
and replace the fabric bearing with a uniform ψ swept through 180°; the spread of the
resulting Δβ is the floor for that seed and that sampling geometry. Estimator is imported
from `anisotropy_null` so this and the tier-1 production floor are the same fit.

**The sweep is not a significance test on a dense azimuth grid.** With azimuths on a full
uniform 0-180 grid, fitting cos²θ at bearing ψ extracts the second Fourier harmonic of β
against azimuth at phase ψ, so Δβ(ψ) = A·cos(2(ψ - ψ_peak)) to within a percent (measured
shape R² 0.993 to 0.997 at all three sites). Sweeping ψ only rotates the phase. The high
percentiles of |Δβ(ψ)| therefore measure A, the anisotropy amplitude itself, and comparing
the observed Δβ against them asks whether the signal exceeds the signal. `p_null` comes out
near 0.5 by construction and `floor95` near A. Neither is a verdict.

What the sweep does give is `harmonic_A`, and its scatter-normalised form `A_norm` =
A / (sd(β) / sqrt(n_azimuths)), which is comparable across sites of differing β scatter. The
valid test is A_norm at a site against the distribution of A_norm over the synthetic seeds in
`null_floor_seeds.csv`, where the truth is zero. That comparison lives outside this script.

    python null_floor.py
    python null_floor.py --site synthetic_isotropic

Weights are unset throughout. A deglaciated or synthetic bed has no flow field, so
`flow_weight` degenerates and the validated object is the unweighted fit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from anisotropy_null import band_floors, delta, sweep  # noqa: E402

DATA = HERE / 'data'
PCT = [50, 90, 95, 99]


def analyse(csv):
    d = pd.read_csv(csv).dropna(subset=['beta', 'azimuth_deg', 'incidence_deg'])
    az, b, inc = d.azimuth_deg.values, d.beta.values, d.incidence_deg.values
    obs = delta(inc, b, None)
    psi, null = sweep(az, b, None)
    a = np.abs(null)

    # A is the second-harmonic amplitude of beta against azimuth; A_norm divides out the
    # per-site beta scatter so sites with different noise levels can be compared.
    A = float((null.max() - null.min()) / 2)
    row = dict(source=csv.parent.parent.name, file=csv.name, n=len(d), n_psi=len(psi),
               delta_obs=round(float(obs), 4),
               harmonic_A=round(A, 4), psi_peak=round(float(psi[np.argmax(null)]), 1),
               A_norm=round(A / (d.beta.std() / np.sqrt(d.azimuth_deg.nunique())), 3),
               p_null=round(float((a >= abs(obs)).mean()), 4))
    row |= {f'floor{p}': round(float(np.percentile(a, p)), 4) for p in PCT}

    # The fabric bearing is one member of the sweep when it is uniform across nodes, so the
    # sweep value there should reproduce delta_obs. A gap means theta varies spatially.
    i = int(np.argmin(np.abs(psi - np.median(az - inc) % 180)))
    row['delta_at_fabric_psi'] = round(float(null[i]), 4)
    return row | band_floors(az, b, None)


if __name__ == '__main__':
    args = sys.argv[1:]
    want = args[args.index('--site') + 1] if '--site' in args else None
    csvs = sorted(DATA.glob('*/prep/*_stage_a_profiles*.csv'))
    if want:
        csvs = [c for c in csvs if c.parent.parent.name == want]
    if not csvs:
        sys.exit(f'no Stage A profile CSVs in {DATA}' + (f' for {want!r}' if want else ''))

    rows = []
    for c in csvs:
        row = analyse(c)
        rows.append(row)
        print(f"{row['source'][:24]:24s} {row['file'][-28:]:28s} n={row['n']:5d}  "
              f"obs={row['delta_obs']:+.3f}  A={row['harmonic_A']:.3f} "
              f"peak={row['psi_peak']:5.1f}  A_norm={row['A_norm']:.2f}")
    print('\nA_norm is the statistic. Compare it against the seed distribution in '
          'null_floor_seeds.csv,\nnot against floor95 in this table.')

    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'null_floor_stage_a.csv', index=False)
    print(f"\n{len(df)} fits -> {HERE / 'null_floor_stage_a.csv'}")
