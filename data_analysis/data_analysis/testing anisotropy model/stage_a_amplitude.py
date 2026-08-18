"""Item 5: amplitude anisotropy on the Stage A profiles, as an independent check on Δβ.

Δβ is slope anisotropy and is the only kind [Hubbard_2000] measures. Cooper's Ω,
[Rippin_2014]'s directionality and [Taylor_2004]'s proto-anisotropy are all amplitude
anisotropy, and Hubbard's Site 2 shows the two can decouple. That is what makes agreement
between them evidence rather than restatement. This runs the same two-parameter contrast on
`psd_amplitude_1km` that `weighted_anisotropy` runs on β, over exactly the same profiles.

cos²θ is kept as the interpolating convention so the two contrasts are comparable rather than
because amplitude is expected to follow it; the `amp_aic_*` columns say whether it does.

Amplitude is already log10 PSD, so delta_amp is a log ratio and `power_ratio` = 10**delta_amp
is the along-fabric to cross-fabric in-band power ratio. That ratio is the properly in-band
version of the bedform-relief over profile-rms proxy in TESTING_ANISOTROPY.

    python stage_a_amplitude.py
    python stage_a_amplitude.py --site "Site F Nunavut"

Needs the `amp` column that stage_a.py writes from the fit intercept. Profile CSVs written
before that patch do not carry it and are skipped with a note.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from anisotropy_null import sweep                        # noqa: E402
from config import FIT_BAND_M                            # noqa: E402
from stage_a import compare_models                       # noqa: E402
from weighted_anisotropy import fit_cos2                 # noqa: E402

DATA = HERE / 'data'


def band_centroid_m(csv):
    """Log-centroid of the fit band, where the fitted amplitude is least correlated with the
    slope. 1 km sits off it, so amp_1km carries some of beta's estimation error. The band
    differs per run, so take it from the JSON carrying this CSV's own suffix."""
    prefix, _, sfx = csv.name[:-4].partition('_stage_a_profiles')
    js = csv.parent / f'{prefix}_stage_a{sfx}.json'
    band = json.loads(js.read_text())['band_m'] if js.exists() else list(FIT_BAND_M)
    return 10 ** np.mean(np.log10(band))


def harmonic_norm(az, y):
    """Second-harmonic amplitude of y against azimuth, over the scatter it sits in. The
    bootstrap SE understates by 1.2x to 2.5x depending on the variable, so this is the
    statistic to carry: it is comparable across sites and against the synthetic seeds in
    null_floor_seeds.csv, where the truth is zero."""
    _, curve = sweep(az, y, None)
    amplitude = float((curve.max() - curve.min()) / 2)
    return amplitude, amplitude / (y.std() / np.sqrt(len(np.unique(az))))


def analyse(csv):
    d = pd.read_csv(csv)
    if 'amp' not in d:
        print(f'  {csv.parent.parent.name}: no amp column, re-run stage_a.py')
        return None
    d = d.dropna(subset=['beta', 'amp', 'incidence_deg'])
    th, b, a = d.incidence_deg.values, d.beta.values, d.amp.values

    # Amplitude re-evaluated at the band centroid: P(w) = amp_1km + beta*(log10 w - 3).
    wc = band_centroid_m(csv)
    ac = a + b * (np.log10(wc) - 3)

    fb, fa = fit_cos2(th, b, quiet=True), fit_cos2(th, a, quiet=True)
    fc = fit_cos2(th, ac, quiet=True)
    if not (fb and fa and fc):
        print(f'  {csv.parent.parent.name}: fit failed')
        return None

    # If beta and amplitude are strongly coupled per profile they are not two degrees of
    # freedom and agreement between the contrasts would be circular.
    rho, rho_p = stats.spearmanr(b, a)

    az = d.azimuth_deg.values
    (A_b, An_b), (A_a, An_a), (A_c, An_c) = (harmonic_norm(az, v) for v in (b, a, ac))

    # Concordance is OPPOSITE signs, not equal ones. A steeper along-fabric slope and lower
    # along-fabric in-band power both say the bed is smoother along the fabric, which is
    # Hubbard's abrasion direction. Equal signs would mean the two channels disagree.
    row = dict(site=csv.parent.parent.name, file=csv.name, n=len(d),
               band_centroid_m=round(float(wc)),
               delta_beta=round(fb['delta'], 4), delta_beta_se=round(fb['delta_se'], 4),
               beta_A=round(A_b, 4), beta_A_norm=round(An_b, 2),
               delta_amp=round(fa['delta'], 4), delta_amp_se=round(fa['delta_se'], 4),
               amp_A=round(A_a, 4), amp_A_norm=round(An_a, 2),
               power_ratio=round(float(10 ** fa['delta']), 3),
               amp_par=round(fa['beta_par'], 4), amp_perp=round(fa['beta_perp'], 4),
               amp_r2=round(fa['r2'], 4),
               delta_amp_c=round(fc['delta'], 4), delta_amp_c_se=round(fc['delta_se'], 4),
               amp_c_A=round(A_c, 4), amp_c_A_norm=round(An_c, 2),
               rho_beta_amp=round(float(rho), 3), rho_p=float(f'{rho_p:.1e}'),
               rho_beta_amp_c=round(float(stats.spearmanr(b, ac).statistic), 3),
               concordant=bool(np.sign(fb['delta']) != np.sign(fa['delta'])),
               concordant_c=bool(np.sign(fb['delta']) != np.sign(fc['delta'])))
    return row | {f'amp_{k}': v for k, v in compare_models(th, a).items()}


if __name__ == '__main__':
    args = sys.argv[1:]
    want = args[args.index('--site') + 1] if '--site' in args else None
    csvs = sorted(DATA.glob('*/prep/*_stage_a_profiles*.csv'))
    if want:
        csvs = [c for c in csvs if c.parent.parent.name == want]
    if not csvs:
        sys.exit(f'no Stage A profile CSVs in {DATA}' + (f' for {want!r}' if want else ''))

    rows = [r for r in (analyse(c) for c in csvs) if r]
    if not rows:
        sys.exit('nothing to report')

    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'stage_a_amplitude.csv', index=False)

    print(f"\n{'site':20s} {'band':>12s} {'n':>5s} {'delta_beta':>16s} {'A_norm':>6s} "
          f"{'delta_amp_c':>16s} {'A_norm':>6s} {'rho':>6s}  conc")
    for r in rows:
        band = r['file'][:-4].partition('_stage_a_profiles')[2] or 'production'
        print(f"{r['site'][:20]:20s} {band[-12:]:>12s} {r['n']:5d} "
              f"{r['delta_beta']:+.3f} +/- {r['delta_beta_se']:.3f} {r['beta_A_norm']:6.2f} "
              f"{r['delta_amp_c']:+.3f} +/- {r['delta_amp_c_se']:.3f} "
              f"{r['amp_c_A_norm']:6.2f} {r['rho_beta_amp_c']:+6.2f}  "
              f"{'yes' if r['concordant_c'] else 'no'}")
    print('\nA_norm null p95 over 20 synthetic seeds: beta 3.29, amp_c 1.74.')
    print(f"\n-> {HERE / 'stage_a_amplitude.csv'}")
