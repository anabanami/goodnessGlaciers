import os, sys, glob, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from config import Tee, PROCESSING_FLAG_NOTE, processing_flag_of
from plotting import flag_suptitle
# Decision constants live with the classifier that applies them; no cycle, landscape_vector
# does not import this module.
from landscape_vector import (K_SIGMA, DELTA_BETA_BOOTSTRAP_INFLATION,
                              DELTA_BETA_MIN_EFFECT, ANISOTROPY_TRUSTED)

# Output configuration - nested inside region output from loading.py
from loading import OUTPUT_BASE_PATH as _REGION_BASE
OUTPUT_BASE_PATH = os.path.join(_REGION_BASE, 'anisotropy/')

"""
Compares cos²(θ) anisotropy fits with and without MEaSUREs-based weighting.

Windows where REMA and MEaSUREs flow directions disagree strongly have
unreliable incidence angles. This script down-weights those windows
in the cos²(θ) fit and shows the effect on the anisotropy signal.

Usage:
  python weighted_anisotropy.py                    # region menu, or walks a tree of
                                                   # region folders if the base holds those
  python weighted_anisotropy.py Aurora              # partial match on region or folder name
  python weighted_anisotropy.py individual_region_TEST  # walk a tree of region folders
  python weighted_anisotropy.py some_window_stats.csv   # direct path

A walked region writes its own <region>/anisotropy/, figures, local Delta_beta CSV and log
alike, exactly as a single-region run does.
"""


def discover_regions(directory='.'):
    """Find all region datasets: window CSVs in window_csvs/, segment CSVs in region subfolders."""
    regions = {}
    # Window CSVs in window_csvs/
    for f in glob.glob(os.path.join(directory, 'window_csvs', '*_window_stats.csv')):
        region = os.path.basename(f).replace('_window_stats.csv', '')
        regions.setdefault(region, {})['window'] = f
    # Segment CSVs in segment_csvs/
    for f in glob.glob(os.path.join(directory, 'segment_csvs', '*_segment_stats.csv')):
        region = os.path.basename(f).replace('_segment_stats.csv', '')
        regions.setdefault(region, {})['segment'] = f
    # Fallback: flat directory (legacy layout)
    if not regions:
        for kind, pattern in [('segment', '*_segment_stats.csv'), ('window', '*_window_stats.csv')]:
            for f in glob.glob(os.path.join(directory, pattern)):
                region = os.path.basename(f).replace(f'_{kind}_stats.csv', '')
                regions.setdefault(region, {})[kind] = f
    return regions


def select_region(regions):
    """Interactive region selection if multiple regions available."""
    if not regions:
        print("No region datasets found (*_segment_stats.csv or *_window_stats.csv)")
        return None
    if len(regions) == 1:
        region = list(regions.keys())[0]
        print(f"Found 1 region: {region}")
        return region

    sorted_regions = sorted(regions.keys())
    print(f"\nFound {len(regions)} regions:")
    for i, r in enumerate(sorted_regions, 1):
        f = regions[r]
        print(f"  {i}. {r} [seg: {'Y' if 'segment' in f else 'N'}, win: {'Y' if 'window' in f else 'N'}]")
    print(f"  0. Process ALL regions")

    while True:
        try:
            choice = int(input("\nSelect region number (or 0 for all): ").strip())
            if choice == 0:
                return 'ALL'
            if 1 <= choice <= len(sorted_regions):
                return sorted_regions[choice - 1]
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")


def flow_weight(flow_error, speed=None, angle_cutoff=60.0, speed_cutoff=5.0):
    """
    Combined weight from flow direction agreement and velocity magnitude.
    - Angle component: linear decay from 1.0 at 0° to 0.0 at angle_cutoff.
    - Speed component: 0.0 below speed_cutoff, linear ramp to 1.0 at 2*speed_cutoff.
    Final weight is the product of both components.
    """
    w = np.clip(1.0 - flow_error / angle_cutoff, 0.0, 1.0)
    w[np.isnan(flow_error)] = 0.0
    if speed is not None:
        w_speed = np.clip((speed - speed_cutoff) / speed_cutoff, 0.0, 1.0)
        w_speed[np.isnan(speed)] = 0.0
        w *= w_speed
    return w


def cos2_model(theta_deg, beta_perp, beta_parallel):
    """β(θ) = β⊥ + (β∥ - β⊥) cos²(θ)"""
    return beta_perp + (beta_parallel - beta_perp) * np.cos(np.radians(theta_deg))**2


def _do_curve_fit(theta, beta, weights, p0):
    if weights is not None:
        sigma = np.divide(1.0, weights, out=np.full_like(weights, 1e10, dtype=float),
                          where=weights > 0)
        return optimize.curve_fit(cos2_model, theta, beta, p0=p0,
                                  sigma=sigma, absolute_sigma=False, maxfev=5000)
    return optimize.curve_fit(cos2_model, theta, beta, p0=p0, maxfev=5000)


def bootstrap_cos2_uncertainty(theta, beta, weights=None, n_boot=2000, block_length=3, seed=0):
    """Block bootstrap for cos²θ fit, optionally weighted.

    SE is the robust half-interval (p84-p16)/2, not the plain std. On small,
    heavily down-weighted samples (Aurora, Maud) the resample distribution of
    Δβ is heavy-tailed, so its std does not converge with n_boot and swings
    run-to-run (Aurora weighted: 0.27-0.60). The percentile spread is stable
    (~0.21) and seed-independent. RNG is seeded so the run reproduces exactly.
    """
    rng = np.random.default_rng(seed)
    n = len(theta)
    n_blocks = int(np.ceil(n / block_length))
    boot_params = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_length, n)) for s in starts])[:n]
        try:
            w_boot = weights[idx] if weights is not None else None
            popt, _ = _do_curve_fit(theta[idx], beta[idx], w_boot, p0=[np.mean(beta), np.mean(beta)])
            boot_params.append(popt)
        except (RuntimeError, ValueError):
            continue

    boot_params = np.array(boot_params)
    robust_se = lambda x: (np.percentile(x, 84, axis=0) - np.percentile(x, 16, axis=0)) / 2
    return robust_se(boot_params), robust_se(boot_params[:, 1] - boot_params[:, 0])


def fit_cos2(theta, beta, weights=None, n_boot=2000, quiet=False):
    """Fit cos²θ model, return dict with fit results or None on failure."""
    low, high = theta < 30, theta > 60
    p0_par = np.mean(beta[low]) if np.any(low) else np.mean(beta)
    p0_perp = np.mean(beta[high]) if np.any(high) else np.mean(beta)

    try:
        popt, _ = _do_curve_fit(theta, beta, weights, p0=[p0_perp, p0_par])
        beta_perp, beta_par = popt
        perr, delta_se = bootstrap_cos2_uncertainty(theta, beta, weights=weights, n_boot=n_boot)

        pred = cos2_model(theta, *popt)
        if weights is not None:
            ss_res = np.sum(weights * (beta - pred)**2)
            ss_tot = np.sum(weights * (beta - np.average(beta, weights=weights))**2)
        else:
            ss_res = np.sum((beta - pred)**2)
            ss_tot = np.sum((beta - np.mean(beta))**2)

        return dict(beta_par=beta_par, beta_perp=beta_perp,
                    delta=beta_par - beta_perp, delta_se=delta_se,
                    perr=perr, r2=1 - ss_res / ss_tot if ss_tot > 0 else 0, popt=popt)
    except (RuntimeError, ValueError) as e:
        if not quiet:
            print(f"  Fit failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Local anisotropy. A segment runs along one trajectory, so it carries one
# incidence angle and no lever arm for cos²θ; the angular spread lives between
# crossing tracks. Pool by distance instead, and gate on the coverage each
# neighbourhood actually has.
NEIGHBOURHOOD_RADIUS_KM = 50.0
MIN_NEIGHBOURS = 6
MIN_THETA_SPREAD_DEG = 30.0
LOCAL_N_BOOT = 300  # per-window, so the region default of 2000 is unaffordable


def delta_beta_label(delta, se, status, k=2.0):
    """Local Δβ -> catalogue axis value. A failed gate is 'not_fitted', deliberately not
    a catalogue value: no angular leverage here is a fact about the survey, not the bed."""
    if status != 'ok' or not np.isfinite(delta) or not np.isfinite(se) or se <= 0:
        return 'not_fitted'
    if abs(delta) >= k * se:
        return 'pos_sig' if delta > 0 else 'neg_sig'
    return 'zero'


def local_anisotropy(csv_path, radius_km=NEIGHBOURHOOD_RADIUS_KM, min_n=MIN_NEIGHBOURS,
                     min_spread=MIN_THETA_SPREAD_DEG, n_boot=LOCAL_N_BOOT, k_sigma=2.0):
    """Δβ per window from a cos²θ fit over every window within radius_km, weighted
    as the region fit is. Writes one row per window; `delta_beta_status` says
    whether the neighbourhood had the coverage to support a fit at all."""
    need = ['incidence_deg', 'beta', 'center_x', 'center_y']
    df = pd.read_csv(csv_path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"  LOCAL Δβ: missing {missing} — skipped.")
        return pd.DataFrame()
    df = df.dropna(subset=need)
    if 'is_transition' in df.columns:
        df = df[~df['is_transition']]
    df = df.reset_index(drop=True)
    if len(df) < min_n:
        print(f"  LOCAL Δβ: only {len(df)} windows, below min_n={min_n} — skipped.")
        return pd.DataFrame()

    theta, beta = df['incidence_deg'].to_numpy(float), df['beta'].to_numpy(float)
    speed = df['measures_speed_mean'].to_numpy(float) if 'measures_speed_mean' in df else None
    w = (flow_weight(df['flow_error_mean'].to_numpy(float), speed=speed)
         if 'flow_error_mean' in df.columns else np.ones(len(df)))
    xy = df[['center_x', 'center_y']].to_numpy(float)
    dist = np.hypot(xy[:, 0][:, None] - xy[:, 0], xy[:, 1][:, None] - xy[:, 1]) / 1000.0

    rows = []
    for i in range(len(df)):
        idx = np.flatnonzero(dist[i] <= radius_km)
        wi = w[idx]
        ok = idx[wi > 0]  # zero-weight points are effectively out of the fit
        spread = float(theta[ok].max() - theta[ok].min()) if ok.size else 0.0
        sw2 = float(np.sum(wi ** 2))
        r = dict(n_neighbours=int(idx.size), n_weighted=int(ok.size),
                 theta_spread_deg=spread, n_eff=(float(wi.sum()) ** 2 / sw2) if sw2 > 0 else 0.0,
                 delta_beta_local=np.nan, delta_beta_local_se=np.nan, r2=np.nan)
        if ok.size == 0:
            r['delta_beta_status'] = 'flow_ambiguous'
        elif ok.size < min_n or spread < min_spread:
            r['delta_beta_status'] = 'low_coverage'
        else:
            fit = fit_cos2(theta[idx], beta[idx], weights=wi, n_boot=n_boot, quiet=True)
            if fit is None:
                r['delta_beta_status'] = 'fit_failed'
            else:
                r.update(delta_beta_local=fit['delta'], delta_beta_local_se=fit['delta_se'],
                         r2=fit['r2'], delta_beta_status='ok')
        r['delta_beta_label'] = delta_beta_label(r['delta_beta_local'],
                                                 r['delta_beta_local_se'],
                                                 r['delta_beta_status'], k=k_sigma)
        rows.append(r)

    ids = [c for c in ('trajectory', 'segment', 'window_id') if c in df.columns]
    out = pd.concat([df[ids + ['center_x', 'center_y']], pd.DataFrame(rows)], axis=1)

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    path = os.path.join(OUTPUT_BASE_PATH, os.path.basename(csv_path)
                        .replace('_window_stats.csv', '_delta_beta_local.csv'))
    out.to_csv(path, index=False)

    n_ok = int((out.delta_beta_status == 'ok').sum())
    print(f"\n  LOCAL Δβ (R={radius_km:.0f} km, gate n≥{min_n} and Δθ≥{min_spread:.0f}°): "
          f"{n_ok}/{len(out)} windows fitted")
    print("    status: " + ', '.join(f'{k} {v}' for k, v in out.delta_beta_status.value_counts().items()))
    print("    labels: " + ', '.join(f'{k} {v}' for k, v in out.delta_beta_label.value_counts().items()))
    if n_ok:
        g = out[out.delta_beta_status == 'ok']
        print(f"    median Δβ = {g.delta_beta_local.median():+.3f}, "
              f"median SE = {g.delta_beta_local_se.median():.3f}, "
              f"median n_eff = {g.n_eff.median():.1f}")
    if n_ok / len(out) < 0.5:
        print(f"  ** ANISOTROPY UNRELIABLE AT THIS SCALE: only {n_ok/len(out):.0%} of windows "
              f"have a neighbourhood with the angular coverage to resolve Δβ. Local Δβ should "
              f"not be read as a landscape variable here — use the region fit or nothing. **")
    print(f"    Saved: {path}")
    return out


_RESOLVED = dict(color='black', ls='-', lw=2.0)
_UNRESOLVED = dict(color='0.45', ls='--', lw=1.5)


def _delta_verdict(fit):
    """observe()'s three-way branch and its x2.52 inflation, applied to this fit's own
    region-wide block bootstrap. Same rule, different estimator: the classifier's sigma comes
    from the local fits (median local SE, inflated, then shrunk over independent patches), so
    these numbers illustrate the decision, they are not the numbers the decision is taken on."""
    se = fit['delta_se']
    half = K_SIGMA * DELTA_BETA_BOOTSTRAP_INFLATION * se
    # observe() short-circuits on the same switch, so the figure must not show a live verdict
    # for an axis the classifier is ignoring.
    if not ANISOTROPY_TRUSTED:
        return dict(se=se, half=np.nan, verdict='not_fitted',
                    words='anisotropy axis switched off (ANISOTROPY_TRUSTED=False)',
                    style=dict(_UNRESOLVED))
    if not np.isfinite(half) or half <= 0:
        return dict(se=se, half=np.nan, verdict='not_fitted',
                    words='no usable envelope', style=dict(_UNRESOLVED))
    if abs(fit['delta']) >= half:
        v, words, style = 'resolved_nonzero', 'envelope excludes zero', dict(_RESOLVED)
    elif half <= DELTA_BETA_MIN_EFFECT:
        v, words, style = ('resolved_zero',
                           f'envelope narrower than the {DELTA_BETA_MIN_EFFECT:.2f} floor',
                           dict(_UNRESOLVED))
    else:
        v, words, style = ('below_floor',
                           'envelope spans zero and is wider than the floor',
                           dict(_UNRESOLVED))
    return dict(se=se, half=half, verdict=v, words=words, style=style)


EXTRAPOLATION_TOL_DEG = 5.0  # how close to an endpoint counts as having observed it


def _coverage(theta_obs):
    """Observed angular range and which reported endpoint, if either, is extrapolated. The
    cos2 model reports beta_par at theta=0 and beta_perp at 90 whether or not either end was
    sampled: MSB spans 57-88 deg, so most of its curve is model, not measurement."""
    lo, hi = float(np.min(theta_obs)), float(np.max(theta_obs))
    return dict(lo=lo, hi=hi, spread=hi - lo,
                par_extrap=lo > EXTRAPOLATION_TOL_DEG,
                perp_extrap=hi < 90.0 - EXTRAPOLATION_TOL_DEG)


def _fit_label(fit, dv, cov=None):
    drawn = 'solid black' if dv['verdict'] == 'resolved_nonzero' else 'dashed grey'
    if np.isfinite(dv['half']):
        env = (f"$\\pm$ {dv['half']:.2f}  ({K_SIGMA:.0f}$\\sigma$, with the same $\\times$"
               f"{DELTA_BETA_BOOTSTRAP_INFLATION:.2f} inflation the classifier applies)")
    else:
        env = '$\\pm$ n/a  (no envelope drawn)'
    star = {'par': '*' if cov and cov['par_extrap'] else '',
            'perp': '*' if cov and cov['perp_extrap'] else ''}
    cover = ''
    if cov:
        which = [n for n, k in (('$\\beta_\\parallel$', 'par_extrap'),
                                ('$\\beta_\\perp$', 'perp_extrap')) if cov[k]]
        cover = (f"\nobserved $\\theta$ {cov['lo']:.0f}–{cov['hi']:.0f}° "
                 f"($\\Delta\\theta$={cov['spread']:.0f}°)"
                 + (f"; *{' and '.join(which)} extrapolated, curve dotted there"
                    if which else '; curve spans the observed range'))
    return (f"$\\beta_\\parallel$={fit['beta_par']:.2f}$\\pm${fit['perr'][1]:.2f}{star['par']}   "
            f"$\\beta_\\perp$={fit['beta_perp']:.2f}$\\pm${fit['perr'][0]:.2f}{star['perp']}   "
            f"R²={fit['r2']:.3f}\n"
            f"$\\Delta\\beta$={fit['delta']:+.2f} {env}\n"
            f"fit-internal $\\sigma$={dv['se']:.2f} (raw block bootstrap, uninflated)\n"
            f"{dv['verdict']}: {dv['words']} [{drawn}]{cover}")


def _plot_fit_curve(ax, fit, dv, x_fit, theta_obs):
    """Solid over the sampled angular range, dotted and faded outside it, with the unobserved
    part of the axis shaded. Delta_beta is the gap between the two model endpoints, so an
    unsampled end is still reported: this makes the reader see which half is measurement."""
    cov = _coverage(theta_obs)
    style = dict(dv['style'])
    inside = (x_fit >= cov['lo']) & (x_fit <= cov['hi'])
    ax.plot(x_fit[inside], cos2_model(x_fit[inside], *fit['popt']),
            label=_fit_label(fit, dv, cov), **style)
    ghost = dict(style, ls=':', lw=max(style['lw'] * 0.7, 1.0), alpha=0.5)
    for seg in ((x_fit <= cov['lo']), (x_fit >= cov['hi'])):
        if seg.sum() > 1:
            ax.plot(x_fit[seg], cos2_model(x_fit[seg], *fit['popt']), **ghost)
    for a, b in ((-2, cov['lo']), (cov['hi'], 92)):
        if b - a > 0.5:
            ax.axvspan(a, b, color='0.5', alpha=0.07, zorder=0)
    return cov


def _ruler_bars(fit, dv):
    """The three quantities the verdict compares, in the order they are drawn. Empty when
    there is no envelope, which is also how the caller learns not to size a ruler."""
    if not np.isfinite(dv['half']):
        return []
    return [(abs(fit['delta']), dv['style']['color'], r'$|\Delta\beta|$'),
            (dv['half'], '0.60', f"{K_SIGMA:.0f}$\\sigma$"),
            (DELTA_BETA_MIN_EFFECT, 'tab:red', 'floor')]


def _draw_ruler(rax, fit, dv, xmax=None):
    """The verdict's test on its own axes, below the data: |delta_beta|, the inflated 2-sigma
    envelope and the floor as horizontal bars. resolved_nonzero is bar 1 >= bar 2 and
    resolved_zero is bar 2 <= bar 3, which is what the branch compares. Deliberately not a
    band about beta_perp: that would draw |delta_beta| + half <= floor, a different test.
    Off the main axes, so a wide envelope cannot stretch the beta scale or cover a point.
    xmax shares one scale across panels: side by side, per-panel scales would draw the same
    0.10 floor at two lengths and the larger |delta_beta| as the shorter bar."""
    for sp in rax.spines.values():
        sp.set_visible(False)
    rax.grid(False)
    bars = _ruler_bars(fit, dv)
    if not bars:
        rax.set_xticks([]); rax.set_yticks([])
        rax.text(0.0, 0.5, f'$\\Delta\\beta$ ruler: no envelope, {dv["words"]}',
                 fontsize=7.5, color='0.35', va='center', transform=rax.transAxes)
        return
    h = [b[0] for b in bars]
    rax.barh(range(len(bars)), h, color=[b[1] for b in bars], height=0.62, alpha=0.9)
    rax.set_yticks(range(len(bars)))
    rax.set_yticklabels([b[2] for b in bars], fontsize=7.5)
    rax.invert_yaxis()
    scale = xmax if xmax else max(h)
    rax.set_xlim(0, scale * 1.30)
    for i, v in enumerate(h):
        rax.text(v + 0.015 * scale, i, f'{v:.2f}', va='center', fontsize=7, color='0.25')
    rax.set_xticks([0, scale])
    rax.tick_params(axis='x', labelsize=6.5, length=2, colors='0.4')
    rax.tick_params(axis='y', length=0)
    units = r'$\beta$ units, shared scale.  ' if xmax else r'$\beta$ units.  '
    rax.set_xlabel(units + r'resolved_nonzero needs $|\Delta\beta|\geq$2$\sigma$;  '
                   r'resolved_zero needs 2$\sigma\leq$floor', fontsize=7, color='0.35')


_VERDICT_TITLE = {'resolved_zero': 'resolved zero',
                  'below_floor': 'below floor, not measured',
                  'not_fitted': 'not fitted'}


def _verdict_title(fit, dv):
    if dv['verdict'] == 'resolved_nonzero':
        return f"resolved, {fit['delta']:+.2f}"
    return _VERDICT_TITLE[dv['verdict']]


def _display_name(csv_path):
    """Readable region name from the dataset label: campaign and figure tags off the front,
    window size off the back."""
    s = re.sub(r'_(window|segment)_stats\.csv$', '', os.path.basename(csv_path))
    s = re.sub(r'_w\d+km$', '', s)
    m = re.search(r'Fig\w+?_(.*)$', s)
    s = m.group(1) if m else re.sub(r'^([A-Za-z0-9\-]+_)*?(19|20)\d{2}_', '', s)
    return s.replace('_lowrelief', ' (low relief)').replace('_', ' ').strip()


_SIGMA_NOTE = ('The envelope here is this fit\'s own region-wide block bootstrap with the '
               'classifier\'s x2.52 inflation applied. The classifier builds its sigma from '
               'the per-window local_anisotropy fits instead (median local SE, inflated, then '
               'shrunk over spatially independent patches), so these numbers illustrate the '
               'decision rather than being the ones it is taken on.')


def _num(x):
    """JSON does not take numpy scalars, and NaN is not valid JSON either."""
    x = float(x)
    return x if np.isfinite(x) else None


def _sidecar_entry(region_label, level, n_total, n_valid, n_eff, pflag,
                   fit_unw, dv_unw, fit_w, dv_w, both_panels, cov_unw=None, cov_w=None):
    d_diff = fit_w['delta'] - fit_unw['delta'] if (fit_unw and fit_w) else np.nan
    has_env = fit_w and np.isfinite(dv_w['half']) and dv_w['half'] > 0
    rel = abs(d_diff) / dv_w['half'] if (has_env and np.isfinite(d_diff)) else np.nan
    panels = ('Left panel unweighted, right panel weighted by flow confidence. '
              if both_panels else 'Weighted by flow confidence. ')
    shift = ''
    if fit_unw and fit_w:
        against = (f' = {rel:.2f}x the weighted fit\'s {K_SIGMA:.0f}-sigma envelope. '
                   if np.isfinite(rel) else ', with no envelope to compare it against. ')
        shift = (f'Weighting moves delta_beta from {fit_unw["delta"]:+.3f} to '
                 f'{fit_w["delta"]:+.3f}, a shift of {d_diff:+.3f}{against}')
    # The caption carries what the figure cannot: the two sample counts, the weighting shift
    # and the scope. Encoding rules and the sigma derivation stay out of it; the encoding is
    # on the figure and the derivation is in sigma_note.
    cover = ''
    if cov_w:
        ends = [n for n, k in (('beta_par at theta=0', 'par_extrap'),
                               ('beta_perp at theta=90', 'perp_extrap')) if cov_w[k]]
        cover = (f'Weighted fit observes theta {cov_w["lo"]:.0f}-{cov_w["hi"]:.0f} deg '
                 f'(spread {cov_w["spread"]:.0f} deg)'
                 + (f'; {" and ".join(ends)} is extrapolated and the curve is dotted there. '
                    if ends else '. '))
    caption = (f'{region_label}, {level} level: region-wide diagnostic cos2(theta) fit over '
               f'n={n_total} {level}s, {n_valid} of them with non-zero flow weight, Kish '
               f'n_eff={n_eff:.1f}. {panels}{cover}{shift}Verdict {dv_w["verdict"]}: '
               f'{dv_w["words"]}. Not the per-window local_anisotropy fit the archetype '
               f'catalogue reads. Processing flag: {pflag or "unknown"}.')
    e = dict(caption=caption, region=region_label, level=level, n=int(n_total),
             n_valid=int(n_valid), n_eff=_num(n_eff), processing_flag=pflag,
             delta_weighted=_num(fit_w['delta']), half_2sigma_weighted=_num(dv_w['half']),
             delta_se_raw_weighted=_num(dv_w['se']), verdict_weighted=dv_w['verdict'],
             r2_weighted=_num(fit_w['r2']),
             beta_par_weighted=_num(fit_w['beta_par']),
             beta_perp_weighted=_num(fit_w['beta_perp']),
             k_sigma=K_SIGMA, bootstrap_inflation=DELTA_BETA_BOOTSTRAP_INFLATION,
             min_effect=DELTA_BETA_MIN_EFFECT, anisotropy_trusted=ANISOTROPY_TRUSTED,
             sigma_note=_SIGMA_NOTE)
    for tag, c in (('weighted', cov_w), ('unweighted', cov_unw)):
        if c:
            e.update({f'theta_min_{tag}': _num(c['lo']), f'theta_max_{tag}': _num(c['hi']),
                      f'theta_spread_{tag}': _num(c['spread']),
                      f'beta_par_extrapolated_{tag}': bool(c['par_extrap']),
                      f'beta_perp_extrapolated_{tag}': bool(c['perp_extrap'])})
    if fit_unw:
        e.update(delta_unweighted=_num(fit_unw['delta']),
                 half_2sigma_unweighted=_num(dv_unw['half']),
                 delta_se_raw_unweighted=_num(dv_unw['se']),
                 verdict_unweighted=dv_unw['verdict'], r2_unweighted=_num(fit_unw['r2']),
                 beta_par_unweighted=_num(fit_unw['beta_par']),
                 beta_perp_unweighted=_num(fit_unw['beta_perp']),
                 weighting_shift=_num(d_diff),
                 weighting_shift_over_weighted_envelope=_num(rel))
    return e


def _save(fig, path):
    """flag_suptitle anchors its title above the canvas and only compensates for overflow in
    width, so a two-line title clips under a plain bbox_inches='tight'. Take the tight bbox
    here, which does include it, and pad. Layout is set by explicit gridspec margins: the
    colorbar axes makes tight_layout a no-op and it warns."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(path, dpi=300, bbox_inches=bb.padded(0.25))
    plt.close(fig)


def _write_sidecar(png_name, entry):
    """One JSON per region folder, keyed by exact PNG filename. Everything the titles no
    longer carry lives here: caption, weighting shift, scope note and the raw numbers."""
    path = os.path.join(OUTPUT_BASE_PATH, 'figure_metadata.json')
    data = {}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[png_name] = entry
    with open(path, 'w') as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def _print_comparison(fit_unw, fit_w):
    print(f"\n{'='*55}")
    print(f"{'':20s} {'Unweighted':>15s} {'Weighted':>15s}")
    print(f"{'-'*55}")
    if fit_unw and fit_w:
        for label, key, idx in [('beta_parallel', 'beta_par', 1), ('beta_perp', 'beta_perp', 0)]:
            print(f"{label:20s} {fit_unw[key]:>8.3f}±{fit_unw['perr'][idx]:<5.3f} {fit_w[key]:>8.3f}±{fit_w['perr'][idx]:<.3f}")
        print(f"{'delta_beta':20s} {fit_unw['delta']:>+8.3f}±{fit_unw['delta_se']:<5.3f} {fit_w['delta']:>+8.3f}±{fit_w['delta_se']:<.3f}")
        print(f"{'R²':20s} {fit_unw['r2']:>14.4f} {fit_w['r2']:>14.4f}")
    print(f"{'='*55}")


def plot_anisotropy(csv_path, level='window'):
    """Unified anisotropy comparison plot for window or segment level data."""
    df = pd.read_csv(csv_path).dropna(subset=['incidence_deg', 'beta'])
    if 'is_transition' in df.columns:
        n_tz = int(df['is_transition'].sum())
        if n_tz:
            df = df[~df['is_transition']].copy()
            print(f"  Excluded {n_tz} transition windows from anisotropy fit ({len(df)} remain)")
    if len(df) == 0:
        print("No valid data."); return

    pflag = processing_flag_of(df)
    if pflag:
        print(f"  processing: {PROCESSING_FLAG_NOTE.get(pflag, pflag)}")

    if 'flow_error_mean' not in df.columns:
        print(f"No flow_error_mean column in {csv_path} — cannot compute weighted fit.")
        print("Run bed_analysis with MEaSUREs validation enabled first.")
        return

    theta = df['incidence_deg'].values
    beta = df['beta'].values
    beta_err = df['beta_uncertainty'].values if 'beta_uncertainty' in df.columns else None
    speed = df['measures_speed_mean'].values if 'measures_speed_mean' in df.columns else None
    weights = flow_weight(df['flow_error_mean'].values, speed=speed)
    if speed is not None:
        n_slow = np.sum(speed < 5.0)
        print(f"  {n_slow} {level}s with MEaSUREs speed < 5 m/yr (down-weighted)")

    n_total = len(theta)
    n_valid = np.sum(weights > 0)
    # Kish effective sample size: (Σw)²/Σw². The survivor count (weights > 0)
    # is not an ESS — it treats a weight of 0.05 as a full sample. ESS ≤ n_valid
    # always, with equality only for uniform weights.
    sum_w, sum_w2 = weights.sum(), np.sum(weights**2)
    n_eff = (sum_w**2 / sum_w2) if sum_w2 > 0 else 0.0
    print(f"Loaded {n_total} {level}s, {n_valid} with non-zero weight "
          f"(Kish n_eff = {n_eff:.1f})")

    if n_valid == 0:
        print(f"  FLOW-AMBIGUOUS: all {level}s have zero weight (ice speed too low).")
        print(f"  Incidence angles unreliable — skipping anisotropy fit.")
        return

    fit_unw = fit_cos2(theta, beta)
    fit_w = fit_cos2(theta, beta, weights=weights)
    if fit_unw is None and fit_w is None:
        print("Both fits failed."); return
    dv_unw = _delta_verdict(fit_unw) if fit_unw else None
    dv_w = _delta_verdict(fit_w) if fit_w else None

    # A weighted fit only sees the angles that survived weighting, so its observed range is
    # the weighted one. Report both: an unsampled endpoint is still reported as a fit value.
    theta_w = theta[weights > 0]
    cov_unw, cov_w = _coverage(theta), _coverage(theta_w)
    print(f"  angular coverage: unweighted θ {cov_unw['lo']:.0f}–{cov_unw['hi']:.0f}° "
          f"(Δθ={cov_unw['spread']:.0f}°), weighted θ {cov_w['lo']:.0f}–{cov_w['hi']:.0f}° "
          f"(Δθ={cov_w['spread']:.0f}°)")
    for tag, c in (('unweighted', cov_unw), ('weighted', cov_w)):
        ends = [n for n, k in (('β∥ (θ=0)', 'par_extrap'), ('β⊥ (θ=90)', 'perp_extrap')) if c[k]]
        if ends:
            print(f"  ** EXTRAPOLATED ENDPOINT ({tag}): {', '.join(ends)} lies outside the "
                  f"observed {c['lo']:.0f}–{c['hi']:.0f}° range — that end of the curve is "
                  f"model, not measurement, and Δβ inherits it. **")
        if c['spread'] < MIN_THETA_SPREAD_DEG:
            print(f"  ** ANGULAR SPREAD ({tag}): Δθ={c['spread']:.0f}° is below the "
                  f"{MIN_THETA_SPREAD_DEG:.0f}° spread local fits are gated on. **")

    # Style per level
    is_seg = level == 'segment'
    color, ms, s = ('darkorange', 5, 40) if is_seg else ('steelblue', 3, 20)
    elw, cap = (0.8, 2) if is_seg else (0.5, 1.5)

    # Ruler row under each panel: its own axes keeps the envelope out of the beta scale.
    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[6, 1], hspace=0.22, wspace=0.10,
                          top=0.92, bottom=0.075, left=0.055, right=0.985)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    rulers = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    x_fit = np.linspace(0, 90, 200)

    # Left: unweighted
    ax = axes[0]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='o', alpha=0.5 + 0.1*is_seg,
                    ms=ms, color=color, ecolor='gray', elinewidth=elw, capsize=cap)
    else:
        ax.scatter(theta, beta, alpha=0.5 + 0.1*is_seg, s=s, c=color)
    ax.set_title('Unweighted (original)', fontsize=12)
    if fit_unw:
        _plot_fit_curve(ax, fit_unw, dv_unw, x_fit, theta)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # Right: weighted
    ax = axes[1]
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='none', ecolor='gray',
                    elinewidth=elw, capsize=cap, alpha=0.5)
    sc = ax.scatter(theta, beta, alpha=0.6, s=s, c=weights, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    ax.set_title('Weighted by flow confidence', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Weight (1=agree, 0=disagree)', fontsize=9)
    if fit_w:
        _plot_fit_curve(ax, fit_w, dv_w, x_fit, theta_w)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # One scale across both strips, so the 0.10 floor draws the same length on each.
    panels = [(rulers[0], fit_unw, dv_unw), (rulers[1], fit_w, dv_w)]
    shared = max([h for _, f, d in panels if f for h, _, _ in _ruler_bars(f, d)], default=0.0)
    for rax, f, d in panels:
        if f:
            _draw_ruler(rax, f, d, xmax=shared or None)
        else:
            rax.axis('off')

    for ax in axes:
        ax.set_xlabel('Incidence Angle (°)')
        ax.set_xlim(-2, 92)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r'Power Law Exponent ($\beta$)')

    region_label = _display_name(csv_path)
    # Two short lines. The weighting shift, the scope note and the raw numbers moved to the
    # JSON sidecar; the legend keeps the per-fit numbers.
    if fit_unw and fit_w:
        vt = (_verdict_title(fit_w, dv_w) if dv_unw['verdict'] == dv_w['verdict']
              else f'unweighted {_verdict_title(fit_unw, dv_unw)}, '
                   f'weighted {_verdict_title(fit_w, dv_w)}')
        flag_suptitle(fig, f'{region_label}: region-wide diagnostic, {level} level, '
                           f'n={n_total}\n'
                           f'$\\Delta\\beta$: {vt}', pflag, fontsize=13)

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    basename = os.path.basename(csv_path)
    suffix = '_seg_weighted_anisotropy.png' if is_seg else '_weighted_anisotropy.png'
    out_name = basename.replace(f'_{level}_stats.csv', suffix)
    if out_name == basename:
        out_name = basename.replace('.csv', suffix)
    output_path = os.path.join(OUTPUT_BASE_PATH, out_name)
    _save(fig, output_path)
    if fit_unw and fit_w:
        _write_sidecar(out_name, _sidecar_entry(
            region_label, level, n_total, n_valid, n_eff, pflag,
            fit_unw, dv_unw, fit_w, dv_w, both_panels=True,
            cov_unw=cov_unw, cov_w=cov_w))

    # ONLY weighted plot
    fig = plt.figure(figsize=(8, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[6, 1], hspace=0.22,
                          top=0.93, bottom=0.075, left=0.10, right=0.97)
    ax, rax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    if beta_err is not None and np.any(np.isfinite(beta_err)):
        ax.errorbar(theta, beta, yerr=beta_err, fmt='none', ecolor='gray',
                    elinewidth=elw, capsize=cap, alpha=0.5)
    sc = ax.scatter(theta, beta, alpha=0.6, s=s, c=weights, cmap='viridis',
                    vmin=0, vmax=1, edgecolors='none')
    ax.set_title('Weighted by flow confidence', fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Weight (1=agree, 0=disagree)', fontsize=9)
    if fit_w:
        _plot_fit_curve(ax, fit_w, dv_w, x_fit, theta_w)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        _draw_ruler(rax, fit_w, dv_w)
    else:
        rax.axis('off')

    ax.set_xlabel('Incidence Angle (°)')
    ax.set_xlim(-2, 92)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel(r'Power Law Exponent ($\beta$)')

    if fit_w:
        flag_suptitle(fig, f'{region_label}: region-wide diagnostic, {level} level, '
                           f'n={n_total}\n'
                           f'$\\Delta\\beta$: {_verdict_title(fit_w, dv_w)}',
                      pflag, fontsize=13)

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    basename = os.path.basename(csv_path)
    suffix = '_seg_ONLY_weighted_anisotropy.png' if is_seg else '_ONLY_weighted_anisotropy.png'
    out_name = basename.replace(f'_{level}_stats.csv', suffix)
    if out_name == basename:
        out_name = basename.replace('.csv', suffix)
    output_path = os.path.join(OUTPUT_BASE_PATH, out_name)
    _save(fig, output_path)
    if fit_w:
        sidecar = _write_sidecar(out_name, _sidecar_entry(
            region_label, level, n_total, n_valid, n_eff, pflag,
            fit_unw, dv_unw, fit_w, dv_w, both_panels=False,
            cov_unw=cov_unw, cov_w=cov_w))
        print(f"  Figure metadata: {sidecar}")

    _print_comparison(fit_unw, fit_w)
    print(f"\nSaved to {output_path}")
    return {'unweighted': fit_unw, 'weighted': fit_w, 'n': n_total,
            'n_valid': int(n_valid), 'n_eff': float(n_eff)}


def _cross_scale_comparison(win_fits, seg_fits, n_win=0, n_seg=0,
                            n_win_eff=0.0, n_seg_eff=0.0, min_n=20):
    """Compare Δβ between window and segment scales via z-score."""
    print(f"\n{'='*55}")
    print("CROSS-SCALE COMPARISON  (window vs segment Δβ)")
    print(f"{'-'*55}")
    low_n = []
    if n_win < min_n:
        low_n.append(f"windows (n={n_win})")
    if n_seg < min_n:
        low_n.append(f"segments (n={n_seg})")
    if low_n:
        print(f"  ** LOW SAMPLE SIZE: {', '.join(low_n)} < {min_n} — "
              f"bootstrap SEs unreliable, interpret with caution **")
    low_n_eff = []
    if n_win_eff < min_n:
        low_n_eff.append(f"windows (n_eff={n_win_eff:.1f})")
    if n_seg_eff < min_n:
        low_n_eff.append(f"segments (n_eff={n_seg_eff:.1f})")
    for label, key in [('Unweighted', 'unweighted'), ('Weighted', 'weighted')]:
        fw, fs = win_fits.get(key), seg_fits.get(key)
        if fw is None or fs is None:
            print(f"  {label}: fit missing — skipped")
            continue
        if key == 'weighted' and low_n_eff:
            print(f"  ** LOW EFFECTIVE SAMPLE SIZE: {', '.join(low_n_eff)} < {min_n} — "
                  f"weighted bootstrap SEs unreliable, interpret with caution **")
        diff = fw['delta'] - fs['delta']
        se = np.sqrt(fw['delta_se']**2 + fs['delta_se']**2)
        z = diff / se if se > 0 else np.inf
        verdict = 'CONSISTENT' if abs(z) < 2 else 'INCONSISTENT'
        print(f"  {label}:")
        print(f"    Window  Δβ = {fw['delta']:+.3f} ± {fw['delta_se']:.3f}")
        print(f"    Segment Δβ = {fs['delta']:+.3f} ± {fs['delta_se']:.3f}")
        print(f"    Difference  = {diff:+.3f},  z = {abs(z):.2f}  →  {verdict} (|z|<2)")
    print(f"{'='*55}")


def walk_tree(root):
    """Region folders one level down (individual_region_TEST/RSL/window_csvs/...), the layout
    landscape_vector already walks. Returns {region_folder: {region: {level: csv}}}."""
    trees = {}
    for d in sorted(glob.glob(os.path.join(root, '*/'))):
        found = discover_regions(d)
        if found:
            trees[os.path.normpath(d)] = found
    return trees


def _region_dir_of(csv_path):
    """The region folder a stats CSV belongs to, so its outputs land beside it."""
    d = os.path.dirname(os.path.abspath(csv_path))
    return os.path.dirname(d) if os.path.basename(d) in ('window_csvs', 'segment_csvs') else d


def _rebind_output(region_dir):
    """Point the module's output dir and log at one region folder, so a walked region gets
    exactly what a single-region run gives it. Restores the real stdout first: a Tee built on
    top of a Tee would keep writing into the previous region's log."""
    global OUTPUT_BASE_PATH
    OUTPUT_BASE_PATH = os.path.join(region_dir, 'anisotropy/')
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    if isinstance(sys.stdout, Tee):
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
    sys.stdout = Tee(os.path.join(OUTPUT_BASE_PATH, 'weighted_anisotropy_log.txt'))


def process_region(region_name, files):
    print(f"\n{'='*60}\nProcessing: {region_name}\n{'='*60}")
    fits = {}
    for level in ['window', 'segment']:
        if level in files:
            fits[level] = plot_anisotropy(files[level], level=level)
        else:
            print(f"  No {level} stats file for {region_name}")
    if 'window' in files:
        local_anisotropy(files['window'])
    if 'window' in fits and 'segment' in fits and fits['window'] and fits['segment']:
        _cross_scale_comparison(fits['window'], fits['segment'],
                                n_win=fits['window']['n'],
                                n_seg=fits['segment']['n'],
                                n_win_eff=fits['window']['n_eff'],
                                n_seg_eff=fits['segment']['n_eff'])


if __name__ == "__main__":
    regions = discover_regions(_REGION_BASE)
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # A direct CSV writes beside its own region folder, not into the base.
    if arg and arg.endswith('.csv'):
        _rebind_output(_region_dir_of(arg))
        plot_anisotropy(arg, level='segment' if 'segment' in arg else 'window')
        sys.exit(0)

    # An explicit directory, or a base holding region folders instead of CSVs: walk it and
    # give each region folder its own anisotropy/ output and log.
    root = arg if (arg and os.path.isdir(arg)) else (None if regions else _REGION_BASE)
    if root:
        trees = walk_tree(root)
        if arg and not os.path.isdir(arg):  # bare name: filter by folder or region name
            trees = {d: {r: f for r, f in found.items()
                         if arg.lower() in r.lower() or arg.lower() in os.path.basename(d).lower()}
                     for d, found in trees.items()}
            trees = {d: f for d, f in trees.items() if f}
        if not trees:
            print(f"No region datasets found under {root}"
                  f"{f' matching {arg!r}' if arg else ''}")
            sys.exit(0)
        print(f"Walking {root}: {len(trees)} region folder(s)")
        for d, found in trees.items():
            _rebind_output(d)
            for r in sorted(found):
                process_region(r, found[r])
        sys.exit(0)

    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    log_path = os.path.join(OUTPUT_BASE_PATH, 'weighted_anisotropy_log.txt')
    sys.stdout = Tee(log_path)

    if arg:
        if arg in regions:
            process_region(arg, regions[arg])
        else:
            matches = [r for r in regions if arg.lower() in r.lower()]
            if len(matches) == 1:
                process_region(matches[0], regions[matches[0]])
            elif matches:
                print(f"Multiple matches for '{arg}':"); [print(f"  - {m}") for m in matches]
            else:
                print(f"Region '{arg}' not found. Available:"); [print(f"  - {r}") for r in sorted(regions)]
    else:
        selection = select_region(regions)
        if selection == 'ALL':
            for r in sorted(regions):
                process_region(r, regions[r])
        elif selection:
            process_region(selection, regions[selection])
