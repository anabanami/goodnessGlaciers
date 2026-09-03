import numpy as np
import matplotlib.pyplot as plt

"""
Plot delta-beta and R² as a function of weighting cutoff angle
for all three regions, at both window and segment level.

Data extracted from the weighted_anisotropy.py results at each cutoff.
"""

cutoffs = [45, 50, 60, 70, 75]

# Unweighted values are the same across all cutoffs (no weighting applied)
# Format: (delta_beta, delta_se, r2)

regions = {
    'ASB (Aurora Subglacial Basin)': {
        'short': 'ASB',
        'window': {
            'unweighted': (0.146, 0.081, 0.0174),
            'weighted': {
                45: (0.529, 0.182, 0.1406),
                50: (0.524, 0.176, 0.1400),
                60: (0.514, 0.166, 0.1392),
                70: (0.507, 0.161, 0.1387),
                75: (0.504, 0.153, 0.1385),
            }
        },
        'segment': {
            'unweighted': (0.028, 0.112, 0.0003),
            'weighted': {
                45: (0.132, 0.291, 0.0025),
                50: (0.130, 0.301, 0.0025),
                60: (0.129, 0.273, 0.0025),
                70: (0.128, 0.259, 0.0024),
                75: (0.128, 0.257, 0.0024),
            }
        }
    },
    'MS (Moller Stream)': {
        'short': 'MS',
        'window': {
            'unweighted': (-0.016, 0.033, 0.0005),
            'weighted': {
                45: (-0.004, 0.039, -0.0000),
                50: (-0.003, 0.041, -0.0000),
                60: (-0.001, 0.039, -0.0000),
                70: (0.000, 0.038, -0.0000),
                75: (0.001, 0.037, -0.0000),
            }
        },
        'segment': {
            'unweighted': (0.003, 0.053, 0.0000),
            'weighted': {
                45: (0.076, 0.063, 0.0024),
                50: (0.074, 0.064, 0.0024),
                60: (0.071, 0.063, 0.0022),
                70: (0.070, 0.062, 0.0022),
                75: (0.069, 0.059, 0.0023),
            }
        }
    },
    'PPB (Pensacola/Pole)': {
        'short': 'PPB',
        'window': {
            'unweighted': (-0.118, 0.094, 0.0162),
            'weighted': {
                45: (-0.118, 0.143, 0.0115),
                50: (-0.118, 0.147, 0.0112),
                60: (-0.119, 0.139, 0.0109),
                70: (-0.119, 0.140, 0.0108),
                75: (-0.119, 0.135, 0.0108),
            }
        },
        'segment': {
            'unweighted': (-0.172, 0.160, 0.0173),
            'weighted': {
                45: (-0.067, 0.222, 0.0020),
                50: (-0.061, 0.212, 0.0015),
                60: (-0.052, 0.222, 0.0009),
                70: (-0.046, 0.219, 0.0006),
                75: (-0.044, 0.218, 0.0005),
            }
        }
    }
}

colours = {
    'ASB (Aurora Subglacial Basin)': '#2166ac',
    'MS (Moller Stream)': '#b2182b',
    'PPB (Pensacola/Pole)': '#4daf4a',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

for col, level in enumerate(['window', 'segment']):
    ax_db = axes[0, col]
    ax_r2 = axes[1, col]

    for region_name, data in regions.items():
        colour = colours[region_name]
        short = data['short']
        level_data = data[level]

        unw_db, unw_se, unw_r2 = level_data['unweighted']

        db_vals = [level_data['weighted'][c][0] for c in cutoffs]
        db_errs = [level_data['weighted'][c][1] for c in cutoffs]
        r2_vals = [level_data['weighted'][c][2] for c in cutoffs]

        # delta-beta
        ax_db.errorbar(cutoffs, db_vals, yerr=db_errs, fmt='o-', color=colour,
                        capsize=4, capthick=1.5, linewidth=1.5, markersize=6,
                        label=short)
        ax_db.axhline(unw_db, color=colour, linestyle='--', alpha=0.5, linewidth=1)
        ax_db.fill_between([cutoffs[0] - 3, cutoffs[-1] + 3],
                           unw_db - unw_se, unw_db + unw_se,
                           color=colour, alpha=0.08)

        # R²
        ax_r2.plot(cutoffs, r2_vals, 'o-', color=colour, linewidth=1.5,
                   markersize=6, label=short)
        ax_r2.axhline(unw_r2, color=colour, linestyle='--', alpha=0.5, linewidth=1)

    ax_db.axhline(0, color='k', linewidth=0.5, zorder=0)
    ax_r2.axhline(0, color='k', linewidth=0.5, zorder=0)

    ax_db.set_title(f'{level.capitalize()}-level', fontsize=13)
    ax_db.set_ylabel(r'Weighted $\Delta\beta$ ($\beta_\parallel - \beta_\perp$)')
    ax_r2.set_ylabel(r'Weighted R²')
    ax_r2.set_xlabel('Weighting cutoff angle (°)')

    ax_db.legend(fontsize=10)
    ax_db.grid(True, alpha=0.3)
    ax_r2.grid(True, alpha=0.3)

    ax_r2.set_xticks(cutoffs)
    ax_r2.set_xlim(42, 78)

fig.suptitle('Sensitivity of weighted anisotropy fit to cutoff angle\n'
             '(dashed lines = unweighted baseline ± 1 SE)',
             fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig('cutoff_sensitivity.png', dpi=300, bbox_inches='tight')
print('Saved to cutoff_sensitivity.png')
plt.close()
