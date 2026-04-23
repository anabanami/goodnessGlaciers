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
            'unweighted': (0.176, 0.080, 0.0274),
            'weighted': {
                45: (0.318, 0.087, 0.0592),
                50: (0.304, 0.084, 0.0537),
                60: (0.281, 0.079, 0.0461),
                70: (0.264, 0.080, 0.0415),
                75: (0.257, 0.079, 0.0401),
            }
        },
        'segment': {
            'unweighted': (0.153, 0.152, 0.0076),
            'weighted': {
                45: (0.212, 0.185, 0.0087),
                50: (0.201, 0.182, 0.0085),
                60: (0.187, 0.175, 0.0076),
                70: (0.175, 0.170, 0.0070),
                75: (0.171, 0.165, 0.0071),
            }
        }
    },
    'MS (Moller Stream)': {
        'short': 'MS',
        'window': {
            'unweighted': (-0.002, 0.031, 0.0000),
            'weighted': {
                45: (0.047, 0.034, 0.0022),
                50: (0.043, 0.033, 0.0008),
                60: (0.035, 0.034, -0.0004),
                70: (0.028, 0.031, -0.0006),
                75: (0.026, 0.032, -0.0006),
            }
        },
        'segment': {
            'unweighted': (-0.029, 0.094, 0.0012),
            'weighted': {
                45: (0.157, 0.082, 0.0116),
                50: (0.140, 0.081, 0.0057),
                60: (0.113, 0.081, -0.0003),
                70: (0.091, 0.082, -0.0019),
                75: (0.082, 0.079, -0.0021),
            }
        }
    },
    'PPB (Pensacola/Pole)': {
        'short': 'PPB',
        'window': {
            'unweighted': (-0.043, 0.082, 0.0023),
            'weighted': {
                45: (-0.030, 0.099, -0.0005),
                50: (-0.029, 0.101, 0.0001),
                60: (-0.029, 0.097, 0.0006),
                70: (-0.029, 0.093, 0.0009),
                75: (-0.030, 0.095, 0.0011),
            }
        },
        'segment': {
            'unweighted': (0.060, 0.203, 0.0028),
            'weighted': {
                45: (-0.066, 0.222, -0.0015),
                50: (-0.050, 0.219, -0.0018),
                60: (-0.025, 0.210, -0.0014),
                70: (-0.008, 0.204, -0.0005),
                75: (-0.001, 0.207, -0.0002),
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
