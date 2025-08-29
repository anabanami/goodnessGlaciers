# Grid Convergence Analysis Report

| Parameter | Value |
| :--- | :--- |
| **Profile** | 165 |
| **Experiment** | S2 |
| **Analysis Date** | 2025-08-25 18:32:12 |
| **Reference Resolution** | `0.75` |

## Key Metrics (from Reference Solution)

| Metric | Value |
| :--- | :--- |
| Max Surface Velocity | 0.013 m/a |
| Max Basal Velocity | 0.000 m/a |

## Convergence Analysis

An absolute metric (RMSE) is used for solutions where max velocity is < 0.1 m/a.

| Resolution | Status | Surface vx Error | Basal vx Error | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `0.875` | ✓ PASSED | 2.08e-05 | 0.00e+00 | Near-zero solution |
| `1.0` | ✓ PASSED | 2.20e-05 | 0.00e+00 | Near-zero solution |
| `1.125` | ✓ PASSED | 3.48e-05 | 0.00e+00 | Near-zero solution |

## Recommendations

**✓ Solution has converged.** The results are consistent and can be considered reliable.
