# Grid Convergence Analysis Report

| Parameter | Value |
| :--- | :--- |
| **Profile** | 165 |
| **Experiment** | S3 |
| **Analysis Date** | 2025-08-25 21:42:00 |
| **Reference Resolution** | `0.75` |

## Key Metrics (from Reference Solution)

| Metric | Value |
| :--- | :--- |
| Max Surface Velocity | 121.039 m/a |
| Max Basal Velocity | 134.744 m/a |

## Convergence Analysis

An absolute metric (RMSE) is used for solutions where max velocity is < 0.1 m/a.

| Resolution | Status | Surface vx Error | Basal vx Error | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `0.875` | ✓ PASSED | 0.12% | 0.19% |  |
| `1.0` | ✓ PASSED | 0.11% | 0.22% |  |
| `1.125` | ✓ PASSED | 0.12% | 0.29% |  |

## Recommendations

**✓ Solution has converged.** The results are consistent and can be considered reliable.
