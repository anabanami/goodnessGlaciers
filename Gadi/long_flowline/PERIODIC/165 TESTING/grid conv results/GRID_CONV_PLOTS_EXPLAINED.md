# Grid Convergence Analysis Plots - Explanation

## Overview

The `analyse_grid_convergence.py` script generates a comprehensive 2×2 plot layout to visualize grid convergence behavior in ice flow simulations. Each subplot provides critical insights into numerical accuracy, computational scaling, and mesh resolution effects.

---

## Plot 1: Surface Velocity Comparison (Top Left)

**Purpose:** Visualizes how surface ice velocity profiles change with mesh resolution

**What it shows:**
- **X-axis:** Normalized distance (x/L) from 0 to 1 across the 160km domain
- **Y-axis:** Surface velocity in m/a (meters per year)
- **Multiple lines:** Each resolution factor (0.5, 0.75, 1.0, 1.25) shown in different colors

**Key insights:**
- **Convergence behavior:** Lines should overlap closely for converged solutions
- **Resolution sensitivity:** Large differences between lines indicate mesh-dependent results
- **Physical validity:** Typical surface velocities range 1-1500 m/a for ice sheets
- **Spatial patterns:** Shows how bedrock topography influences surface flow

**Interpretation:**
- ✅ **Good convergence:** All resolution lines nearly identical
- ⚠️ **Poor convergence:** Visible separation between different resolutions
- 🔍 **Critical regions:** Areas where lines diverge most indicate resolution-sensitive zones

---

## Plot 2: Basal Velocity Comparison (Top Right)

**Purpose:** Visualizes how basal (sliding) ice velocity profiles change with mesh resolution

**What it shows:**
- **X-axis:** Normalized distance (x/L) from 0 to 1 across the domain
- **Y-axis:** Basal velocity in m/a (meters per year)
- **Multiple lines:** Each resolution factor shown in different colors

**Key insights:**
- **Sliding behavior:** Shows ice-bed interface velocity (sliding component)
- **Mesh sensitivity:** Basal velocities often more sensitive to resolution than surface
- **Physical validity:** Basal velocities typically 100-500 m/a for sliding ice sheets
- **Boundary effects:** May show artifacts near domain boundaries

**Interpretation:**
- ✅ **Converged:** Minimal variation between resolution lines
- ❌ **Not converged:** Significant differences between resolutions
- 🎯 **Critical for S3/S4 experiments:** Sliding boundary conditions require adequate resolution

---

## Plot 3: Convergence Metrics (Bottom Left)

**Purpose:** Quantifies convergence quality using L2 relative error metrics

**What it shows:**
- **X-axis:** Resolution factors (0.75, 1.0, 1.25) - excludes 0.5 (reference)
- **Y-axis:** L2 Relative Error (%) on logarithmic scale
- **Blue bars:** Surface velocity (vx) errors
- **Orange bars:** Basal velocity (vx) errors
- **Red dashed line:** 1% convergence threshold

**Key insights:**
- **Error quantification:** Numerical measure of how much solutions differ from finest mesh
- **Convergence threshold:** Errors below 1% indicate acceptable convergence
- **Field comparison:** Surface vs basal velocity convergence behavior
- **Resolution ranking:** Lower bars = better convergence

**Interpretation:**
- ✅ **Excellent:** Both bars below 1% threshold (green zone)
- ⚠️ **Marginal:** Some bars slightly above 1% threshold
- ❌ **Inadequate:** Bars significantly above 1% threshold
- 📈 **Trend analysis:** Systematic error increase indicates consistent mesh dependence

---

## Plot 4: Computational Scaling (Bottom Right)

**Purpose:** Shows computational cost vs mesh resolution relationship using actual mesh complexity

**What it shows:**
- **X-axis:** Resolution factors (0.5, 0.75, 1.0, 1.25)
- **Y-axis:** Mesh vertices (actual simulation mesh nodes)
- **Blue line with circles:** Actual mesh vertex count for each resolution
- **Data source:** Extracted from simulation logs, not output file lengths

**Key insights:**
- **True computational cost:** Shows actual mesh complexity used in simulations
- **Proper scaling relationship:** Demonstrates how resolution factor affects mesh density
- **Resource planning:** Accurate prediction of computational requirements
- **Cost-benefit analysis:** Real computational cost to compare with convergence metrics

**Expected scaling behavior:**
- **Resolution 0.5:** ~118,000 vertices (finest, highest cost)
- **Resolution 0.75:** ~50,000 vertices (moderate cost)
- **Resolution 1.0:** ~43,000 vertices (coarser, lower cost)
- **Resolution 1.25:** ~42,500 vertices (coarsest, lowest cost)

**Interpretation:**
- 📊 **Proper scaling:** Smaller resolution factors = exponentially more vertices
- 💰 **True cost scaling:** Reflects actual computational burden (not surface sampling)
- ⚖️ **Optimization target:** Balance between accuracy (Plot 3) and real cost (Plot 4)
- 🎯 **Production planning:** Accurate estimate of computational resources needed
- 🔧 **Fixed issue:** Previous version incorrectly plotted surface output points (~2100) instead of mesh vertices

**Important note:** This plot now correctly shows mesh vertices from simulation logs rather than the constant surface sampling points from output files.

---

## Integrated Analysis Strategy

### How to Use All Four Plots Together:

1. **Start with Plots 1 & 2:** Visual assessment of convergence quality
   - Look for overlapping lines across resolutions
   - Identify regions of high sensitivity

2. **Quantify with Plot 3:** Numerical convergence metrics
   - Check if errors fall below 1% threshold
   - Compare surface vs basal convergence behavior

3. **Consider Plot 4:** Computational cost implications
   - Balance accuracy requirements with available resources
   - Plan computational budgets for production runs

4. **Make resolution decision:** 
   - ✅ Choose finest resolution that passes convergence criteria
   - ⚠️ Consider stability implications (transient analysis needed)
   - 💰 Balance accuracy requirements with computational constraints

### Critical Success Criteria:

- **Visual convergence:** Plots 1 & 2 show overlapping lines
- **Quantitative convergence:** Plot 3 shows errors < 1%
- **Computational feasibility:** Plot 4 shows acceptable node counts
- **Stability validation:** Must confirm with transient analysis

---

## Example Interpretation

**Profile 001 S4 Results (after fix):**
- **Plot 1:** Surface velocities converge well at resolution 0.75
- **Plot 2:** Basal velocities show good agreement across resolutions
- **Plot 3:** Resolution 0.75 passes 1% threshold for both fields
- **Plot 4:** Resolution 0.75 uses ~50,200 mesh vertices (moderate computational cost, 2.4x more than resolution 1.0)

**Conclusion:** Resolution 0.75 optimal for Profile 001 based on diagnostic analysis with proper computational scaling assessment

**⚠️ Important Note:** These diagnostic convergence results must be validated with transient stability testing, as the grid convergence study revealed that diagnostic convergence does not guarantee transient stability.