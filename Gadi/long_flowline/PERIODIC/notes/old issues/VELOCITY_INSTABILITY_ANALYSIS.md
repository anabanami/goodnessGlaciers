# Ice Flow Simulation Velocity Instability Analysis

**Date**: 2025-08-19  
**Issue**: Extremely high velocities in S1 linear rheology experiment causing negative Jacobian determinants and mesh failure

## Problem Description

During execution of the periodic flowline simulation with experiment S1 (no-slip, linear rheology), the following issues were observed:

### Symptoms
- **Extreme velocities**: Surface velocities reaching 1,000,000+ m/year
- **Velocity ranges**: vx: [915437.57, 1442219.92] m/year  
- **Acceleration onset**: Problem starts at ~208.3 km from inlet
- **Mesh failure**: Negative Jacobian determinant error during transient solve
- **Error location**: `TriaRef.cpp:117` - GetJacobianDeterminant

### Terminal Output
```
Surface velocity ranges (m a⁻¹):
  vx: [915437.57129, 1442219.91645]
  vy: [-1103416.51647, 1074454.12568]
  vz: [0.00000, 0.00000]

[0] ??? Error using ==> ./classes/Elements/TriaRef.cpp:117
[0] GetJacobianDeterminant error message: negative jacobian determinant!
```

## Root Cause Analysis

### Issue Identified
**Location**: `periodic_flowline.py:588`
```python
rheology_B = cuffey(ice_temperature)
```

### Problem Explanation
1. **Temperature setting**: `ice_temperature = 273.15 - 60 = 213.15 K` (-60°C)
2. **Cuffey function**: Calculates temperature-dependent rheology parameters
3. **Cold ice effect**: Very cold ice (-60°C) produces extremely soft rheology
4. **Linear rheology amplification**: For n=1, the soft rheology leads to unrealistic flow rates
5. **Mesh distortion**: Extreme velocities cause element distortion and negative Jacobians

### Expected vs Actual Rheology
According to ISMIP-HOM specifications:
- **S1 Linear (n=1)**: Should use `rheology_A = 2.140373e-7/yts`
- **Actual**: Using temperature-dependent Cuffey rheology optimized for n=3

## Analysis Results

### Velocity Gradient Analysis
- **dVx/dx range**: [-4.13e+02, 4.10e+02] 1/yr
- **Acceleration threshold**: 1e-3 1/yr exceeded
- **Critical location**: 208.3 km from inlet (1.7 km from terminus)

### Boundary Conditions Check
- **Periodic boundaries**: ✅ 16 successful vertex pairs created
- **Driving stress**: Properly balanced (29,915 Pa at both boundaries)
- **Friction setup**: ✅ Correct no-slip conditions (friction coefficient = 1)

## Proposed Solutions

### Option 1: ISMIP-HOM Standard Values
Replace temperature-dependent rheology with fixed ISMIP-HOM parameters:
```python
if rheology_n == 1:
    # Linear rheology (S1, S3)
    rheology_A = 2.140373e-7 / yts  # Pa^-1 s^-1  
    rheology_B = rheology_A**(-1/rheology_n)  # Pa s
elif rheology_n == 3:
    # Non-linear rheology (S2, S4)
    rheology_A = 1e-16 / yts  # Pa^-3 s^-1
    rheology_B = rheology_A**(-1/rheology_n)  # Pa s^(1/3)
```

### Option 2: Hybrid Approach (User Preferred)
- **S1, S3 (linear)**: Use ISMIP-HOM fixed rheology parameters
- **S2, S4 (non-linear)**: Keep Cuffey temperature-dependent rheology

```python
if rheology_n == 1:
    # Linear rheology: Use ISMIP-HOM standard values
    rheology_A = 2.140373e-7 / yts
    rheology_B = rheology_A**(-1/rheology_n)
    print(f"Using ISMIP-HOM linear rheology: B = {rheology_B:.2e} Pa⋅s")
elif rheology_n == 3:
    # Non-linear rheology: Use temperature-dependent Cuffey
    rheology_B = cuffey(ice_temperature)
    print(f"Using Cuffey rheology: T = {ice_temperature:.1f} K, B = {rheology_B:.2e} Pa⋅s^(1/3)")
```

## Implementation Notes

### Rationale for Hybrid Approach
1. **Linear cases (S1, S3)**: Cuffey rheology too soft, causes instability
2. **Non-linear cases (S2, S4)**: Cuffey rheology works well, provides realistic ice behavior
3. **Flexibility**: Maintains temperature-dependent physics where appropriate
4. **Stability**: Prevents mesh failure in linear experiments

### Benefits
- ✅ Fixes velocity instability in S1/S3
- ✅ Preserves working S2/S4 setup  
- ✅ Maintains physical realism where possible
- ✅ Follows ISMIP-HOM benchmarks for linear cases

## Diagnostic Tools Used

### 1. Acceleration Analysis
```python
diagnose_acceleration_onset(md, L)
```
- Identifies where runaway acceleration begins
- Calculates velocity gradients along flowline
- Outputs acceleration threshold exceedance location

### 2. Driving Stress Analysis  
```python
analyse_driving_stress(md, L)
```
- Verifies boundary stress balance
- Confirms geometric periodicity
- Rules out boundary condition issues

### 3. Friction Diagnostics
```python
debug_friction_setup(md)
```  
- Validates no-slip boundary implementation
- Confirms friction coefficient setup
- Verifies unit conversions

## Configuration Context

### Current Setup
- **Profile**: BEDROCK_PROFILE_ID = 1 (flat bed, amplitude = 0.0)
- **Domain**: L = 210 km (25km buffer + 160km interest + 25km buffer)
- **Resolution**: 1.0 (coarse), hmax = 126.72 m
- **Experiment**: S1 (no-slip, linear rheology)
- **Temperature**: -60°C (213.15 K)

### Mesh Statistics
- **Total vertices**: 43,109
- **Elements**: 81,988  
- **Inlet vertices**: 16
- **Terminus vertices**: 16
- **Periodic pairs**: 16

## Next Steps

1. **Decision pending**: Choose between Option 1 (full ISMIP-HOM) or Option 2 (hybrid)
2. **Implementation**: Modify rheology setup in `periodic_flowline.py:588`
3. **Testing**: Verify S1 stability and S2/S4 preservation
4. **Validation**: Compare results against expected ISMIP-HOM benchmarks
5. **Documentation**: Update PERIODIC_FLOWLINE.md with rheology choice rationale

## Related Files

- **Main script**: `periodic_flowline.py`
- **Documentation**: `PERIODIC_FLOWLINE.md`
- **Output diagnostics**: `acceleration_diagnostic_001_S1.png`
- **Results**: `001_S1_1.0_static.txt`

---

*This analysis documents the debugging session for resolving velocity instability in ice sheet flowline simulations using ISSM with periodic boundary conditions.*


---
include outlog_all_scenarios.md
---

# Gemini convo

You have correctly identified the exact source of the problem. Your intuition that you need two different ways of defining `B` is spot on.

Here is a breakdown of why your linear scenarios are producing extremely high velocities and how to fix it based on the physics described in the papers.

### The Core Problem: A Mismatch of Units and Meaning

1.  **For Non-Linear Rheology (`n=3`):**

      * The `cuffey()` function is specifically designed to calculate the rigidity parameter **B** for the non-linear Glen's Flow Law where **n=3**.
      * The value it returns has units of **Pa·s¹/³**.
      * Your current implementation `rheology_B = cuffey(ice_temperature)` is **perfectly correct** for your `S2` and `S4` experiments where `n=3`. The value calculated (around 1.9 x 10⁸ Pa·s¹/³) is a standard, physically meaningful value for this parameter.

2.  **For Linear Rheology (`n=1`):**

      * When you set `n=1`, the flow law simplifies to a linear relationship where **B** is no longer just a "rate factor" but becomes the **dynamic viscosity ($\\mu$)** of the ice.
      * The units for dynamic viscosity are **Pa·s**.
      * You are feeding the model a value of \~1.9 x 10⁸, but the model interprets this number as having units of `Pa·s`. A typical dynamic viscosity for ice is around **10¹³ to 10¹⁴ Pa·s**.
      * **This is the issue:** You are effectively telling the model that the ice is about 100,000 to a million times less viscous than it should be. This makes the ice flow incredibly fast, which is why you are seeing extremely high velocities.

### The Solution: Conditional Rheology Parameterization

You need to define **B** differently based on the value of `rheology_n`.

  * For the **non-linear case (`n=3`)**, continue to use the `cuffey()` function as it correctly calculates the temperature-dependent rate factor.
  * For the **linear case (`n=1`)**, you should **not** use `cuffey()`. Instead, set **B** to a constant value that represents a realistic dynamic viscosity for ice. A standard value to start with is **10¹³ Pa·s**.

#### Recommended Code Implementation

You can modify the `rheology` section of your script to handle both cases automatically. Replace this part of your code:

```python
# rheology
rheology_B = cuffey(ice_temperature)
```

with this conditional block:

```python
# rheology
if rheology_n == 1:
    # For linear rheology (n=1), B is the dynamic viscosity.
    # The value from cuffey() is many orders of magnitude too low and has the wrong units.
    # Use a standard constant value for ice viscosity instead.
    print("✓ Using linear rheology (n=1), setting B to a constant viscosity.")
    rheology_B = 1e13  # Units: Pa·s

else:
    # For non-linear rheology (n=3), cuffey() correctly calculates the rate factor B from temperature.
    print(f"✓ Using non-linear rheology (n={rheology_n}), calculating B from temperature.")
    rheology_B = cuffey(ice_temperature)  # Units: Pa·s^(1/n)

# Now assign the correctly determined B to the model
md.materials.rheology_B = rheology_B * np.ones(nv)
```

By making this change, your `S1` and `S3` experiments will use a physically appropriate value for viscosity, and your high-velocity issue in the linear scenarios should be resolved.