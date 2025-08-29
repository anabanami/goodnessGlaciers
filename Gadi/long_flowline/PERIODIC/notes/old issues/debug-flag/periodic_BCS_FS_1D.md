# ISSM solve() Debug Assertion Issue

======================================================================================
## TLDR

The specific issue with Full Stokes and periodic boundary conditions stems from a debug assertion that may be overly restrictive (currently investigating this). Local simulations succeed because debug assertions are disabled, while Gadi cluster ISSM crashes due to active assertion checking. I mainly want to know if the mathematical simulation is valid and whether the restriction on FS 1D simulations is necessary.

It is possible to choose from the following finite element types for FS equations by setting `md.flowequation.fe_FS` as:

  1. 'P1P1' - P1 velocity/P1 pressure (debugging only - not stable)
  2. 'P1P1GLS' - P1 velocity/P1 pressure with GLS stabilization
  3. 'MINIcondensed' - MINI elements with condensed formulation (**DEFAULT**)
  4. 'MINI' - MINI elements (P1 + bubble for velocity, P1 for pressure)
  5. 'TaylorHood' - Taylor-Hood elements (P2 velocity, P1 pressure)
  6. 'LATaylorHood' - Locally Augmented Taylor-Hood
  7. 'XTaylorHood' - Extended Taylor-Hood with θ-method
  8. 'OneLayerP4z' - Special vertical discretization
  9. 'CrouzeixRaviart' - Crouzeix-Raviart elements
  10. 'LACrouzeixRaviart' - Locally Augmented Crouzeix-Raviart

I have tested `md.flowequation.fe_FS = 'TaylorHood'` and the default ('MINIcondensed'). 

After setting Full Stokes (`setflowequation('FS','all')`) with periodic boundary conditions and either `'TaylorHood'` or `'MINIcondensed'`. I see the following behaviour: 

**Local Behavior**: Sim runs successfully and produces reasonable (?) results.

**Cluster Behavior**: The exact same simulation crashes with error:

```
CreateNodes error message: Assertion "numvertex_pairing==0 || finite_element==P1Enum" failed
```

======================================================================================


This is an nvestigation into ISSM's `solve()` function behavior and a specific issue with Flowband Full Stokes simulations using periodic boundary conditions that work locally but crash on HPC clusters. For more context, see original forum post on this issue: https://issm.ess.uci.edu/forum/d/587-error-1d-freesurfacebaseanalysis/8

## Debug Assertion Issue: FS + Periodic BCs

### Problem Description

**Local Behavior**: 
Full Stokes (`setflowequation('FS','all')`) with periodic boundary conditions runs (for both `solve(md, "sb")` and `solve(md, "tr")`) successfully and produces *reasonable (?) results*.

**Cluster Behavior**: Same simulation (for both `solve(md, "sb")` and `solve(md, "tr")`) crashes with error:
```
CreateNodes error message: Assertion "numvertex_pairing==0 || finite_element==P1Enum" failed
```

### Root Cause Analysis

#### The Assertion
Located in `/home/ana/ISSM/src/c/modules/ModelProcessorx/CreateNodes.cpp:610`:
```cpp
_assert_(numvertex_pairing==0 || finite_element==P1Enum);
```

**Translation**: "If you have vertex pairing (periodic BCs), you MUST use P1 finite elements."

#### The Conflict
- **Full Stokes default finite element**: `MINIcondensed` (not P1)
- **Periodic BCs**: Implemented via `vertex_pairing` 
- **Assertion Logic**: Only allows P1 elements with periodic BCs

### Debug Assertion Mechanism

From `/home/ana/ISSM/src/c/shared/Exceptions/exceptions.h`:
```cpp
#ifdef _ISSM_DEBUG_ 
#define _assert_(statement)\
  if (!(statement)) _error_("Assertion failed...")
#else
#define _assert_(ignore)\
  ((void) 0)  // Does absolutely nothing!
#endif
```

### Local vs Cluster Difference

| Environment | Compilation | Assertion Behavior | Result |
|-------------|-------------|-------------------|---------|
| **Local** | Without `_ISSM_DEBUG_` | All assertions ignored | Simulation runs |
| **Cluster** | With `_ISSM_DEBUG_` | Assertions active | Crashes on violation |

### What Actually Runs Locally

When assertions are disabled locally:

1. **Full Stokes with MINIcondensed elements**: Default FS finite element
2. **Periodic boundary conditions**: Applied via vertex pairing  
3. **No assertion checking**: Debug assertions completely disabled
4. **Valid simulation (?)**: Mathematical problem is well-posed and solvable
     > I WANT TO VERIFY THAT THIS IS THE CASE. If so then the restriction may be overly conservative.

### Finite Element Options for FS

From `flowequation.py`, supported FS finite elements:
- `'P1P1'` (debugging only)
- `'P1P1GLS'` 
- `'MINIcondensed'` ← **default**
- `'MINI'`
- `'TaylorHood'`
- `'LATaylorHood'`
- `'XTaylorHood'`
- `'OneLayerP4z'`
- `'CrouzeixRaviart'`
- `'LACrouzeixRaviart'`

Evidence of succesful local runs suggests that at least TaylorHood and MINIcondensed element types work fine with periodic BCs despite the assertion in HPC cluster.

MINIcondensed is neither P1 nor P2

TaylorHood elements use:
  - Velocity field: P2 (quadratic polynomials)
  - Pressure field: P1 (linear polynomials)

  ------------------------------------------------------------------------------------
  | Element Type       | Velocity     | Pressure | Description                        |
  |--------------------|--------------|----------|-----------------------------------|
  | MINI/MINIcondensed | P1 + bubbles | P1       | Enhanced P1 with bubble functions |
  | TaylorHood         | P2           | P1       | Pure polynomial elements (P2-P1)  |
  | P1P1               | P1           | P1       | Pure polynomial (unstable)        |
  | P2                 | P2           | P2       | Pure polynomial (over-constrained)|
  ------------------------------------------------------------------------------------

P1: Linear velocity, linear pressure (unstable without stabilization)
P2: Quadratic velocity, linear pressure
MINI: Linear + bubble velocity, linear pressure (stable)
MINIcondensed: Same as MINI but with bubbles* eliminated locally (stable + efficient)

*Bubble functions are polynomials that:
  1. Are non-zero only inside an element
  2. Are exactly zero on all element boundaries (edges/faces)
  3. Have a maximum value at the element center ("bubble up")

--------------------------------------------------------------------------------------

# Possible Solutions

- [x]1. **Test different finite elements with periodic BCs**
  test_cases = [
      {'fe_FS': 'MINIcondensed', 'description': 'Default FS element'}, <- MOST ROBUST
      {'fe_FS': 'TaylorHood', 'description': 'Standard mixed element'},
      {'fe_FS': 'P1P1GLS', 'description': 'Stabilized P1-P1 elements'},
  ]

  Neither 'TaylorHood' nor 'MiniCondensed' allow for the simulation to run on Gadi 
  (For either transient nor stressbalance)
   ```python
   md = setflowequation(md, 'FS', 'all')
   md.flowequation.fe_FS = 'P1P1GLS'
   ```
    This appears to run locally (I did not check the full run) but I submitted to Gadi and the same error persists: 
    ```
    [1] ??? Error using ==> ./modules/ModelProcessorx/CreateNodes.cpp:610
    [1] CreateNodes error message: Assertion "numvertex_pairing==0 || finite_element==P1Enum" failed, please report bug at https://issm.ess.uci.edu/forum/
```
  There appears to be no work around to the error that is this simple.


2. **Alternative flow equations**:
   - **Higher-Order (HO)**: Typically uses P1 elements by default
   ==FOR THIS:==
  I need to reconfigure my sim a little bit ... because now I am getting this error:
  ```
  [0] ??? Error using ==> ./classes/Elements/Tria.cpp:181
  [0] AddBasalInput error message: not implemented yet

  --------------------------------------------------------------------------
  Primary job  terminated normally, but 1 process returned
  a non-zero exit code. Per user-direction, the job has been aborted.
  --------------------------------------------------------------------------
  --------------------------------------------------------------------------
  mpiexec detected that one or more processes exited with non-zero status, thus causing
  the job to be terminated. The first process to do so was:

    Process name: [[28705,1],0]
    Exit code:    1
  ```   

- [x]3. **Alternative boundary conditions**: 
  Replace periodic BCs if mathematically acceptable

  THIS IS UNVIABLE AND SOMEWHAT PARADOXICAL(???) 

  1. Running my simulation with dirichlet inlet and neumann (free flow) terminus yields unphysically high ice velocity magnitudes (~1e8 m/a)
  2. Running my simulation with dirichlet inlet and diriclet (restricted flow / max cap velocity) terminus yields unphysically low ice velocity magnitudes (~1e-7 m/a)
   
       I tested two methods to restrict ice flow velocity:

       1. Imposing a max velocity at the terminus nodes
      ```python
        # simple velocity cap
        if exp in ('S2', 'S4'):  # n=3 experiments only
            max_velocity_ma = 100000  # m/a - physically reasonable upper limit (?)
            max_velocity_ms = max_velocity_ma / md.constants.yts
            md.stressbalance.spcvx[terminus_nodes] = max_velocity_ms
        ```
        
       2. Attempting to estimate appropriate outflow velocities at the terminus and avoid over-constraining the system while maintaining numerical stability

        ```python
        # Estimate surface slope at terminus using surface vertices only

        # Use last N km of the domain for slope estimation
        window_km = 2  # or 5
        window_m = window_km * 1000

        # Get surface vertices only
        surface_idx = np.where(md.mesh.vertexonsurface == 1)[0]
        x_surf = md.mesh.x[surface_idx]
        s_surf = md.geometry.surface[surface_idx] 

        print(f"x_surf shape: {x_surf.shape}, s_surf shape: {s_surf.shape}")

        # pick nodes within [L - window_m, L]
        terminus_mask = (x_surf > L - window_m)
        x_term = x_surf[terminus_mask]
        s_term = s_surf[terminus_mask]
        
        print(f"terminus_mask shape: {terminus_mask.shape}, sum: {terminus_mask.sum()}")
        print(f"x_term length: {len(x_term)}")
        print(f"s_term length: {len(s_term)}")
        x_term_centered = x_term - np.mean(x_term)
        
        # fit for the driving force !
        slope = np.polyfit(x_term_centered, s_term, 1)[0]

        # Driving stress estimate
        rho = md.materials.rho_ice
        g = md.constants.g
        H = md.geometry.thickness[terminus_nodes]
        tau_d = rho * g * H * abs(slope) # force pushing ice downslope

        # β² from current friction field
        beta = md.friction.coefficient[terminus_nodes] # still in Pa·s·m⁻¹
        beta2 = beta**2

        # Stabilizing fallback: prevent div by zero
        beta2 = np.clip(beta2, 1e-4, None)

        # Basal sliding estimate: v ≈ τ_d / β²  (from linear sliding law)
        v_est = tau_d / beta2
        v_est = np.clip(v_est, 0, 50000 / md.constants.yts)  #50,000 m/year (≈137 m/s)

        # Apply soft outflow BC
        md.stressbalance.spcvx[terminus_nodes] = v_est
      ```
## In short

Stressbalance solve with capped velocity:
  - S1/S2: Reasonable velocities (10-50 m/a range)
  - S3/S4: Higher but reasonable velocities (<100 m/a range)

Stressbalance solve without velocity cap:
  - S1: ~1,800 m/a (unreasonably high)
  - S2: ~8+ million m/a (completely unphysical)
  - S3: ~3,200 m/a (unreasonably high)
  - S4: ~9+ million m/a (unphysical)

Transient solve with capped velocities:
  - All experiments: Much lower velocities than expected from ISMIP-HOM (~1e-7 m/a)
  - cap might be ovely restrictive or interacting with the transient evolution in an unexpected way

  Transient solve with uncapped velocities:  
  - crashes

======================================================================================

# Using PERIODIC BCS
Based on the stress balance solve results (recorded in 13-08-25.md) for the velocity magnitude ranges of all experiments (periodic BCS) are

  S1 (Linear, No-slip) - Various Resolutions:
  - Resolution 1.25: vx: [1.84, 23.56] m/a
  - Resolution 1.0: vx: [1.85, 23.54] m/a
  - Resolution 0.75: vx: [1.83, 23.55] m/a
  - Resolution 0.5: vx: [1.86, 23.54] m/a

  S2 (Nonlinear n=3, No-slip) - Various Resolutions:
  - Resolution 1.25: vx: [9.44, 49.15] m/a
  - Resolution 1.0: vx: [9.44, 49.19] m/a
  - Resolution 0.75: vx: [9.42, 49.04] m/a
  - Resolution 0.5: vx: [9.41, 49.00] m/a

  S3 (Linear, Sliding) - Various Resolutions:
  - Resolution 1.25: vx: [23.06, 41.53] m/a
  - Resolution 1.0: vx: [23.10, 41.51] m/a
  - Resolution 0.75: vx: [23.08, 41.51] m/a
  - Resolution 0.5: vx: [23.12, 41.51] m/a

  S4 (Nonlinear n=3, Sliding) - Various Resolutions:
  - Resolution 1.25: vx: [32.14, 63.97] m/a
  - Resolution 1.0: vx: [32.15, 63.78] m/a
  - Resolution 0.75: vx: [32.15, 63.95] m/a
  - Resolution 0.5: vx: [32.13, 64.04] m/a

### Pros of Periodic BCS
1. There is consistency across different mesh resolutions
2. The magnitudes ranges appear similar to those reported in ismip-hom
3. The velocity magnitudes follow: S1 < S3 < S2 < S4 (as expected from physics)
    Nonlinear rheology effect:
    S2 > S1  
    S4 > S3 
  (nonlinear rheology is expected to be faster)

4. Sliding effect: S3 > S1 and S4 > S2, showing higher velocities with basal sliding

### Cons of Periodic BCS
1. Doesn't run on Gadi

THIS APPROACH DOESN'T APPEAR TO WORK UPON INSPECTION! see periodic!/ in RESULTS/
======================================================================================

# Long-term Investigation: Using Periodic BCS with Bamgflowband 
(transient and stressbalance)

1. **Code review**: Examine why this restriction exists in the codebase
2. **Verify assertion necessity**: The restriction may be overly conservative

### Trust in Results

**Are Local results valid?**
- The mathematical problem (2D FS + periodic BCs) is well-posed
- ISSM's vertex pairing implementation appears to handle non-P1 elements correctly
- The HPC assertion appears to be a conservative safeguard rather than a fundamental limitation

## Code Locations

### Key Files
- **solve() implementations**: 
  - `/home/ana/ISSM/src/m/solve/solve.py:12-168`
- **Transient core**: `/home/ana/ISSM/src/c/cores/transient_core.cpp:23-100`
- **Assertion location**: `/home/ana/ISSM/src/c/modules/ModelProcessorx/CreateNodes.cpp:610`
- **Debug macro definition**: `/home/ana/ISSM/src/c/shared/Exceptions/exceptions.h`
- **Flow equation class**: `/home/ana/ISSM/src/m/classes/flowequation.py`

### Key Functions
- **Finite element defaults**: `flowequation.py:setdefaultparameters()`

======================================================================================
