# ISSM solve() Function Analysis 

### Purpose
The `solve()` function is the main entry point for running ice sheet simulations in ISSM (Ice Sheet System Model). It orchestrates the entire workflow from model preparation through cluster execution to result retrieval.

### Function Signature
```python
md = solve(md, solutionstring, *args)
```

### Supported Solution Types

| Short Form | Full Form | Description |
|------------|-----------|-------------|
| 'sb' | 'Stressbalance' | Ice flow equations |
| 'mt' | 'Masstransport' | Ice mass transport |
| 'oceant' | 'Oceantransport' | Ocean transport |
| 'th' | 'Thermal' | Temperature evolution |
| 'ss' | 'Steadystate' | Equilibrium solutions |
| 'tr' | 'Transient' | Time-dependent simulations |
| 'mc' | 'Balancethickness' | Mass balance thickness |
| 'bv' | 'Balancevelocity' | Balance velocity |
| 'bsl' | 'BedSlope' | Bed slope |
| 'ssl' | 'SurfaceSlope' | Surface slope |
| 'hy' | 'Hydrology' | Hydrology |
| 'da' | 'DamageEvolution' | Damage evolution |
| 'gia' | 'Gia' | Glacial isostatic adjustment |
| 'lv' | 'Love' | Love numbers |
| 'esa' | 'Esa' | Earth system approach |
| 'smp' | 'Sampling' | Sampling |

### Main Workflow

1. **Solution Type Mapping**: Maps short forms (e.g., 'sb') to full solution names (e.g., 'StressbalanceSolution')
2. **Model Consistency Check**: Validates model configuration via `ismodelselfconsistent()`
3. **Runtime Setup**: Creates unique runtime names with timestamps for job identification
4. **File Preparation**:
   - Marshalls model data to binary files (.bin)
   - Creates toolkit configuration files (.toolkits)  
   - Builds queue/batch scripts for cluster execution (.queue/.bat)
5. **Job Submission**: Uploads files and launches computation on cluster
6. **Result Handling**: Waits for completion and loads results back

### Key Options

- `loadonly`: Skip computation, just load existing results
- `batch`: Submit job without waiting for completion
- `restart`: Resume from previous computation  
- `checkconsistency`: Validate model before solving (default: 'yes')
- `runtimename`: Generate unique runtime names (default: true)

## Steadystate vs Transient Solutions

### Steadystate ('ss') Solution

**Physics Included**: Exactly **two coupled physics**:
- **Stressbalance**: Ice flow/velocity field
- **Thermal**: Temperature field

**Algorithm** (from `steadystate_core.cpp:48-77`):
```cpp
for(;;){
    // Step 1: Compute velocity
    stressbalance_core(femmodel);
    GetSolutionFromInputsx(&ug,femmodel);
    
    // Step 2: Compute temperature  
    thermal_core(femmodel);
    GetSolutionFromInputsx(&tg,femmodel);
    
    // Check convergence of both fields
    if(steadystateconvergence(tg,tg_old,ug,ug_old,reltol)) break;
    
    // Update and continue
    step++;
}
```

**Key Features**:
- Iterative coupling between velocity and temperature until convergence
- Accounts for thermomechanical coupling (velocity affects heat generation, temperature affects viscosity)
- Fixed point iteration with convergence tolerance
- Maximum iteration limit (default: 100)

**Default Outputs**: Combines stressbalance + thermal outputs:
- **Velocity**: Vx, Vy, (Vz), Vel, Pressure  
- **Thermal**: Temperature, BasalforcingsGroundediceMeltingRate (+ Enthalpy fields if enabled)

### Transient ('tr') Solution

**Physics Included**: **Multiple physics simultaneously**  evaluated over a specified time period:
- Stressbalance, Masstransport, Thermal, Hydrology, Damage, etc.
- Any combination of time-dependent physics

**Algorithm** (from `transient_core.cpp:55-91`):
```cpp
while(time < finaltime){
    // Adaptive or fixed time stepping
    switch(timestepping){
        case AdaptiveTimesteppingEnum: femmodel->TimeAdaptx(&dt); break;
        case FixedTimesteppingEnum: /* use fixed dt */ break;
    }
    
    // Run transient step for all active physics
    transient_step(femmodel);
    
    // Save results at specified frequency
    if(save_results) OutputResultsx(femmodel);
    
    time += dt;
}
```

**Key Features**:
- Time-stepping with adaptive/fixed time steps
- Handles complex multi-physics evolution
- Checkpointing and restart capabilities
- Adaptive mesh refinement support
- Output frequency control

### Key Differences

| Aspect | Steadystate | Transient |
|--------|-------------|-----------|
| **Physics** | Exactly 2: stressbalance + thermal | Any combination of time-dependent physics |
| **Time** | Finds equilibrium state | Evolves through time |
| **Coupling** | Thermomechanical only | Full multi-physics |
| **Algorithm** | Fixed-point iteration | Time-stepping |
| **Use Case** | Equilibrium analysis | Evolution studies |

This showsd that steadystate is **not** like transient in multi-physics capability. Steadystate is a **specialized coupling** for thermomechanical equilibrium, while transient is the **general framework** for time-evolution of any ice sheet processes.
