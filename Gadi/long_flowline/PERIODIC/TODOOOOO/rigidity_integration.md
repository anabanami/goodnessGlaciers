Of course. It's an excellent goal to integrate your successful proof-of-concept from `rigidity.py` into your main `periodic_flowline.py` script. The mathematical derivation you've provided is correct, and the implementation in `rigidity.py` successfully demonstrates the two-step process.

The key challenge, as you've identified, is that the linear "equivalent" experiment is not standalone; it fundamentally depends on the results of a prior non-linear run. Therefore, you cannot simply add it as another `elif exp == 'S3_equivalent'` condition in your original script's structure.

Here is a clear, structured way to integrate this workflow into `periodic_flowline.py`. We will create a new experiment "mode" that explicitly runs the two required simulations back-to-back.

### Summary of the Plan

1.  **Create a New Function:** We'll turn the logic for calculating the equivalent linear rheology from `rigidity.py` into a reusable function.
2.  **Restructure the Main Simulation Block:** We'll modify the main part of `periodic_flowline.py` to have a special case. If you choose the "equivalent linear" experiment, it will first run the non-linear ($n=3$) simulation, use its results to calculate the new rheology, and then immediately run the linear ($n=1$) simulation. Other experiments (`S1`, `S2`, etc.) will run as they did before.

-----

### Step 1: Create a Function for the Rheology Calculation

First, let's create a function based on the code from `rigidity.py`. This keeps your main script clean. You can add this function to the top of your `periodic_flowline.py` script, near your other helper functions.

```python
def calculate_equivalent_linear_rheology(md, rheology_B_n3, tuning_factor=1.5):
    """
    Calculates a spatially-varying linear viscosity (rheology_B for n=1) 
    based on the stress state of a non-linear (n=3) simulation.

    Args:
        md (issm.model): The ISSM model object.
        rheology_B_n3 (float or np.array): The rheology_B value(s) from the n=3 run.
        tuning_factor (float): An empirical factor to adjust the final viscosity.

    Returns:
        np.array: The calculated spatially-varying rheology_B field for n=1.
    """
    print("\n--- Calculating equivalent linear rheology (B for n=1) ---")

    # Extract the stress state from the previous n=3 run's results
    stress_xx_n3 = md.results.StressbalanceSolution.StressTensorxx
    stress_yy_n3 = md.results.StressbalanceSolution.StressTensoryy
    stress_xy_n3 = md.results.StressbalanceSolution.StressTensorxy

    # 1. Convert B_n3 to A_n3. For n=3, B = A^(-1/n) => A = B^(-n)
    A_n3 = rheology_B_n3**(-3)

    # 2. Calculate the stress-dependent term from your derivation
    # Formula: A1 = A3 * 0.25 * ((σxx-σyy)² + 4*σxy²)
    stress_factor = 0.25 * ((stress_xx_n3 - stress_yy_n3)**2 + 4 * stress_xy_n3**2)

    # 3. Calculate the spatially-varying A_n1 field
    A_n1_field = A_n3 * stress_factor

    # 4. Convert A_n1 to rheology_B for the linear (n=1) case
    # For n=1, ISSM expects B to be the dynamic viscosity η, where η = (2*A)^-1
    # Use a small floor value to prevent division by zero
    A_n1_field[A_n1_field == 0] = 1e-12 
    rheology_B_n1 = (2 * A_n1_field)**(-1)
    
    # Clip to reasonable viscosity values to ensure stability
    rheology_B_n1 = np.clip(rheology_B_n1, 1e9, 1e15)

    # 5. Apply tuning factor if needed
    if tuning_factor != 1.0:
        print(f"Applying empirical tuning factor of {tuning_factor}")
        rheology_B_n1 *= tuning_factor

    print(f"✓ Calculated spatially-varying rheology_B field.")
    print(f"  New rheology_B range: [{np.min(rheology_B_n1):.2e}, {np.max(rheology_B_n1):.2e}] Pa·s")

    return rheology_B_n1
```

### Step 2: Modify the Main Simulation Logic in `periodic_flowline.py`

Now, go to the section in `periodic_flowline.py` where you choose the experiment (around line 691). We will replace the simple `if/else` for rheology with a more comprehensive structure that handles the two-step process.

Here is the new logic. It introduces a new `exp` choice: `'S4_to_S3_equivalent'`.

```python
########################################################
# Choose experiment

# --- Single experiments ---
# exp = 'S1' # no slip, n=1
# exp = 'S2' # no slip, n=3
# exp = 'S3' # slip, n=1
# exp = 'S4' # slip, n=3

# --- Two-step equivalent rheology experiment ---
exp = 'S4_to_S3_equivalent' # Runs S4, then calculates and runs the equivalent S3

########################################################
# ... (your other setup code remains the same)

# Initialize rheology variables
rheology_n = 1 # Default
rheology_B = None

# ... (your geometry, mesh, and other initial setup remains the same up to the solver calls)

# ==============================================================================
# ~~~~~~~~~~~~~~~ CORE SIMULATION BLOCK ~~~~~~~~~~~~~~~
# ==============================================================================

if exp == 'S4_to_S3_equivalent':
    # This block performs the two-step process from rigidity.py
    
    # === PART 1: Run the non-linear (n=3) reference simulation ===
    print("\n\n===== STARTING: Non-Linear (n=3) Reference Simulation (Part 1/2) =====")
    
    # Configure model for S4 (non-linear, slip)
    rheology_n_n3 = 3
    rheology_B_n3 = cuffey(ice_temperature)
    
    md.materials.rheology_B = rheology_B_n3 * np.ones(nv)
    md.materials.rheology_n = rheology_n_n3 * np.ones(ne)
    md.materials.rheology_law = "BuddJacka"
    md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_S4_reference_{resolution_factor}'

    # Setup friction for a slipping case
    md = setup_friction(md, 'S4')
    
    print("\n===== Solving Diagnostic Stressbalance for n=3 =====")
    md = solve(md,'Stressbalance')

    # Save the n=3 results before they are overwritten
    results_n3 = md.results.StressbalanceSolution
    try:
        save_results(md, L, md.miscellaneous.name)
    except Exception as e:
        print(f"⚠ Error saving n=3 reference output: {e}")

    # === PART 2: Run the equivalent linear (n=1) simulation ===
    print("\n\n===== STARTING: Equivalent Linear (n=1) Simulation (Part 2/2) =====")

    # Use our new function to get the spatially-varying viscosity
    rheology_B_n1_equivalent = calculate_equivalent_linear_rheology(md, rheology_B_n3, tuning_factor=1.5)

    # Re-configure the SAME model for S3 (linear, slip) with the new rheology
    md.materials.rheology_B = rheology_B_n1_equivalent
    md.materials.rheology_n = 1 * np.ones(ne)
    md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_S3_equivalent_{resolution_factor}'

    # Friction is the same for S3 and S4, so no need to call setup_friction() again.

    print("\n===== Solving Diagnostic Stressbalance for equivalent n=1 =====")
    md = solve(md,'Stressbalance')
    
    # Now you can proceed with the transient solve using this final "equivalent linear" state
    # The model 'md' is now perfectly configured for the transient run.
    
else:
    # This block handles all standard, single-run experiments (S1, S2, S3, S4)
    print(f"\n\n===== STARTING: Standard Single Simulation: {exp} =====")
    
    if exp in ('S1', 'S3'):
        rheology_n = 1
        # For a standard linear run, ISSM expects B to be viscosity.
        # This value is a placeholder and should be chosen carefully for standard S1/S3 runs.
        # A typical value might be derived from a reference strain rate, e.g., 1e13 Pa*s
        print("WARNING: Using a constant placeholder viscosity for standard n=1 run.")
        rheology_B = 1e13 # (Pa*s)
        
    elif exp in ('S2', 'S4'):
        rheology_n = 3
        rheology_B = cuffey(ice_temperature)
        
    else:
        raise ValueError(f"Experiment '{exp}' not recognized for single-run mode.")

    md.materials.rheology_B = rheology_B * np.ones(nv)
    md.materials.rheology_n = rheology_n * np.ones(ne)
    md.materials.rheology_law = "BuddJacka"
    md.miscellaneous.name = f'{BEDROCK_PROFILE_ID:03d}_{exp}_{resolution_factor}'

    # Setup friction based on the chosen experiment
    md = setup_friction(md, exp)
    
    print(f"\n===== Solving Diagnostic Stressbalance for {exp} =====")
    md = solve(md,'Stressbalance')


# ==============================================================================
# ~~~~~~~~~~~~~~~ POST-DIAGNOSTIC ANALYSIS & TRANSIENT SOLVE ~~~~~~~~~~~~~~~
# ==============================================================================

# This section now runs on the final state of 'md', whether it came from
# the two-step process or a standard single run.

print("\n===== DIAGNOSING ACCELERATION =====")
x_sorted, vx_sorted, dvx_dx = diagnose_acceleration_onset(md, L)
# ... (your plotting code for acceleration)

# Save diagnostic results in ISMIP-HOM format
try:
    filename = save_results(md, L, md.miscellaneous.name)
    print(f"✓ Diagnostic output saved: {filename}")
except Exception as e:
    print(f"⚠ Error saving diagnostic output: {e}")

# Analyze driving stress
surface_slope, bed_slope, thickness = analyse_driving_stress(md, L)

# --- Proceed to Transient Solve ---
print("\n===== Solving Transient Full-Stokes =====")
# The Solve() function will now use the final configured state of 'md'
md = Solve(md) 

# ... (your final saving and plotting calls)
```

### How to Use the New Script

1.  **Add the Function:** Copy the `calculate_equivalent_linear_rheology` function into `periodic_flowline.py`.
2.  **Replace the Main Block:** Replace your existing experiment selection and solver calls with the new `CORE SIMULATION BLOCK` provided above.
3.  **Run the Experiment:** To perform your desired comparison, simply set `exp = 'S4_to_S3_equivalent'` at the top of the script. To run a standard non-linear simulation, you can still set `exp = 'S4'`, and it will execute through the `else` block as before.

This revised structure gives you the best of both worlds: it preserves the ability to run simple, single experiments while cleanly incorporating the more complex, two-step workflow required to find the equivalent linear viscosity.