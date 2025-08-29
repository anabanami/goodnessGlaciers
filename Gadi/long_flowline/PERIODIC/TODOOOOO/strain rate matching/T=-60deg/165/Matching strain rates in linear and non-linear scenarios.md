# Glen's Flow law
This calculations are for a 2D system 
$$\dot{\varepsilon}_{ij} \,=\, A_{n}(T)\,\tau^{n-1}\,\tau_{ij},$$

let $n = 1$:

$$\dot{\varepsilon}_{ij} =  A_{1}(T)\,\tau^{1-1}\,\tau_{ij},$$

$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,\tau_{ij},$$

from the definition of the full stress (i.e. $\sigma_{ij}\,=\,-p\,\delta_{ij}\,+\,\tau_{ij}$ ) , substitute the deviatoric stress tensor $\tau_{ij}$ in the strain rate equation
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,+\,p\,\delta_{ij}),$$
here the pressure is the negative average of the stress in 2 dimensions $p\,=\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{1}(T)\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{1}(T)\left (\begin{bmatrix}
 \sigma_{xx} & \sigma_{xy} \\
\sigma_{yx} & \sigma_{yy} \\
\end{bmatrix}\,-\,\frac{1}{2}(\sigma_{xx}\,+\,\sigma_{yy})\begin{bmatrix} 1& 0\\ 0& 1\\ \end{bmatrix}\right ),$$
$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{1}(T)\left (\begin{bmatrix}
 \frac{1}{2}(\sigma_{xx}\,-\,\sigma_{yy}) & \sigma_{xy} \\
\sigma_{yx} & \frac{1}{2}(\sigma_{yy}\,-\,\sigma_{xx}) \\
\end{bmatrix}\right ),$$

While letting $n\,=\,3$ in Glen's flow law returns

$$\dot{\varepsilon}_{ij} =  A_{3}(T)\,\tau^{3-1}\,\tau_{ij},$$

$$\Rightarrow\,\dot{\varepsilon}_{ij}\,= \,A_{3}(T)\,\tau^{2}\,\tau_{ij},$$

once again, substitute the deviatoric stress tensor $\tau_{ij}$ in the strain rate equation
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{3}(T)\,\tau^{2}\,(\sigma_{ij}\,+\,p\,\delta_{ij}),$$
here the pressure is the negative average of the stress in 2 dimensions $p\,=\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})$
$$\Rightarrow\,\dot{\varepsilon}_{ij}\,=\,A_{3}(T)\,\tau^{2}\,(\sigma_{ij}\,-\frac{1}{2}\,(\Sigma_{i}\,\sigma_{ii})\,\delta_{ij}),$$
$$\Rightarrow\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,A_{3}(T)\,\tau^{2}\,\begin{bmatrix}
 \frac{1}{2}(\sigma_{xx}\,-\,\sigma_{yy}) & \sigma_{xy} \\
\sigma_{yx} & \frac{1}{2}(\sigma_{yy}\,-\,\sigma_{xx}) \\
\end{bmatrix},$$
here it is useful to write the effective stress as $\tau\,=\,\sqrt{\frac{1}{2}\tau_{ij}\tau_{ij}}$  $\Rightarrow\,\tau^2\,=\,\frac{1}{2}\tau_{ij}\tau_{ij}$.

The term $\tau_{ij}\tau_{ij}$ represents the sum of the squares of all components of the deviatoric stress tensor matrix (Frobenius norm)
$$\dot{\varepsilon}_{ij}\,=\, A_3(T)\left(  \frac{1}{2}\left( \left( \frac{1}{2}\left(\sigma_{xx}-\sigma_{yy}\right)\right)^2\,+\,\sigma_{xy}^2\,+ \sigma_{yx}^2\,\left( \frac{1}{2}\left(\sigma_{yy}-\sigma_{xx}\right)\right)^2\right)\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix} $$
Assuming a symmetric stress tensor, where $\sigma_{xy}\,=\,\sigma_{yx}$,
$$\dot{\varepsilon}_{ij}\,=\, A_3(T)\left(   \frac{1}{2}\left(  \frac{1}{4}\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,2\sigma_{xy}^2\,+\,\frac{1}{4}\left(\sigma_{yy}-\sigma_{xx}\right)^2\right)\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$$
$$\Leftrightarrow\,\begin{bmatrix}
 \dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{yx} & \dot{\varepsilon}_{yy} \\
\end{bmatrix}\,=\,\frac{1}{4}A_{3}(T)\, \left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,\sigma_{xy}^2\right) \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$$

We now match the strain rates for both the linear and non-linear cases:

$$\dot{\varepsilon}_{ij}(n=1)\,=\,\dot{\varepsilon}_{ij}(n=3)$$

let $M = \begin{bmatrix} \frac{1}{2}(\sigma_{xx}-\sigma_{yy}) & \sigma_{xy} \\ \sigma_{yx} & \frac{1}{2}(\sigma_{yy}-\sigma_{xx}) \end{bmatrix}$ and simplify the equation:
$$\Leftrightarrow\,A_{1}(T)M \,=\, \,\frac{1}{4}A_{3}(T)\, \left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,\sigma_{xy}^2\right)M$$

$$\therefore\,A_{1}(T) \,=\, \,\frac{1}{4}A_{3}(T)\,\left(\left(\sigma_{xx}-\sigma_{yy}\right)^2\,+\,\sigma_{xy}^2\right)$$

# IMPLEMENTING THIS IN CODE
the workflow should be a two-step process:

1. **Run the physical, non-linear (`n=3`) simulation first.** This is "reference" or "ground-truth" experiment.
	**`reference_nonlinear_flowline.py`:**
     run the non-linear experiment (`exp = 'S4'`, `rheology_n = 3`).
     It uses the standard `cuffey` function to determine the ice rigidity (`rheology_B`) based on temperature.
        
 Crucially, after the diagnostic solve,  add a new section to save the essential results: `stress_xx`, `stress_yy`, `stress_xy`, and the `rheology_B_n3` field into a `.npz` file. This perfectly captures the "reference state" needed for the next step.
```python
# ... after existing diagnostic solve ...
print("\n===== Solving Diagnostic Stressbalance =====")
# diagnostic solve
md.stressbalance.requested_outputs = ['StressTensorxx', 'StressTensoryy', 'StressTensorxy']
md = solve(md,'Stressbalance')

# --- NEW SECTION: SAVE REFERENCE RESULTS FOR EQUIVALENT LINEAR RUN ---
if rheology_n == 3:
    print("\n===== SAVING N=3 REFERENCE STATE =====")
    
    # Extract the full stress tensor components from the diagnostic solution
    # These are the 'sigma' values in your derivation
    stress_xx = md.results.StressbalanceSolution.StressTensorxx
    stress_yy = md.results.StressbalanceSolution.StressTensoryy
    stress_xy = md.results.StressbalanceSolution.StressTensorxy
    
    # Also save the original rheology_B field used for the n=3 run
    rheology_B_n3 = md.materials.rheology_B
    
    # Save these fields to a file
    reference_filename = f'reference_state_{md.miscellaneous.name}.npz'
    np.savez(
        reference_filename,
        stress_xx=stress_xx,
        stress_yy=stress_yy,
        stress_xy=stress_xy,
        rheology_B_n3=rheology_B_n3
    )
    print(f"✓ Saved reference stress and rheology to {reference_filename}")
# --- END OF NEW SECTION ---
# ... rest of script continues as normal ...
```

2. **Use the formula to calculate the equivalent, spatially varying `rheology_B` field** for the `n=1` case.
**`linear_periodic_flowline.py`:**
This script is configured to run the linear experiment (`exp = 'S3'`, `rheology_n = 1`).        
Instead of using a constant value for the ice rigidity, it loads the `reference_state_...npz` file generated by the first script.

The script then assigns this newly calculated `rheology_B` field to `md.materials.rheology_B` before solving.
```python
# ... in main script logic ...

# Choose experiment
exp = 'S3_equivalent' # Use a new name for clarity
rheology_n = 1
# constants & material properties
yts = 31556926 # s/yr

# ... (skip the old rheology block) ...

# Run the geometry and meshing setup as before
# ...
# md, nv, ne, resolution_factor = adaptive_bamg(md, x_1D, s0, b, 1.0) 
# ...

# --- NEW RHEOLOGY SETUP FOR EQUIVALENT LINEAR RUN ---
print("\n===== CONFIGURING EQUIVALENT LINEAR RHEOLOGY =====")

# This assumes you are running this after an n=3 simulation with the same mesh settings
# The name should match the output from the n=3 run (e.g., '03d_S4_1.0')
reference_model_name = f'{BEDROCK_PROFILE_ID:03d}_S4_1.0' # IMPORTANT: Match this to your n=3 run name
reference_filename = f'reference_state_{reference_model_name}.npz'

try:
    print(f"Loading reference state from: {reference_filename}")
    data = np.load(reference_filename)
    stress_xx = data['stress_xx']
    stress_yy = data['stress_yy']
    stress_xy = data['stress_xy']
    rheology_B_n3 = data['rheology_B_n3']

    # Your derivation: A1 = A3 * 0.25 * ((σxx-σyy)² + σxy²)
    # Note: I corrected a small typo in your PDF where the final formula missed a '4' on the shear term
    # Based on line 26 of your PDF, the term is (1/4)*((σxx-σyy)² + 4*σxy²), I will use your final derived formula
    
    # 1. Convert B_n3 to A_n3. For n=3, B = A^(-1/n) => A = B^(-n)
    A_n3 = rheology_B_n3**(-3)

    # 2. Calculate the stress-dependent part of your formula
    stress_factor = 0.25 * ((stress_xx - stress_yy)**2 + 4 * stress_xy**2)

    # 3. Calculate the spatially-varying A_n1 field
    A_n1_field = A_n3 * stress_factor

    # 4. Convert A_n1 to rheology_B for the linear (n=1) case
    # For n=1, ISSM expects B to be the dynamic viscosity η, where η = (2*A)^-1
    # Avoid division by zero if A is zero anywhere
    A_n1_field[A_n1_field == 0] = 1e-30 # Add a small floor to prevent errors
    rheology_B = (2 * A_n1_field)**(-1)

    print("✓ Successfully calculated spatially-varying rheology_B field.")
    print(f"  New rheology_B range: [{np.min(rheology_B):.2e}, {np.max(rheology_B):.2e}]")

except FileNotFoundError:
    print(f"FATAL ERROR: Could not find reference file {reference_filename}.")
    print("Please run the corresponding n=3 simulation first to generate this file.")
    sys.exit(1)

# --- END OF NEW RHEOLOGY SETUP ---

# Now, assign this new field to the model
md.materials.rheology_B = rheology_B
md.materials.rheology_n = rheology_n * np.ones(ne)

# ... continue with the rest of the model setup (friction, BCs, etc.) and solve
```

To verify the success of this implementation, compare the `Vx` output from the diagnostic solves of both models. They should be nearly identical