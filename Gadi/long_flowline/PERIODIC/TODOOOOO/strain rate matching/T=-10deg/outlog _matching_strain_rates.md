# S4 & S3
resolution 1.0
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python rigidity.py 

Loaded bedrock profile 165 with parameters:
  amplitude: 22.400000000000002
  wavelength: 6336.0
  skewness: 0.0
  kurtosis: 0.10000000000000003
  noise_level: 0.0
  initial_elevation: 1.0

============ SETTING MESH==============
Construction of a mesh from a given geometry

[ADAPTIVE_BAMG] FINAL Mesh statistics:
  wavelength_thickness_ratio: 3.3
  hmax: 126.72
  resolution_factor: 1.0
  refinement_factor: 50
  Total vertices: 43437
  Elements: 82644
  inlet vertices: 16
  terminus vertices: 16
========================================
Found 16 left, 16 right vertices
✅ Created 16 flowband pairs using relative depth matching


===== STARTING: Non-Linear (n=3) Reference Simulation =====

============ SETUP FRICTION ==============

Experiment S4: β² = 1500 field
β² field statistics:
  Array size: 1
  Range: [1500.0, 1500.0] Pa·a·m⁻¹
  Mean: 1500.0 Pa·a·m⁻¹

Final friction coefficient:
  Array size: 43437
  Range: [0.0, 217567.0] Pa·s·m⁻¹
checking model consistency
marshalling file '165_S4_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...
   computing slope
   computing basal mass balance
=================================================================
WARNING: Could not find the output "BasalDragx"
         - either this output is unavailable for this model run 
         - or there may be a spelling error in your requested_outputs 
         - or there may be a spelling error in an outputdefinition 
           object name (unlikely)
=================================================================
=================================================================
WARNING: Could not find the output "BasalDragy"
         - either this output is unavailable for this model run 
         - or there may be a spelling error in your requested_outputs 
         - or there may be a spelling error in an outputdefinition 
           object name (unlikely)
=================================================================
write lock file:

   FemModel initialization elapsed time:   0.541275
   Total Core solution elapsed time:       55.4586
   Linear solver elapsed time:             18.049  (33%)

   Total elapsed time: 0 hrs 0 min 56 sec
loading results from cluster
--- Storing n=3 results for comparison ---

--- Saving results for non-linear run ---

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (43437, 1)
  SolutionType: length 21, type <class 'str'>
  StrainRatexx: shape (43437, 1)
  StrainRatexy: shape (43437, 1)
  StressTensorxx: shape (43437, 1)
  StressTensorxy: shape (43437, 1)
  StressTensoryy: shape (43437, 1)
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (43437, 1)
  Vel: shape (43437, 1)
  Vx: shape (43437, 1)
  Vy: shape (43437, 1)
  checkconsistency: <class 'method'>
  errlog: length 0, type <class 'list'>
  getfieldnames: <class 'method'>
  getlongestfieldname: <class 'method'>
  marshall: <class 'method'>
  outlog: length 0, type <class 'list'>
  setdefaultparameters: <class 'method'>
  step: <class 'int'>
  time: <class 'float'>
Coordinate ranges: x_hat=[0.000, 1.000],y_hat=[-0.002, 0.009]
Warning: No Vz field found, using zeros
Surface velocity ranges (m a⁻¹):
  vx: [4.23546, 41.02462]
  vy: [-26.19322, 27.97066]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-97.52982, 41.40952]
✓ Saved 165_S4_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[  1.          16.34218135   0.         -97.52981677]
 [  0.99952358  16.33824048   0.         -79.55655675]
 [  0.99904717  16.04924821   0.         -62.11415812]
 [  0.99857075  15.66027953   0.         -48.29105841]
 [  0.99809433  14.86385301   0.         -39.53441218]]

--- Extracting reference state for equivalent linear run ---


===== STARTING: Equivalent Linear (n=1) Simulation =====
--- Calculating equivalent linear rheology (B for n=1) ---
✓ Calculated spatially-varying rheology_B field.
  New rheology_B range: [2.29e+13, 1.00e+15]
checking model consistency
marshalling file '165_S3_equivalent_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...
   computing slope
   computing basal mass balance
=================================================================
WARNING: Could not find the output "BasalDragx"
         - either this output is unavailable for this model run 
         - or there may be a spelling error in your requested_outputs 
         - or there may be a spelling error in an outputdefinition 
           object name (unlikely)
=================================================================
=================================================================
WARNING: Could not find the output "BasalDragy"
         - either this output is unavailable for this model run 
         - or there may be a spelling error in your requested_outputs 
         - or there may be a spelling error in an outputdefinition 
           object name (unlikely)
=================================================================
write lock file:

   FemModel initialization elapsed time:   0.550604
   Total Core solution elapsed time:       6.86652
   Linear solver elapsed time:             1.99488 (29%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

--- Saving results for equivalent linear run ---

--- Extracting reference state for equivalent linear run ---

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (43437, 1)
  SolutionType: length 21, type <class 'str'>
  StrainRatexx: shape (43437, 1)
  StrainRatexy: shape (43437, 1)
  StressTensorxx: shape (43437, 1)
  StressTensorxy: shape (43437, 1)
  StressTensoryy: shape (43437, 1)
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (43437, 1)
  Vel: shape (43437, 1)
  Vx: shape (43437, 1)
  Vy: shape (43437, 1)
  checkconsistency: <class 'method'>
  errlog: length 0, type <class 'list'>
  getfieldnames: <class 'method'>
  getlongestfieldname: <class 'method'>
  marshall: <class 'method'>
  outlog: length 0, type <class 'list'>
  setdefaultparameters: <class 'method'>
  step: <class 'int'>
  time: <class 'float'>
Coordinate ranges: x_hat=[0.000, 1.000],y_hat=[-0.002, 0.009]
Warning: No Vz field found, using zeros
Surface velocity ranges (m a⁻¹):
  vx: [-1.53698, 55.14661]
  vy: [-39.14917, 39.40731]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-145.31817, 52.15531]
✓ Saved 165_S3_equivalent_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           20.2318936     0.         -145.31816954]
 [   0.99952358   20.2539608     0.         -126.18711383]
 [   0.99904717   19.82612782    0.          -97.26190303]
 [   0.99857075   19.15279547    0.          -74.45583661]
 [   0.99809433   17.9005032     0.          -61.84481258]]


===== COMPARING S4 (n=3) vs S3 (n=1) RESULTS =====

--- STATISTICAL COMPARISONS ---

=== Vx COMPARISON ===
  n=3 range: [-9.753e+01, 4.514e+01]
  n=1 range: [-1.453e+02, 6.211e+01]
  Max abs diff: 4.796e+01
  Mean rel diff: 0.518
  RMS rel diff: 16.754

=== Vel COMPARISON ===
  n=3 range: [4.529e-02, 9.753e+01]
  n=1 range: [3.128e-02, 1.453e+02]
  Max abs diff: 4.779e+01
  Mean rel diff: 0.269
  RMS rel diff: 1.556

=== StrainRatexx COMPARISON ===
  n=3 range: [-7.167e-09, 5.525e-09]
  n=1 range: [-9.586e-09, 9.160e-09]
  Max abs diff: 3.635e-09
  Mean rel diff: 1.081
  RMS rel diff: 1.399

=== StrainRatexy COMPARISON ===
  n=3 range: [-1.209e-09, 4.771e-09]
  n=1 range: [-1.934e-09, 6.031e-09]
  Max abs diff: 1.552e-09
  Mean rel diff: 1.203
  RMS rel diff: 1.860

=== StressTensorxx COMPARISON ===
  n=3 range: [-1.890e+07, 2.023e+05]
  n=1 range: [-1.888e+07, 1.739e+05]
  Max abs diff: 2.796e+05
  Mean rel diff: 0.075
  RMS rel diff: 5.080

=== StressTensorxy COMPARISON ===
  n=3 range: [-1.434e+05, 1.879e+05]
  n=1 range: [-1.203e+05, 1.803e+05]
  Max abs diff: 8.712e+04
  Mean rel diff: 0.475
  RMS rel diff: 8.176
⚠ Field BasalDragx not available in one or both solutions

--- GENERATING COMPARISON PLOTS ---

✓ Comparison complete! Check generated plots and statistics above.
✓ Plots saved with prefix: profile_165_


===== Workflow Complete =====

```
