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

   FemModel initialization elapsed time:   0.539561
   Total Core solution elapsed time:       55.4591
   Linear solver elapsed time:             18.1215 (33%)

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
Applying empirical tuning factor of 1.5
✓ Calculated spatially-varying rheology_B field.
  New rheology_B range: [3.43e+13, 1.50e+15]
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

   FemModel initialization elapsed time:   0.540178
   Total Core solution elapsed time:       6.88037
   Linear solver elapsed time:             1.99625 (29%)

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
  vx: [2.04987, 45.99808]
  vy: [-27.33625, 27.36168]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-106.51601, 45.52893]
✓ Saved 165_S3_equivalent_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           17.16453395    0.         -106.51600857]
 [   0.99952358   17.17042555    0.          -93.50208116]
 [   0.99904717   16.85033802    0.          -73.39926324]
 [   0.99857075   16.34729328    0.          -57.05291632]
 [   0.99809433   15.41950526    0.          -47.47827167]]


===== COMPARING S4 (n=3) vs S3 (n=1) RESULTS =====

--- STATISTICAL COMPARISONS ---

=== Vx COMPARISON ===
  n=3 range: [-9.753e+01, 4.514e+01]
  n=1 range: [-1.065e+02, 5.112e+01]
  Max abs diff: 1.395e+01
  Mean rel diff: 0.195
  RMS rel diff: 6.269

=== Vel COMPARISON ===
  n=3 range: [4.529e-02, 9.753e+01]
  n=1 range: [1.847e-02, 1.065e+02]
  Max abs diff: 1.395e+01
  Mean rel diff: 0.092
  RMS rel diff: 0.572

=== StrainRatexx COMPARISON ===
  n=3 range: [-7.167e-09, 5.525e-09]
  n=1 range: [-6.678e-09, 6.369e-09]
  Max abs diff: 8.436e-10
  Mean rel diff: 0.474
  RMS rel diff: 0.771

=== StrainRatexy COMPARISON ===
  n=3 range: [-1.209e-09, 4.771e-09]
  n=1 range: [-1.229e-09, 4.305e-09]
  Max abs diff: 7.712e-10
  Mean rel diff: 0.485
  RMS rel diff: 0.878

=== StressTensorxx COMPARISON ===
  n=3 range: [-1.890e+07, 2.023e+05]
  n=1 range: [-1.886e+07, 1.942e+05]
  Max abs diff: 2.344e+05
  Mean rel diff: 0.084
  RMS rel diff: 6.641

=== StressTensorxy COMPARISON ===
  n=3 range: [-1.434e+05, 1.879e+05]
  n=1 range: [-1.229e+05, 2.167e+05]
  Max abs diff: 1.401e+05
  Mean rel diff: 0.252
  RMS rel diff: 5.571
⚠ Field BasalDragx not available in one or both solutions

--- GENERATING COMPARISON PLOTS ---

✓ Comparison complete! Check generated plots and statistics above.
✓ Plots saved with prefix: profile_165_


===== Workflow Complete =====

```
