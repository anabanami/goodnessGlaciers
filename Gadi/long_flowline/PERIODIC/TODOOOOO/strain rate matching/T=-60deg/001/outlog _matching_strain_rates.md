# S4 & S3
resolution 1.0
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python rigidity.py 

Loaded bedrock profile 1 with parameters:
  amplitude: 0.0
  wavelength: 6336.0
  skewness: 0.0
  kurtosis: 0.0
  noise_level: 0.0
  initial_elevation: 1.0

============ SETTING MESH==============
Construction of a mesh from a given geometry

[ADAPTIVE_BAMG] FINAL Mesh statistics:
  wavelength_thickness_ratio: 3.3
  hmax: 126.72
  resolution_factor: 1.0
  refinement_factor: 50
  Total vertices: 43109
  Elements: 81988
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
  Array size: 43109
  Range: [0.0, 217567.0] Pa·s·m⁻¹
checking model consistency
marshalling file '001_S4_1.0'.bin
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

   FemModel initialization elapsed time:   0.612988
   Total Core solution elapsed time:       61.9238
   Linear solver elapsed time:             19.4975 (31%)

   Total elapsed time: 0 hrs 1 min 2 sec
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
  BedSlopeX: shape (43109, 1)
  SolutionType: length 21, type <class 'str'>
  StrainRatexx: shape (43109, 1)
  StrainRatexy: shape (43109, 1)
  StressTensorxx: shape (43109, 1)
  StressTensorxy: shape (43109, 1)
  StressTensoryy: shape (43109, 1)
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (43109, 1)
  Vel: shape (43109, 1)
  Vx: shape (43109, 1)
  Vy: shape (43109, 1)
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
  vx: [17.59794, 17.87798]
  vy: [-0.12058, 0.05919]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [17.11601, 17.87771]
✓ Saved 001_S4_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         17.63837871  0.         17.11600555]
 [ 0.99952358 17.63822266  0.         17.16161284]
 [ 0.99904717 17.63754816  0.         17.21237264]
 [ 0.99857075 17.63639987  0.         17.26444194]
 [ 0.99809433 17.63411297  0.         17.30185277]]

--- Extracting reference state for equivalent linear run ---


===== STARTING: Equivalent Linear (n=1) Simulation =====
--- Calculating equivalent linear rheology (B for n=1) ---
✓ Calculated spatially-varying rheology_B field.
  New rheology_B range: [1.00e+15, 1.00e+15]
checking model consistency
marshalling file '001_S3_equivalent_1.0'.bin
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

   FemModel initialization elapsed time:   0.612167
   Total Core solution elapsed time:       7.69744
   Linear solver elapsed time:             2.21708 (29%)

   Total elapsed time: 0 hrs 0 min 8 sec
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
  BedSlopeX: shape (43109, 1)
  SolutionType: length 21, type <class 'str'>
  StrainRatexx: shape (43109, 1)
  StrainRatexy: shape (43109, 1)
  StressTensorxx: shape (43109, 1)
  StressTensorxy: shape (43109, 1)
  StressTensoryy: shape (43109, 1)
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (43109, 1)
  Vel: shape (43109, 1)
  Vx: shape (43109, 1)
  Vy: shape (43109, 1)
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
  vx: [2.37754, 21.75563]
  vy: [-4.90489, 4.89654]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-15.96142, 19.94330]
✓ Saved 001_S3_equivalent_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[  1.           3.85157813   0.         -15.96141838]
 [  0.99952358   3.84178203   0.         -15.00817314]
 [  0.99904717   3.79441821   0.         -13.74459968]
 [  0.99857075   3.72568581   0.         -12.25643079]
 [  0.99809433   3.62207037   0.         -10.99346548]]


===== COMPARING S4 (n=3) vs S3 (n=1) RESULTS =====

--- STATISTICAL COMPARISONS ---

=== Vx COMPARISON ===
  n=3 range: [1.712e+01, 1.788e+01]
  n=1 range: [-1.596e+01, 2.176e+01]
  Max abs diff: 3.308e+01
  Mean rel diff: 0.206
  RMS rel diff: 0.271

=== Vel COMPARISON ===
  n=3 range: [1.712e+01, 1.788e+01]
  n=1 range: [1.144e-01, 2.176e+01]
  Max abs diff: 1.745e+01
  Mean rel diff: 0.199
  RMS rel diff: 0.244

=== StrainRatexx COMPARISON ===
  n=3 range: [-2.123e-11, 1.651e-11]
  n=1 range: [-4.643e-10, 4.716e-10]
  Max abs diff: 4.551e-10
  Mean rel diff: 5.015
  RMS rel diff: 10.991

=== StrainRatexy COMPARISON ===
  n=3 range: [-2.237e-12, 1.538e-11]
  n=1 range: [-4.053e-11, 3.847e-10]
  Max abs diff: 3.696e-10
  Mean rel diff: 13.743
  RMS rel diff: 16.180

=== StressTensorxx COMPARISON ===
  n=3 range: [-1.891e+07, 2.648e+05]
  n=1 range: [-1.891e+07, 1.103e+05]
  Max abs diff: 2.334e+05
  Mean rel diff: 0.060
  RMS rel diff: 0.208

=== StressTensorxy COMPARISON ===
  n=3 range: [-1.544e+05, 2.657e+05]
  n=1 range: [-3.932e+04, 2.810e+05]
  Max abs diff: 1.337e+05
  Mean rel diff: 0.241
  RMS rel diff: 3.099
⚠ Field BasalDragx not available in one or both solutions

--- GENERATING COMPARISON PLOTS ---

✓ Comparison complete! Check generated plots and statistics above.
✓ Plots saved with prefix: profile_001_


===== Workflow Complete =====

```
