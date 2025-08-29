1.125
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python periodic_flowline.py 

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
  hmax: 142.56
  resolution_factor: 1.125
  refinement_factor: 50
  Total vertices: 44039
  Elements: 83852
  inlet vertices: 14
  terminus vertices: 14
========================================

============ SETUP FRICTION ==============

Experiment S4: β² = 1500 field
β² field statistics:
  Array size: 1
  Range: [1500.0, 1500.0] Pa·a·m⁻¹
  Mean: 1500.0 Pa·a·m⁻¹

Final friction coefficient:
  Array size: 44039
  Range: [0.0, 217567.0] Pa·s·m⁻¹

============ FRICTION DIAGNOSTIC ==============
INTENDED FRICTION:
  β² = 1500 Pa·a·m⁻¹
  β² (ISSM) = 4.73e+10 Pa·s·m⁻¹
  friction coeff = 217567 Pa·s·m⁻¹

ACTUAL FRICTION:
  friction coeff range = [217567, 217567]
  β² (ISSM) range = [4.73e+10, 4.73e+10]
  β² (annual) range = [1500, 1500]
Found 14 left, 14 right vertices
✅ Created 14 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S4_1.125'.bin
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
write lock file:

   FemModel initialization elapsed time:   0.562944
   Total Core solution elapsed time:       66.6482
   Linear solver elapsed time:             21.329  (32%)

   Total elapsed time: 0 hrs 1 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [15.252, 16.953] m/yr
  dvx_dx range: [-2.58e-04, 2.27e-04] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (44039, 1)
  Pressure: shape (44039, 1)
  SolutionType: length 21, type <class 'str'>
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (44039, 1)
  Vel: shape (44039, 1)
  Vx: shape (44039, 1)
  Vy: shape (44039, 1)
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
  vx: [15.25222, 16.95317]
  vy: [-0.42362, 0.35749]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [14.99475, 16.86621]
✓ Saved 165_S4_1.125_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         15.49038214  0.         14.99474508]
 [ 0.99952358 15.48560249  0.         15.04608516]
 [ 0.99904717 15.47983503  0.         15.09266134]
 [ 0.99857075 15.47337804  0.         15.14600435]
 [ 0.99809433 15.46578676  0.         15.18427177]]
✓ output saved: 165_S4_1.125_static.txt

=== DRIVING STRESS DIAGNOSTIC S4 ===
Surface elevation: 1943.400 to 1578.368 m
Bed elevation: 23.400 to -341.632 m
Ice thickness: 1920.000 to 1920.000 m

Systematic trends over 210 km:
  Surface: -365.032 m (-1.738 m/km)
  Bed: -365.032 m (-1.738 m/km)
  Thickness: 0.000 m (0.000 m/km)

Driving stress at boundaries:
  Left (x=0): 29915.0 Pa
  Right (x=L): 29915.0 Pa
  Difference: 0.0 Pa

======================================

===== Solving Transient Full-Stokes =====
sb_coupling_frequency: 1
output_frequency: 112.5
isstressbalance: 1
Δt (yr): 0.0030821917808219177 Tfinal (yr): 300 ≈nsteps: 97333
checking model consistency
marshalling file '165_S4_1.125'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/97334  time [yr]: 0.00 (time step: 0.00)

...


```

1.0
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python periodic_flowline.py 

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

============ SETUP FRICTION ==============

Experiment S4: β² = 1500 field
β² field statistics:
  Array size: 1
  Range: [1500.0, 1500.0] Pa·a·m⁻¹
  Mean: 1500.0 Pa·a·m⁻¹

Final friction coefficient:
  Array size: 43437
  Range: [0.0, 217567.0] Pa·s·m⁻¹

============ FRICTION DIAGNOSTIC ==============
INTENDED FRICTION:
  β² = 1500 Pa·a·m⁻¹
  β² (ISSM) = 4.73e+10 Pa·s·m⁻¹
  friction coeff = 217567 Pa·s·m⁻¹

ACTUAL FRICTION:
  friction coeff range = [217567, 217567]
  β² (ISSM) range = [4.73e+10, 4.73e+10]
  β² (annual) range = [1500, 1500]
Found 16 left, 16 right vertices
✅ Created 16 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
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
write lock file:

   FemModel initialization elapsed time:   0.545865
   Total Core solution elapsed time:       60.5662
   Linear solver elapsed time:             19.6591 (32%)

   Total elapsed time: 0 hrs 1 min 1 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [15.252, 16.953] m/yr
  dvx_dx range: [-2.55e-04, 2.29e-04] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (43437, 1)
  Pressure: shape (43437, 1)
  SolutionType: length 21, type <class 'str'>
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
  vx: [15.25210, 16.95326]
  vy: [-0.42380, 0.35808]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [15.00190, 16.86673]
✓ Saved 165_S4_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         15.48992044  0.         15.00189998]
 [ 0.99952358 15.48529289  0.         15.04685588]
 [ 0.99904717 15.47928305  0.         15.09724821]
 [ 0.99857075 15.47291105  0.         15.14211995]
 [ 0.99809433 15.46518663  0.         15.18098906]]
✓ output saved: 165_S4_1.0_static.txt

=== DRIVING STRESS DIAGNOSTIC S4 ===
Surface elevation: 1943.400 to 1578.368 m
Bed elevation: 23.400 to -341.632 m
Ice thickness: 1920.000 to 1920.000 m

Systematic trends over 210 km:
  Surface: -365.032 m (-1.738 m/km)
  Bed: -365.032 m (-1.738 m/km)
  Thickness: 0.000 m (0.000 m/km)

Driving stress at boundaries:
  Left (x=0): 29915.0 Pa
  Right (x=L): 29915.0 Pa
  Difference: 0.0 Pa

======================================

===== Solving Transient Full-Stokes =====
sb_coupling_frequency: 1
output_frequency: 100.0
isstressbalance: 1
Δt (yr): 0.0027397260273972603 Tfinal (yr): 300 ≈nsteps: 109500
checking model consistency
marshalling file '165_S4_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/109501  time [yr]: 0.00 (time step: 0.00)

```

0.875
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python periodic_flowline.py 

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
  hmax: 110.88
  resolution_factor: 0.875
  refinement_factor: 50
  Total vertices: 43479
  Elements: 82724
  inlet vertices: 18
  terminus vertices: 18
========================================

============ SETUP FRICTION ==============

Experiment S4: β² = 1500 field
β² field statistics:
  Array size: 1
  Range: [1500.0, 1500.0] Pa·a·m⁻¹
  Mean: 1500.0 Pa·a·m⁻¹

Final friction coefficient:
  Array size: 43479
  Range: [0.0, 217567.0] Pa·s·m⁻¹

============ FRICTION DIAGNOSTIC ==============
INTENDED FRICTION:
  β² = 1500 Pa·a·m⁻¹
  β² (ISSM) = 4.73e+10 Pa·s·m⁻¹
  friction coeff = 217567 Pa·s·m⁻¹

ACTUAL FRICTION:
  friction coeff range = [217567, 217567]
  β² (ISSM) range = [4.73e+10, 4.73e+10]
  β² (annual) range = [1500, 1500]
Found 18 left, 18 right vertices
✅ Created 18 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S4_0.875'.bin
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
write lock file:

   FemModel initialization elapsed time:   0.555827
   Total Core solution elapsed time:       58.0131
   Linear solver elapsed time:             18.5862 (32%)

   Total elapsed time: 0 hrs 0 min 58 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [15.252, 16.953] m/yr
  dvx_dx range: [-2.54e-04, 2.31e-04] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (43479, 1)
  Pressure: shape (43479, 1)
  SolutionType: length 21, type <class 'str'>
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (43479, 1)
  Vel: shape (43479, 1)
  Vx: shape (43479, 1)
  Vy: shape (43479, 1)
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
  vx: [15.25208, 16.95311]
  vy: [-0.42357, 0.35744]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [14.99471, 16.86646]
✓ Saved 165_S4_0.875_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         15.49106318  0.         14.99470577]
 [ 0.99952358 15.48633669  0.         15.04208023]
 [ 0.99904717 15.48064521  0.         15.09339264]
 [ 0.99857075 15.47405563  0.         15.13916962]
 [ 0.99809433 15.46635426  0.         15.17851928]]
✓ output saved: 165_S4_0.875_static.txt

=== DRIVING STRESS DIAGNOSTIC S4 ===
Surface elevation: 1943.400 to 1578.368 m
Bed elevation: 23.400 to -341.632 m
Ice thickness: 1920.000 to 1920.000 m

Systematic trends over 210 km:
  Surface: -365.032 m (-1.738 m/km)
  Bed: -365.032 m (-1.738 m/km)
  Thickness: 0.000 m (0.000 m/km)

Driving stress at boundaries:
  Left (x=0): 29915.0 Pa
  Right (x=L): 29915.0 Pa
  Difference: 0.0 Pa

======================================

===== Solving Transient Full-Stokes =====
sb_coupling_frequency: 1
output_frequency: 87.5
isstressbalance: 1
Δt (yr): 0.002397260273972603 Tfinal (yr): 300 ≈nsteps: 125142
checking model consistency
marshalling file '165_S4_0.875'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/125143  time [yr]: 0.00 (time step: 0.00)

```

0.75
-----------------------------------------------------------------------------------------
```
(.venv) ana@MU00236940:~/Desktop/code/Gadi/long_flowline/PERIODIC$ python periodic_flowline.py 

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
  hmax: 95.03999999999999
  resolution_factor: 0.75
  refinement_factor: 50
  Total vertices: 51103
  Elements: 97966
  inlet vertices: 21
  terminus vertices: 21
========================================

============ SETUP FRICTION ==============

Experiment S4: β² = 1500 field
β² field statistics:
  Array size: 1
  Range: [1500.0, 1500.0] Pa·a·m⁻¹
  Mean: 1500.0 Pa·a·m⁻¹

Final friction coefficient:
  Array size: 51103
  Range: [0.0, 217567.0] Pa·s·m⁻¹

============ FRICTION DIAGNOSTIC ==============
INTENDED FRICTION:
  β² = 1500 Pa·a·m⁻¹
  β² (ISSM) = 4.73e+10 Pa·s·m⁻¹
  friction coeff = 217567 Pa·s·m⁻¹

ACTUAL FRICTION:
  friction coeff range = [217567, 217567]
  β² (ISSM) range = [4.73e+10, 4.73e+10]
  β² (annual) range = [1500, 1500]
Found 21 left, 21 right vertices
✅ Created 21 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S4_0.75'.bin
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
write lock file:

   FemModel initialization elapsed time:   0.947373
   Total Core solution elapsed time:       107.123
   Linear solver elapsed time:             33.7469 (32%)

   Total elapsed time: 0 hrs 1 min 48 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [15.252, 16.953] m/yr
  dvx_dx range: [-2.54e-04, 2.30e-04] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (51103, 1)
  Pressure: shape (51103, 1)
  SolutionType: length 21, type <class 'str'>
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (51103, 1)
  Vel: shape (51103, 1)
  Vx: shape (51103, 1)
  Vy: shape (51103, 1)
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
  vx: [15.25174, 16.95299]
  vy: [-0.42310, 0.35765]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [14.99415, 16.86643]
✓ Saved 165_S4_0.75_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         15.49114689  0.         14.99415489]
 [ 0.99952358 15.48629213  0.         15.04008967]
 [ 0.99904717 15.4806      0.         15.09222738]
 [ 0.99857075 15.47404327  0.         15.1386901 ]
 [ 0.99809433 15.46640952  0.         15.17904041]]
✓ output saved: 165_S4_0.75_static.txt

=== DRIVING STRESS DIAGNOSTIC S4 ===
Surface elevation: 1943.400 to 1578.368 m
Bed elevation: 23.400 to -341.632 m
Ice thickness: 1920.000 to 1920.000 m

Systematic trends over 210 km:
  Surface: -365.032 m (-1.738 m/km)
  Bed: -365.032 m (-1.738 m/km)
  Thickness: 0.000 m (0.000 m/km)

Driving stress at boundaries:
  Left (x=0): 29915.0 Pa
  Right (x=L): 29915.0 Pa
  Difference: 0.0 Pa

======================================

===== Solving Transient Full-Stokes =====
sb_coupling_frequency: 1
output_frequency: 75.0
isstressbalance: 1
Δt (yr): 0.0020547945205479454 Tfinal (yr): 300 ≈nsteps: 145999
checking model consistency
marshalling file '165_S4_0.75'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/146000  time [yr]: 0.00 (time step: 0.00)

```