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

Experiment S3: β² = 1500 field
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
marshalling file '165_S3_1.125'.bin
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

   FemModel initialization elapsed time:   0.555321
   Total Core solution elapsed time:       6.66838
   Linear solver elapsed time:             1.98041 (30%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-29.118, 121.039] m/yr
  dvx_dx range: [-2.81e-02, 1.57e-02] 1/yr
Acceleration problem starts at 208.1 km

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
  vx: [-29.11762, 121.03880]
  vy: [-64.79917, 79.15948]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-136.24999, 69.59665]
✓ Saved 165_S3_1.125_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           20.31971113    0.         -136.24999275]
 [   0.99952358   20.2412648     0.         -122.71068848]
 [   0.99904717   19.75528431    0.         -107.705733  ]
 [   0.99857075   19.0224815     0.          -90.83829737]
 [   0.99809433   17.90102381    0.          -78.36330287]]
✓ output saved: 165_S3_1.125_static.txt

=== DRIVING STRESS DIAGNOSTIC S3 ===
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
Δt (yr): 0.0015410958904109589 Tfinal (yr): 300 ≈nsteps: 194666
checking model consistency
marshalling file '165_S3_1.125'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/194667  time [yr]: 0.00 (time step: 0.00)

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

Experiment S3: β² = 1500 field
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
marshalling file '165_S3_1.0'.bin
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

   FemModel initialization elapsed time:   0.542346
   Total Core solution elapsed time:       6.62612
   Linear solver elapsed time:             2.01484 (30%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-29.134, 120.989] m/yr
  dvx_dx range: [-2.79e-02, 1.58e-02] 1/yr
Acceleration problem starts at 208.1 km

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
  vx: [-29.13381, 120.98895]
  vy: [-64.82515, 79.23689]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-133.78185, 69.58855]
✓ Saved 165_S3_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           20.32340355    0.         -133.78184825]
 [   0.99952358   20.31120907    0.         -121.55767301]
 [   0.99904717   19.71599563    0.         -106.55611895]
 [   0.99857075   18.99244874    0.          -91.49320611]
 [   0.99809433   17.84768589    0.          -79.24116473]]
✓ output saved: 165_S3_1.0_static.txt

=== DRIVING STRESS DIAGNOSTIC S3 ===
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
Δt (yr): 0.0013698630136986301 Tfinal (yr): 300 ≈nsteps: 219000
checking model consistency
marshalling file '165_S3_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/219000  time [yr]: 0.00 (time step: 0.00)

...

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

Experiment S3: β² = 1500 field
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
marshalling file '165_S3_0.875'.bin
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

   FemModel initialization elapsed time:   0.552876
   Total Core solution elapsed time:       6.62386
   Linear solver elapsed time:             1.94585 (29%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-29.138, 121.031] m/yr
  dvx_dx range: [-2.80e-02, 1.58e-02] 1/yr
Acceleration problem starts at 208.1 km

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
  vx: [-29.13814, 121.03053]
  vy: [-64.85122, 79.20599]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-134.86196, 69.58712]
✓ Saved 165_S3_0.875_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           20.39024844    0.         -134.86196494]
 [   0.99952358   20.32055354    0.         -122.39607104]
 [   0.99904717   19.80867586    0.         -106.76238332]
 [   0.99857075   19.03271158    0.          -91.59055122]
 [   0.99809433   17.94889417    0.          -79.03576961]]
✓ output saved: 165_S3_0.875_static.txt

=== DRIVING STRESS DIAGNOSTIC S3 ===
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
Δt (yr): 0.0011986301369863014 Tfinal (yr): 300 ≈nsteps: 250285
checking model consistency
marshalling file '165_S3_0.875'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/250286  time [yr]: 0.00 (time step: 0.00)

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

Experiment S3: β² = 1500 field
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
marshalling file '165_S3_0.75'.bin
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

   FemModel initialization elapsed time:   0.662826
   Total Core solution elapsed time:       7.81789
   Linear solver elapsed time:             2.40417 (31%)

   Total elapsed time: 0 hrs 0 min 8 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-29.140, 121.039] m/yr
  dvx_dx range: [-2.79e-02, 1.56e-02] 1/yr
Acceleration problem starts at 208.1 km

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
  vx: [-29.14010, 121.03866]
  vy: [-65.01924, 79.23513]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [-134.74380, 69.60649]
✓ Saved 165_S3_0.75_static.txt with shape (2100, 4)
First 5 rows:
[[   1.           20.44986266    0.         -134.74379595]
 [   0.99952358   20.33885057    0.         -122.47038447]
 [   0.99904717   19.81782793    0.         -106.64558448]
 [   0.99857075   19.03658631    0.          -91.68812521]
 [   0.99809433   17.95076749    0.          -78.85227195]]
✓ output saved: 165_S3_0.75_static.txt

=== DRIVING STRESS DIAGNOSTIC S3 ===
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
Δt (yr): 0.0010273972602739727 Tfinal (yr): 300 ≈nsteps: 291999
checking model consistency
marshalling file '165_S3_0.75'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/292000  time [yr]: 0.00 (time step: 0.00)

...


```