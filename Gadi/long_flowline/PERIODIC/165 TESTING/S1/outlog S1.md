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

EXPERIMENT S1: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 14 left, 14 right vertices
✅ Created 14 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S1_1.125'.bin
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

   FemModel initialization elapsed time:   0.556249
   Total Core solution elapsed time:       6.57651
   Linear solver elapsed time:             1.93406 (29%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-32.582, 83.535] m/yr
  dvx_dx range: [-2.96e-02, 2.04e-02] 1/yr
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
  vx: [-32.58234, 83.53534]
  vy: [-54.31208, 61.89718]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S1_1.125_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         23.09993822  0.          0.        ]
 [ 0.99952358 23.05618311  0.          0.        ]
 [ 0.99904717 22.85162084  0.          0.        ]
 [ 0.99857075 22.54198702  0.          0.        ]
 [ 0.99809433 22.09282391  0.          0.        ]]
✓ output saved: 165_S1_1.125_static.txt

=== DRIVING STRESS DIAGNOSTIC S1 ===
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
marshalling file '165_S1_1.125'.bin
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

EXPERIMENT S1: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 16 left, 16 right vertices
✅ Created 16 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S1_1.0'.bin
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

   FemModel initialization elapsed time:   0.553139
   Total Core solution elapsed time:       6.60953
   Linear solver elapsed time:             1.97325 (30%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-32.678, 83.674] m/yr
  dvx_dx range: [-2.95e-02, 2.03e-02] 1/yr
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
  vx: [-32.67781, 83.67381]
  vy: [-54.30288, 61.96310]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S1_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         23.16899992  0.          0.        ]
 [ 0.99952358 23.15252575  0.          0.        ]
 [ 0.99904717 22.90512501  0.          0.        ]
 [ 0.99857075 22.59751563  0.          0.        ]
 [ 0.99809433 22.12953797  0.          0.        ]]
✓ output saved: 165_S1_1.0_static.txt

=== DRIVING STRESS DIAGNOSTIC S1 ===
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
marshalling file '165_S1_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/219000  time [yr]: 0.00 (time step: 0.00)

...


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
  hmax: 110.88
  resolution_factor: 0.875
  refinement_factor: 50
  Total vertices: 43479
  Elements: 82724
  inlet vertices: 18
  terminus vertices: 18
========================================

============ SETUP FRICTION ==============

EXPERIMENT S1: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 18 left, 18 right vertices
✅ Created 18 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S1_0.875'.bin
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

   FemModel initialization elapsed time:   0.547826
   Total Core solution elapsed time:       6.53012
   Linear solver elapsed time:             1.93114 (30%)

   Total elapsed time: 0 hrs 0 min 7 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-32.670, 83.627] m/yr
  dvx_dx range: [-2.95e-02, 2.02e-02] 1/yr
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
  vx: [-32.66974, 83.62747]
  vy: [-54.46132, 61.93113]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S1_0.875_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         23.19187513  0.          0.        ]
 [ 0.99952358 23.15432489  0.          0.        ]
 [ 0.99904717 22.93949102  0.          0.        ]
 [ 0.99857075 22.6129375   0.          0.        ]
 [ 0.99809433 22.16344337  0.          0.        ]]
✓ output saved: 165_S1_0.875_static.txt

=== DRIVING STRESS DIAGNOSTIC S1 ===
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
marshalling file '165_S1_0.875'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/250286  time [yr]: 0.00 (time step: 0.00)

...


```

0.5
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

EXPERIMENT S1: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 21 left, 21 right vertices
✅ Created 21 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S1_0.75'.bin
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

   FemModel initialization elapsed time:   0.652395
   Total Core solution elapsed time:       7.75689
   Linear solver elapsed time:             2.37288 (31%)

   Total elapsed time: 0 hrs 0 min 8 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-32.664, 83.749] m/yr
  dvx_dx range: [-2.95e-02, 2.03e-02] 1/yr
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
  vx: [-32.66441, 83.74917]
  vy: [-54.43748, 61.92074]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S1_0.75_static.txt with shape (2100, 4)
First 5 rows:
[[ 1.         23.24757207  0.          0.        ]
 [ 0.99952358 23.19517044  0.          0.        ]
 [ 0.99904717 22.97309927  0.          0.        ]
 [ 0.99857075 22.64171364  0.          0.        ]
 [ 0.99809433 22.18575481  0.          0.        ]]
✓ output saved: 165_S1_0.75_static.txt

=== DRIVING STRESS DIAGNOSTIC S1 ===
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
marshalling file '165_S1_0.75'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/292000  time [yr]: 0.00 (time step: 0.00)

...


```