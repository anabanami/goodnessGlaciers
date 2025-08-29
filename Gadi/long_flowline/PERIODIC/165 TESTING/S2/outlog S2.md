1.25 <--- CRASHED
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
  hmax: 158.4
  resolution_factor: 1.25
  refinement_factor: 50
  Total vertices: 42685
  Elements: 81146
  inlet vertices: 13
  terminus vertices: 13
========================================

============ SETUP FRICTION ==============

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 13 left, 13 right vertices
✅ Created 13 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_1.25'.bin
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

   FemModel initialization elapsed time:   1.71499
   Total Core solution elapsed time:       229.798
   Linear solver elapsed time:             68.3001 (30%)

   Total elapsed time: 0 hrs 3 min 51 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-5.89e-06, 3.95e-06] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (42685, 1)
  Pressure: shape (42685, 1)
  SolutionType: length 21, type <class 'str'>
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (42685, 1)
  Vel: shape (42685, 1)
  Vx: shape (42685, 1)
  Vy: shape (42685, 1)
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
  vx: [-0.00269, 0.01347]
  vy: [-0.01090, 0.01412]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_1.25_static.txt with shape (2100, 4)
First 5 rows:
[[1.         0.00153231 0.         0.        ]
 [0.99952358 0.00153686 0.         0.        ]
 [0.99904717 0.00145973 0.         0.        ]
 [0.99857075 0.00140289 0.         0.        ]
 [0.99809433 0.0012401  0.         0.        ]]
✓ output saved: 165_S2_1.25_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
output_frequency: 125.0
isstressbalance: 1
Δt (yr): 0.003424657534246575 Tfinal (yr): 300 ≈nsteps: 87600
checking model consistency
marshalling file '165_S2_1.25'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/87601  time [yr]: 0.00 (time step: 0.00)
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...
   computing slope
   computing basal mass balance
   computing basal mass balance
   computing mass transport
   call free surface computational core
   extruding Base from base...
   extruding solution from top...
   extruding solution from top...
   extruding solution from top...
   updating vertices positions
   computing transient requested outputs
   saving temporary results
iteration 2/87600  time [yr]: 0.01 (time step: 0.00)
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...

[0] ??? Error using ==> ./classes/Elements/TriaRef.cpp:117
[0] GetJacobianDeterminant error message: negative jacobian determinant!

--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[42807,1],0]
  Exit code:    1
--------------------------------------------------------------------------
loading results from cluster
Solving complete - saving results
field md.solidearth.external is None
qmu is skipped until it is more stable
✓ Full results saved to 165_S2_1.25_final_time=300_yrs_timestep=0.00274_yrs.nc

```


1.125 <--- DOESN'T break (yet)
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

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 14 left, 14 right vertices
✅ Created 14 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_1.125'.bin
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

   FemModel initialization elapsed time:   0.936833
   Total Core solution elapsed time:       130.121
   Linear solver elapsed time:             39.06   (30%)

   Total elapsed time: 0 hrs 2 min 11 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-6.16e-06, 3.99e-06] 1/yr

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
  vx: [-0.00269, 0.01347]
  vy: [-0.01056, 0.01406]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_1.125_static.txt with shape (2100, 4)
First 5 rows:
[[1.         0.00153193 0.         0.        ]
 [0.99952358 0.00152979 0.         0.        ]
 [0.99904717 0.00148551 0.         0.        ]
 [0.99857075 0.00139935 0.         0.        ]
 [0.99809433 0.00125405 0.         0.        ]]
✓ output saved: 165_S2_1.125_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
marshalling file '165_S2_1.125'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/97334  time [yr]: 0.00 (time step: 0.00)

...


```

1.0 <--- DOESN'T break for at least 500 time steps
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

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 16 left, 16 right vertices
✅ Created 16 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_1.0'.bin
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

   FemModel initialization elapsed time:   1.35008
   Total Core solution elapsed time:       233.814
   Linear solver elapsed time:             71.772  (31%)

   Total elapsed time: 0 hrs 3 min 55 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-5.95e-06, 3.81e-06] 1/yr

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
  vx: [-0.00269, 0.01347]
  vy: [-0.01355, 0.01406]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_1.0_static.txt with shape (2100, 4)
First 5 rows:
[[1.         0.00172903 0.         0.        ]
 [0.99952358 0.00172868 0.         0.        ]
 [0.99904717 0.00165448 0.         0.        ]
 [0.99857075 0.00155883 0.         0.        ]
 [0.99809433 0.00136583 0.         0.        ]]
✓ output saved: 165_S2_1.0_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
marshalling file '165_S2_1.0'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/109501  time [yr]: 0.00 (time step: 0.00)

...

```

0.875 <--- DOESN'T break (yet) 
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

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 18 left, 18 right vertices
✅ Created 18 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_0.875'.bin
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

   FemModel initialization elapsed time:   1.15602
   Total Core solution elapsed time:       128.001
   Linear solver elapsed time:             38.604  (30%)

   Total elapsed time: 0 hrs 2 min 9 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-5.97e-06, 3.97e-06] 1/yr

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
  vx: [-0.00269, 0.01347]
  vy: [-0.01370, 0.01678]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_0.875_static.txt with shape (2100, 4)
First 5 rows:
[[1.         0.00176316 0.         0.        ]
 [0.99952358 0.00175764 0.         0.        ]
 [0.99904717 0.00169573 0.         0.        ]
 [0.99857075 0.00158223 0.         0.        ]
 [0.99809433 0.00139849 0.         0.        ]]
✓ output saved: 165_S2_0.875_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
marshalling file '165_S2_0.875'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/125143  time [yr]: 0.00 (time step: 0.00)

...


```

0.75 <--- DOESN'T break (yet) 
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

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 21 left, 21 right vertices
✅ Created 21 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_0.75'.bin
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

   FemModel initialization elapsed time:   1.13126
   Total Core solution elapsed time:       263.157
   Linear solver elapsed time:             82.5063 (31%)

   Total elapsed time: 0 hrs 4 min 24 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 2100
  x shape: (2100,)
  vx shape: (2100, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-6.01e-06, 3.99e-06] 1/yr

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
  vx: [-0.00269, 0.01348]
  vy: [-0.01618, 0.01681]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_0.75_static.txt with shape (2100, 4)
First 5 rows:
[[1.         0.0019179  0.         0.        ]
 [0.99952358 0.00191143 0.         0.        ]
 [0.99904717 0.0018472  0.         0.        ]
 [0.99857075 0.00172214 0.         0.        ]
 [0.99809433 0.00152227 0.         0.        ]]
✓ output saved: 165_S2_0.75_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
marshalling file '165_S2_0.75'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/146000  time [yr]: 0.00 (time step: 0.00)

```

0.5 <--- CRASHED
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
  hmax: 63.36
  resolution_factor: 0.5
  refinement_factor: 50
  Total vertices: 118395
  Elements: 228332
  inlet vertices: 31
  terminus vertices: 31
========================================

============ SETUP FRICTION ==============

EXPERIMENT S2: basal boundary is frozen (no slip)

============ FRICTION DIAGNOSTIC ==============

ACTUAL FRICTION:
  friction coeff range = [1, 1]
  β² (ISSM) range = [1.00e+00, 1.00e+00]
  β² (annual) range = [0, 0]
Found 31 left, 31 right vertices
✅ Created 31 flowband pairs using relative depth matching

===== Solving Diagnostic Stressbalance =====
checking model consistency
marshalling file '165_S2_0.5'.bin
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

   FemModel initialization elapsed time:   1.63512
   Total Core solution elapsed time:       566.637
   Linear solver elapsed time:             211.253 (37%)

   Total elapsed time: 0 hrs 9 min 28 sec
loading results from cluster

===== DIAGNOSING ACCELERATION =====
Debug info:
  Surface nodes: 4199
  x shape: (4199,)
  vx shape: (4199, 1)
  x range: [0.0, 210000.0] m
  vx range: [-0.003, 0.013] m/yr
  dvx_dx range: [-6.00e-06, 3.91e-06] 1/yr

=== SAVING STATIC OUTPUT ===
Available results in md.results:
  md.results.StressbalanceSolution
  md.results.checkconsistency
  md.results.marshall
  md.results.setdefaultparameters

Available results in md.results.StressbalanceSolution:
  BedSlopeX: shape (118395, 1)
  Pressure: shape (118395, 1)
  SolutionType: length 21, type <class 'str'>
  StressbalanceConvergenceNumSteps: shape (1,)
  SurfaceSlopeX: shape (118395, 1)
  Vel: shape (118395, 1)
  Vx: shape (118395, 1)
  Vy: shape (118395, 1)
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
  vx: [-0.00270, 0.01348]
  vy: [-0.02007, 0.01996]
  vz: [0.00000, 0.00000]
Basal velocity ranges (m a⁻¹):
  vx_basal: [0.00000, 0.00000]
✓ Saved 165_S2_0.5_static.txt with shape (4199, 4)
First 5 rows:
[[1.         0.00205291 0.         0.        ]
 [0.99952358 0.00204335 0.         0.        ]
 [0.99904717 0.00196916 0.         0.        ]
 [0.99857075 0.00181608 0.         0.        ]
 [0.99809433 0.00160819 0.         0.        ]]
✓ output saved: 165_S2_0.5_static.txt

=== DRIVING STRESS DIAGNOSTIC S2 ===
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
output_frequency: 50.0
isstressbalance: 1
Δt (yr): 0.0013698630136986301 Tfinal (yr): 300 ≈nsteps: 219000
checking model consistency
marshalling file '165_S2_0.5'.bin
uploading input file and queuing script
launching solution sequence on remote cluster

Ice-sheet and Sea-level System Model (ISSM) version  4.24
(website: http://issm.jpl.nasa.gov forum: https://issm.ess.uci.edu/forum/)

call computational core:
iteration 1/219000  time [yr]: 0.00 (time step: 0.00)
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...
   computing slope
   computing basal mass balance
   computing basal mass balance
   computing mass transport
   call free surface computational core
   extruding Base from base...
   extruding solution from top...
   extruding solution from top...
   extruding solution from top...
   updating vertices positions
   computing transient requested outputs
   saving temporary results
iteration 2/219001  time [yr]: 0.00 (time step: 0.00)
   computing new velocity
computing slope...
   extruding SurfaceSlopeX from base...

[0] ??? Error using ==> ./classes/Elements/TriaRef.cpp:117
[0] GetJacobianDeterminant error message: negative jacobian determinant!

--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[42669,1],0]
  Exit code:    1
--------------------------------------------------------------------------
loading results from cluster
Solving complete - saving results

```