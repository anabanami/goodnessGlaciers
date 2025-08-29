

● ISSM PETSc Matrix Allocation Error Investigation

  Problem Description

  Date: August 25, 2025System: Gadi HPCUser: ah3716

  Error Details

  - Failed Simulations: S3, S4 (with sliding)
  - Working Simulations: S1, S2 (no sliding)
  - Error Type: PETSc matrix allocation error
  - Specific Error: New nonzero at (0,82472) caused a malloc

##  Error Message

```bash
[8]PETSC ERROR: --------------------- Error Message --------------------------------------------------------------
[8]PETSC ERROR: Argument out of range
[8]PETSC ERROR: New nonzero at (0,82472) caused a malloc. Use MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check
[8]PETSC ERROR: WARNING! There are unused option(s) set! Could be the program crashed before usage or a spelling mistake, etc!
[8]PETSC ERROR:   Option left: name:-ksp_type value: preonly source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_14 value: 120 source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_28 value: 1 source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_29 value: 2 source: code
[8]PETSC ERROR:   Option left: name:-pc_factor_mat_solver_type value: mumps source: code
[8]PETSC ERROR:   Option left: name:-pc_type value: lu source: code
[8]PETSC ERROR: See https://petsc.org/release/faq/ for trouble shooting.
[8]PETSC ERROR: Petsc Release Version 3.21.1, Apr 26, 2024 
[8]PETSC ERROR: issm.exe on a  named gadi-cpu-clx-2608.gadi.nci.org.au by ah3716 Mon Aug 25 13:24:49 2025
[8]PETSC ERROR: Configure options --prefix=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n --with-ssl=0 --download-c2html=0 --download-sowing=0 --download-hwloc=0 --with-make-exec=make --with-cc=/apps/openmpi/4.1.7/bin/mpicc --with-cxx=/apps/openmpi/4.1.7/bin/mpic++ --with-fc=/apps/openmpi/4.1.7/bin/mpif90 --with-precision=double --with-scalar-type=real --with-shared-libraries=1 --with-debugging=0 --with-openmp=1 --with-64-bit-indices=0 --with-blaslapack-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/openblas-0.3.26-fb2vqjju3kd3cb6c5b5gyr4dbk527jqb/lib/libopenblas.so --with-x=0 --with-sycl=0 --with-clanguage=C --with-cuda=0 --with-hip=0 --with-metis=1 --with-metis-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/metis-5.1.0-tjf2qsrnu4ncpcjoguu7vko7qyeileif/include --with-metis-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/metis-5.1.0-tjf2qsrnu4ncpcjoguu7vko7qyeileif/lib/libmetis.so --with-hypre=1 --with-hypre-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hypre-2.31.0-v6tukbt5kiecuh6w3qehj2nt2xixtst5/include --with-hypre-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hypre-2.31.0-v6tukbt5kiecuh6w3qehj2nt2xixtst5/lib/libHYPRE.so --with-parmetis=1 --with-parmetis-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/parmetis-4.0.3-2mfcskk6gia2nz7vfnc7wivtzoj3wqwb/include --with-parmetis-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/parmetis-4.0.3-2mfcskk6gia2nz7vfnc7wivtzoj3wqwb/lib/libparmetis.so --with-kokkos=0 --with-kokkos-kernels=0 --with-superlu_dist=1 --with-superlu_dist-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/superlu-dist-8.2.1-k6pijk653j6gvdyr6xyz5jswm7tmh5u7/include --with-superlu_dist-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/superlu-dist-8.2.1-k6pijk653j6gvdyr6xyz5jswm7tmh5u7/lib/libsuperlu_dist.so --with-ptscotch=0 --with-suitesparse=0 --with-hdf5=1 --with-hdf5-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hdf5-1.14.3-2rsjdtyey7ym6i44kfvkj5bmlxfb3n2q/include --with-hdf5-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hdf5-1.14.3-2rsjdtyey7ym6i44kfvkj5bmlxfb3n2q/lib/libhdf5.so --with-zlib=0 --with-mumps=1 --with-mumps-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/include --with-mumps-lib="/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libcmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libsmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libzmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libdmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libmumps_common.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libpord.so" --with-trilinos=0 --with-fftw=0 --with-valgrind=0 --with-gmp=0 --with-libpng=0 --with-giflib=0 --with-mpfr=0 --with-netcdf=0 --with-pnetcdf=0 --with-moab=0 --with-random123=0 --with-exodusii=0 --with-cgns=0 --with-memkind=0 --with-p4est=0 --with-saws=0 --with-yaml=0 --with-hwloc=0 --with-libjpeg=0 --with-scalapack=1 --with-scalapack-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/netlib-scalapack-2.2.0-xc4yzxrjbfuufus5lkmkumvxzyllenve/lib/libscalapack.so --with-strumpack=0 --with-mmg=0 --with-parmmg=0 --with-tetgen=0 --with-zoltan=0
[8]PETSC ERROR: #1 MatSetValues_MPIAIJ() at /scratch/su58/ah3716/tmp/spack-stage/spack-stage-petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n/spack-src/src/mat/impls/aij/mpi/mpiaij.c:598
[8]PETSC ERROR: #2 MatSetValues() at /scratch/su58/ah3716/tmp/spack-stage/spack-stage-petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n/spack-src/src/mat/interface/matrix.c:1510

[8] ??? Error using ==> ./toolkits/petsc/objects/PetscMat.cpp:191
[8] SetValues error message: PETSc's MatSetValues reported an error

--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[3633,1],8]
  Exit code:    1
--------------------------------------------------------------------------
```

  Key Findings

  Mesh Analysis

  - Meshes are identical between experiments at same resolution
  - Resolution 0.75: Both S1 and S3 have 51,103 vertices
  - Resolution 1.0: Both S1 and S3 have 43,437 vertices
  - Issue is not mesh-related

  Physics Differences

  | Experiment | Sliding         | Glen's n | Basal Velocity   | Status     |
  |------------|-----------------|----------|------------------|------------|
  | S1         | No (frozen bed) | n=1      | 0.00000          | ✅ Working  |
  | S2         | No (frozen bed) | n=3      | 0.00000          | ✅ Working  |
  | S3         | Yes (β²=1500)   | n=1      | [-133.78, 69.59] | ❌ Crashing |
  | S4         | Yes (β²=1500)   | n=3      | Not reached      | ❌ Crashing |

  Root Cause

  Basal sliding changes the sparse matrix structure:
  - No-slip boundary conditions create simpler finite element coupling
  - Sliding introduces friction law equations that create new matrix entries
  - PETSc pre-allocates matrix sparsity based on physics model
  - Position (0,82472) exists in sliding cases but wasn't pre-allocated

  Timeline Context

  Recent Changes

  - Yesterday: I rebuilt ISSM using custom spack package
  - Previous experience: Non-linear sliding experiments worked fine on Gadi before
  - Key difference hypothesis: New PETSc build configuration

  Spack Package Configuration

  # From gadi's package.py
  depends_on("petsc~examples+metis+mumps+scalapack", when="~ad")

  PETSc Version: 3.21.1 with --with-debugging=0 (strict mode)

  Why This Worked Before vs. Now

  Previous Setup

  - Different PETSc version or configuration
  - More permissive memory allocation defaults
  - Possibly debugging enabled

  Current Setup

  - PETSc 3.21.1 with debugging OFF
  - Stricter matrix allocation checking

  Proposed Solutions

  Option 1: Rebuild with Permissive PETSc (Recommended)

  Modify package.py:
  depends_on("petsc~examples+metis+mumps+scalapack+debugging", when="~ad")

  Option 2: Runtime Configuration

  Status: Unknown if this syntax exists in ISSM
  # Theoretical - needs verification
  md.settings.petsc_options = ['-mat_new_nonzero_allocation_err', 'false']

  Assistant Error Analysis

  Problem with AI Responses

  Both Claude instances made the same error:
  1. Suggested solutions without verification
  2. Fabricated ISSM API syntax (e.g.,
  md.stressbalance.matrix_assembly.mat_new_nonzero_allocation_err = False)
  3. When challenged, admitted the parameter doesn't exist in ISSM codebase
  4. Presented guesses as established facts

  Lesson Learned

  - AI suggestions for specific software APIs should be treated as hypotheses requiring
  verification
  - The most reliable solution addresses the root cause (PETSc build configuration)
  rather than runtime workarounds

  Conclusion

  The issue stems from rebuilding ISSM with a stricter PETSc configuration. The sliding
  physics creates matrix sparsity patterns that the new, more restrictive PETSc build
  cannot handle. The most straightforward solution is rebuilding ISSM with more
  permissive PETSc options.



I investigated further and decided to try the fix

```python
# Conditionally apply PETSc fix only for sliding experiments
if exp in ('S3', 'S4'):
    print("Applying PETSc runtime fix for sliding simulation...")
    md.settings.petsc_options = ['-mat_new_nonzero_allocation_err', 'false']

````



```bash
[8]PETSC ERROR: --------------------- Error Message --------------------------------------------------------------
[8]PETSC ERROR: Argument out of range
[8]PETSC ERROR: New nonzero at (0,82472) caused a malloc. Use MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE) to turn off this check
[8]PETSC ERROR: WARNING! There are unused option(s) set! Could be the program crashed before usage or a spelling mistake, etc!
[8]PETSC ERROR:   Option left: name:-ksp_type value: preonly source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_14 value: 120 source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_28 value: 1 source: code
[8]PETSC ERROR:   Option left: name:-mat_mumps_icntl_29 value: 2 source: code
[8]PETSC ERROR:   Option left: name:-pc_factor_mat_solver_type value: mumps source: code
[8]PETSC ERROR:   Option left: name:-pc_type value: lu source: code
[8]PETSC ERROR: See https://petsc.org/release/faq/ for trouble shooting.
[8]PETSC ERROR: Petsc Release Version 3.21.1, Apr 26, 2024 
[8]PETSC ERROR: issm.exe on a  named gadi-cpu-clx-0683.gadi.nci.org.au by ah3716 Mon Aug 25 15:21:23 2025
[8]PETSC ERROR: Configure options --prefix=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n --with-ssl=0 --download-c2html=0 --download-sowing=0 --download-hwloc=0 --with-make-exec=make --with-cc=/apps/openmpi/4.1.7/bin/mpicc --with-cxx=/apps/openmpi/4.1.7/bin/mpic++ --with-fc=/apps/openmpi/4.1.7/bin/mpif90 --with-precision=double --with-scalar-type=real --with-shared-libraries=1 --with-debugging=0 --with-openmp=1 --with-64-bit-indices=0 --with-blaslapack-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/openblas-0.3.26-fb2vqjju3kd3cb6c5b5gyr4dbk527jqb/lib/libopenblas.so --with-x=0 --with-sycl=0 --with-clanguage=C --with-cuda=0 --with-hip=0 --with-metis=1 --with-metis-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/metis-5.1.0-tjf2qsrnu4ncpcjoguu7vko7qyeileif/include --with-metis-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/metis-5.1.0-tjf2qsrnu4ncpcjoguu7vko7qyeileif/lib/libmetis.so --with-hypre=1 --with-hypre-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hypre-2.31.0-v6tukbt5kiecuh6w3qehj2nt2xixtst5/include --with-hypre-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hypre-2.31.0-v6tukbt5kiecuh6w3qehj2nt2xixtst5/lib/libHYPRE.so --with-parmetis=1 --with-parmetis-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/parmetis-4.0.3-2mfcskk6gia2nz7vfnc7wivtzoj3wqwb/include --with-parmetis-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/parmetis-4.0.3-2mfcskk6gia2nz7vfnc7wivtzoj3wqwb/lib/libparmetis.so --with-kokkos=0 --with-kokkos-kernels=0 --with-superlu_dist=1 --with-superlu_dist-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/superlu-dist-8.2.1-k6pijk653j6gvdyr6xyz5jswm7tmh5u7/include --with-superlu_dist-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/superlu-dist-8.2.1-k6pijk653j6gvdyr6xyz5jswm7tmh5u7/lib/libsuperlu_dist.so --with-ptscotch=0 --with-suitesparse=0 --with-hdf5=1 --with-hdf5-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hdf5-1.14.3-2rsjdtyey7ym6i44kfvkj5bmlxfb3n2q/include --with-hdf5-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/hdf5-1.14.3-2rsjdtyey7ym6i44kfvkj5bmlxfb3n2q/lib/libhdf5.so --with-zlib=0 --with-mumps=1 --with-mumps-include=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/include --with-mumps-lib="/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libcmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libsmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libzmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libdmumps.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libmumps_common.so /home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/mumps-5.6.2-mw2vnxgurbcz7na67b6ti3eyysq7l75m/lib/libpord.so" --with-trilinos=0 --with-fftw=0 --with-valgrind=0 --with-gmp=0 --with-libpng=0 --with-giflib=0 --with-mpfr=0 --with-netcdf=0 --with-pnetcdf=0 --with-moab=0 --with-random123=0 --with-exodusii=0 --with-cgns=0 --with-memkind=0 --with-p4est=0 --with-saws=0 --with-yaml=0 --with-hwloc=0 --with-libjpeg=0 --with-scalapack=1 --with-scalapack-lib=/home/565/ah3716/spack/0.22/release/linux-rocky8-x86_64_v4/gcc-13.2.0/netlib-scalapack-2.2.0-xc4yzxrjbfuufus5lkmkumvxzyllenve/lib/libscalapack.so --with-strumpack=0 --with-mmg=0 --with-parmmg=0 --with-tetgen=0 --with-zoltan=0
[8]PETSC ERROR: #1 MatSetValues_MPIAIJ() at /scratch/su58/ah3716/tmp/spack-stage/spack-stage-petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n/spack-src/src/mat/impls/aij/mpi/mpiaij.c:598
[8]PETSC ERROR: #2 MatSetValues() at /scratch/su58/ah3716/tmp/spack-stage/spack-stage-petsc-3.21.1-funiamyutwedxliv7n7qpxhfz4mjvy7n/spack-src/src/mat/interface/matrix.c:1510

[8] ??? Error using ==> ./toolkits/petsc/objects/PetscMat.cpp:191
[8] SetValues error message: PETSc's MatSetValues reported an error

--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpiexec detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[10348,1],8]
  Exit code:    1
--------------------------------------------------------------------------

  ```


Gemini's fix is to change settings for transient in this block:

```python
# Conditionally apply PETSc fix only for sliding experiments
if exp in ('S3', 'S4'):
    print("Applying PETSc runtime fix for sliding simulation...")
    # The target is changed from .settings to .transient
    md.transient.petsc_options = ['-mat_new_nonzero_allocation_err', 'false']

````

that did not work after we inspected the toolkits file

we also moved this clock inside the Solve funtion I have made in periodic_flowline.py that also didnt work
The required PETSc option -mat_new_nonzero_allocation_err false is still not being written... We cannot modify this file directly in this mode it seems.

```bash
%Toolkits options file: 165_S4_1.125.toolkits written from Python toolkits array

+DefaultAnalysis
-toolkit petsc
-mat_type mpiaij
-ksp_type preonly
-pc_type lu
-pc_factor_mat_solver_type mumps
-mat_mumps_icntl_14 120
-mat_mumps_icntl_28 1
-mat_mumps_icntl_29 2

+RecoveryAnalysis
-toolkit petsc
-mat_type mpiaij
-ksp_type preonly
-pc_type lu
-pc_factor_mat_solver_type mumps
-mat_mumps_icntl_14 120
-mat_mumps_icntl_28 1
-mat_mumps_icntl_29 2

```


The chatbots have no idea what I can do to fix this.. neither do I,

I tried running sim on gadi without mpiexec in my queue file and that runs fine. 
I tried running sim on gadi with mpiexec -np 6 (matching local computer test) and that crashes.


I tried running sim in my local computer with mpiexec -np 6 and that runs fine.
I tried running sim in my local computer without mpiexec and that runs fine.


Chris's fix add this line in queue file instead of regular ana-local-version line 
spack load issm@ana-local-version-allocation-bugfix %gcc@13

