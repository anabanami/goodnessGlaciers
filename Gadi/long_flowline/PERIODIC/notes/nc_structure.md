netcdf /home/ana/Desktop/code/Gadi/long_flowline/PERIODIC/165_S1_0.75.nc {
  group: results {
    group: TransientSolution {
      dimensions:
        time = 309;
        Vx_dim1 = 51103;
        Vx_dim2 = 1;
        Vy_dim1 = 51103;
        Vy_dim2 = 1;
        Vel_dim1 = 51103;
        Vel_dim2 = 1;
        Pressure_dim1 = 51103;
        Pressure_dim2 = 1;
        Thickness_dim1 = 51103;
        Thickness_dim2 = 1;
        Surface_dim1 = 51103;
        Surface_dim2 = 1;
        Base_dim1 = 51103;
        Base_dim2 = 1;
        SurfaceSlopeX_dim1 = 51103;
        SurfaceSlopeX_dim2 = 1;
        BedSlopeX_dim1 = 51103;
        BedSlopeX_dim2 = 1;
        SolutionType_strlen = 17;
      variables:
        double time(time=309);

        int step(time=309);

        double StressbalanceConvergenceNumSteps(time=309);

        double Vx(time=309, Vx_dim1=51103, Vx_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double Vy(time=309, Vy_dim1=51103, Vy_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double Vel(time=309, Vel_dim1=51103, Vel_dim2=1);
          :original_shape = 51103L, 1L; // long
          :issm_type = 3L; // long

        double Pressure(time=309, Pressure_dim1=51103, Pressure_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double Thickness(time=309, Thickness_dim1=51103, Thickness_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double Surface(time=309, Surface_dim1=51103, Surface_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double Base(time=309, Base_dim1=51103, Base_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double SurfaceSlopeX(time=309, SurfaceSlopeX_dim1=51103, SurfaceSlopeX_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        double BedSlopeX(time=309, BedSlopeX_dim1=51103, BedSlopeX_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 51103L, 1L; // long

        char SolutionType(time=309, SolutionType_strlen=17);

    }

  }

  // global attributes:
  :description = "Converted from ISSM .outbin format: 165_S1_0.75.outbin";
  :creation_date = "2025-08-26T09:49:33.310554";
  :source = "ISSM";
  :converter = "convert_to_nc_grouped.py";
  :original_file = "165_S1_0.75.outbin";
  :title = "ISSM Simulation Output";
}