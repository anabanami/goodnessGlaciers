netcdf /home/ana/Desktop/code/ISMIP/processing/IsmipF_S1.nc {
  group: results {
    group: TransientSolution {
      dimensions:
        time = 3602;
        Vx_dim1 = 4500;
        Vx_dim2 = 1;
        Vy_dim1 = 4500;
        Vy_dim2 = 1;
        Vz_dim1 = 4500;
        Vz_dim2 = 1;
        Vel_dim1 = 4500;
        Vel_dim2 = 1;
        Pressure_dim1 = 4500;
        Pressure_dim2 = 1;
        SolutionType_strlen = 21;
        Thickness_dim1 = 4500;
        Thickness_dim2 = 1;
        Surface_dim1 = 4500;
        Surface_dim2 = 1;
        Base_dim1 = 4500;
        Base_dim2 = 1;
      variables:
        double time(time=3602);

        int step(time=3602);

        double StressbalanceConvergenceNumSteps(time=3602);

        double Vx(time=3602, Vx_dim1=4500, Vx_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 4500L, 1L; // long

        double Vy(time=3602, Vy_dim1=4500, Vy_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 4500L, 1L; // long

        double Vz(time=3602, Vz_dim1=4500, Vz_dim2=1);
          :original_shape = 4500L, 1L; // long
          :issm_type = 3L; // long

        double Vel(time=3602, Vel_dim1=4500, Vel_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 4500L, 1L; // long

        double Pressure(time=3602, Pressure_dim1=4500, Pressure_dim2=1);
          :issm_type = 3L; // long
          :original_shape = 4500L, 1L; // long

        char SolutionType(time=3602, SolutionType_strlen=21);

        double Thickness(time=3602, Thickness_dim1=4500, Thickness_dim2=1);
          :original_shape = 4500L, 1L; // long
          :issm_type = 3L; // long

        double Surface(time=3602, Surface_dim1=4500, Surface_dim2=1);
          :original_shape = 4500L, 1L; // long
          :issm_type = 3L; // long

        double Base(time=3602, Base_dim1=4500, Base_dim2=1);
          :original_shape = 4500L, 1L; // long
          :issm_type = 3L; // long

    }

  }

  // global attributes:
  :description = "Converted from ISSM .outbin format: IsmipF.outbin";
  :creation_date = "2025-09-01T21:11:07.499399";
  :source = "ISSM";
  :converter = "convert_to_nc_grouped.py";
  :original_file = "IsmipF.outbin";
  :title = "ISSM Simulation Output";
}